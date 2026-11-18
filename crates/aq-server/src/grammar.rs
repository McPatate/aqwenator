//! Constrained decoding: the output must match a JSON schema (`response_format`, and the
//! tool calls of `tool_choice` "required" or a named function).
//!
//! The grammar engine (llguidance) says which tokens may come next; the others are masked
//! out of the logits row before the request's sampler picks, so the output is valid by
//! construction and greedy or sampled as the request asked. A request that thinks first is
//! free until `</think>`; the schema applies to what follows.

use aq_engine::sample::{Picker, Sampler};
use llguidance::api::TopLevelGrammar;
use llguidance::{Matcher, ParserFactory};
use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use toktrie::{ApproximateTokEnv, TokEnv, TokRxInfo, TokTrie};

pub struct Grammars {
    /// `Err`: why constrained decoding is not available for this model.
    factory: Result<ParserFactory, String>,
    eos: Vec<u32>,
    /// Tokens a reasoning block may not contain: `<tool_call>` (and end-of-sequence).
    no_think: Vec<u32>,
    think_end: Option<u32>,
    /// The bytes of every token, by id (empty if the tokenizer is not byte-level BPE).
    pub words: Arc<Vec<Vec<u8>>>,
}

/// The bytes of every token of a byte-level BPE tokenizer.json, by id; special tokens carry
/// the grammar engine's marker byte so text can never match them.
fn token_bytes(dir: &Path) -> Result<(Vec<Vec<u8>>, Vec<(String, u32)>), String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    if v["decoder"]["type"] != "ByteLevel" {
        return Err("constrained decoding needs a byte-level BPE tokenizer".into());
    }
    // The byte-level alphabet: printable bytes stand for themselves, the rest are moved up
    // from U+0100 in order.
    let mut byte_of = std::collections::HashMap::new();
    let mut moved = 0u32;
    for b in 0..=255u32 {
        let plain = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        byte_of.insert(char::from_u32(if plain { b } else { 256 + moved }).unwrap(), b as u8);
        moved += !plain as u32;
    }
    let vocab = v["model"]["vocab"].as_object().ok_or("tokenizer.json has no model.vocab")?;
    let added = v["added_tokens"].as_array().cloned().unwrap_or_default();
    let n = vocab.values().chain(added.iter().map(|a| &a["id"])).filter_map(|i| i.as_u64()).max().ok_or("empty vocabulary")? as usize + 1;
    let mut words = vec![Vec::new(); n];
    for (s, id) in vocab {
        let id = id.as_u64().ok_or("bad token id")? as usize;
        words[id] = s.chars().map(|c| byte_of.get(&c).copied().ok_or("token outside the byte-level alphabet")).collect::<Result<_, _>>()?;
    }
    let mut names = Vec::new();
    for a in &added {
        let (Some(id), Some(s)) = (a["id"].as_u64(), a["content"].as_str()) else { continue };
        words[id as usize] = if a["special"] == true { [&[TokTrie::SPECIAL_TOKEN_MARKER], s.as_bytes()].concat() } else { s.as_bytes().to_vec() };
        names.push((s.to_string(), id as u32));
    }
    Ok((words, names))
}

impl Grammars {
    pub fn new(dir: &Path, eos: &[u32]) -> Self {
        let mut me = Self { factory: Err("no end-of-sequence token".into()), eos: eos.to_vec(), no_think: eos.to_vec(), think_end: None, words: Arc::default() };
        me.factory = token_bytes(dir).and_then(|(words, names)| {
            let id = |n: &str| names.iter().find(|x| x.0 == n).map(|x| x.1);
            // For `logprobs`: the text of a special token without the grammar engine's marker.
            me.words = Arc::new(words.iter().map(|w| w.strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER]).unwrap_or(w).to_vec()).collect());
            me.think_end = id("</think>");
            me.no_think.extend(id("<tool_call>"));
            // The grammar ends the output with the last end-of-sequence id (the chat one).
            let info = TokRxInfo::new(words.len() as u32, *eos.last().ok_or("no end-of-sequence token")?);
            let env: TokEnv = Arc::new(ApproximateTokEnv::new(TokTrie::from(&info, &words)));
            let mut f = ParserFactory::new_simple(&env).map_err(|e| e.to_string())?;
            f.quiet();
            Ok(f)
        });
        if let Err(e) = &me.factory {
            eprintln!("constrained decoding is off: {e}");
        }
        me
    }

    /// A picker whose output matches `schema`. `thinking`: the output starts inside a
    /// reasoning block. Fails on a schema the grammar engine cannot compile.
    pub fn constrain(&self, schema: &Value, thinking: bool, sampler: Sampler) -> Result<Box<dyn Picker>, String> {
        let factory = self.factory.as_ref().map_err(|e| e.clone())?;
        let mut m = Matcher::new(factory.create_parser(TopLevelGrammar::from_json_schema(schema.clone())));
        if let Some(e) = m.get_error() {
            return Err(e);
        }
        // Compiles the first mask: a schema too large for the engine's limits fails here.
        m.compute_mask().map_err(|e| e.to_string())?;
        // (No `</think>` token: nothing can end a reasoning block, so there is none.)
        let thinking = thinking && self.think_end.is_some();
        Ok(Box::new(Constrained { m, sampler, thinking, think_end: self.think_end, no_think: self.no_think.clone(), eos: self.eos.last().copied().unwrap_or(0) }))
    }
}

struct Constrained {
    m: Matcher,
    sampler: Sampler,
    thinking: bool,
    think_end: Option<u32>,
    no_think: Vec<u32>,
    eos: u32,
}

impl Picker for Constrained {
    fn pick(&mut self, logits: &mut [f32]) -> u32 {
        if self.thinking {
            for &t in &self.no_think {
                logits[t as usize] = f32::NEG_INFINITY;
            }
            let t = self.sampler.pick(logits);
            self.thinking = Some(t) != self.think_end;
            return t;
        }
        // Once the grammar is complete the only token left is end-of-sequence.
        let Ok(mask) = self.m.compute_mask_or_eos() else { return self.eos };
        for (i, l) in logits.iter_mut().enumerate() {
            if i >= mask.len() || !mask.is_allowed(i as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        let t = self.sampler.pick(logits);
        if self.m.consume_token(t).is_err() { self.eos } else { t }
    }
}
