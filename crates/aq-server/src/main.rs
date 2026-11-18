//! aq-serve: OpenAI-compatible front end for the aqwenator engine.
//!
//! The engine runs on its own OS thread, which owns the CUDA context and loops
//! `step()` while there is work. HTTP handlers (tokio) submit requests over a channel and
//! receive token events over a per-request channel; detokenization happens on the tokio
//! side so the engine thread never touches text.
//!
//! Endpoints: `/v1/completions` (prompt as text or token ids), `/v1/chat/completions` (the
//! model's own chat template, reasoning split, tool calls: see `chat.rs`), `/v1/models`,
//! `/health` (503 once the engine thread has ended), `/metrics` (Prometheus text).
//!
//! Request fields, both endpoints: `max_tokens` (default: what is left of the context),
//! `temperature`, `top_p`, `top_k` (defaults: see `default_sampling`), `seed`, `stop`, `stop_token_ids`,
//! `ignore_eos`, `skip_special_tokens`, `stream`, `stream_options.include_usage`,
//! `response_format` (`json_object`, `json_schema`: constrained decoding, see `grammar.rs`).
//! `logprobs` / `top_logprobs`, `logit_bias`, `presence_penalty`, `frequency_penalty`,
//! `repetition_penalty`, `min_tokens` (see `shape.rs`).
//! A field the server does not know, or knows and cannot honour (`n` > 1, `best_of`, `echo`,
//! `suffix`, `min_p`, ...), is a 400 that names it. The OpenAI fields that change nothing on a
//! server like this one (`user`, `store`, `metadata`, `service_tier`, `prompt_cache_key`,
//! `safety_identifier`) are accepted and ignored, and `reasoning_effort` only switches the
//! template's thinking ("none": off, any other level: on); `--strict-requests true` makes
//! all of these a 400 too, so that nothing a client sends is dropped unseen.
//!
//! Sampling: `temperature` 0 is the engine's greedy path, token-identical to what it always
//! was (and what a request without `temperature` gets under `--default-temperature 0`). With a temperature the token is drawn on the host from the logits row
//! (`aq_engine::sample`); `seed` makes a request reproducible as far as its logits are (they
//! can move by a rounding step with what else shares the batch). Speculative decoding stays
//! on and exact for sampled requests; fewer drafts are accepted as the temperature rises.
//!
//!     aq-serve <model> [--host 127.0.0.1] [--port 8000] [--served-model-name NAME]
//!              [--max-seqs 64] [--max-ctx 32768] [--max-tokens 8192] [--mixed-prefill N]
//!              [--max-running N] [--session-cache true|false] [--kv-gb X] [--kv-reserve N]
//!              [--spec-k N] [--vllm-metrics true|false] [--cache-slots N]
//!              [--default-temperature T] [--strict-requests true|false]
//!
//! `--cache-slots`: state slots for the session cache beyond `--max-seqs` (parked
//! conversations, prompt checkpoints, shared-prefix snapshots); each costs the recurrent state
//! only, no KV pages.
//!
//! `--host`: address to listen on (default 127.0.0.1, or `AQ_HOST`; 0.0.0.0 for every
//! interface).
//!
//! `--spec-k` (default 4, 0 = off): exact speculative decoding with prompt-lookup drafts;
//! outputs are token-identical to plain greedy decoding.
//!
//! `--max-seqs` is the number of state slots (running + parked sessions); `--max-running`
//! caps how many run at once.
//!
//! `--kv-reserve N` (default 4096, 0 = the whole token limit): output tokens whose KV pages
//! a request holds from admission on. Beyond that it takes pages as it generates. If the pool
//! runs dry it waits for the pages of the next request to end (new requests queue behind
//! it); only when every running request is out of pages does one end there, with
//! `finish_reason` "length" (`aq_requests_cut_total`), which hands its pages on. Without
//! this, every request with no `max_tokens` would hold a full context of pages, and a pool
//! smaller than `max-running` full contexts would run that many fewer requests at once.

mod chat;
mod grammar;
mod shape;
mod text;

use anyhow::{Context, Result};
use aq_engine::engine::{Control, Engine, Request, TokenEvent};
use aq_engine::model::{Limits, Model};
use aq_engine::sample::{Picker, Sampler, Sampling};
use aq_engine::weights::resolve_model;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tokenizers::tokenizer::pipeline::{EncodeOptions, PipelineDecodeStream, PipelineTokenizer};
use tokio::sync::mpsc;

enum Msg {
    /// The engine took the request (it may still wait for a slot).
    Accepted,
    Token(TokenEvent),
    /// The engine refused the request (too long for the context or the KV pool).
    Refused(String),
    /// The engine failed while the request was running.
    Failed(String),
    /// The request ended while it waited for KV pages (`--kv-reserve`): finish "length".
    Cut,
}

struct Submit {
    req: Request,
    ctl: Control,
    tx: mpsc::UnboundedSender<Msg>,
}

#[derive(Default)]
struct Metrics {
    /// Requests submitted and not finished (running or waiting).
    open: AtomicUsize,
    /// Sequences the engine is running (published by the engine thread).
    running: AtomicUsize,
    requests: AtomicU64,
    cut: AtomicU64,
    prompt_tokens: AtomicU64,
    cached_tokens: AtomicU64,
    generation_tokens: AtomicU64,
}

pub struct App {
    submit: std::sync::mpsc::Sender<Submit>,
    pub tok: Arc<PipelineTokenizer>,
    next_id: AtomicU64,
    pub model: String,
    /// False once the engine thread has ended (the process exits right after).
    alive: Arc<AtomicBool>,
    /// End-of-sequence token ids (config.json, generation_config.json, the tokenizer's eos).
    eos: Vec<u32>,
    max_ctx: usize,
    metrics: Arc<Metrics>,
    vllm_metrics: bool,
    pub chat: Option<chat::Template>,
    grammar: grammar::Grammars,
    /// Sampling of a request that does not say (`default_sampling`); the seed is unused.
    defaults: Sampling,
    /// `--strict-requests`: refuse the fields that are otherwise accepted and ignored.
    strict: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Prompt {
    Ids(Vec<u32>),
    Text(String),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamOptions {
    include_usage: Option<bool>,
    continuous_usage_stats: Option<bool>,
}

/// A request body of either endpoint. Unknown fields are an error (`deny_unknown_fields`);
/// `refusal` rejects the fields of the other endpoint and the known ones this server cannot
/// honour.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Body {
    model: Option<String>,
    // /v1/completions
    prompt: Option<Prompt>,
    return_token_ids: Option<bool>,
    // /v1/chat/completions
    pub messages: Option<Vec<Value>>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub chat_template_kwargs: Option<serde_json::Map<String, Value>>,
    max_completion_tokens: Option<usize>,
    // both
    pub response_format: Option<Value>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    seed: Option<i64>,
    stop: Option<Stop>,
    stop_token_ids: Option<Vec<u32>>,
    pub stream: Option<bool>,
    stream_options: Option<StreamOptions>,
    ignore_eos: Option<bool>,
    pub skip_special_tokens: Option<bool>,
    // Logit shaping and log-probabilities (see `shape.rs`).
    /// Chat: true or false, with `top_logprobs` (0 to 20). Completions: the number of most
    /// likely tokens to report per position (0 to 20).
    logprobs: Option<Value>,
    top_logprobs: Option<u64>,
    /// Token id (as a string, the OpenAI way) to a bias in [-100, 100].
    logit_bias: Option<HashMap<String, f32>>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    min_tokens: Option<usize>,
    // OpenAI fields that change nothing here (bookkeeping on the provider's side): accepted
    // and ignored, or refused under `--strict-requests` (see `ignored`).
    user: Option<Value>,
    store: Option<Value>,
    metadata: Option<Value>,
    service_tier: Option<Value>,
    prompt_cache_key: Option<Value>,
    safety_identifier: Option<Value>,
    /// Chat: "none" turns the template's thinking off, any other level on, unless
    /// `chat_template_kwargs.enable_thinking` says; ignored by templates without the switch.
    pub reasoning_effort: Option<String>,
    // Known OpenAI and vLLM fields this server cannot honour: accepted only with a value
    // that asks for nothing (see `refusal`).
    n: Option<Value>,
    best_of: Option<Value>,
    min_p: Option<Value>,
    echo: Option<Value>,
    suffix: Option<Value>,
    add_generation_prompt: Option<Value>,
    continue_final_message: Option<Value>,
    include_stop_str_in_output: Option<Value>,
    truncate_prompt_tokens: Option<Value>,
    spaces_between_special_tokens: Option<Value>,
}

impl Body {
    /// Why this request cannot be served as asked: (field, message).
    fn refusal(&self, chat: bool) -> Option<(&'static str, String)> {
        // (field, value, the values that ask for nothing, besides null).
        let neutral: [(&'static str, &Option<Value>, &[Value]); 10] = [
            ("n", &self.n, &[json!(1)]),
            ("best_of", &self.best_of, &[json!(1)]),
            ("min_p", &self.min_p, &[json!(0)]),
            ("echo", &self.echo, &[json!(false)]),
            ("suffix", &self.suffix, &[]),
            ("add_generation_prompt", &self.add_generation_prompt, &[json!(true)]),
            ("continue_final_message", &self.continue_final_message, &[json!(false)]),
            ("include_stop_str_in_output", &self.include_stop_str_in_output, &[json!(false)]),
            ("truncate_prompt_tokens", &self.truncate_prompt_tokens, &[]),
            ("spaces_between_special_tokens", &self.spaces_between_special_tokens, &[json!(true)]),
        ];
        let same = |a: &Value, b: &Value| a == b || matches!((a.as_f64(), b.as_f64()), (Some(x), Some(y)) if x == y);
        for (name, v, ok) in neutral {
            if let Some(v) = v.as_ref().filter(|v| !v.is_null() && !ok.iter().any(|o| same(v, o))) {
                return Some((name, format!("`{name}` = {v} is not supported by this server")));
            }
        }
        let foreign: &[(&'static str, bool)] = if chat {
            &[("prompt", self.prompt.is_some()), ("return_token_ids", self.return_token_ids.is_some())]
        } else {
            &[
                ("messages", self.messages.is_some()),
                ("tools", self.tools.is_some()),
                ("tool_choice", self.tool_choice.is_some()),
                ("parallel_tool_calls", self.parallel_tool_calls.is_some()),
                ("chat_template_kwargs", self.chat_template_kwargs.is_some()),
                ("max_completion_tokens", self.max_completion_tokens.is_some()),
                ("reasoning_effort", self.reasoning_effort.is_some()),
            ]
        };
        if let Some((name, _)) = foreign.iter().find(|f| f.1) {
            return Some((name, format!("`{name}` is not a field of this endpoint")));
        }
        if self.reasoning_effort.as_deref().is_some_and(|e| !["none", "minimal", "low", "medium", "high", "xhigh"].contains(&e)) {
            return Some(("reasoning_effort", "`reasoning_effort` must be none, minimal, low, medium, high or xhigh".into()));
        }
        if self.stream_options.as_ref().is_some_and(|o| o.continuous_usage_stats == Some(true)) {
            return Some(("stream_options", "`stream_options.continuous_usage_stats` is not supported by this server".into()));
        }
        if self.stream_options.is_some() && self.stream != Some(true) {
            return Some(("stream_options", "`stream_options` needs `stream`: true".into()));
        }
        if self.temperature.is_some_and(|t| !(0.0..=100.0).contains(&t)) {
            return Some(("temperature", "`temperature` must be 0 or more".into()));
        }
        if self.top_p.is_some_and(|p| !(p > 0.0 && p <= 1.0)) {
            return Some(("top_p", "`top_p` must be in (0, 1]".into()));
        }
        if self.top_k.is_some_and(|k| k < -1) {
            return Some(("top_k", "`top_k` must be -1 or 0 (off), or positive".into()));
        }
        if self.max_tokens.or(self.max_completion_tokens) == Some(0) {
            return Some(("max_tokens", "`max_tokens` must be at least 1".into()));
        }
        for (name, v) in [("presence_penalty", self.presence_penalty), ("frequency_penalty", self.frequency_penalty)] {
            if v.is_some_and(|x| !(-2.0..=2.0).contains(&x)) {
                return Some((name, format!("`{name}` must be in [-2, 2]")));
            }
        }
        if self.repetition_penalty.is_some_and(|x| !(x > 0.0 && x.is_finite())) {
            return Some(("repetition_penalty", "`repetition_penalty` must be positive".into()));
        }
        if self.logit_bias.iter().flatten().any(|(k, b)| k.parse::<u32>().is_err() || !(-100.0..=100.0).contains(b)) {
            return Some(("logit_bias", "`logit_bias` maps token ids to a bias in [-100, 100]".into()));
        }
        if self.min_tokens.unwrap_or(0) > self.max_tokens.or(self.max_completion_tokens).unwrap_or(usize::MAX) {
            return Some(("min_tokens", "`min_tokens` is more than `max_tokens`".into()));
        }
        let shown = match (&self.logprobs, chat) {
            (None | Some(Value::Null), _) => Ok(None),
            (Some(Value::Bool(b)), true) => Ok(b.then_some(self.top_logprobs.unwrap_or(0))),
            (Some(Value::Bool(false)), false) => Ok(None),
            (Some(Value::Number(n)), false) => n.as_u64().ok_or(()).map(Some),
            _ => Err(()),
        };
        if !matches!(shown, Ok(None | Some(0..=20))) {
            let how = if chat { "true or false, with `top_logprobs` from 0 to 20" } else { "a number from 0 to 20" };
            return Some(("logprobs", format!("`logprobs` must be {how}")));
        }
        if self.top_logprobs.is_some_and(|k| k > 0) && (!chat || shown == Ok(None)) {
            return Some(("top_logprobs", "`top_logprobs` needs `logprobs`: true (chat completions)".into()));
        }
        None
    }

    pub fn logprobs_asked(&self) -> bool {
        self.top_logprobs().is_some()
    }

    /// How many most likely tokens to report per position, if log-probabilities are asked for.
    fn top_logprobs(&self) -> Option<usize> {
        match &self.logprobs {
            Some(Value::Bool(true)) => Some(self.top_logprobs.unwrap_or(0) as usize),
            Some(Value::Number(n)) => n.as_u64().map(|n| n as usize),
            _ => None,
        }
    }

    /// The first field present that this server accepts and ignores.
    fn ignored(&self) -> Option<&'static str> {
        [("user", &self.user), ("store", &self.store), ("metadata", &self.metadata), ("service_tier", &self.service_tier),
         ("prompt_cache_key", &self.prompt_cache_key), ("safety_identifier", &self.safety_identifier)]
            .into_iter()
            .find(|f| f.1.as_ref().is_some_and(|v| !v.is_null()))
            .map(|f| f.0)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options.as_ref().is_some_and(|o| o.include_usage == Some(true))
    }

    /// The JSON schema `response_format` constrains the output to, if it asks for one.
    pub fn format_schema(&self) -> Result<Option<Value>, Response> {
        let Some(f) = self.response_format.as_ref().filter(|f| !f.is_null()) else { return Ok(None) };
        match f["type"].as_str() {
            Some("text") => Ok(None),
            Some("json_object") => Ok(Some(json!({"type": "object"}))),
            Some("json_schema") if f["json_schema"]["schema"].is_object() => Ok(Some(f["json_schema"]["schema"].clone())),
            Some("json_schema") => Err(bad(Some("response_format"), "`response_format.json_schema.schema` must be a JSON schema object")),
            _ => Err(bad(Some("response_format"), "`response_format.type` must be text, json_object or json_schema")),
        }
    }
}

pub fn error(status: StatusCode, kind: &str, param: Option<&str>, msg: impl Into<String>) -> Response {
    (status, Json(json!({"error": {"message": msg.into(), "type": kind, "param": param, "code": status.as_u16()}}))).into_response()
}

pub fn bad(param: Option<&str>, msg: impl Into<String>) -> Response {
    error(StatusCode::BAD_REQUEST, "invalid_request_error", param, msg)
}

/// A generation that ended in an error instead of a finish reason.
pub enum Fail {
    /// The request cannot be served (400).
    Refused(String),
    /// The engine failed or is gone (500).
    Engine(String),
}

impl Fail {
    pub fn response(&self) -> Response {
        match self {
            Fail::Refused(m) => bad(None, m.clone()),
            Fail::Engine(m) => error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", None, m.clone()),
        }
    }
    pub fn event(&self) -> String {
        let (Fail::Refused(m) | Fail::Engine(m)) = self;
        json!({"error": {"message": m, "type": if matches!(self, Fail::Refused(_)) { "invalid_request_error" } else { "internal_error" }}}).to_string()
    }
}

pub struct Finish {
    /// "stop" or "length".
    pub reason: &'static str,
    /// The stop string or the `stop_token_ids` entry that ended the request (null for
    /// end-of-sequence and for the token limit).
    pub stop_reason: Value,
}

/// A step of a response: new text (possibly empty: held back, or a token that is not text
/// yet), the token ids behind it, and the finish once the generation is over.
pub struct Piece {
    pub text: String,
    pub ids: Vec<u32>,
    /// One per id, when the request asked for log-probabilities.
    pub logprobs: Vec<shape::LogProb>,
    pub finish: Option<Finish>,
}

/// One running generation, from the handler's side: token events in, text pieces out.
/// Dropping it cancels the request in the engine (its channel closes).
pub struct Output {
    rx: mpsc::UnboundedReceiver<Msg>,
    tok: Arc<PipelineTokenizer>,
    stream: PipelineDecodeStream,
    stops: text::StopText,
    eos: Vec<u32>,
    metrics: Arc<Metrics>,
    pub id: u64,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub completion_tokens: usize,
    done: bool,
    /// Log-probabilities of the tokens picked so far, in output order (`shape::Shaper`).
    probs: Option<shape::Queue>,
    words: Arc<Vec<Vec<u8>>>,
}

impl Output {
    /// A token as `logprobs` shows it: its text (lossy if it is part of a character) and bytes.
    fn word(&self, id: u32) -> (String, &[u8]) {
        let b = self.words.get(id as usize).map_or(&[][..], |w| &w[..]);
        (String::from_utf8_lossy(b).into_owned(), b)
    }

    /// `logprobs.content` entries of a chat completion for the tokens of `p`.
    pub fn chat_logprobs(&self, p: &Piece) -> Vec<Value> {
        let entry = |id: u32, lp: f32| {
            let (token, bytes) = self.word(id);
            json!({"token": token, "logprob": lp, "bytes": bytes})
        };
        p.ids.iter().zip(&p.logprobs).map(|(&id, l)| {
            let mut e = entry(id, l.logprob);
            e["top_logprobs"] = l.top.iter().map(|&(t, lp)| entry(t, lp)).collect();
            e
        }).collect()
    }

    /// Adds the tokens of `p` to the `logprobs` object of a text completion.
    fn text_logprobs(&self, p: &Piece, into: &mut Value, offset: &mut usize) {
        for (&id, l) in p.ids.iter().zip(&p.logprobs) {
            let (token, _) = self.word(id);
            let top: serde_json::Map<String, Value> = l.top.iter().map(|&(t, lp)| (self.word(t).0, json!(lp))).collect();
            let len = token.len();
            for (key, v) in [("tokens", json!(token)), ("token_logprobs", json!(l.logprob)), ("top_logprobs", json!(top)), ("text_offset", json!(*offset))] {
                into[key].as_array_mut().unwrap().push(v);
            }
            *offset += len;
        }
    }

    /// The next piece, or the error that ended the generation. Not to be called after a
    /// piece with a finish.
    pub async fn next(&mut self) -> Result<Piece, Fail> {
        loop {
            let e = match self.rx.recv().await {
                Some(Msg::Token(e)) => e,
                Some(Msg::Accepted) => continue,
                Some(Msg::Cut) => {
                    self.done = true;
                    self.rx.close();
                    return Ok(Piece { text: self.stops.flush(), ids: Vec::new(), logprobs: Vec::new(), finish: Some(Finish { reason: "length", stop_reason: Value::Null }) });
                }
                Some(Msg::Refused(m)) => return Err(Fail::Refused(m)),
                Some(Msg::Failed(m)) => return Err(Fail::Engine(m)),
                // The channel closed before the request finished: the engine thread is gone.
                None => return Err(Fail::Engine("the engine stopped".into())),
            };
            self.completion_tokens += 1;
            self.cached_tokens = e.cached;
            let logprobs = self.probs.as_ref().and_then(|q| q.lock().unwrap().pop_front()).into_iter().collect();
            let mut piece = Piece { text: String::new(), ids: vec![e.token], logprobs, finish: None };
            if e.stop {
                // An end-of-sequence or stop token is not text.
                piece.text = self.stops.flush();
                let stop_reason = if self.eos.contains(&e.token) { Value::Null } else { json!(e.token) };
                piece.finish = Some(Finish { reason: "stop", stop_reason });
            } else {
                let new = self.stream.step(&self.tok, &[e.token]).ok().flatten().unwrap_or_default();
                let (text, hit) = self.stops.push(&new);
                piece.text = text;
                if let Some(s) = hit {
                    piece.finish = Some(Finish { reason: "stop", stop_reason: json!(s) });
                } else if e.finished {
                    piece.text.push_str(&self.stops.flush());
                    piece.finish = Some(Finish { reason: "length", stop_reason: Value::Null });
                }
            }
            if piece.finish.is_some() {
                self.done = true;
                // Closes the channel: after a stop string the engine must not go on.
                self.rx.close();
            }
            return Ok(piece);
        }
    }

    pub fn usage(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
        })
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        let m = &self.metrics;
        m.open.fetch_sub(1, Ordering::Relaxed);
        m.requests.fetch_add(self.done as u64, Ordering::Relaxed);
        m.prompt_tokens.fetch_add(self.prompt_tokens as u64, Ordering::Relaxed);
        m.cached_tokens.fetch_add(self.cached_tokens as u64, Ordering::Relaxed);
        m.generation_tokens.fetch_add(self.completion_tokens as u64, Ordering::Relaxed);
    }
}

impl App {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let opts = EncodeOptions { add_special_tokens: false, ..Default::default() };
        match self.tok.encode(text, &opts).wait() {
            Ok(encs) => Ok(encs.first().map(|e| e.ids().iter().map(|&t| u32::from(t)).collect()).unwrap_or_default()),
            Err(e) => Err(format!("tokenization failed: {e}")),
        }
    }

    /// Parse and vet a request body.
    pub fn body(&self, raw: &[u8], chat: bool) -> Result<Body, Response> {
        // serde names an unknown field in its message ("unknown field `x`, expected one of ...").
        let body: Body = serde_json::from_slice(raw).map_err(|e| bad(None, format!("invalid request: {e}")))?;
        if let Some((param, msg)) = body.refusal(chat) {
            return Err(bad(Some(param), msg));
        }
        if let Some(name) = body.ignored().or(body.reasoning_effort.as_ref().map(|_| "reasoning_effort")).filter(|_| self.strict) {
            return Err(bad(Some(name), format!("`{name}` has no effect on this server and --strict-requests is on")));
        }
        if let Some(m) = body.model.as_ref().filter(|m| **m != self.model) {
            return Err(error(StatusCode::NOT_FOUND, "not_found_error", Some("model"), format!("the model `{m}` does not exist; this server serves `{}`", self.model)));
        }
        Ok(body)
    }

    /// Submit a generation and wait until the engine has taken or refused it, so a refusal
    /// is an HTTP status rather than an event in a 200 stream. `constraint`: (JSON schema the
    /// output must match, whether the output starts inside a reasoning block; the schema then
    /// applies after it) with the field that asked for it.
    pub async fn start(&self, body: &Body, prompt: Vec<u32>, skip_special: bool, constraint: Option<(Value, bool, &str)>) -> Result<Output, Response> {
        if prompt.is_empty() {
            return Err(bad(Some(if body.messages.is_some() { "messages" } else { "prompt" }), "empty prompt"));
        }
        if prompt.len() >= self.max_ctx {
            return Err(bad(None, format!("the prompt has {} tokens and the context is {} (--max-ctx): no room to generate", prompt.len(), self.max_ctx)));
        }
        let max_new_tokens = body.max_tokens.or(body.max_completion_tokens).unwrap_or(self.max_ctx - prompt.len());
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        // No seed: a fresh one per request.
        let seed = body.seed.map(|s| s as u64).unwrap_or_else(|| {
            use std::hash::{BuildHasher, Hasher};
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(id);
            h.finish()
        });
        let d = self.defaults;
        let sampling = Sampling {
            temperature: body.temperature.unwrap_or(d.temperature),
            top_p: body.top_p.unwrap_or(d.top_p),
            top_k: body.top_k.map_or(d.top_k, |k| k.max(0) as usize),
            seed,
        };
        // Greedy and unconstrained: no picker, the device decides (the engine's own path).
        let mut pick: Option<Box<dyn Picker>> = match constraint {
            Some((schema, thinking, field)) => Some(self.grammar.constrain(&schema, thinking, Sampler::new(sampling)).map_err(|e| bad(Some(field), format!("`{field}`: {e}")))?),
            None if sampling.temperature > 0.0 => Some(Box::new(Sampler::new(sampling))),
            None => None,
        };
        let stop_ids = body.stop_token_ids.clone().unwrap_or_default();
        let probs = body.top_logprobs().map(|k| (k, shape::Queue::default()));
        if probs.is_some() && self.grammar.words.is_empty() {
            return Err(bad(Some("logprobs"), "`logprobs` needs a byte-level BPE tokenizer"));
        }
        let repetition = body.repetition_penalty.unwrap_or(1.0);
        let shape = shape::Shape {
            bias: body.logit_bias.iter().flatten().map(|(k, &b)| (k.parse().unwrap_or(u32::MAX), b)).collect(),
            presence: body.presence_penalty.unwrap_or(0.0),
            frequency: body.frequency_penalty.unwrap_or(0.0),
            repetition,
            prompt: if repetition != 1.0 { prompt.iter().copied().collect() } else { Default::default() },
            min_tokens: body.min_tokens.unwrap_or(0),
            hold: self.eos.iter().chain(&stop_ids).copied().collect(),
            top: probs.clone(),
        };
        // Any of these needs the logits row on the host, also for a greedy request.
        if !shape.bias.is_empty() || shape.presence != 0.0 || shape.frequency != 0.0 || repetition != 1.0 || shape.min_tokens > 0 || probs.is_some() {
            let inner = pick.unwrap_or_else(|| Box::new(Sampler::new(sampling)));
            pick = Some(Box::new(shape::Shaper::new(shape, inner)));
        }
        let ctl = Control { pick, stop_ids };
        let stops = match &body.stop {
            None => Vec::new(),
            Some(Stop::One(s)) => vec![s.clone()],
            Some(Stop::Many(v)) => v.clone(),
        };
        let prompt_tokens = prompt.len();
        let req = Request { id, prompt, max_new_tokens, ignore_eos: body.ignore_eos.unwrap_or(false) };
        let (tx, rx) = mpsc::unbounded_channel();
        if self.submit.send(Submit { req, ctl, tx }).is_err() {
            return Err(Fail::Engine("the engine stopped".into()).response());
        }
        self.metrics.open.fetch_add(1, Ordering::Relaxed);
        let mut out = Output {
            rx,
            tok: self.tok.clone(),
            stream: PipelineDecodeStream::new(Vec::new(), skip_special),
            stops: text::StopText::new(stops),
            eos: self.eos.clone(),
            metrics: self.metrics.clone(),
            id,
            prompt_tokens,
            cached_tokens: 0,
            completion_tokens: 0,
            done: false,
            probs: probs.map(|p| p.1),
            words: self.grammar.words.clone(),
        };
        match out.rx.recv().await {
            Some(Msg::Accepted) => Ok(out),
            Some(Msg::Refused(m)) => Err(Fail::Refused(m).response()),
            Some(Msg::Failed(m)) => Err(Fail::Engine(m).response()),
            _ => Err(Fail::Engine("the engine stopped".into()).response()),
        }
    }
}

/// An SSE response: `chunks` turns each piece (or the failure) into the JSON chunks to send;
/// `[DONE]` follows the last one.
pub fn sse<F>(out: Output, chunks: F) -> Response
where
    F: FnMut(&Output, Result<Piece, Fail>) -> Vec<String> + Send + 'static,
{
    let stream = futures::stream::unfold(Some((out, chunks)), |st| async move {
        let (mut out, mut chunks) = st?;
        let piece = out.next().await;
        let last = !matches!(&piece, Ok(p) if p.finish.is_none());
        let mut events = chunks(&out, piece);
        if last {
            events.push("[DONE]".into());
        }
        let events = futures::stream::iter(events.into_iter().map(|d| Ok::<_, Infallible>(Event::default().data(d))));
        Some((events, (!last).then_some((out, chunks))))
    })
    .flatten();
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
}

pub fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

async fn completions(State(app): State<Arc<App>>, raw: Bytes) -> Response {
    let body = match app.body(&raw, false) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let schema = match body.format_schema() {
        Ok(s) => s,
        Err(r) => return r,
    };
    let prompt = match &body.prompt {
        Some(Prompt::Ids(ids)) => ids.clone(),
        Some(Prompt::Text(text)) => match app.encode(text) {
            Ok(ids) => ids,
            Err(e) => return bad(Some("prompt"), e),
        },
        None => return bad(Some("prompt"), "`prompt` is required"),
    };
    // Completions return special tokens as text unless asked not to (as this server always did).
    let skip = body.skip_special_tokens.unwrap_or(false);
    let mut out = match app.start(&body, prompt, skip, schema.map(|s| (s, false, "response_format"))).await {
        Ok(o) => o,
        Err(r) => return r,
    };
    let with_ids = body.return_token_ids.unwrap_or(false);
    let (created, rid, model) = (now(), format!("cmpl-{}", out.id), app.model.clone());
    let with_probs = body.top_logprobs().is_some();
    let probs = move || if with_probs { json!({"tokens": [], "token_logprobs": [], "top_logprobs": [], "text_offset": []}) } else { Value::Null };
    let choice = move |text: &str, ids: &[u32], logprobs: Value, finish: Option<&Finish>| {
        let mut c = json!({"index": 0, "text": text, "logprobs": logprobs, "finish_reason": finish.map(|f| f.reason), "stop_reason": finish.map(|f| f.stop_reason.clone())});
        if with_ids {
            c["token_ids"] = json!(ids);
        }
        c
    };

    if body.stream.unwrap_or(false) {
        let usage = body.include_usage();
        let mut offset = 0;
        return sse(out, move |out, piece| match piece {
            Err(f) => vec![f.event()],
            Ok(p) => {
                let base = json!({"id": rid, "object": "text_completion", "created": created, "model": model});
                let mut c = base.clone();
                let mut lp = probs();
                if with_probs {
                    out.text_logprobs(&p, &mut lp, &mut offset);
                }
                c["choices"] = json!([choice(&p.text, &p.ids, lp, p.finish.as_ref())]);
                let mut v = vec![c.to_string()];
                if p.finish.is_some() && usage {
                    let mut u = base;
                    u["choices"] = json!([]);
                    u["usage"] = out.usage();
                    v.push(u.to_string());
                }
                v
            }
        });
    }

    let (mut text, mut ids, mut lp, mut offset) = (String::new(), Vec::new(), probs(), 0);
    loop {
        match out.next().await {
            Err(f) => return f.response(),
            Ok(p) => {
                if with_probs {
                    out.text_logprobs(&p, &mut lp, &mut offset);
                }
                text.push_str(&p.text);
                ids.extend(p.ids);
                if let Some(f) = p.finish {
                    return Json(json!({"id": rid, "object": "text_completion", "created": created, "model": model,
                        "choices": [choice(&text, &ids, lp, Some(&f))], "usage": out.usage()}))
                    .into_response();
                }
            }
        }
    }
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({"object": "list", "data": [{"id": app.model, "object": "model", "owned_by": "aqwenator", "max_model_len": app.max_ctx}]}))
}

/// 200 while the engine thread runs, 503 once it has ended.
async fn health(State(app): State<Arc<App>>) -> Response {
    if app.alive.load(Ordering::SeqCst) { "ok".into_response() } else { (StatusCode::SERVICE_UNAVAILABLE, "engine stopped").into_response() }
}

/// Prometheus text format. With `--vllm-metrics true` the gauges and token counters are also
/// exposed under vLLM's names, for tooling that reads those.
async fn metrics_text(State(app): State<Arc<App>>) -> Response {
    let m = &app.metrics;
    let running = m.running.load(Ordering::Relaxed);
    let waiting = m.open.load(Ordering::Relaxed).saturating_sub(running);
    let rows: [(&str, &str, &str, u64, Option<&str>); 7] = [
        ("aq_requests_running", "gauge", "Requests the engine is running.", running as u64, Some("vllm:num_requests_running")),
        ("aq_requests_waiting", "gauge", "Requests waiting for a slot or for KV pages.", waiting as u64, Some("vllm:num_requests_waiting")),
        ("aq_requests_total", "counter", "Requests that finished (stop or length).", m.requests.load(Ordering::Relaxed), None),
        ("aq_requests_cut_total", "counter", "Requests ended early because the KV pool had no page left (--kv-reserve).", m.cut.load(Ordering::Relaxed), None),
        ("aq_prompt_tokens_total", "counter", "Prompt tokens of the requests served.", m.prompt_tokens.load(Ordering::Relaxed), Some("vllm:prompt_tokens_total")),
        ("aq_prompt_tokens_cached_total", "counter", "Prompt tokens served from the session cache or a shared prefix.", m.cached_tokens.load(Ordering::Relaxed), None),
        ("aq_generation_tokens_total", "counter", "Tokens generated.", m.generation_tokens.load(Ordering::Relaxed), Some("vllm:generation_tokens_total")),
    ];
    let mut out = String::new();
    for (name, kind, help, v, alias) in rows {
        for name in std::iter::once(name).chain(alias.filter(|_| app.vllm_metrics)) {
            out += &format!("# HELP {name} {help}\n# TYPE {name} {kind}\n{name}{{model_name=\"{}\"}} {v}\n", app.model);
        }
    }
    ([(axum::http::header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")], out).into_response()
}

/// Engine thread: owns the CUDA context; steps while there is work, blocks on the channel
/// when idle.
#[derive(Clone, Copy)]
struct EngineOpts {
    mixed: Option<usize>,
    max_running: Option<usize>,
    session_cache: bool,
    spec_k: usize,
    policy: aq_engine::engine::Policy,
    max_snapshots: Option<usize>,
    itl_target_ms: Option<f64>,
    adaptive: Option<(usize, usize)>,
    prewarm: bool,
    /// Burst prefill for long prompts: tokens per burst step (0 = off) and steps between bursts.
    burst: (usize, usize),
    /// Output tokens reserved in the KV pool at admission (0 = the whole token limit).
    kv_reserve: usize,
}

fn engine_loop(dir: PathBuf, limits: Limits, opts: EngineOpts, eos: Vec<u32>, metrics: Arc<Metrics>, rx: std::sync::mpsc::Receiver<Submit>, ready: std::sync::mpsc::Sender<Result<()>>) {
    let t0 = Instant::now();
    let model = match Model::load(&dir, limits) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let mut model = model;
    if opts.prewarm {
        let t = Instant::now();
        match model.prewarm_gemms() {
            Ok(n) => eprintln!("prewarmed GEMM plans for {n} M buckets in {:.1}s", t.elapsed().as_secs_f64()),
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        }
    }
    let mut eng = Engine::new(model, eos);
    if let Some(m) = opts.mixed {
        eng.mixed_prefill_budget = m;
    }
    if let Some(r) = opts.max_running {
        eng.max_running = r;
    }
    eng.session_cache = opts.session_cache;
    eng.spec_k = opts.spec_k;
    eng.policy = opts.policy;
    eng.kv_reserve = if opts.kv_reserve == 0 { usize::MAX } else { opts.kv_reserve };
    // AQ_BURST_PREFILL / AQ_BURST_EVERY (read by the engine) win over the flags when set.
    if std::env::var("AQ_BURST_PREFILL").is_err() {
        eng.burst_prefill = opts.burst.0;
    }
    if std::env::var("AQ_BURST_EVERY").is_err() {
        eng.burst_every = opts.burst.1;
    }
    if let Some(n) = opts.max_snapshots {
        eng.max_snapshots = n;
    }
    eng.itl_target_us = opts.itl_target_ms.map(|m| m * 1000.0);
    eng.mixed_prefill_adaptive = opts.adaptive;
    eprintln!("policy: {:?} max_snapshots {} itl_target_ms {:?}", eng.policy, eng.max_snapshots, opts.itl_target_ms);
    let mut last_report = Instant::now();
    eprintln!("engine ready in {:.2}s", t0.elapsed().as_secs_f64());
    let _ = ready.send(Ok(()));
    let mut sinks: HashMap<u64, mpsc::UnboundedSender<Msg>> = HashMap::new();
    let mut accept = |eng: &mut Engine, s: Submit, sinks: &mut HashMap<u64, mpsc::UnboundedSender<Msg>>| {
        let id = s.req.id;
        match eng.add_with(s.req, s.ctl) {
            Ok(()) => {
                let _ = s.tx.send(Msg::Accepted);
                sinks.insert(id, s.tx);
            }
            Err(e) => {
                let _ = s.tx.send(Msg::Refused(e.to_string()));
            }
        }
    };
    let mut reports = 0u32;
    // AQ_FAULT_PANIC_STEP=N: fault injection for the liveness check, the engine thread panics
    // at its Nth step.
    let panic_at: u64 = std::env::var("AQ_FAULT_PANIC_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut steps = 0u64;
    loop {
        if !eng.has_work() {
            metrics.running.store(0, Ordering::Relaxed);
            match rx.recv() {
                Ok(s) => accept(&mut eng, s, &mut sinks),
                Err(_) => return,
            }
        }
        while let Ok(s) = rx.try_recv() {
            accept(&mut eng, s, &mut sinks);
        }
        if last_report.elapsed().as_secs() >= 10 {
            let c = eng.cache;
            eprintln!(
                "[stats] running {} parked {} | admitted {} hits {} (forks {}) reused {} prefilled {} snapshots {} checkpoints {} evictions {} (snapshot {}) deferred {} cut {} stalled {} host {:.0}us/step | spec drafted {} accepted {} (cross-session {} / {}, mtp {} / {}) tok/step {:.2}",
                eng.num_running(), eng.num_parked(), c.admitted, c.hits, c.forks, c.reused_tokens, c.prefilled_tokens, c.snapshots, c.checkpoints, c.evictions,
                c.snapshot_evictions, c.deferred, c.cut, c.stalled, c.host_us as f64 / c.steps.max(1) as f64,
                eng.spec.drafted, eng.spec.accepted, eng.spec.global_accepted, eng.spec.global_drafted, eng.spec.mtp_accepted, eng.spec.mtp_drafted, eng.spec.emitted as f64 / eng.spec.decode_entries.max(1) as f64
            );
            last_report = Instant::now();
            // AQ_PROFILE=1|decode|prefill: per-op GPU time of the steps since the last report.
            reports += 1;
            if reports % 6 == 0 {
                if let Some(p) = eng.model.prof.as_mut().filter(|p| p.steps > 0) {
                    eprintln!("[profile]\n{}", p.report());
                }
            }
        }
        steps += 1;
        if steps == panic_at {
            panic!("AQ_FAULT_PANIC_STEP={panic_at}");
        }
        match eng.step() {
            Ok(events) => {
                for e in events {
                    // A failed send means the handler dropped its receiver: the client went
                    // away, or a stop string ended the request. Stop generating for it instead of
                    // running to its token limit.
                    let gone = sinks.get(&e.id).is_some_and(|tx| tx.send(Msg::Token(e)).is_err());
                    if gone && !e.finished {
                        eng.cancel(e.id);
                    }
                    if gone || e.finished {
                        sinks.remove(&e.id);
                    }
                }
                // Clients that left before their first token (still queued or prefilling).
                let closed: Vec<u64> = sinks.iter().filter(|(_, tx)| tx.is_closed()).map(|(&id, _)| id).collect();
                for id in closed {
                    eng.cancel(id);
                    sinks.remove(&id);
                }
                // Ended while waiting for KV pages: no token event carries that.
                for id in eng.take_cut() {
                    if let Some(tx) = sinks.remove(&id) {
                        let _ = tx.send(Msg::Cut);
                    }
                }
                metrics.running.store(eng.num_running(), Ordering::Relaxed);
                metrics.cut.store(eng.cache.cut, Ordering::Relaxed);
            }
            Err(e) => {
                eprintln!("engine step failed: {e:#}");
                for (_, tx) in sinks.drain() {
                    let _ = tx.send(Msg::Failed(format!("engine error: {e}")));
                }
                return;
            }
        }
    }
}

/// End-of-sequence ids: `eos_token_id` of generation_config.json and config.json, and the
/// tokenizer's `eos_token` (for a chat model, the token that ends a turn).
fn eos_ids(dir: &std::path::Path, tok: &PipelineTokenizer) -> Vec<u32> {
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok().and_then(|s| serde_json::from_str::<Value>(&s).ok());
    let mut out = Vec::new();
    for v in ["generation_config.json", "config.json"].into_iter().filter_map(read) {
        for key in [&v["eos_token_id"], &v["text_config"]["eos_token_id"]] {
            match key {
                Value::Number(n) => out.extend(n.as_u64().map(|x| x as u32)),
                Value::Array(a) => out.extend(a.iter().filter_map(|x| x.as_u64().map(|x| x as u32))),
                _ => {}
            }
        }
    }
    if let Some(v) = read("tokenizer_config.json") {
        let name = v["eos_token"].as_str().or(v["eos_token"]["content"].as_str());
        out.extend(name.and_then(|n| tok.token_to_id(n)));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Sampling of a request that does not set `temperature`, `top_p` or `top_k`: the values of
/// the model's generation_config.json where it has them (as `transformers` and vLLM do), else
/// temperature 1 and no truncation (the OpenAI default). `--default-temperature` replaces the
/// temperature; 0 makes such requests greedy.
fn default_sampling(dir: &std::path::Path, temperature: Option<f32>) -> Sampling {
    let g: Value = std::fs::read_to_string(dir.join("generation_config.json")).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(Value::Null);
    Sampling {
        temperature: temperature.or(g["temperature"].as_f64().map(|t| t as f32)).unwrap_or(1.0),
        top_p: g["top_p"].as_f64().map_or(1.0, |p| p as f32),
        top_k: g["top_k"].as_u64().unwrap_or(0) as usize,
        seed: 0,
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().context("usage: aq-serve <model> [--host ADDR] [--port N] [--max-seqs N] [--max-ctx N] [--max-tokens N] [--mixed-prefill N]")?;
    let (mut port, mut max_seqs, mut max_ctx, mut max_tokens) = (8000u16, 64usize, 32768usize, 8192usize);
    let mut opts = EngineOpts { mixed: None, max_running: None, session_cache: true, spec_k: 4, policy: Default::default(), max_snapshots: None, itl_target_ms: None, adaptive: None, prewarm: false, burst: (4096, 3), kv_reserve: 4096 };
    let mut host = std::env::var("AQ_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let (mut served, mut vllm_metrics) = (None, false);
    // Shortest remaining prefill first by default (`--srpt false` restores admission order):
    // on real agent traces it keeps small turns from queueing behind a long tool result, at
    // the same throughput (Spark, 16 agents: TTFT p90 2633 -> 1506 ms).
    opts.policy.srpt = true;
    let mut kv_gb = 0f64;
    let mut cache_slots = 0usize;
    let mut default_temperature = None;
    let mut strict = false;
    while let Some(k) = args.next() {
        let v = args.next().with_context(|| format!("missing value for {k}"))?;
        match k.as_str() {
            "--host" => host = v,
            "--port" => port = v.parse()?,
            "--served-model-name" => served = Some(v),
            "--vllm-metrics" => vllm_metrics = v.parse()?,
            "--kv-reserve" => opts.kv_reserve = v.parse()?,
            "--max-seqs" => max_seqs = v.parse()?,
            "--max-ctx" => max_ctx = v.parse()?,
            "--max-tokens" => max_tokens = v.parse()?,
            "--mixed-prefill" => opts.mixed = Some(v.parse()?),
            "--max-running" => opts.max_running = Some(v.parse()?),
            "--session-cache" => opts.session_cache = v.parse()?,
            "--spec-k" => opts.spec_k = v.parse()?,
            "--evict" => {
                opts.policy.evict = match v.as_str() {
                    "lru" => aq_engine::engine::Evict::Lru,
                    "mru" => aq_engine::engine::Evict::Mru,
                    "idle" => aq_engine::engine::Evict::Idle,
                    _ => anyhow::bail!("--evict lru|mru|idle"),
                }
            }
            "--pin-snapshots" => opts.policy.pin_snapshots = v.parse()?,
            "--load-control" => opts.policy.load_control = v.parse()?,
            "--srpt" => opts.policy.srpt = v.parse()?,
            // A long prompt gets a whole step of up to N prefill tokens to itself every
            // `--burst-every` steps (default 4096 every 3; 0 = off): on real traces it halves
            // the time to first token of turns with 3k+ new tokens for the same throughput.
            "--burst-prefill" => opts.burst.0 = v.parse()?,
            "--burst-every" => opts.burst.1 = v.parse()?,
            "--max-wait-ms" => opts.policy.max_wait_ms = v.parse()?,
            "--dead-after-ms" => opts.policy.dead_after_ms = v.parse()?,
            "--quantum" => opts.policy.quantum = v.parse()?,
            "--prewarm" => opts.prewarm = v.parse()?,
            "--mixed-prefill-adaptive" => {
                // "HI:THRESHOLD", e.g. 8192:16384
                let (a, b) = v.split_once(':').context("--mixed-prefill-adaptive HI:THRESHOLD")?;
                opts.adaptive = Some((a.parse()?, b.parse()?));
            }
            "--max-snapshots" => opts.max_snapshots = Some(v.parse()?),
            "--itl-target-ms" => opts.itl_target_ms = Some(v.parse()?),
            "--kv-gb" => kv_gb = v.parse()?,
            "--cache-slots" => cache_slots = v.parse()?,
            "--strict-requests" => strict = v.parse()?,
            "--default-temperature" => default_temperature = Some(v.parse::<f32>()?).filter(|t| *t >= 0.0),
            _ => anyhow::bail!("unknown option {k}"),
        }
    }
    let dir = resolve_model(&model)?;
    // Hub tokenizer.json files are format 1.0; tokenizers v1 reads 2.0, so upgrade in memory.
    let json = tokenizers::canonicalize_file(dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{e}"))?;
    let tok = Arc::new(tokenizers::from_json(&json).map_err(|e| anyhow::anyhow!("{e}"))?);
    let eos = eos_ids(&dir, &tok);
    eprintln!("eos ids: {eos:?}");
    let mut defaults = default_sampling(&dir, default_temperature);
    // These modes pick tokens on the device only (`Engine::add_with`): a request that does
    // not say stays servable, greedy, as before there was a default temperature.
    let lanes = std::env::var("AQ_COEXEC").as_deref() == Ok("1") || std::env::var("AQ_BG_PREFILL").is_ok_and(|v| v.parse::<usize>().is_ok_and(|n| n > 0));
    if lanes && default_temperature.is_none() && defaults.temperature > 0.0 {
        eprintln!("AQ_COEXEC / AQ_BG_PREFILL: requests without `temperature` are greedy; one that asks for sampling gets a 400");
        defaults.temperature = 0.0;
    }
    eprintln!("default sampling: temperature {} top_p {} top_k {}", defaults.temperature, defaults.top_p, defaults.top_k);
    let limits = Limits { max_tokens, max_seqs, max_ctx, block_size: 16, kv_budget_bytes: (kv_gb * (1u64 << 30) as f64) as usize, kv_blocks: 0, cache_slots };
    let metrics = Arc::new(Metrics::default());
    let (tx, rx) = std::sync::mpsc::channel();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (edir, eeos, emetrics) = (dir.clone(), eos.clone(), metrics.clone());
    let engine = std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_loop(edir, limits, opts, eeos, emetrics, rx, ready_tx))?;
    // While the weights load: the chat template and the token table of the grammar engine.
    let chat = chat::Template::load(&dir)?;
    let grammar = grammar::Grammars::new(&dir, &eos);
    ready_rx.recv()??;
    // The engine thread only ends on a failed step or a panic. Nothing can be served after
    // that: /health answers 503 at once and the process exits non-zero (after a moment, so
    // open requests get their error), for a supervisor to restart it.
    let alive = Arc::new(AtomicBool::new(true));
    let flag = alive.clone();
    std::thread::spawn(move || {
        let how = if engine.join().is_err() { "panicked" } else { "stopped on an error" };
        flag.store(false, Ordering::SeqCst);
        eprintln!("the engine thread {how}: exiting");
        std::thread::sleep(std::time::Duration::from_millis(500));
        std::process::exit(1);
    });

    let app = Arc::new(App { submit: tx, tok, next_id: AtomicU64::new(0), model: served.unwrap_or(model), alive, eos, max_ctx, metrics, vllm_metrics, chat, grammar, defaults, strict });
    let router = Router::new()
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics_text))
        .route("/health", get(health))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await.with_context(|| format!("cannot listen on {host}:{port}"))?;
    eprintln!("listening on {host}:{port}");
    axum::serve(listener, router).await?;
    Ok(())
}
