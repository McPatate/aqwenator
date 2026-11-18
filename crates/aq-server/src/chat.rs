//! `/v1/chat/completions`: the model's own chat template, and its output split into
//! reasoning, content and tool calls.
//!
//! The prompt is the model's `chat_template.jinja` rendered as transformers renders it. The
//! output is read the way the model writes it (Qwen 3.x): reasoning up to `</think>` when the
//! prompt opened a `<think>` block, then content, then tool calls as
//! `<tool_call><function=NAME><parameter=KEY>value</parameter>...</function></tool_call>`,
//! their values typed by the tool's JSON schema. `tool_choice` "required" or a named
//! function constrains the output to a JSON list of calls (or the arguments of the named
//! one) instead, see `grammar.rs`.

use crate::{App, Fail, Piece, bad, now, sse};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::{Value, json};
use std::sync::Arc;

pub struct Template {
    env: minijinja::Environment<'static>,
}

/// JSON as Python's `json.dumps` writes it (", " and ": "), which is what the template's
/// `tojson` produced when the model was trained, and what tool call arguments look like.
pub fn pyjson(v: &impl Serialize) -> String {
    struct Py;
    impl serde_json::ser::Formatter for Py {
        fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    match v.serialize(&mut serde_json::Serializer::with_formatter(&mut out, Py)) {
        Ok(()) => String::from_utf8(out).unwrap_or_default(),
        Err(_) => "null".into(),
    }
}

impl Template {
    /// The model's chat template (chat_template.jinja, or the one in tokenizer_config.json);
    /// `None` if it has none.
    pub fn load(dir: &std::path::Path) -> anyhow::Result<Option<Self>> {
        let config = || {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?).ok()?;
            Some(v["chat_template"].as_str()?.to_string())
        };
        let Some(src) = std::fs::read_to_string(dir.join("chat_template.jinja")).ok().or_else(config) else { return Ok(None) };
        let mut env = minijinja::Environment::new();
        // As transformers sets its Jinja environment up.
        env.set_syntax(minijinja::syntax::SyntaxConfig::builder().trim_blocks(true).lstrip_blocks(true).build()?);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |m: String| -> Result<String, minijinja::Error> { Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, m)) });
        env.add_filter("tojson", |v: minijinja::Value| pyjson(&v));
        env.add_template_owned("chat", src)?;
        Ok(Some(Self { env }))
    }

    pub fn render(&self, ctx: &Value) -> Result<String, String> {
        let t = self.env.get_template("chat").map_err(|e| e.to_string())?;
        t.render(minijinja::value::Serde(ctx)).map_err(|e| e.to_string())
    }
}

#[derive(PartialEq)]
enum Mode {
    /// No tools, or `tool_choice` "none": the output is content as it is.
    Raw,
    /// `tool_choice` "auto": tool calls in the model's own format.
    Auto,
    /// `tool_choice` "required": the output is a JSON list of {name, parameters}.
    Required,
    /// A named function: the output is its arguments.
    Named(String),
}

/// What a response holds so far.
#[derive(Default)]
struct View {
    reasoning: Option<String>,
    content: Option<String>,
    /// (name, arguments as JSON text)
    calls: Vec<(String, String)>,
}

/// The output text of one response, read as reasoning, content and tool calls.
struct Parser {
    acc: String,
    /// The prompt left a `<think>` block open: the output starts as reasoning.
    thinking: bool,
    mode: Mode,
    tools: Vec<Value>,
    one_call: bool,
    /// Streaming: bytes of reasoning and of content, and tool calls, already sent.
    sent: (usize, usize, usize),
}

/// One `<tool_call>` block: its function and arguments, each value typed by the tool's schema
/// (a value that is not a string there, or has no schema, is read as JSON when it is JSON).
fn parse_call(block: &str, tools: &[Value]) -> Option<(String, String)> {
    let Some(f) = block.find("<function=") else {
        // The JSON form of a call: {"name": ..., "arguments": {...}}.
        let v: Value = serde_json::from_str(block.trim()).ok()?;
        return Some((v["name"].as_str()?.to_string(), pyjson(v.get("arguments").unwrap_or(&json!({})))));
    };
    let (name, body) = block[f + "<function=".len()..].split_once('>')?;
    let name = name.trim();
    let mut body = body.split("</function>").next().unwrap_or(body);
    let props = tools.iter().find(|t| t["function"]["name"] == name).map(|t| &t["function"]["parameters"]["properties"]);
    let mut args = serde_json::Map::new();
    while let Some((_, after)) = body.split_once("<parameter=") {
        let (key, after) = after.split_once('>')?;
        let (val, next) = after.split_once("</parameter>").unwrap_or((after, ""));
        let val = val.strip_prefix('\n').unwrap_or(val);
        let val = val.strip_suffix('\n').unwrap_or(val);
        let is_string = props.is_some_and(|p| p[key.trim()]["type"] == "string");
        let v = if is_string { json!(val) } else { serde_json::from_str(val).unwrap_or_else(|_| json!(val)) };
        args.insert(key.trim().to_string(), v);
        body = next;
    }
    Some((name.to_string(), pyjson(&args)))
}

/// The length of `s` without a trailing start of `tag`.
fn without_partial(s: &str, tag: &str) -> usize {
    (1..tag.len()).rev().find(|&n| s.ends_with(&tag[..n])).map_or(s.len(), |n| s.len() - n)
}

impl Parser {
    /// `done`: the generation is over (nothing is held back any more).
    fn view(&self, done: bool) -> View {
        let mut v = View::default();
        let rest = if !self.thinking {
            self.acc.as_str()
        } else if let Some((r, c)) = self.acc.split_once("</think>") {
            v.reasoning = Some(r.to_string());
            c
        } else {
            // Still reasoning (or cut off inside it): there is no content.
            let n = if done { self.acc.len() } else { without_partial(&self.acc, "</think>") };
            v.reasoning = Some(self.acc[..n].to_string());
            return v;
        };
        match &self.mode {
            Mode::Raw => v.content = Some(rest.to_string()),
            Mode::Auto => {
                let (head, mut tail) = rest.split_once("<tool_call>").map_or((rest, None), |(c, b)| (c, Some(b)));
                let mut content = head.to_string();
                while let Some(b) = tail {
                    let (block, next) = b.split_once("</tool_call>").map_or((b, None), |(x, n)| (x, Some(n)));
                    if next.is_none() && !done {
                        break; // still being written
                    }
                    // (At the end, the last block may lack its closing tag.)
                    match parse_call(block, &self.tools) {
                        Some(c) => v.calls.push(c),
                        None => {
                            // Not a call after all: it, and what follows, stays text.
                            content = format!("{content}<tool_call>{b}");
                            break;
                        }
                    }
                    tail = next.and_then(|n| n.split_once("<tool_call>")).map(|x| x.1);
                }
                if self.one_call {
                    v.calls.truncate(1);
                }
                // White space before a call is not content; while a call may still follow,
                // trailing white space (and the start of a tag) waits.
                if !done {
                    content.truncate(without_partial(&content, "<tool_call>"));
                }
                if !v.calls.is_empty() || !done {
                    content.truncate(content.trim_end().len());
                }
                v.content = (v.calls.is_empty() || !content.is_empty()).then_some(content);
            }
            // The output is JSON: nothing can be said of it before it is complete.
            Mode::Required | Mode::Named(_) if !done => v.content = Some(String::new()),
            Mode::Required => {
                let list: Option<Vec<Value>> = serde_json::from_str(rest).ok();
                v.calls = list.unwrap_or_default().iter().filter_map(|c| Some((c["name"].as_str()?.to_string(), pyjson(&c["parameters"])))).collect();
                v.content = Some(if v.calls.is_empty() { rest.to_string() } else { String::new() });
            }
            Mode::Named(name) => match serde_json::from_str::<Value>(rest) {
                Ok(args) => (v.content, v.calls) = (Some(String::new()), vec![(name.clone(), pyjson(&args))]),
                // Cut off before the arguments were complete.
                Err(_) => v.content = Some(rest.to_string()),
            },
        }
        v
    }

    /// Streaming: the deltas for what `text` adds.
    fn deltas(&mut self, text: &str, done: bool, id: u64) -> Vec<Value> {
        self.acc.push_str(text);
        let v = self.view(done);
        let mut out = Vec::new();
        if let Some(r) = v.reasoning.as_deref().and_then(|r| r.get(self.sent.0..)).filter(|r| !r.is_empty()) {
            out.push(json!({"reasoning": r, "reasoning_content": r}));
            self.sent.0 += r.len();
        }
        if let Some(c) = v.content.as_deref().and_then(|c| c.get(self.sent.1..)).filter(|c| !c.is_empty()) {
            out.push(json!({"content": c}));
            self.sent.1 += c.len();
        }
        for (i, (name, args)) in v.calls.iter().enumerate().skip(self.sent.2) {
            out.push(json!({"tool_calls": [{"index": i, "id": call_id(id, i), "type": "function", "function": {"name": name, "arguments": args}}]}));
            self.sent.2 = i + 1;
        }
        out
    }

    fn finish_reason(&self, reason: &'static str, calls: usize) -> &'static str {
        // (A named function is reported as "stop", as the OpenAI API does.)
        if calls > 0 && reason == "stop" && !matches!(self.mode, Mode::Named(_)) { "tool_calls" } else { reason }
    }
}

fn call_id(id: u64, i: usize) -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(id);
    format!("chatcmpl-tool-{:016x}{i:02x}", h.finish())
}

/// The messages as the template expects them, or why they cannot be served.
fn messages(body: &crate::Body) -> Result<Vec<Value>, String> {
    let mut out = body.messages.clone().filter(|m| !m.is_empty()).ok_or("`messages` must be a non-empty list")?;
    for m in &mut out {
        let m = m.as_object_mut().ok_or("each message must be an object")?;
        if let Some(parts) = m.get("content").and_then(|c| c.as_array()) {
            if parts.iter().any(|p| p["type"] != "text") {
                return Err("only text content is supported".into());
            }
        }
        // The template iterates over the arguments: the API carries them as JSON text.
        for call in m.get_mut("tool_calls").and_then(|c| c.as_array_mut()).into_iter().flatten() {
            let args = &mut call["function"]["arguments"];
            if let Some(s) = args.as_str() {
                *args = serde_json::from_str(s).map_err(|e| format!("tool call arguments are not JSON: {e}"))?;
            }
        }
        if let (None, Some(r)) = (m.get("reasoning_content"), m.get("reasoning").cloned()) {
            m.insert("reasoning_content".into(), r);
        }
    }
    Ok(out)
}

/// How tool calls are to be produced, and the schema that constrains the output if any.
fn tool_mode(body: &crate::Body, tools: &[Value]) -> Result<(Mode, Option<Value>), String> {
    let named = |name: &str| tools.iter().find(|t| t["function"]["name"] == name);
    let params = |t: &Value| t["function"].get("parameters").filter(|p| p.is_object()).cloned().unwrap_or(json!({"type": "object"}));
    let mode = match body.tool_choice.as_ref().filter(|c| !c.is_null()) {
        None => return Ok((if tools.is_empty() { Mode::Raw } else { Mode::Auto }, None)),
        Some(Value::String(s)) if s == "none" => return Ok((Mode::Raw, None)),
        Some(Value::String(s)) if s == "auto" => return Ok((if tools.is_empty() { Mode::Raw } else { Mode::Auto }, None)),
        Some(Value::String(s)) if s == "required" => Mode::Required,
        Some(c) if c["type"] == "function" && c["function"]["name"].is_string() => Mode::Named(c["function"]["name"].as_str().unwrap_or_default().to_string()),
        Some(_) => return Err("must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}".into()),
    };
    if tools.is_empty() {
        return Err("needs `tools`".into());
    }
    let schema = match &mode {
        Mode::Named(name) => params(named(name).ok_or(format!("no tool is named `{name}`"))?),
        _ => {
            let calls: Vec<Value> = tools.iter().map(|t| json!({"properties": {"name": {"type": "string", "enum": [t["function"]["name"]]}, "parameters": params(t)}, "required": ["name", "parameters"]})).collect();
            let mut s = json!({"type": "array", "minItems": 1, "items": {"type": "object", "anyOf": calls}});
            if body.parallel_tool_calls == Some(false) {
                s["maxItems"] = json!(1);
            }
            s
        }
    };
    Ok((mode, Some(schema)))
}

pub async fn completions(State(app): State<Arc<App>>, raw: Bytes) -> Response {
    let body = match app.body(&raw, true) {
        Ok(b) => b,
        Err(r) => return r,
    };
    let Some(template) = &app.chat else { return bad(None, "this model has no chat template") };
    let msgs = match messages(&body) {
        Ok(m) => m,
        Err(e) => return bad(Some("messages"), e),
    };
    let tools = body.tools.clone().unwrap_or_default();
    if tools.iter().any(|t| !t["function"]["name"].is_string()) {
        return bad(Some("tools"), "each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}");
    }
    let (mode, tool_schema) = match tool_mode(&body, &tools) {
        Ok(m) => m,
        Err(e) => return bad(Some("tool_choice"), format!("`tool_choice`: {e}")),
    };
    let constraint = match (tool_schema, body.format_schema()) {
        (_, Err(r)) => return r,
        (Some(_), Ok(Some(_))) => return bad(Some("response_format"), "`response_format` cannot be combined with a `tool_choice` that forces a call"),
        (Some(s), _) => Some((s, "tool_choice")),
        (None, Ok(s)) => s.map(|s| (s, "response_format")),
    };
    let mut ctx = json!({"messages": msgs, "tools": if tools.is_empty() { Value::Null } else { json!(tools) }, "add_generation_prompt": true});
    if let Some(effort) = &body.reasoning_effort {
        ctx["reasoning_effort"] = json!(effort);
        ctx["enable_thinking"] = json!(effort != "none");
    }
    for (k, v) in body.chat_template_kwargs.iter().flatten() {
        ctx[k.as_str()] = v.clone();
    }
    let prompt = match template.render(&ctx) {
        Ok(p) => p,
        Err(e) => return bad(Some("messages"), format!("chat template: {e}")),
    };
    let thinking = prompt.ends_with("<think>\n");
    let ids = match app.encode(&prompt) {
        Ok(i) => i,
        Err(e) => return bad(Some("messages"), e),
    };
    // `<think>` and `<tool_call>` are ordinary added tokens; the special ones are not text.
    let skip = body.skip_special_tokens.unwrap_or(true);
    let mut out = match app.start(&body, ids, skip, constraint.map(|(s, f)| (s, thinking, f))).await {
        Ok(o) => o,
        Err(r) => return r,
    };
    let one_call = body.parallel_tool_calls == Some(false);
    let with_probs = body.logprobs_asked();
    let mut parser = Parser { acc: String::new(), thinking, mode, tools, one_call, sent: (0, 0, 0) };
    let (created, rid, model) = (now(), format!("chatcmpl-{}", out.id), app.model.clone());

    if body.stream.unwrap_or(false) {
        let usage = body.include_usage();
        let mut first = true;
        // Log-probabilities of the tokens since the last chunk (a token may yield no delta).
        let mut lp: Vec<Value> = Vec::new();
        return sse(out, move |out, piece: Result<Piece, Fail>| {
            let base = json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model});
            if let Ok(p) = &piece {
                lp.extend(out.chat_logprobs(p));
            }
            let mut chunk = |delta: Value, finish: Value, stop: Value| {
                let mut c = base.clone();
                let probs = if with_probs && !delta.as_object().is_some_and(|d| d.contains_key("role")) { json!({"content": std::mem::take(&mut lp)}) } else { Value::Null };
                c["choices"] = json!([{"index": 0, "delta": delta, "logprobs": probs, "finish_reason": finish, "stop_reason": stop}]);
                c.to_string()
            };
            let mut v = Vec::new();
            if std::mem::take(&mut first) {
                v.push(chunk(json!({"role": "assistant", "content": ""}), Value::Null, Value::Null));
            }
            match piece {
                Err(f) => v.push(f.event()),
                Ok(p) => {
                    for d in parser.deltas(&p.text, p.finish.is_some(), out.id) {
                        v.push(chunk(d, Value::Null, Value::Null));
                    }
                    if let Some(f) = p.finish {
                        v.push(chunk(json!({}), json!(parser.finish_reason(f.reason, parser.sent.2)), f.stop_reason));
                        if usage {
                            let mut u = base.clone();
                            u["choices"] = json!([]);
                            u["usage"] = out.usage();
                            v.push(u.to_string());
                        }
                    }
                }
            }
            v
        });
    }

    let mut lp: Vec<Value> = Vec::new();
    loop {
        match out.next().await {
            Err(f) => return f.response(),
            Ok(p) => {
                lp.extend(out.chat_logprobs(&p));
                parser.acc.push_str(&p.text);
                let Some(f) = p.finish else { continue };
                let v = parser.view(true);
                let mut message = json!({"role": "assistant", "content": v.content, "reasoning": v.reasoning, "reasoning_content": v.reasoning});
                if !v.calls.is_empty() {
                    let calls: Vec<Value> = v.calls.iter().enumerate().map(|(i, (name, args))| json!({"id": call_id(out.id, i), "type": "function", "function": {"name": name, "arguments": args}})).collect();
                    message["tool_calls"] = json!(calls);
                }
                return Json(json!({"id": rid, "object": "chat.completion", "created": created, "model": model,
                    "choices": [{"index": 0, "message": message, "logprobs": if with_probs { json!({"content": lp}) } else { Value::Null }, "finish_reason": parser.finish_reason(f.reason, v.calls.len()), "stop_reason": f.stop_reason}],
                    "usage": out.usage()}))
                .into_response();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![json!({"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}, "zip": {"type": "string"}}}}})]
    }

    fn parser(thinking: bool, mode: Mode) -> Parser {
        Parser { acc: String::new(), thinking, mode, tools: tools(), one_call: false, sent: (0, 0, 0) }
    }

    const CALL: &str = "<tool_call>\n<function=get_weather>\n<parameter=city>\nLyon\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=zip>\n69001\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn reasoning_content_and_typed_tool_call() {
        let mut p = parser(true, Mode::Auto);
        p.acc = format!("I need the weather.\n</think>\n\nLet me look.\n\n{CALL}");
        let v = p.view(true);
        assert_eq!(v.reasoning.as_deref(), Some("I need the weather.\n"));
        assert_eq!(v.content.as_deref(), Some("\n\nLet me look."));
        // `days` is an integer in the schema, `zip` a string although it looks like a number.
        assert_eq!(v.calls, [("get_weather".to_string(), r#"{"city": "Lyon", "days": 3, "zip": "69001"}"#.to_string())]);
        assert_eq!(p.finish_reason("stop", 1), "tool_calls");
    }

    #[test]
    fn a_call_alone_has_no_content_and_cut_off_reasoning_neither() {
        let mut p = parser(false, Mode::Auto);
        p.acc = format!("\n\n{CALL}\n{CALL}");
        let v = p.view(true);
        assert_eq!((v.content, v.calls.len()), (None, 2));
        let mut p = parser(true, Mode::Auto);
        p.acc = "Thinking Process:\n\n1".into();
        let v = p.view(true);
        assert_eq!((v.reasoning.as_deref(), v.content), (Some("Thinking Process:\n\n1"), None));
    }

    #[test]
    fn streamed_deltas_add_up_to_the_whole() {
        let text = format!("Let me think.\n</think>\n\nSure. \n\n{CALL}");
        let whole = { let mut p = parser(true, Mode::Auto); p.acc = text.clone(); p.view(true) };
        // Fed a few bytes at a time (on character boundaries), tags split or not.
        for step in [1, 3, 7, 1000] {
            let mut p = parser(true, Mode::Auto);
            let (mut r, mut c, mut calls) = (String::new(), String::new(), Vec::new());
            let chars: Vec<char> = text.chars().collect();
            let n = chars.len().div_ceil(step);
            for (i, piece) in chars.chunks(step).enumerate() {
                for d in p.deltas(&piece.iter().collect::<String>(), i + 1 == n, 0) {
                    r += d["reasoning"].as_str().unwrap_or("");
                    c += d["content"].as_str().unwrap_or("");
                    if let Some(t) = d["tool_calls"][0]["function"].as_object() {
                        // Nothing of an unfinished call is ever sent.
                        calls.push((t["name"].as_str().unwrap().to_string(), t["arguments"].as_str().unwrap().to_string()));
                    }
                    assert!(!c.contains("<tool_call>") && !r.contains("</think>"), "step {step}: {c:?} {r:?}");
                }
            }
            assert_eq!((Some(r), Some(c), calls), (whole.reasoning.clone(), whole.content.clone(), whole.calls.clone()), "step {step}");
        }
    }

    #[test]
    fn forced_calls_are_json() {
        let mut p = parser(false, Mode::Required);
        p.acc = r#"[{"name": "get_weather", "parameters": {"city": "Lyon"}}, {"name": "get_time", "parameters": {}}]"#.into();
        assert_eq!(p.view(false).calls.len(), 0);
        let v = p.view(true);
        assert_eq!(v.calls, [("get_weather".to_string(), r#"{"city": "Lyon"}"#.to_string()), ("get_time".to_string(), "{}".to_string())]);
        assert_eq!(v.content.as_deref(), Some(""));
        let mut p = parser(false, Mode::Named("get_weather".into()));
        p.acc = r#"{"city": "New York"}"#.into();
        assert_eq!(p.view(true).calls, [("get_weather".to_string(), r#"{"city": "New York"}"#.to_string())]);
        assert_eq!(p.finish_reason("stop", 1), "stop");
    }

    #[test]
    fn text_that_is_not_a_call_stays_text() {
        let mut p = parser(false, Mode::Auto);
        p.acc = "Use <tool_call> like this".into();
        assert_eq!(p.view(false).content.as_deref(), Some("Use"));
        let v = p.view(true);
        assert_eq!((v.content.as_deref(), v.calls.len()), (Some("Use <tool_call> like this"), 0));
        let mut p = parser(false, Mode::Raw);
        p.acc = CALL.into();
        assert_eq!(p.view(true).content.as_deref(), Some(CALL));
    }
}
