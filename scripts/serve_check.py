"""HTTP checks of a running aq-serve (standard library only).

    python3 scripts/serve_check.py --url http://127.0.0.1:8000 --model Qwen/Qwen3.5-0.8B
    python3 scripts/serve_check.py --url ... --model ... --dead

Start the server fresh, with `--vllm-metrics true` and a context of 8192 or more. Prints one line per check and exits non-zero if
one fails. `--dead` is for a server started with AQ_FAULT_PANIC_STEP=N: the request that hits
the fault must get an error (not a cut-off 200) and /health must then be 503 or gone; the
caller checks the server's exit status.
"""
import argparse, json, sys, threading, time, urllib.error, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", required=True)
ap.add_argument("--dead", action="store_true")
ap.add_argument("--greedy-default", action="store_true", help="the server runs with --default-temperature 0")
ap.add_argument("--strict", action="store_true", help="the server runs with --strict-requests true")
args = ap.parse_args()
fails = []


def call(path, body=None, timeout=300, raw=False):
    """(status, parsed JSON | text | list of SSE events). Requests are greedy unless they say otherwise or `raw`."""
    if body is not None and not raw:
        body = {"temperature": 0, **body}
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(args.url + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status = r.read().decode(), r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(), e.code
    except OSError as e:
        return 0, str(e)
    if body and body.get("stream") and status == 200:
        return status, [l[6:] for l in raw.splitlines() if l.startswith("data: ")]
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name}{' | ' + str(detail)[:300] if detail else ''}", flush=True)
    if not ok:
        fails.append(name)


def chat(**kw):
    kw.setdefault("chat_template_kwargs", {"enable_thinking": False})
    return call("/v1/chat/completions", {"model": args.model, **kw})


def msg(text):
    return [{"role": "user", "content": text}]


def joined(events):
    """Streamed chat events -> (reasoning, content, tool calls, finish_reason, usage, saw [DONE])"""
    r, c, calls, fin, usage = "", "", [], None, None
    for e in events:
        if e == "[DONE]":
            continue
        j = json.loads(e)
        usage = j.get("usage") or usage
        for ch in j.get("choices", []):
            d = ch.get("delta", {})
            r += d.get("reasoning_content") or ""
            c += d.get("content") or ""
            calls += [t["function"] for t in d.get("tool_calls", [])]
            fin = ch.get("finish_reason") or fin
    return r, c, calls, fin, usage, events[-1:] == ["[DONE]"]


class section:
    """A group of checks; an exception in it (a missing field, say) fails it instead of ending the run."""
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        return self
    def __exit__(self, kind, exc, tb):
        if exc is not None and not isinstance(exc, SystemExit):
            check(f"{self.name}: checks could not run", False, f"{kind.__name__}: {exc}")
            return True


if args.dead:
    s, b = call("/v1/completions", {"prompt": [9707, 11, 1879, 0], "max_tokens": 400, "ignore_eos": True})
    check("dead engine: the request gets an error, not a cut-off 200", s in (500, 0), f"status {s} {b}")
    time.sleep(1.5)
    s, b = call("/health", timeout=5)
    check("dead engine: /health is 503 or the process is gone", s in (503, 0), f"status {s}")
    sys.exit(1 if fails else 0)

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Current time in a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
PERSON = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"], "additionalProperties": False}

s, b = call("/health")
check("health is 200 while the engine runs", s == 200, s)

# First, while the server has free slots: a prompt checkpoint needs one.
with section('Usage and cached tokens'):
    long = msg("Here is a list of numbers: " + ", ".join(str(i) for i in range(400)) + ". What is the first number?")
    a = chat(messages=long, max_tokens=8)[1]["usage"]
    b2 = chat(messages=long, max_tokens=8)[1]["usage"]
    check("usage: the same prompt again reports cached tokens", a["prompt_tokens"] > 800 and a["total_tokens"] == a["prompt_tokens"] + a["completion_tokens"] and b2["prompt_tokens_details"]["cached_tokens"] > 700, (a, b2))
    s, ev = call("/v1/completions", {"prompt": "One, two, three,", "max_tokens": 6, "stream": True, "stream_options": {"include_usage": True}})
    last = json.loads(ev[-2]) if s == 200 and len(ev) > 2 else {}
    check("completions, streamed: final usage chunk", last.get("choices") == [] and last.get("usage", {}).get("completion_tokens") == 6 and ev[-1] == "[DONE]", ev[-2:] if s == 200 else ev)
    s, ev = call("/v1/completions", {"prompt": "One, two, three,", "max_tokens": 6, "stream": True})
    check("completions, streamed: no usage chunk unless asked", s == 200 and all("usage" not in json.loads(e) for e in ev[:-1]), ev[-2:] if s == 200 else ev)

with section('Text prompts tokenize'):
    s, b = call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 8})
    check("text prompt: tokenized and completed", s == 200 and b["choices"][0]["text"], b)
    check("completions usage: prompt, completion and cached counts", s == 200 and b["usage"]["prompt_tokens"] >= 4 and b["usage"]["completion_tokens"] >= 1 and "cached_tokens" in b["usage"]["prompt_tokens_details"], b.get("usage") if s == 200 else b)

with section('Nothing is silently dropped'):
    for path, base in (("/v1/completions", {"prompt": "Hi"}), ("/v1/chat/completions", {"messages": msg("Hi")})):
        s, b = call(path, {**base, "max_tokens": 4, "bogus_field": 1})
        check(f"{path}: unknown field is a 400 naming it", s == 400 and "bogus_field" in json.dumps(b), f"{s} {b}")
        s, b = call(path, {**base, "max_tokens": 4, "n": 2})
        check(f"{path}: n=2 is a 400 naming n", s == 400 and b["error"]["param"] == "n", f"{s} {b}")
        s, b = call(path, {**base, "max_tokens": 4, "n": 1, "presence_penalty": 0, "logprobs": False})
        check(f"{path}: fields that ask for nothing are accepted", s == 200, f"{s}")
        quiet = {"user": "u1", "store": False, "metadata": {"k": "v"}, "service_tier": "auto", "prompt_cache_key": "k", "safety_identifier": "s"}
        s, b = call(path, {**base, "max_tokens": 4, **quiet})
        if args.strict:
            check(f"{path}: --strict-requests refuses the fields that have no effect", s == 400 and b["error"]["param"] == "user", f"{s} {b}")
        else:
            check(f"{path}: OpenAI fields that have no effect here are accepted", s == 200, f"{s} {b}")
    for effort, want in (("none", False), ("high", True)):
        s, b = call("/v1/chat/completions", {"messages": msg("What is 17 + 4?"), "max_tokens": 200, "reasoning_effort": effort})
        got = bool(b["choices"][0]["message"]["reasoning_content"]) if s == 200 else None
        if args.strict:
            check(f"--strict-requests refuses reasoning_effort {effort}", s == 400 and b["error"]["param"] == "reasoning_effort", f"{s} {b}")
        else:
            check(f"reasoning_effort {effort}: thinking {'on' if want else 'off'}", got is want, b["choices"][0] if s == 200 else b)
    s, b = chat(messages=msg("Hi"), max_tokens=4, reasoning_effort="max")
    check("unknown reasoning_effort is a 400 naming it", s == 400 and b["error"]["param"] == "reasoning_effort", f"{s} {b}")
    s, b = chat(messages=msg("Hi"), max_tokens=4, response_format={"type": "xml"})
    check("unknown response_format type is a 400 naming response_format", s == 400 and b["error"]["param"] == "response_format", f"{s} {b}")
    s, b = chat(messages=msg("Hi"), max_tokens=4, tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}})
    check("tool_choice naming no tool is a 400 naming tool_choice", s == 400 and b["error"]["param"] == "tool_choice", f"{s} {b}")
    s, b = call("/v1/completions", {"model": "not-this-one", "prompt": "Hi", "max_tokens": 2})
    check("another model name is a 404", s == 404, s)

with section('End of turn: no max_tokens, the model must stop by itself and the stop token is not text'):
    ends = [chat(messages=msg(q)) for q in ("Say hi.", "Name the capital of France in one word.", "What is 2+2? Answer with the number only.")]
    ok = all(s == 200 and b["choices"][0]["finish_reason"] == "stop" and "<|im_end|>" not in b["choices"][0]["message"]["content"] and b["usage"]["completion_tokens"] < 300 for s, b in ends)
    check("chat ends on the end-of-turn token (no max_tokens given)", ok, [(b["choices"][0]["finish_reason"], b["usage"]["completion_tokens"], b["choices"][0]["message"]["content"][:40]) if s == 200 else (s, b) for s, b in ends])

with section('Stop strings and stop token ids'):
    count = msg("Count from 1 to 10, comma separated, digits only.")
    s, b = chat(messages=count, max_tokens=60)
    full = b["choices"][0]["message"]["content"] if s == 200 else ""
    s, b = chat(messages=count, max_tokens=60, stop=["6"])
    c = b["choices"][0] if s == 200 else {}
    check("stop string: output ends before it, stop_reason names it", s == 200 and "6" in full and c["message"]["content"] == full.split("6")[0] and c["finish_reason"] == "stop" and c["stop_reason"] == "6", f"{full!r} -> {c}")
    s, ev = chat(messages=count, max_tokens=60, stop=["6"], stream=True)
    check("stop string, streamed: same text, nothing of the stop string sent", s == 200 and joined(ev)[1] == full.split("6")[0] and joined(ev)[5], joined(ev) if s == 200 else ev)
    s, b = chat(messages=count, max_tokens=60, stop=[", 6", "never"])
    check("stop string across tokens", s == 200 and b["choices"][0]["message"]["content"] == full.split(", 6")[0], b["choices"][0] if s == 200 else b)
    s, b = call("/v1/completions", {"prompt": "1, 2, 3, 4, 5, 6, 7,", "max_tokens": 12, "return_token_ids": True})
    ids = b["choices"][0]["token_ids"]
    s, b = call("/v1/completions", {"prompt": "1, 2, 3, 4, 5, 6, 7,", "max_tokens": 12, "return_token_ids": True, "stop_token_ids": [ids[3]]})
    c = b["choices"][0]
    first = ids.index(ids[3])
    check("stop_token_ids: ends at the token, which is not text", s == 200 and c["token_ids"] == ids[: first + 1] and c["finish_reason"] == "stop" and c["stop_reason"] == ids[3], c)

with section('Sampling'):
    story = msg("Write one sentence about the sea.")
    g = [chat(messages=story, max_tokens=24, **kw)[1]["choices"][0]["message"]["content"] for kw in ({}, {"temperature": 0}, {"temperature": 0, "seed": 5})]
    check("temperature 0 is greedy and repeatable", g[0] == g[1] == g[2], g)
    t = [chat(messages=story, max_tokens=24, temperature=1.0, top_p=0.95, seed=sd)[1]["choices"][0]["message"]["content"] for sd in (1, 1, 2, 3, 4)]
    check("seeded sampling repeats; other seeds differ", t[0] == t[1] and len(set(t)) >= 3, t)
    d = [call("/v1/chat/completions", {"messages": story, "max_tokens": 24, "chat_template_kwargs": {"enable_thinking": False}}, raw=True)[1]["choices"][0]["message"]["content"] for _ in range(4)]
    if args.greedy_default:
        check("no temperature under --default-temperature 0: greedy", len(set(d)) == 1 and d[0] == g[0], d)
    else:
        check("no temperature: sampled at the server default (generation_config.json, else 1.0)", len(set(d)) >= 2, d)
    s, b = chat(messages=story, max_tokens=4, temperature=-1)
    check("negative temperature is a 400", s == 400 and b["error"]["param"] == "temperature", f"{s} {b}")

with section('Logit shaping and log-probabilities'):
    base = {"prompt": "1, 1, 1, 1, 1, 1,", "max_tokens": 16, "return_token_ids": True}
    s, b = call("/v1/completions", base)
    g = b["choices"][0]["token_ids"]
    s, b = call("/v1/completions", {**base, "logit_bias": {str(g[0]): -100}})
    check("logit_bias -100 bans a token", s == 200 and g[0] not in b["choices"][0]["token_ids"], (g, b["choices"][0] if s == 200 else b))
    other = next(t for t in range(1000, 1100) if t not in g)
    s, b = call("/v1/completions", {**base, "max_tokens": 3, "logit_bias": {str(other): 100}})
    check("logit_bias +100 forces a token", s == 200 and b["choices"][0]["token_ids"] == [other] * 3, b["choices"][0] if s == 200 else b)
    most = lambda ids: max(ids.count(t) for t in ids)
    sea = [chat(messages=msg("Write three sentences about the sea."), max_tokens=60, **kw)[1]["choices"][0]["message"]["content"] for kw in ({}, {"presence_penalty": 2})]
    check("presence_penalty changes a text that repeats words", sea[0] != sea[1] and len(sea[1]) > 20, sea)
    for field, v in (("frequency_penalty", 2), ("repetition_penalty", 1.8)):
        s, b = call("/v1/completions", {**base, field: v})
        ids = b["choices"][0]["token_ids"] if s == 200 else []
        check(f"{field}: the output repeats less than the greedy one", s == 200 and ids != g and len(set(ids)) > len(set(g)), (g, ids or b))
    s, b = call("/v1/completions", {**base, "repetition_penalty": 0})
    check("repetition_penalty 0 is a 400 naming it", s == 400 and b["error"]["param"] == "repetition_penalty", f"{s} {b}")
    s, b = chat(messages=msg("Say hi."), max_tokens=60)
    short = b["usage"]["completion_tokens"]
    s, b = chat(messages=msg("Say hi."), max_tokens=60, min_tokens=40)
    check("min_tokens holds the end of the turn back", s == 200 and short < 40 <= b["usage"]["completion_tokens"], (short, b["usage"] if s == 200 else b))
    s, b = chat(messages=msg("Name three colours."), max_tokens=8, logprobs=True, top_logprobs=3)
    lp = b["choices"][0]["logprobs"]["content"] if s == 200 else []
    ok = len(lp) == b["usage"]["completion_tokens"] and all(e["logprob"] <= 0 and len(e["top_logprobs"]) == 3 and e["top_logprobs"][0]["token"] == e["token"]
                                                               and e["top_logprobs"][0]["logprob"] >= e["top_logprobs"][2]["logprob"] and bytes(e["bytes"]).decode(errors="replace") == e["token"] for e in lp)
    check("chat logprobs: one entry per token, the greedy token is the most likely, top 3 in order", ok and "".join(e["token"] for e in lp) == b["choices"][0]["message"]["content"], lp[:2] or b)
    s, ev = chat(messages=msg("Name three colours."), max_tokens=8, logprobs=True, top_logprobs=3, stream=True)
    got = [e for x in ev[:-1] for c in json.loads(x).get("choices", []) for e in (c.get("logprobs") or {}).get("content", [])] if s == 200 else []
    check("chat logprobs, streamed: the same entries", [e["token"] for e in got] == [e["token"] for e in lp] and len(lp) > 0, [e["token"] for e in got])
    s, b = call("/v1/completions", {"prompt": "The capital of France is", "max_tokens": 6, "logprobs": 2})
    lp = b["choices"][0]["logprobs"] if s == 200 else {}
    check("completions logprobs: tokens, token_logprobs, top_logprobs, text_offset", s == 200 and "".join(lp["tokens"]) == b["choices"][0]["text"] and len(lp["token_logprobs"]) == 6
          and all(len(t) == 2 for t in lp["top_logprobs"]) and lp["text_offset"][0] == 0 and lp["text_offset"] == sorted(lp["text_offset"]), lp or b)
    s, b = chat(messages=msg("Hi"), max_tokens=4, top_logprobs=3)
    check("top_logprobs without logprobs is a 400", s == 400 and b["error"]["param"] == "top_logprobs", f"{s} {b}")
    s, b = chat(messages=msg("Hi"), max_tokens=4, logprobs=True, top_logprobs=30)
    check("top_logprobs over 20 is a 400", s == 400 and b["error"]["param"] == "logprobs", f"{s} {b}")

with section('Reasoning split'):
    s, b = chat(messages=msg("What is 17 + 4?"), max_tokens=600, chat_template_kwargs={"enable_thinking": True})
    m = b["choices"][0]["message"] if s == 200 else {}
    check("thinking on: reasoning_content apart, no think tags in content", s == 200 and m["reasoning_content"] and m["reasoning"] == m["reasoning_content"] and "</think>" not in (m["content"] or "") + m["reasoning_content"], str(m)[:200])
    s, b = chat(messages=msg("What is 17 + 4?"), max_tokens=60)
    m = b["choices"][0]["message"] if s == 200 else {}
    check("thinking off: no reasoning, content only", s == 200 and m["reasoning_content"] is None and m["content"], m)
    s, ev = chat(messages=msg("What is 17 + 4?"), max_tokens=600, chat_template_kwargs={"enable_thinking": True}, stream=True, stream_options={"include_usage": True})
    r, c, _, fin, usage, done = joined(ev) if s == 200 else ("", "", [], None, None, False)
    check("thinking on, streamed: reasoning deltas, then content, usage chunk, [DONE]", bool(r) and "</think>" not in r + c and fin and usage and usage["completion_tokens"] > 0 and done, (r[:60], c[:60], fin, usage))

with section('Tool calls'):
    ask = msg("What is the weather in Lyon for the next 3 days?")
    s, b = chat(messages=ask, tools=[TIME, WEATHER], max_tokens=200)
    c = b["choices"][0] if s == 200 else {}
    calls = c.get("message", {}).get("tool_calls") or []
    ok = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]).get("city") == "Lyon" and c["finish_reason"] == "tool_calls" and "<tool_call>" not in (c["message"]["content"] or "")
    check("tool_choice auto: the call is parsed, arguments typed by the schema", ok and json.loads(calls[0]["function"]["arguments"]).get("days", 3) == 3, c)
    s, ev = chat(messages=ask, tools=[TIME, WEATHER], max_tokens=200, stream=True)
    r, cc, scalls, fin, _, done = joined(ev) if s == 200 else ("", "", [], None, None, False)
    check("tool_choice auto, streamed: same call, no tag in content", ok and len(scalls) == 1 and scalls[0]["name"] == "get_weather" and scalls[0]["arguments"] == calls[0]["function"]["arguments"] and fin == "tool_calls" and "<" not in cc and done, (cc, scalls, fin))
    s, b = chat(messages=ask, tools=[TIME, WEATHER], tool_choice="none", max_tokens=60)
    check("tool_choice none: no tool_calls", s == 200 and not b["choices"][0]["message"].get("tool_calls"), b["choices"][0] if s == 200 else b)
    history = ask + [{"role": "assistant", "content": None, "tool_calls": calls or [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Lyon\", \"days\": 3}"}}]},
                     {"role": "tool", "tool_call_id": (calls or [{"id": "c1"}])[0]["id"], "content": "{\"forecast\": \"sunny, 21 C, all three days\"}"}]
    s, b = chat(messages=history, tools=[TIME, WEATHER], max_tokens=120)
    check("a tool result is rendered back and answered", s == 200 and "sunny" in (b["choices"][0]["message"]["content"] or "").lower(), b["choices"][0] if s == 200 else b)

with section('Constrained output'):
    s, b = chat(messages=msg("Hello there."), tools=[TIME, WEATHER], tool_choice="required", max_tokens=200)
    c = b["choices"][0] if s == 200 else {}
    calls = c.get("message", {}).get("tool_calls") or []
    check("tool_choice required: at least one valid call although the user asked for none", s == 200 and calls and all(k["function"]["name"] in ("get_time", "get_weather") and "city" in json.loads(k["function"]["arguments"]) for k in calls) and c["finish_reason"] == "tool_calls", c or b)
    s, b = chat(messages=msg("Hello there."), tools=[TIME, WEATHER], tool_choice={"type": "function", "function": {"name": "get_time"}}, max_tokens=200)
    c = b["choices"][0] if s == 200 else {}
    calls = c.get("message", {}).get("tool_calls") or []
    check("tool_choice named: that function, valid arguments", s == 200 and len(calls) == 1 and calls[0]["function"]["name"] == "get_time" and isinstance(json.loads(calls[0]["function"]["arguments"])["city"], str), c or b)
    person = msg("Ada Lovelace was 36 and lived in London. Give me the person.")
    fmt = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}
    for label, kw in (("greedy", {}), ("sampled at temperature 1.2", {"temperature": 1.2, "seed": 3}), ("after thinking", {"chat_template_kwargs": {"enable_thinking": True}, "max_tokens": 4000})):
        s, b = chat(**{"messages": person, "max_tokens": 200, "response_format": fmt, **kw})
        try:
            v = json.loads(b["choices"][0]["message"]["content"])
            ok = set(v) == {"name", "age"} and isinstance(v["name"], str) and isinstance(v["age"], int) and b["choices"][0]["finish_reason"] == "stop"
        except Exception:
            ok = False
        c = b["choices"][0] if s == 200 else {}
        if "thinking" in label and c.get("finish_reason") == "length" and c["message"]["content"] is None:
            # A small model may never close its reasoning; the schema applies after it.
            print(f"skip response_format json_schema, {label}: the model was still reasoning at max_tokens", flush=True)
            continue
        check(f"response_format json_schema, {label}: content matches the schema", ok, c or b)
    s, b = chat(messages=msg("Say hello in plain words, not JSON."), max_tokens=120, response_format={"type": "json_object"})
    try:
        ok = isinstance(json.loads(b["choices"][0]["message"]["content"]), dict)
    except Exception:
        ok = False
    check("response_format json_object: content is a JSON object", ok, b["choices"][0] if s == 200 else b)
    s, b = chat(messages=person, max_tokens=50, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "string", "pattern": "(unclosed"}}})
    check("a schema the grammar cannot compile is a 400 naming response_format", s == 400 and b["error"]["param"] == "response_format", f"{s} {str(b)[:200]}")

with section('Two streams at once'):
    res = [None, None]
    def one(i):
        res[i] = chat(messages=msg(f"Count from {i + 1} to {i + 12}, comma separated."), max_tokens=80, stream=True)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    check("two concurrent streams both finish", all(r and r[0] == 200 and joined(r[1])[3] and joined(r[1])[5] for r in res), [joined(r[1])[1:4] if r and r[0] == 200 else r for r in res])

with section('Metrics'):
    s, b = call("/metrics")
    names = {l.split("{")[0] for l in str(b).splitlines() if l and not l.startswith("#")}
    want = {"aq_requests_running", "aq_requests_waiting", "aq_requests_total", "aq_prompt_tokens_total", "aq_prompt_tokens_cached_total", "aq_generation_tokens_total", "vllm:num_requests_running", "vllm:num_requests_waiting"}
    val = {l.split("{")[0]: float(l.rsplit(" ", 1)[1]) for l in str(b).splitlines() if l and not l.startswith("#")}
    check("/metrics: Prometheus text with own names and the vLLM aliases", s == 200 and want <= names and val.get("aq_requests_total", 0) > 20 and val.get("aq_generation_tokens_total", 0) > 100 and val.get("aq_requests_running") == 0, sorted(names) if s == 200 else b)

print(f"SERVE_CHECK {'FAIL ' + str(len(fails)) if fails else 'PASS'}")
sys.exit(1 if fails else 0)
