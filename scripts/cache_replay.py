"""Prompt-cache replay against a running OpenAI-compatible server (standard library only).

    python3 scripts/cache_replay.py --url http://127.0.0.1:8000 --model Qwen/Qwen3.5-9B

The workload of an agent fleet: every conversation starts with the same system prompt and
tool schemas (--prefix-words sizes it, about 6k tokens by default), then diverges. Phase 1
plays --convs conversations for --turns turns, --concurrency at a time, round by round, each
turn sending the whole history again as a chat client does, plus a new user turn that carries
a tool result (--turn-words). Phase 2 sends near-duplicates:
the full history of a few conversations with a different last user turn each time.

Reports, per phase, the share of prompt tokens the server reused (`usage.prompt_tokens_details.
cached_tokens`), time to first token (p50/p99, streamed), and prefill throughput seen from the
client (uncached prompt tokens / time to first token, so it includes queueing).
"""
import argparse, json, random, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--model", required=True)
ap.add_argument("--convs", type=int, default=16)
ap.add_argument("--turns", type=int, default=10)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--prefix-words", type=int, default=2600, help="size of the shared system prompt")
ap.add_argument("--turn-words", type=int, default=150, help="size of the tool result each user turn carries")
ap.add_argument("--warm", type=int, default=0, help="unrelated short requests first, so that no state slot is free (a server that has been up for a while)")
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--dups", type=int, default=6, help="near-duplicates per conversation in phase 2")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--no-think", action="store_true", help="send chat_template_kwargs.enable_thinking=false")
ap.add_argument("--out", help="write the per-request rows as JSON")
args = ap.parse_args()
rng = random.Random(args.seed)

WORDS = "agent tool file path query result error retry plan step user system search index token page cache state slot batch model".split()
TOOLS = [
    {"type": "function", "function": {
        "name": f"tool_{k}",
        "description": f"Tool number {k}: " + " ".join(rng.choice(WORDS) for _ in range(40)),
        "parameters": {"type": "object", "required": ["target"], "properties": {
            "target": {"type": "string", "description": " ".join(rng.choice(WORDS) for _ in range(12))},
            "limit": {"type": "integer"}, "mode": {"type": "string", "enum": ["fast", "full", "dry"]}}}}}
    for k in range(16)
]
SYSTEM = "You are a careful assistant. Rules:\n" + "\n".join(
    f"{k}. " + " ".join(rng.choice(WORDS) for _ in range(12)) for k in range(args.prefix_words // 13))
rows, lock = [], threading.Lock()


def turn(phase, conv, k, messages):
    body = {"model": args.model, "messages": messages, "tools": TOOLS, "tool_choice": "none", "max_tokens": args.max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t0, ttft, text, usage = time.time(), None, "", {}
    req = urllib.request.Request(args.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                if ttft is None and (d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls")):
                    ttft = time.time() - t0
                text += d.get("content") or ""
    row = {"phase": phase, "conv": conv, "turn": k, "prompt": usage.get("prompt_tokens", 0),
           "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
           "ttft": ttft if ttft is not None else time.time() - t0, "wall": time.time() - t0}
    with lock:
        rows.append(row)
    return text


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0.0


def report(name, sel):
    r = [x for x in rows if sel(x)]
    if not r:
        return
    prompt, cached = sum(x["prompt"] for x in r), sum(x["cached"] for x in r)
    ttft = [x["ttft"] for x in r]
    print(f"{name:26s} n={len(r):3d} prompt_tokens={prompt:8d} cached={cached:8d} share={cached / max(prompt, 1):6.1%} "
          f"ttft_p50={pct(ttft, 0.5) * 1e3:7.0f}ms ttft_p99={pct(ttft, 0.99) * 1e3:7.0f}ms "
          f"prefill_tok_s={(prompt - cached) / max(sum(ttft), 1e-9):7.0f}", flush=True)


hist = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Conversation {c}. Say the number {c * 7 + 3} and one word about caches."}]
        for c in range(args.convs)]
with ThreadPoolExecutor(args.concurrency) as pool:
    list(pool.map(lambda w: turn(0, w, 0, [{"role": "user", "content": f"Warm-up {w}: say {w}."}]), range(args.warm)))
    t_start = time.time()
    for k in range(args.turns):
        order = list(range(args.convs))
        rng.shuffle(order)
        for c, text in zip(order, pool.map(lambda c: turn(1, c, k, hist[c]), order)):
            hist[c] += [{"role": "assistant", "content": text or "ok"},
                        {"role": "user", "content": f"Step {k + 1} of conversation {c}, result: " + " ".join(rng.choice(WORDS) for _ in range(args.turn_words))
                         + f". Now the number {c * 7 + 4 + k}, and why."}]
    t_phase1 = time.time() - t_start
    jobs = [(c, d) for c in range(min(3, args.convs)) for d in range(args.dups)]
    list(pool.map(lambda j: turn(2, j[0], j[1], hist[j[0]][:-1] + [{"role": "user", "content": f"Variant {j[1]}: {rng.choice(WORDS)} {j[1] * 31} instead, in a sentence about {WORDS[j[1] % len(WORDS)]}."}]), jobs))

rows = [x for x in rows if x["phase"]]
print(f"convs={args.convs} turns={args.turns} concurrency={args.concurrency} warm={args.warm} first prompt={min(x['prompt'] for x in rows)} tokens, "
      f"last prompt={max(x['prompt'] for x in rows)} tokens, phase 1 wall {t_phase1:.1f}s")
report("turns: first", lambda x: x["phase"] == 1 and x["turn"] == 0)
report("turns: steady (2nd on)", lambda x: x["phase"] == 1 and x["turn"] >= 1)
report("turns: all", lambda x: x["phase"] == 1)
report("near-duplicates: 1st+2nd", lambda x: x["phase"] == 2 and x["turn"] < 2)
report("near-duplicates: 3rd on", lambda x: x["phase"] == 2 and x["turn"] >= 2)
if args.out:
    json.dump(rows, open(args.out, "w"))
