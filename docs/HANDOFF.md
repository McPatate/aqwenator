# Direction and roadmap (agreed 2026-11-18)

**Target:** the moment of "this should not run on this hardware", at good speed: high-quality
models on constrained machines, using the whole machine (VRAM, RAM, SSD) and the structure of
the model to move bytes to where they fit. Speed stays a requirement, not a trade.
**Edge to build on:** many agents on one box (session cache, checkpoints, exact speculation),
which is where the engine is measurably ahead of llama.cpp and, on bandwidth-limited hardware,
of vLLM. **Scope:** Qwen models for the moment. Other models are added opportunistically while
they reuse the existing kernels and scheduler; the day one "requires significant rework" is a
decision point, not a default yes. Keep the code lean: fastest path only, trim the rest.
The competitors are Strata, llama.cpp and gufo; vLLM is the server baseline. Do not rebuild
Strata: where it is already good (one user, largest model on a gaming PC), measure before
building.

1. **Why vLLM is ahead at 32 agents on the B200** (795 against 739 turns/min; our TPOT goes
   2.4 -> 4.6 ms from 16 to 32 agents, vLLM's stays near 3.0): one per-op profile at 32 agents.
2. **Low-bit weights (4-bit) with their own kernel**, quality measured first (others report
   plain 4-bit dense at +9% perplexity against +0.2% for 8-bit). Goal: the 9B on an 8 GB card
   (an RTX 3070 laptop is the test machine). Also what expert weights need later.
3. **Parked sessions in host RAM, then on SSD**: evicted sessions and restarts resume from the
   longest exact prefix instead of re-running prefill. Safetensors `main` has a prefetch
   engine (still changing) for the disk tier.
4. **8-bit attention cache** (then 8-bit keys / 4-bit values): halves the bytes decode attention
   reads and the cache's memory.
5. **Larger models, opportunistically.** Qwen3.8-27B if it has the 9B's layout. Then a
   mixture-of-experts model of the same family (Qwen3.6-35B-A3B: Strata, llama.cpp and vLLM
   all run it) with the experts tiered across VRAM and RAM, then Qwen3.8-Flash-Next (also
   hyper-connections, sparse attention, a 51 GB n-gram table on disk). Known lessons: one
   preallocated expert slot pool with routing remapped to slots so the grouped GEMM is
   unchanged; LRU is within 3 points of the optimal cache policy; router read-ahead wastes
   about 40% of its reads; the fastest design keeps all experts in pinned host RAM, the hot
   ones in VRAM, and computes misses on the CPU.
6. **Strata under the multi-agent replay**, on a model and machine it supports, before building
   anything it already does. llama.cpp is in `scripts/baseline_llamacpp.sh`.
7. **Draft lengths on a bandwidth-limited device.** Chained MTP drafts (`AQ_MTP_K=3`) and
   lookup drafts (`--spec-k 4`, the total budget) give 4.22 tokens per decode pass on the
   traces. On the B200 budget 8 with 3 / 5 / 7 chained drafts gives 4.38 / 4.51 / 4.54 tokens
   per pass and 435 / 428 / 425 turns/min at 16 agents against 433 (budget 16: 412). Chaining
   was +10% on the Spark, so repeat there or on the laptop before changing defaults.
8. **Research: zml `attnd`** ("sparse logarithmic attention, on CPU, over UDP"). Only the
   announcement was read; the mechanism is in its video.

# Update 2026-11-16: branch `mythos-review` (main has everything up to the Docker/CI commits; FP8 is branch-only)

Everything below this section describes main as of 2026-11-13. Since then all work has gone to
`mythos-review` (worktree `.claude/worktrees/mythos-review`), which contains main, the
memory-policy lane, and:

- **Real-trace benchmark** (`scripts/trace_bench.py`, `scripts/trace_sweep.sh`): replay of recorded
  OpenHands trajectories. This replaced the synthetic agentic sweep as the benchmark that
  matters. Traces: `~/aq/traces-9b-40.json` on the Spark and the B200.
- **Serving**: prompt checkpoints, in-kernel state tap, wider prefill attention tiles,
  background prefill lane (`AQ_BG_PREFILL`), `AQ_TRACE` step/request logs, SRPT prefill order
  with an elder share (`--srpt true`, no measurable gain on real traces).
- **Speculation**: cross-session n-gram draft model and adaptive draft length; the model's own
  MTP draft head (`AQ_MTP=1`, opt-in; `AQ_MTP_SHORT` shortlist size, `AQ_MTP_K` chained drafts,
  default 1 because chaining did not raise throughput). ~90% first-draft acceptance on real
  traces, 2.89 -> 3.78 tokens per step.
- **Cold start**: weight loader with direct I/O (O_DIRECT reads into pinned staging buffers,
  overlapped with device allocation). 9B on the Spark: weights resident at 2.4 s with a warm or
  cold page cache, first token at 3.2 s (the mmap loader it replaced: 2.1 s warm, 16.7 s cold).
  The drive reads at ~10 GB/s, so 1.8 s is the floor for 17.7 GiB; allocation is ~0.9 s.
- **Numerics**: cuBLASLt algorithms used to be picked by timing three runs, so a busy GPU gave
  each process different rounding; at a bifurcation point of repetitive text (9b-long, sequence
  1, token 223) that alone moves logits by 2-3 and showed up as a rare speculation "REAL
  MISMATCH". A stabilized timing pick (first candidate unless another is 3% faster, best of 5)
  still let one gate leg in eleven fail on the B200 (9b-long, `AQ_MTP=0`: two or three different
  token paths across identical runs), so there is no timing any more: always the heuristic's
  first candidate. Six runs of that leg then take one path, token-identical to plain decoding.
  Speed on real traces, 16 agents: Spark 92.2 and 93.2 turns/min vs 93.0 timed; B200 434.4 vs
  438.6 (noise). `aq-check` judges a divergence against how far its two plain paths (decode,
  prefill) disagree there. Diagnostics: `AQ_GEMM_PICK=seed` (random plans), `AQ_GEMM_MIN_M`,
  `AQ_SPEC_FAULT=N` (fault injection: the check must fail), `AQ_SPEC_TRACE`,
  `AQ_CHECK_DRIFT_AT=seq:tok`, `AQ_CHECK_ROWDEP=N`, `AQ_CHECK_TF_ONLY`, `AQ_CHECK_SPEC_REPEAT=N`.

Results (Qwen3.5-9B, real traces):

| Machine | Agents | Engine | Turns/min | Tok/s | TTFT p50/p99 (ms) |
|---|---|---|---|---|---|
| Spark | 16 | aqwenator (cross-session drafts, SRPT) | 78.6 | 182 | 411 / 3834 |
| Spark | 16 | vLLM 0.33, n-gram speculation | 47.6 | 127 | 722 / 2292 |
| Spark | 16 | vLLM 0.33, MTP speculation (3) | 62.4 | 137 | 977 / 3729 |
| B200 | 16 | aqwenator, no MTP | 319.4 | 842 | 67 / 1898 |
| B200 | 16 | aqwenator, MTP head (1 draft) | 338.8 | 896 | 72 / 1688 |

Spark, quiet window, 24 GB KV pool, 5 min per leg (2026-11-15):

| Agents | Build | Turns/min | Tok/s | TPOT p50/p99 (ms) | TTFT p50/p99 (ms) | Tok/step |
|---|---|---|---|---|---|---|
| 8 | no MTP | 50.8 | 117 | 47.3 / 127.9 | 342 / 3975 | 2.62 |
| 8 | MTP head | 58.4 | 151 | 36.7 / 74.0 | 337 / 2426 | 3.47 |
| 8 | MTP + background prefill lane | 54.0 | 135 | 41.6 / 96.6 | 364 / 3171 | 3.45 |
| 16 | no MTP | 73.2 | 168 | 73.2 / 161.1 | 547 / 4444 | 2.62 |
| 16 | MTP head | 80.0 | 193 | 58.4 / 189.6 | 520 / 6850 | 3.47 |
| 16 | MTP + background prefill lane | 78.0 | 188 | 59.9 / 157.2 | 533 / 6287 | 3.45 |

So the MTP head is on by default when the checkpoint has one (`AQ_MTP=0` disables it). The
background prefill lane (`AQ_BG_PREFILL=4096`) stays opt-in: on the B200 it cuts the 16-agent
TTFT p99 from 1236 ms to 351-378 ms (the tail there is head-of-line blocking behind 20-30k-token
first turns), but on the Spark, which has no spare GPU capacity for a second stream, it costs
throughput and does not help the tail. `scripts/trace_tail.py` dissects the tail from AQ_TRACE logs.

Same-window head-to-head on the Spark (both with MTP speculation, 5 min per leg, 2026-11-15):

| Agents | Engine | Turns/min | Tok/s | TPOT p50 (ms) | TTFT p50/p90/p99 (ms) | ITL p99 (ms) |
|---|---|---|---|---|---|---|
| 8 | aqwenator | 54.0 | 138 | 39 | 348 / 1428 / 3444 | 395 |
| 8 | vLLM 0.33 | 44.6 | 102 | 58 | 665 / 1583 / 2601 | 489 |
| 16 | aqwenator (FIFO prefill) | 81.4 | 195 | 60 | 597 / 2633 / 5791 | 492 |
| 16 | aqwenator (shortest prefill first) | 82.0 | 197 | 60 | 469 / 1506 / 5631 | 464 |
| 16 | vLLM 0.33 | 57.4 | 122 | 108 | 1022 / 1882 / 2909 | 763 |

`aq-serve` now orders prefill shortest-first by default (`--srpt false` for admission order).
A larger per-step prefill budget (2560) pushed ITL p99 to ~800 ms without helping.

Burst prefill (default in `aq-serve`: `--burst-prefill 4096 --burst-every 3`, 0 = off; engine
knobs `AQ_BURST_PREFILL` / `AQ_BURST_EVERY`): every few steps the longest-waiting prompt that
needs more than the per-step budget gets a whole step of up to N prefill tokens to itself.
Spark, 16 agents, 5 min, turns with 3,000+ new tokens (p50/p90 TTFT) and overall:

| Setting | Turns/min | TTFT p50/p90/p99 (ms) | 3k+ turns p50/p90 (ms) | ITL p99 (ms) |
|---|---|---|---|---|
| no burst (shortest first) | 82.0 | 469 / 1506 / 5631 | 3409 / 6478 | 464 |
| burst 4096 every 3 | 85.8 | 455 / 1565 / 4128 | 2355 / 4266 | 482 |
| burst 3072 every 2 | 83.2 | 533 / 1861 / 3741 | 2566 / 3861 | 511 |
| burst 6144 every 4 | 79.4 | 552 / 1864 / 3745 | 1725 / 3288 | 496 |
| vLLM 0.33, MTP | 57.4 | 1022 / 1882 / 2909 | 1347 / 2909 | 763 |

Bursts are rare enough that ITL p99 hardly moves. Larger bursts close more of the gap on big
turns but make small turns wait a little longer (p90 for <1k-token turns 658-805 ms without,
782-1054 with 4096, 1138-1439 with 6144; vLLM 1707).

10-minute run of the final defaults (tag `final-defaults`): at 8 agents 53.9 turns/min, 147
tok/s, TTFT p50/p90/p99 331/964/2546 ms, ITL p99 340 ms, and every turn-size bucket ahead of
vLLM (44.6 turns/min, 665/1583/2601, 489), including turns with 3k+ new tokens (p50/p90
1954/2835 vs 2434/3133 ms). The 16-agent leg ran while other jobs were on the GPU
(91% busy before the start) and came out at 70.1 turns/min, TTFT p99 7.0 s, ITL p99 833 ms: not
a valid confirmation of the 5-minute quiet-window numbers above; rerun it on a quiet machine.

Open: at 16 agents vLLM still has the better TTFT p99 (2.9 s vs 3.7-4.1 s in quiet 5-minute
runs) and the better p90 for turns with 1k+ new tokens. Also: the GEMM plan fix has not been re-verified on
the Spark with 9B; merge to main.

Gate status: `aq-check` on the B200, 6 goldens x {default, small budget, SRPT}, passes with
and without `AQ_MTP=1`.

Remote jobs: launch detached with a DONE marker and a timeout per process; never a wait loop
keyed on log lines only (one sat dead for 11 hours).

---

## OpenAI-compatible serving (2026-11-16, branch `feat/openai-serving`)

What a client written for vLLM or OpenAI needs from `aq-serve`, and what was missing. The
reference is the README section of the same name (fields, flags, limits, cache behaviour);
`scripts/serve_check.py` checks it against a running server.

- **Fixed: engine panic on a stop token inside an accepted draft** (`engine.rs`, the
  `overrun` case). Any request without `ignore_eos` could kill the engine thread; on the 9B,
  20 chat prompts: 3 served before the panic, 20 of 20 after.
- **Server**: `/v1/chat/completions` (chat template through minijinja, reasoning split, tool
  calls parsed or grammar-constrained), sampling (`aq_engine::sample`, exact under
  speculation), stop strings and stop token ids, `response_format` (llguidance),
  `logit_bias`, penalties, `min_tokens`, `logprobs` (`shape.rs`), `/health` that turns 503
  and a process that exits when the engine thread ends, `/metrics`, strict request parsing.
- **Defaults changed**: a request without `temperature` samples (generation_config.json,
  else 1.0; `--default-temperature 0` restores greedy); the server listens on 127.0.0.1
  (`--host`, `AQ_HOST`); `--kv-reserve 4096` instead of a full-context reservation.
- **Prompt cache in steady state** (`engine.rs`: `cache_slot`, `evict_redundant`, `deeper`
  in `try_admit`; `Limits::cache_slots`). Checkpoints and snapshots only took a free state
  slot, so a server whose slots were all parked reused nothing, and a short snapshot kept a
  longer shared prefix from ever being snapshotted. aq-check leg 4c reproduces both
  (`AQ_NO_CACHE_EVICT=1` restores the old slot rule and fails the leg).
  `scripts/cache_replay.py` is the measurement; numbers in the README.
- **KV reservation**: a request out of pages waits for the next one to end instead of ending
  with `length` (`Engine::unstall`, `take_cut`; aq-check leg 8).
- **Not done**: sampling, constraints and logit shaping under `AQ_COEXEC` / `AQ_BG_PREFILL`
  (the second lane's batches are collected steps later and take their tokens from the
  device; a picker would need that batch's logits rows, which the lane overwrites); `n` > 1;
  a state at message boundaries, so that the second of two near-duplicate requests is
  already a cache hit; multimodal input.

## Hardware ceilings (2026-11-15, uncommitted at the time of writing: commit signing needs the user)

`aq-ceiling [<model>] [--gb N] [--distinct] [--out f.json]` (crates/aq-engine/src/bin/aq-ceiling.rs)
measures what the device can do and what that means for one step of a model: read/copy/write
bandwidth, peak bf16 GEMM rate, launch overhead, and per row count M two lower bounds on the
step's GEMM time (every weight byte read once; every multiply-add at the peak rate) next to
what cuBLASLt takes for the same GEMMs. `--distinct` gives each GEMM its own weight buffer
(the engine's situation; costs the model's weight footprint in memory).

| | DGX Spark (GB10, 48 SMs) | B200 (148 SMs) |
|---|---|---|
| Read bandwidth | 237 GB/s | 6.7 TB/s |
| Peak bf16 GEMM | 98 TFLOP/s | 2.1 PFLOP/s |
| Launch (enqueue / launch + sync) | 1.9 / 4.9 us | 1.7 / 6.9 us |
| 9B weights read once (15.9 GB) | 67 ms | 2.4 ms (3.0 ms with distinct buffers) |
| KV read, 16 seqs x 20k ctx (10.5 GB) | 44 ms | 1.6 ms |

Qwen3.5-9B: 15.9 GB of GEMM weights per step (lm_head alone is 2.0 GB), 32 KiB of attention
KV per context token, 49 MiB of recurrent state per sequence.

Where the engine stands (decode step time against the bandwidth floor = weights + KV + state):
- Spark: 84 ms single stream (floor 70 ms), 141 ms at 16 seqs x 16k context (floor about
  110 ms). About 80% of the ceiling; cuBLASLt itself runs at 92-99% of the weight-read bound
  for M <= 64. The Spark is bandwidth-bound: more throughput has to come from more tokens per
  weight pass (drafts, batch) or fewer bytes per pass (quantized weights, not exact).
- B200: 4.1 ms single stream (floor 2.4-3.0 ms), 5.6 ms at 16 seqs, 10.7 ms at 64 seqs x 4k
  context (floor 4.7 ms). 45-58% of the ceiling before the attention work below.
- B200 prefill at 16k context: `attn_prefill` is 31% of the step and far below both ceilings
  (a block scans the whole context serially; no split-K in prefill). Open.

### Decode attention on tensor cores (`attn_decode_tq`; the split kernel it replaced is deleted)

One block per (sequence tile, KV head, slice) serves every row of the sequence (the token plus
its drafts) and all query heads of the GQA group: K/V is fetched once per sequence instead of
once per row and head. Slices come from a fixed table of absolute key bounds (32 fine slices
of `AQ_DEC_TC_SLICE` = 128 keys, then 32 coarse ones up to max_ctx), so a row's result does not
depend on the batch or on its drafts (this replaces `AQ_ATTN_SPLIT_KEYS`, `AQ_ATTN_FILL` and the
row-local slicing mode, all removed).

Findings on the way:
- The K/V page lookup did two 64-bit divisions per 16-byte copy; that was most of a tile's
  time. `kv_tile_cp_async` does one 32-bit division per thread and tile. `attention_tc`
  (prefill), `attn_prep` and the reference kernels still divide per element.
- Per 16-key tile on the B200: about 1.75 us fetching K/V, 0.5 us of MMAs. Double buffering
  and fetching four tiles per round trip did not help (more shared memory, fewer blocks).

B200, 9B, decode attention per step: 64 seqs x 4k ctx 4.77 -> 1.61 ms (floor 1.36), 16 seqs
1.31 -> 0.60 ms, single stream 0.17 -> 0.28 ms. End to end: 64 streams 2267 -> 2547 tok/s (ITL p50
10.75 -> 7.69 ms), 16 streams 1576 -> 1672, single stream 227 -> 224; with k=4 speculation at 16k
context 291 -> 322 (1 stream) and 521 -> 567 tok/s (16 streams).

Spark, real traces (2026-11-15, 24 GB KV pool): quiet 10-minute 16-agent run of the previous
defaults 82.4 turns/min, 206 tok/s, TTFT p50/p90/p99 459/1673/3596 ms, ITL p99 489 ms (vLLM:
57.4, 122, 1022/1882/2909, 763). With the new decode attention, 5-minute legs: 8 agents 59.0
turns/min, 161 tok/s, TTFT 309/924/2344, ITL p99 300 (before: 53.9, 147, 331/964/2546, 340);
16 agents 84.4 turns/min, 199 tok/s against 83.2 and 200 for the split kernel in the same
session (an earlier A/B of a first version gave 91.6 against 80.8: trace runs differ by several
percent between sessions). Per-op profile, 16 seqs x 16k context: decode attention 38.6 ms
against 40.1 ms with almost no drafts in flight (the floor is 36 ms); the gain grows with
the number of draft rows per sequence.

Not at the ceiling yet (largest first): B200 prefill attention (20% of its bandwidth floor;
one block scans the whole context serially); B200 small-kernel overhead in decode (about
1.1 ms of a 4.2 ms single-stream step, ~300 launches); GDN decode at high concurrency (1.46 ms
against a 0.94 ms floor at 64 streams); cuBLASLt at 128-512 rows (67-80% of its bound). On the
Spark: lm_head reads 2 GB per step (13% of the weight bytes) and prefill has not been
measured against the FLOP ceiling.

### Ceiling-gap investigation and chained drafts (2026-11-15, afternoon)

Verdicts on the gaps listed above:
- Prefill attention (B200): per tile, fetch 28%, MMAs 43%, softmax and bookkeeping 25%. The
  kernel needs 252 registers per thread, so one block (8 warps) per SM. Double-buffered K/V
  gave nothing. On the Spark it is 8% of a prefill step. No cheap win found.
- Small kernels in decode (B200 only): on real traces about 1.9 ms of a 7.9 ms step at ~8 us
  each. Fusing conv1d with the GDN step would save about 3%; not done.
- GDN decode: at its bandwidth floor on the Spark, 65% on the B200 (at most 6% of a step).
- cuBLASLt at 128-512 rows: not a gap. Workspace size (4 MB to 1 GB) changes nothing; it is
  the range where the bandwidth and FLOP limits both bind.
- lm_head: an exact shortcut (low-precision screen, exact recheck) would save 3-5% of a decode
  step; not worth the machinery.
- Spark prefill (16 seqs x 16k, 4096 rows per step, 950 ms): GEMMs 666 ms against a 591 ms
  FLOP floor (88%); attention 77 ms, GDN 83 ms, silu 40 ms and norm 35 ms (both at their
  bandwidth floor), conv1d 18 ms.
- Tried and dropped: 32-bit divisions in silu_mul and attn_prep (no change), padded instead of
  XOR-swizzled shared tiles in attn_decode_tq (170 -> 130 registers, 225 -> 220 us per call),
  four-tile K/V fetches.

Real-trace profile (`AQ_PROFILE=decode` with aq-serve, B200, 16 agents, mean context 20.9k):
decode step 7.9 ms; attention 2.0 ms (floor 1.64), gate_up 1.4, down 0.84, gdn_in 0.62,
gdn 0.58, norm 0.54, lm_head 0.36.

Chained MTP drafts are now the default (`AQ_MTP_K=3`, was 1). With K/V read once per sequence
the extra draft rows are nearly free on a bandwidth-bound machine. Spark, 16 agents, 5 min:
K=1 82.8 turns/min, 195 tok/s, 3.37 tok/step; K=2 90.6, 221, 3.77; K=3 91.4, 222, 3.97;
K=4 89.2, 215, 4.12; `--spec-k 8` alone 85.0, 203; K=3 with `--spec-k 8` 90.8, 220. B200:
437.6 / 435.4 / 441.2 / 443.2 turns/min for K=1 / K=3 / K=3 + k8 / k8 (neutral). Gate on the
B200: 30/30 PASS (6 goldens x default, SRPT + burst, small budget, AQ_MTP=0, AQ_MTP_K=1), fault
injection fails as required.

Spark, 10-minute runs with K=3 (tags mtpk-k3-16, mtpk-k3-8): 16 agents 91.9 turns/min, 229
tok/s, TTFT p50/p90/p99 483/1573/3864 ms, ITL p99 466 ms; 8 agents 57.4 turns/min, 157 tok/s,
TTFT 365/981/2445, ITL p99 360. vLLM (5 min): 57.4, 122, 1022/1882/2909, 763 at 16 agents;
44.6, 102, 665/1583/2601, 489 at 8. B200, 16 agents: 437.6 turns/min, 1146 tok/s (368.6 and
973 in the morning).

### Where the time goes on the Spark (2026-11-15, evening)

16 agents on real traces, K=3 (tag prof-k3-16: 93.4 turns/min, 232 tok/s; the GPU is busy for
the whole run). A least-squares fit of the `[step]` lines of `AQ_TRACE`: a decode-only step
costs 78 ms + 0.25 ms per row + 153 ms per million keys read (once per sequence); a prefill
token costs 0.22 ms. Mean step: 11 decoding sequences, 75 rows (6.7 per sequence: 2.6 pending,
4.1 drafts), 20.7k keys per sequence.

| Share of GPU time | What | Against its limit |
|---|---|---|
| 40% | weights read once per step | 78 ms vs a 67 ms floor |
| 33% | prefill | GEMMs at 88% of the FLOP ceiling |
| 18% | K/V read by decode attention | 214 GB/s vs 237 GB/s measured peak |
| 9% | per-row compute | a third of the rows are rejected drafts or re-fed tokens |

Draft acceptance by position (`AQ_SPEC_TRACE`, 16.8k decode entries): 90% / 85% / 87% / 86%
at positions 1-4 and 93-95% at 5-8 (only n-gram copies get that deep, 22% of entries). Mean
4.15 drafted, 3.03 accepted. Deeper head chains do not pay (K=4 measured no better than K=3):
the head feeding on its own guesses is what limits positions 2+, not the row budget.

Dead ends on the K/V traffic:
- Lossless KV compression: 1.28x on these values, at most 4% of step time before the cost of
  decompressing.
- Skipping the V read of key tiles with negligible attention weight: an instrumented kernel
  counted tiles where no query slot of the block has a score within 2^-20 / 2^-30 / 2^-40 of
  its running maximum. On real traces 99.9996% of 625M tiles are needed even at 2^-20, so
  there is nothing to skip.

What is left on the Spark with exact bf16 is a handful of items worth <= 3% each (base step
78 -> 67 ms, wasted draft rows, non-GEMM prefill ops). The step change is fewer bytes per
weight (8-bit weights would halve the 40% share), which gives up exact outputs.

### Release and 8-bit weights (2026-11-16)

**Release.** `Dockerfile` (CUDA 13.0.3 devel ->
base + `libcublas-13-0`, 1.23 GB), `ci.yml` (image build on amd64 and arm64), `release.yml`
(tag `vX.Y.Z` -> `ghcr.io/<owner>/aqwenator` + binaries on a GitHub release; no tag pushed
yet). One build per CPU type: kernels target `sm_80` and the driver compiles them for the
device (Spark: checks pass, 19.9 vs 19.5 tok/s for the `sm_121a` build). The arm64 image's
`aq-check` passes on the Spark; the amd64 image has not run on a GPU yet.

**8-bit weights: what the hardware offers** (`aq-ceiling` now times the 9B step's GEMMs with
8-bit operands; Spark, ms per step):

| Rows M | bf16 | FP8 E4M3 x E4M3 -> bf16 | int8 x int8 -> i32 |
|---|---|---|---|
| <= 128 | 75-82 | 38-42 | 38-46 |
| 512 | 107 | 55 | 81 |
| 4096 | 648-666 | 304-310 | 407 |

cuBLASLt 13.1 has no 8-bit weights x bf16 activations, and its per-row scale modes
(`OUTER_VEC_32F`) are rejected for this problem (heuristic status 7).

**`AQ_FP8=1|2`** (experimental, opt-in, not exact; `AQ_FP8_SET` picks the GEMMs): E4M3
weights with a power-of-two scale per output channel, quantized as each weight lands and the
bf16 copy freed (`gpu::Weight`); activations quantized per row before each GEMM
(`quant_e4m3`), output rescaled after (`scale_outer`). 2 = output head too.
- Speed, Spark, real traces, 16 agents, 5 min: bf16 83.6 turns/min 204 tok/s (a low run; 92-93
  is usual); FP8 layers 103.6 / 278; layers + head 106.8 / 286. Synthetic 16 x 16k: 25.1 vs
  20.0 out tok/s.
- Memory, 9B with a 4096-token KV pool: 11.70 GiB (layers) vs 18.05 GiB. Weights resident at
  4.4 s vs 2.9 s (uploads drained per weight; a pre-quantized checkpoint would fix it).
- Exact gate: 9B golden passes; resume-vs-cold identity checks fail on 0.8B and 9B-long
  (the decode and prefill paths now differ by quantization noise, not ulps).

**Tiers on real traces** (Spark, 16 agents, 5 min each, same binary, 2026-11-16 02:10-02:36;
`AQ_QUANT=fp8|int8` replaced `AQ_FP8`):

| Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|
| bf16 | 92.6 | 224 | 450 / 6040 | 50.9 / 126.2 |
| bf16 (second run) | 93.2 | 231 | 478 / 5146 | 50.5 / 110.4 |
| fp8 (layers) | 106.2 | 293 | 397 / 3748 | 40.7 / 98.0 |
| int8 (layers, whole-weight expansion) | 70.6 | 150 | 709 / 5371 | 84.5 / 181.3 |

**Final matrix** (Spark, 10 min each, one binary with int8 expanded 1024 rows at a time,
2026-11-16 03:10-04:25, 0 errors; a rebuild used CPU during the first leg):

| Agents | Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 16 | bf16 | 91.5 | 228 | 491 / 4550 | 50.6 / 117.9 |
| 16 | fp8 | 113.7 | 295 | 374 / 3374 | 38.1 / 80.3 |
| 16 | int8 | 71.6 | 182 | 619 / 3946 | 67.0 / 132.6 |
| 8 | bf16 | 60.6 | 166 | 342 / 2780 | 32.9 / 65.0 |
| 8 | fp8 | 79.3 | 209 | 256 / 2075 | 23.4 / 49.9 |
| 8 | int8 | 47.7 | 124 | 456 / 3383 | 46.1 / 105.1 |

fp8 is +24% (16 agents) and +31% (8) turns/min over bf16 with better latency everywhere; int8
is -22% and -21% (chunked expansion took its tok/s from 150 to 182 at 16 agents).

**vLLM on the same 8-bit scheme** (vLLM 0.33, `RedHatAI/Qwen3.5-9B-FP8-dynamic`, MTP
speculation 3 tokens, same replay, 10 min each, 05:16-05:40, 0 errors):

| Agents | Engine | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 16 | vLLM FP8 | 51.9 | 131 | 1052 / 3739 | 101.2 / 135.8 |
| 16 | aqwenator fp8 | 113.7 | 295 | 374 / 3374 | 38.1 / 80.3 |
| 8 | vLLM FP8 | 47.4 | 132 | 480 / 1719 | 45.1 / 60.4 |
| 8 | aqwenator fp8 | 79.3 | 209 | 256 / 2075 | 23.4 / 49.9 |

2.2x (16 agents) and 1.7x (8) turns/min. vLLM keeps the better TTFT p99 at 8 agents (1719 vs
2075 ms). vLLM gained little from FP8 at 16 agents (bf16 earlier: 57.4 turns/min, 122 tok/s).

**Scaling points** (4 min each, 06:15-06:37, 0 errors):

| Agents | Tier | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|---|
| 4 | bf16 | 35.2 | 90 | 258 / 5203 | 26.8 / 57.6 |
| 4 | fp8 | 49.8 | 132 | 197 / 1938 | 17.7 / 33.2 |
| 32 | bf16 | 137.5 | 247 | 964 / 10992 | 103.7 / 180.7 |
| 32 | fp8 | 148.2 | 297 | 808 / 9102 | 84.5 / 151.4 |

fp8 tok/s is flat from 16 to 32 agents (295 -> 297): the machine is saturated at 16, and more
agents only add queueing (TTFT p99 9 s). The fp8 gain shrinks with load (+41% turns/min at 4
agents, +8% at 32) as attention over the cached context, which fp8 does not touch, takes over.

**Extra tier points** (16 agents, 5 min each, 04:31-05:13; short runs, turns/min moves a few
percent with the sessions drawn):

| Variant | Turns/min | Tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) |
|---|---|---|---|---|
| fp8 + FP8 output head (`AQ_QUANT_HEAD=1`) | 108.6 | 293 | 422 / 4546 | 39.1 / 87.4 |
| fp8, MLP only (`AQ_QUANT_SET=ud`) | 103.4 | 273 | 450 / 4286 | 43.1 / 93.8 |
| int8, MLP only | 81.0 | 193 | 580 / 5415 | 61.1 / 125.1 |

The FP8 head buys nothing measurable (293 vs 295 tok/s) and is not worth its quality cost.
fp8 on the MLP alone keeps about two thirds of the tok/s gain at 0.018 nats. int8 on the MLP
alone is a little faster than int8 everywhere (193 vs 182 tok/s); its memory was not measured.

**Where the step goes** (`AQ_PROFILE=1`, 16 agents, last report window, us per call):

| Op | bf16 | fp8 | int8 |
|---|---|---|---|
| `gemm_gate_up` | 1244 | 870 | 1892 |
| `gemm_down` | 645 | 461 | 1190 |
| `gemm_gdn_in` | 645 | 451 | 1029 |
| `gemm_qkv` | 520 | 403 | 732 |
| `gemm_lm_head` (bf16 in all three) | 3308 | 3408 | 3401 |
| `attn_dec` | 4774 | 4573 | 4918 |

Decode-only step fits from the 10-minute runs: base 77 ms (bf16), 62 (fp8), 127 (int8); the
per-key attention term is unchanged (138-163 ms per million keys). With fp8, decode attention
is the largest single op (30% of step time) and the bf16 head is 8%: those two, not the layer
GEMMs, are the next targets. The profiled GEMM gain (~30%) is well under the 2x the bare
cuBLASLt timing shows, because the quantize and rescale passes sit around each small GEMM
(fusing them is open).

**Quality** is measured teacher-forced against transformers on three real trace turns (8k,
16k, 26k prompt tokens, 256 greedy tokens each; `scripts/golden.py --ids-file`, goldens
`~/aq/golden/qwen35-9b-trace-{a,b,c}.safetensors` on the Spark; `aq-check` prints
KL(ref||ours)). Mean KL in nats over the three, and top-1 agreement out of 768:

| Variant | Mean KL | Top-1 |
|---|---|---|
| bf16 | 0.002 | 764 |
| FP8, every layer GEMM | 0.028 | 739 |
| FP8, MLP only | 0.018 | 750 |
| FP8, one GEMM kind only (down / GDN in / qkv / gate-up) | 0.007 / 0.010 / 0.011 / 0.015 | |
| bf16 GEMMs, weights rounded through E4M3 (`AQ_QSIM=w:f`) | 0.022 | 751 |
| bf16 GEMMs, activations rounded through E4M3 (`a:f`) | 0.022 | 740 |
| bf16 GEMMs, weights int8 per output row (`w:i`) | 0.009 | 749 |
| bf16 GEMMs, activations int8 per row (`a:i`) | 0.141 | 710 |
| bf16 GEMMs, weights int8 with a scale per 32 (`w:b`) | 0.003 | 763 |
| bf16 GEMMs, weights int8 with row x column power-of-two scales (one pass; code not kept) | 0.059 | 739 |

So: E4M3's ~4 significant bits are the limit (weights and activations cost the same, no GEMM
is the culprit); linear int8 activations are unusable; block-scaled int8 weights with bf16
activations are at the noise floor. The fast FP8 path therefore costs ~0.03 nats and ~3% of
top-1 tokens, some at reference margins of 2-5 logits (often message / tool-call terminators).

**Open design question** (the user's call): how to multiply 8-bit weights by 16-bit
activations. Candidates: (a) int8 GEMM with each activation row split into two int8 digit
rows: decode ~50-58 ms of GEMM vs 78, but the i32 output pass makes it a loss for prefill, and
a per-32 scale along K cannot be factored out, which leaves per-row scales (0.009) since
one-pass row x column scaling measured worse (0.059);
(b) dequantize each weight to a bf16 scratch buffer and use the bf16 GEMM: clean quality and
half the memory, but ~2.3x the decode weight traffic; (c) our own tensor-core GEMM reading
int8 weights. `AQ_QSIM` and the `qsim` kernel are diagnostics to delete once this is settled.

### int8-weight GEMM kernel (`gemm_i8w`, 2026-11-16, B200 only)

With `AQ_QUANT=int8`, GEMMs of at most 96 rows (`AQ_I8_KERNEL_M`, 0 = off) on weights of at least
6144 rows (`AQ_I8_KERNEL_MIN_N`) run in a kernel that feeds the weight bytes to the integer tensor
cores (`mma.m16n8k32` s8), with no bf16 copy of the weight. Activations become two int8 digit
planes with one f32 scale per 32-wide block (`quant_x16`); each 32-block partial sum is scaled by
the weight block scale and the activation block scale and accumulated in f32. Everything else
keeps the expansion path (int8 to bf16 1024 rows at a time, then the bf16 GEMM).

Dead ends: converting int8 to bf16 per weight inside the kernel (compute-bound, 0.23–0.25 ms at
16 rows); one activation scale per row (one large element in a row cost logits rel_l2 1.14 at a
decode position; per-block scales fixed it).

One 24576 x 4096 weight, ms: bf16 GEMM 0.044 / 0.045 / 0.047 at 16 / 64 / 128 rows, expansion
0.347 flat, kernel 0.076 / 0.134 / 0.247 (0.370 at 256, 0.688 at 512: it costs per row, hence
the row cap).

Decode steps of `aq-serve` under the trace replay (16 agents, us per call, kernel / expansion):
gate_up 106 / 262, gdn_in 82 / 153, qkv 65 / 92, but down 205 / 81, gdn_out 81 / 52, o 62 / 41.
On 4096-row weights the kernel launches 32 blocks on 148 SMs and expanding them is cheap, hence
the minimum weight size; with it the decode step is 5.3 ms against 7.8 ms. The 6144 threshold
was measured on the B200 only; a 48-SM device probably wants a lower one.

Trace replay, 9B, B200, 4 minutes per point (turns/min, tok/s, TPOT p50 ms):

| | 16 agents | 32 agents |
|---|---|---|
| int8, kernel (cap 128) | 363.8 / 955 / 4.8 | 356.0 / 931 / 20.4 |
| int8, kernel (cap 96) | 362.5 / 949 / 4.9 | 373.5 / 1008 / 18.7 |
| int8, kernel (cap 64) | 363.0 / 950 / 5.1 | 363.5 / 941 / 20.1 |
| int8, expansion only (two runs) | 324.0 / 866 / 6.9 and 322.8 / 865 / 6.8 | 383.5 / 1004 / 18.4 and 368.8 / 976 / 18.9 |
| bf16 | 437.8 / 1158 / 2.3 | 424.8 / 1176 / 16.6 |

12% more turns at 16 agents; at 32 agents (saturated) the runs differ by less than the
expansion path's own run-to-run spread. bf16 stays the fast tier on the B200; int8 is for memory.
The point of the kernel is bandwidth-limited devices, where it reads half the bytes of the bf16
GEMM; that has not been measured (no Spark access since 2026-11-16).

Quality, kernel / expansion, teacher-forced on the three trace references (mean KL, nats):
a 4.17e-3 / 3.87e-3, b 1.42e-3 / 1.65e-3, c 3.04e-3 / 3.93e-3; top-1 253 / 254, 252 / 253,
254 / 255 of 256. `aq-check` passes with `AQ_QUANT=int8` on the three 9B references; the exact
gate is 30/30 with the fault leg failing as it must.

### Three engines on one B200, and state-slot starvation (2026-11-17)

Qwen3.5-9B, trace replay, one GPU per run, 4 minutes per point, zero errors everywhere.
vLLM 0.33 on the 16-bit checkpoint (`scripts/trace_sweep.sh vllm`); llama.cpp 1.0.0-dev
(build 12478) through `scripts/baseline_llamacpp.sh`: CUDA, all layers on the GPU, one slot per
agent with 33792 tokens of context each, f16 GGUF converted from the same checkpoint,
`-b 4096 -ub 2048` (the best of six settings). `trace_bench.py` counts tokens from `token_ids`
(vLLM, aq-serve), `tokens` (llama.cpp) or one per text chunk.

turns/min, tok/s, TTFT p50 / p99 ms, TPOT p50 ms:

| Agents | aqwenator bf16 | vLLM, MTP speculation (3) | vLLM, no speculation | llama.cpp |
|---|---|---|---|---|
| 1 | 28.2 / 76 / 42 / 205 / 1.6 | 27.5 / 71 / 118 / 172 / 1.5 | 24.2 / 57 / 102 / 169 / 4.1 | 20.5 / 51 / 98 / 475 / 6.6 |
| 4 | 114.8 / 323 / 43 / 166 / 1.6 | 105.8 / 294 / 132 / 623 / 1.6 | 91.8 / 247 / 114 / 217 / 4.4 | 51.2 / 138 / 361 / 1254 / 15.4 |
| 16 | 432.8 / 1140 / 52 / 571 / 2.4 | 384.5 / 1007 / 217 / 399 / 3.0 | 313.0 / 836 / 165 / 287 / 7.2 | 80.2 / 185 / 921 / 8089 / 68.8 |

On this card the lead over vLLM is 13% of turns at 16 agents (9% at 4, 3% at 1), against
60–75% on the Spark, and vLLM has the lower TTFT p99 at 1 and 16 agents. At 1 and 4 agents the
replay bounds turns per minute for the fast engines. Other llama.cpp settings at 16 agents: f16
defaults 79.0 / 175, bf16 defaults 78.5 / 175, `--spec-type ngram-mod` 75.5 / 165,
`--spec-type draft-mtp` 63.2 / 137 (TTFT p50 3100), Q8_0 63.2 / 135 (TTFT p50 3441). Its
settings come from its help text; nobody who tunes llama.cpp for a living has looked at them.
Strata and gufo were not run: Strata lists only Qwen3.8-Flash-Next and Qwen3.6-35B-A3B, gufo
runs on Strix Halo only.

32 agents, with the slot reserve below (turns/min, tok/s, TTFT p50 / p99 ms, TPOT p50 ms):
vLLM with MTP speculation 795.2 / 2105 / 124 / 273 / 3.1, aqwenator 738.5 / 1945 / 67 / 405 / 4.6,
vLLM without speculation 576.2 / 1553 / 117 / 322 / 8.8, llama.cpp 79.5 / 112 / 1453 / 17016 /
227.7. vLLM is ahead at this load on this card.

**State-slot starvation.** Every 32-agent number before 2026-11-17 was taken with it. Running
and parked sessions and their checkpoints each hold one of `--max-seqs` state slots (128 in the
sweeps), and checkpoints, taps and forks were skipped when no slot was free. With a KV pool
large enough that nothing is evicted for memory, parked entries filled the slots; a turn then
consumed the checkpoint it resumed from without leaving a new one, and the session's next turn
re-read about 16k tokens. A smaller pool hid it because memory pressure evicted stale entries.
32 agents, B200, bf16 (turns/min, tok/s, TTFT p50 / p99 ms, prompt tokens re-read):

| KV pool, slots | before | with the slot reserve |
|---|---|---|
| 48 GB, 128 | 481.3 / 1270 / 152 / 3475 / 6.1M | 738.5 / 1945 / 67 / 405 / 3.1M |
| 120 GB, 128 | 447.2 / 1201 / 154 / 3876 / 6.1M | 739.2 / 1943 / 68 / 406 / 3.1M |
| 24 GB, 128 | 750.0 / 1968 / 64 / 384 / 3.1M | |
| 48 GB, 256 | 752.0 / 1971 / 63 / 371 / 3.1M | |

The fix (`Engine::reserve_slots`) evicts the least-recently-used parked entry before admission
until an eighth of the slots is free. 16 agents (434.0) and 4 agents (113.8) are unchanged;
gate 31/31. The eighth was reasoned, not swept. The B200 is therefore not saturated at 16
agents, and the Spark's 32-agent points (same 128 slots) are probably low for the same reason.

### 32 agents on the B200: where the time goes; prefill attention; low-bit weights (2026-11-18)

**Profile at 32 agents** (bf16, `AQ_PROFILE=decode|prefill` under the replay). Steps that carry
prefill are about half of the GPU time (1699 steps of 46.6 ms against 6569 decode-only steps of
10.9 ms), and decoding sequences wait through them: that is the TPOT gap to vLLM. In those
steps `attn_prefill` is 50% (2.5–3.0 ms per call for about 1000 new tokens over 20k keys),
`attn_dec` 10%, `gemm_gate_up` 10%. In decode-only steps `attn_dec` is 43%. By the rates in
the ceilings table, prefill attention runs at about 6% of the card's GEMM rate and 7% of its
bandwidth bound.

- Tile width is not it: `AQ_ATTN_QT` 1 / 2 / 4 gives 758 / 746 / 748 turns/min and 2.30 / 2.47 /
  2.52 ms per call. Twice the blocks of half the warps take the same time.
- **Second K/V buffer pair** in `attention_tc` (the next tile is requested before the current
  one is waited for; on by default where the device allows the shared memory, 128 KiB for the
  9B; `AQ_ATTN_PIPE=0` turns it off; the GB10's 100 KiB keeps one pair): 2.56 against 2.95 ms
  per call (-13%), 755.0 / 756.8 against 741.8 / 739.8 turns/min in an ABAB run, TTFT p50 64
  against 71 ms. Gate 31/31. vLLM is still ahead at this load (795).
- What is left: the kernel has one warp per (16 tokens, head) and each scans the whole context
  serially, so 1000 tokens are about 1000 warps on 148 SMs, each bound by its own dependency
  chain. More warps per SM needs the key range split across blocks with a combine pass, as
  the decode kernel does. Not done.

**Low-bit weights, quality only** (teacher-forced mean KL in nats against transformers, the
three 20k-token trace references of 256 positions; rounding-only schemes emulated on the int8
grid, one scale per 32 weights; top-1 agreement in brackets):

| Scheme | KL |
|---|---|
| 8-bit | 0.002–0.004 (99%) |
| 6-bit uniform | 0.015–0.045 (96–99%) |
| 5-bit uniform | 0.11–0.14 (90–95%) |
| 4-bit uniform | 0.12–0.21 (91–95%) |
| 4-bit, NF4 levels | 0.15–0.19 |
| 4-bit uniform, scale clipped to 0.9 … 0.6 of the block maximum | 0.13–0.36 … 0.8–1.3 |
| 2-bit uniform | 0.5–1.0 (75–86%) |
| `Intel/Qwen3.5-9B-int4-AutoRound` (calibrated, one scale per 128), dequantized to bf16 | 0.08–0.14 (91–94%) |

4-bit on one group at a time (uniform): g 0.06, o 0.06, q 0.04–0.15, a 0.02–0.04, u 0.07–0.14,
d 0.03–0.05; NF4 halves the MLP's error (u+d 0.05–0.08 against 0.10–0.13) but not the whole
model's. No group is the culprit, so mixing 4-bit and 8-bit groups does not rescue it. The
error grows with context: AutoRound is 0.005 on the short reference and 0.022 at 16k (plain
rounding 0.013 and 0.038; 32 positions each). About 0.12 nats is the price of 4 bits on this
model on agent traces, calibrated or not; 6-bit rounding is at the fp8 tier's level.
The embedding table and the output head are separate 2 GB tensors (not tied).
Plan for an 8 GB card: load the AutoRound format directly (4.125 bits per weight, 3.6 GB of
layer weights), 8-bit head, embedding table in host RAM (it is only gathered from), about 7 GB
with a 1.5 GB KV pool. That budget is arithmetic on tensor sizes, not a measured load.

### 4-bit weights from a pre-quantized checkpoint (2026-11-18)

`scripts/convert_autoround.py <quantized dir> <original dir> <out dir>` turns a 4-bit AutoRound /
GPTQ checkpoint (packing `auto_round:auto_gptq`, symmetric, group size 128; tested with
`Intel/Qwen3.5-9B-int4-AutoRound`) into `<name>.weight.q4` (uint8 `[out, in / 2]`, two codes per
byte, value = code - 8) and `<name>.weight.q4s` (f32 `[out, in / 128]`), plus everything else
as it was (the MTP head comes from the original checkpoint, which the quantized one lacks).
`gemm_weight` uses those tensors when a checkpoint has them (`Fmt::Int4`); nothing is selected
by environment. `Gpu::gemm` expands 1024 rows at a time with `dequant_i4g` into the int8 path's
buffer and runs the bf16 GEMM, as the int8 expansion does. No quantizer of our own.

9B: 8.52 GiB on disk, of which the embedding table and the output head are 2 GB each in bf16.
Quality is the checkpoint's: mean KL 0.147 / 0.079 / 0.129 on the three trace references (the
same checkpoint dequantized to bf16 offline gives 0.142 / 0.079 / 0.127), top-1 235 / 232 /
241 of 256. `aq-check` on the short reference passes: greedy 32/32 identical to the
reference, batched, resumed and speculative paths identical to the single path. Exact gate
with this build: see the commit. B200 replay (turns/min, tok/s, TPOT p50 ms): 4 agents 91.2 /
247 / 4.9, 16 agents 334.0 / 881 / 6.4 (bf16 114.8 and 432.8; int8 with its kernel 362.5).

**8-bit head, embedding table in host memory.** A 4-bit checkpoint leaves the embedding table
and the output head in bf16 (2 GB each for the 9B, not tied). With such a checkpoint the head
is stored as int8 with a scale per 32 (`quant_i8b`) as soon as it has been uploaded, and the
embedding table is not uploaded at all: the checkpoint stays mapped, each step copies the rows
of its tokens to `x` (there is no `embed` kernel then), and the draft head keeps the rows of
its shortlist on the device (`short_e`, with a token-to-slot map; `mtp_prep` and
`mtp_chain_prep` read a token outside the shortlist as zeros, which can only cost a rejected
draft). The main path therefore uses the exact bf16 embeddings. Quality is unchanged (0.146 /
0.078 / 0.129 on the traces), as is the replay (1 / 4 / 16 agents: 23.8 / 91.8 / 331.0
turns/min); gate 31/31.

Device memory of `aq-serve` with the 4-bit 9B on a B200: what the engine allocates (its load
lines report it) and what `nvidia-smi` shows, which adds about 0.73 GiB of CUDA context and
libraries on that card:

| Settings | Allocated | Device |
|---|---|---|
| 8 slots, 1 GB KV pool, `--max-tokens 8192` (1.5 GiB of scratch) | 8.44 GiB | 9.4 GB |
| the same with `--max-tokens 1024` | 7.23 GiB | 8.1 GB |
| 4 slots, 1 GB pool, `--max-tokens 1024` | 6.97 GiB | 7.9 GB |
| the same without the MTP head (`AQ_MTP=0`) | 6.01 GiB | 6.9 GB |
| 4 slots, 0.5 GB pool, `--max-tokens 512`, `--max-ctx 16384`, no MTP head | 5.43 GiB | 6.3 GB |

The weights are 5.5 GiB with the draft head (layers 3.6, head 1.0, draft head with its two
shortlists 0.9) and 4.6 without. With the embedding table on the device in 8 bits each row
above was 0.75 GiB larger. Whether the third row fits an 8 GB card depends on that card's
context size and on whether it drives a display; not measured on one yet. Next reduction: an
8-bit KV cache.

**Direct kernel.** `gemm_i8w` has a 4-bit mode (`w4`): a ring slot holds the tile's packed
codes as stored (128 rows x 32 bytes) and the rows' f32 group scales; when a chunk is used
each thread unpacks its 16 bytes to int8 (flip bit 3, extend it through the high nibble) into
one shared operand tile, then the MMAs run as for int8. Shared memory is
`8192 + 4 * (4608 + ceil(m / 16) * 2176)` bytes, less than the int8 mode, whose request is now
sized from the row cap (and the cap lowered to what the device allows) instead of a constant
that exceeded the GB10's 100 KiB. The minimum weight size for the kernel is now 32 rows per SM
(`AQ_I8_KERNEL_MIN_N` overrides), the same choices as before on the B200.
B200 replay, kernel on / off (turns/min; TPOT p50 ms): 1 agent 24.8 / 22.0 (3.8 / 5.6), 4
agents 96.0 / 84.0 (4.1 / 6.0), 16 agents 339.0 / 309.8 (6.3 / 7.7), both runs with the decode
profile on. us per call: gate_up 175 / 247, gdn_in 101 / 145, lm_head (int8) 279 / 685; the
int8 mode does gate_up in 100 and the bf16 GEMM in 45, so on this card the unpack is the cost.
Quality is the same either way (0.147 / 0.078 / 0.127 against 0.146 / 0.078 / 0.129).
What it is for is a bandwidth-limited card, where it reads 0.5 bytes per weight against about
4.5 for the expansion; not measured on one yet.

### 8-bit KV cache (`AQ_KV=int8`, 2026-11-18)

Opt-in, not exact. A key's or a value's row is 272 bytes (256 int8 and its f32 scale, the
row's largest magnitude / 127) instead of 512, so a pool holds 1.88 times the tokens.
`attn_prep` writes the rows; `attention_tc` and `attn_decode_tq` request them as stored into a
staging area (`kv_tile8_cp_async`) and expand a tile to the bf16 tiles their MMAs already read
(`kv_tile8_unpack`). In `attention_tc` the staging area doubles as the second buffer: the next
tile is fetched while the current one is used. The reference kernels (`AQ_ATTN_NAIVE`,
`AQ_ATTN_TILE`) do not read it. A prefill tile is 32 keys and a page 16, so it is requested as
two halves like the bf16 tile; the first version requested 32 keys at once, read past the page
whenever two pages were not adjacent, and was caught by the prompt-checkpoint check. The gate
now has an `AQ_KV=int8` leg.

Quality is at the bf16 noise floor: mean KL 0.0025 / 0.0013 / 0.0016 on the trace references
(bf16 cache, same build: 0.0020 / 0.0022 / 0.0011), top-1 255 / 255 / 256 of 256; with the
4-bit checkpoint 0.145 / 0.076 / 0.124, as without it. `aq-check` passes on the three 9B
references.

B200 replay, bf16 weights (turns/min, 8-bit / bf16 cache): 4 agents 108.0 / 114.8, 16 agents
358.8 / 430.2, 32 agents 486.5 / 713.0. Slower on this card: the expansion is compute, and
the staging area leaves room for one sub-tile instead of two within 100 KiB. It is for memory
and for bandwidth-limited devices, where decode attention reads half the bytes; not measured
on one.

Device memory with the 4-bit 9B, 4 slots, `--max-tokens 1024` on a B200 (0.73 GiB of it is
the CUDA context): 8-bit cache with a 0.5 GB pool (about 27k tokens with the draft head's
layer) 7.36 GB, the same without the draft head 6.38 GB; a 1 GB pool 7.86 GB.

# Handoff: aqwenator state as of 2026-11-13

## FIRST: known correctness bug on main (fix exists, unmerged)
The memory-policy lane (branch `worktree-agent-ada1630c33961379c`) found that a shared-prefix
**fork admission that evicts to get a slot reuses a stale snapshot index**, so after an eviction it
can fork the *wrong prefix state* (silent wrong output under memory pressure). Fixed in `da01b7c`
(snapshot re-looked-up by content), together with a feasibility fix (`2d0338c`: an admission that
cannot fit no longer evicts parked sessions one by one and then fails) and a host-time fix
(`527332c`: prefix matching was O(waiting × parked × 18k tokens) per step). That branch is based on
c6f9b86 (before the speculation and co-execution merges): rebase onto main, run the full aq-check
suite (its new memory-pressure test + `AQ_CHECK_POLICY`), and merge these fixes first. Until then,
avoid memory-pressure workloads with session cache + snapshots on main.

Lane findings worth keeping: snapshot pinning + dedupe (151 snapshots → 1 at 32 agents) and
`--prewarm` (tune all cuBLASLt plans at startup; mid-run autotune is the likely cause of multi-second
ITL p99.9 spikes) should become defaults; idle-aware eviction, load control with `--quantum`, and
`--mixed-prefill-adaptive` as opt-in; MRU eviction, SRPT (until re-measured on the fixed build) and
25 ms frame pacing are not worth it. Best variants still trail vLLM at 16/32 agents (163 vs 189,
92 vs 188 turns/min); the remaining gap is long-context decode attention (~1 TB/s effective, ITL p50
11.5 vs 5.8 ms at 16 agents / 18k ctx) and re-prefill throughput. Its gate auto-runs on the B200 when
the driver recovers: `~/aq/logs/mem_gate.log`; sweeps: `a local sweep script`.

## What exists (main @ 7771327, all gated)
- Rust workspace: `aq-kernels` (cuda-oxide `#[cuda_module]` kernels), `aq-engine` (model, scheduler,
  session cache, aq-check, aq-bench), `aq-server` (`aq-serve`, OpenAI `/v1/completions`, SSE).
- Model family: `qwen3_5` dense (Qwen3.5 0.8B–27B, Qwen3.6-27B, Qwen3.8-27B), bf16 only (no FP8 yet).
- Kernels: tensor-core prefill attention (`attention_tc`), split-K decode attention with coalesced
  128-bit loads, tensor-core chunked gated delta rule (`gdn_tc_*`), single-token `gdn_decode`,
  vectorized norm/SwiGLU, two-phase argmax. cuBLASLt for GEMMs (M-bucketed, autotuned plans).
- Engine: continuous batching + chunked prefill, paged KV (refcounted pages), DeltaNet state slots,
  CUDA graphs for decode, session cache (park/resume exact prefixes, LRU), shared-prefix snapshots
  (fork state + share pages), KV pool sized from `--kv-gb`.
- Exact speculative decoding (prompt-lookup drafts, k=4 default in `aq-serve`): token-identical to
  greedy; +25–62% on random prompts, 2.3–4× on quote-back workloads.
- Decode/prefill co-execution (`AQ_COEXEC=1`, off by default): ~10× better ITL tail on B200 at equal
  throughput, worse TTFT; hurts throughput on the Spark (needs green contexts / SM partitioning).

## Correctness (the #1 rule)
`aq-check <model> <golden>` compares against transformers references in `~/aq/golden/*.safetensors`
(made by `scripts/golden.py`): teacher-forced logits at every generated position, free-running and
batched greedy, session resume, shared-prefix fork. Divergences are classified by top-2 margin
(near-tie < 0.25). Always run the full suite: 6 goldens × {default, `AQ_CHECK_BUDGET=1000
AQ_CHECK_CHUNK=777`} (the odd budget found a real upload/compute stream race), plus `AQ_SPEC_K=4`.
(2026-11-15: cuBLASLt plans are no longer picked by timing, so runs are repeatable without a switch.)

## Open branches (not merged)
- `worktree-agent-ae9e174bba1af161f` @ d7e410c — lossless KV compression (400 B rows, 1.28× tokens,
  logits byte-identical to raw). Rebased on 7771327, opt-in (`AQ_KV_COMPRESS=1`). **Gate not yet run**
  (B200 driver hang): run `scripts/kvc_gate.sh` before merging. Decode attention is ~2.3× slower than
  raw, so it only pays when capacity avoids evictions (memory-tight agentic runs, the Spark).
- `worktree-agent-ada1630c33961379c` — memory policy / scheduler lane; **contains the stale-fork-index correctness fix** (see top). Eviction policies, snapshot
  pinning, load control, SRPT). See its final report / commit messages; results in
  `~/aq/results/agentic/mem-*` on the B200.
- `worktree-agent-a6d76573bade9b155` — skinny GEMM experiments (did not beat cuBLASLt; keep unmerged).

## Benchmarks (where things stand)
- B200, 9B, 1024/256: we win at c ≤ 4 (e.g. c=1 252 vs 235 tok/s), lose at c ≥ 16 (prefill speed).
- DGX Spark, 9B, 1024/256: 93–108% of vLLM throughput; ITL p99 better at c ≤ 16; vLLM's 2048-token
  mixed steps are ~30% cheaper on GB10 (profile Spark prefill next). Cold start 6 s vs vLLM 163 s.
- Agentic (B200, 9B, 32 GB budget): we win at 4 agents, lose at 8–32 (memory policy; 32 agents
  collapses). Spark agentic head-to-head was running at handoff: results in
  `~/aq/results/agentic/spark-*` on the Spark, log `~/aq/logs/run_spark_queue.log`.
  It uses a pre-speculation build (`aq-serve-current` = session cache only).
- Progress page: https://claude.ai/artifact/MNkgXru1gcTGbZz3Zw3zwN (data files under `docs/progress/`,
  regenerate `bench.json`/`agentic.json` with `scripts/collect.py`, `AQ_GPU_LABEL=GB10` on the Spark).

## Machines and gotchas
- Two machines were used: a B200 box (x86_64, `sm_100`, shared with other users) and a DGX Spark
  (aarch64, `sm_121`, unified memory). How to reach them is not in this repo; ssh wrappers and
  sync scripts live in an untracked `.local/` directory.
- Both use the user-space toolchain from `scripts/setup.sh` under `~/aq`; `source scripts/env.sh`,
  then `cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins` (the
  numbers in this file were measured with `--arch sm_100a` / `sm_121a` builds; `sm_80` runs at
  the same speed).
- On a shared machine: never `pkill`/`pgrep -f` (self-matches, other users); kill only your own
  PIDs; use `grep "[p]attern"` in wait loops; pick an idle GPU first.
- Nsight Compute counters may be blocked (`ERR_NVGPUCTRPERM`); use `AQ_PROFILE=1|decode|prefill`
  (CUDA events, works inside graphs) and `AQ_KINFO=1` (registers/spills per kernel).
- vLLM baselines: `scripts/baseline_vllm.sh` (readiness wait 900 s, `--temperature 0`),
  `scripts/agentic_sweep.sh`, `scripts/agentic_bench.py`.
- Correctness gate: `scripts/gate.sh <aq-check> <reference dir>`.

## Research
`docs/research/sota-survey.md`, `docs/research/moonshots.md`, `docs/research/compression-and-skinny-gemm.md`.
Top unexplored ideas: prompt-end snapshots + runahead for chat-template re-renders, cooperative scan
over shared prefixes, batch-invariant numerics, NVMe session tier (safetensors I/O), green-context
SM partitioning on the Spark, FP8 weights for Qwen3.8-27B.
