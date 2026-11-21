# Development

## Build

With Docker:

```sh
docker build -t aqwenator .                                         # image
docker build --target binaries --output type=local,dest=bin .       # binaries only
```

Without: the CUDA 13 toolkit, clang 21, Rust 1.99.1 and `cargo-oxide`. `scripts/setup.sh`
installs them in user space under `~/aq`.

```sh
source scripts/env.sh
cargo oxide build --arch sm_80 -- --release -p aq-engine -p aq-server --bins
```

Kernels target `sm_80` and the driver compiles them for the device, so one build serves every
supported GPU. Do not run `cargo fmt --all`.

| Binary | Use |
|---|---|
| `aq-serve` | The server |
| `aq-check` | Correctness against reference files ([correctness](correctness.md)) |
| `aq-bench` | Synthetic load without HTTP |
| `aq-ceiling` | What a device allows: bandwidth, GEMM rate, bounds for one step of a model |

## Server options

```
aq-serve <model> [options]
```

| Option | Default | Meaning |
|---|---|---|
| `--port` | 8000 | |
| `--max-seqs` | 64 | State slots: running plus parked sessions |
| `--max-running` | | Sequences running at once |
| `--max-ctx` | 32768 | Longest context |
| `--max-tokens` | 8192 | Prefill tokens per step; also sizes the scratch memory |
| `--kv-gb` | | Attention cache pool |
| `--mixed-prefill` | | Prefill tokens allowed in a step that also decodes |
| `--spec-k` | 4 | Draft budget per sequence, 0 = off |
| `--session-cache` | true | Park finished sessions for reuse |
| `--burst-prefill`, `--burst-every` | 4096, 3 | A long prompt gets a whole step every few steps |
| `--srpt` | true | Shortest prefill first |
| `--prewarm` | false | Create every GEMM plan at startup |
| `--evict`, `--pin-snapshots`, `--load-control` | | Cache policy |

Environment:

| Variable | Meaning |
|---|---|
| `AQ_QUANT=fp8\|int8` | Weight tier ([quantization](quantization.md)) |
| `AQ_KV=int8` | 8-bit attention cache |
| `AQ_EXPERT_GB=N` | 4-bit mixture-of-experts checkpoints: keep only N GiB of experts on the device (the least recently used ones are replaced), the rest in page-locked host memory. Outputs do not change |
| `AQ_EXPERT_RAM_GB=N` | With `AQ_EXPERT_GB`: keep only N GiB of experts in host memory too, and read the others from the checkpoint with direct I/O |
| `AQ_EXPERT_AHEAD=N` | With `AQ_EXPERT_GB`: experts per token uploaded ahead for the next layer (8; 0 = none) |
| `AQ_MTP=0` | Do not load the draft head |
| `AQ_MTP_K`, `AQ_MTP_SHORT` | Chained drafts (3), draft shortlist size (32768) |
| `AQ_BG_PREFILL=N` | Background prefill lane |
| `AQ_PROFILE=1\|decode\|prefill` | Per-operation GPU time, printed periodically |
| `AQ_TRACE=1` | Step and request logs |
| `AQ_EXPERT_TRACE=<file>` | With `AQ_EXPERT_GB`: record what every layer asked of the expert pool (records needed, guesses for the next two layers), for `aq-replay` |
| `AQ_LAYERS=N` | Load the first N layers only; `aq-check` then prints each layer's error against the reference and stops |
| `AQ_KINFO=1` | Registers and shared memory of every kernel |
| `AQ_DEVICE_GB=N`, `AQ_SMEM_KB=N` | Refuse to allocate past N GiB, and to use more than N KB of shared memory per block: sizes a configuration for a smaller card |

`AQ_PROFILE` records CUDA events between operations and turns CUDA graphs off, so small
operations include their launch latency.

## Scripts

| Script | Use |
|---|---|
| `setup.sh`, `env.sh` | Toolchain in user space |
| `golden.py`, `gate.sh` | Reference files and the correctness gate |
| `trace_bench.py`, `trace_sweep.sh`, `trace_tail.py` | The replay benchmark ([benchmarks](benchmarks.md)) |
| `baseline_vllm.sh`, `baseline_llamacpp.sh`, `setup_ref.sh`, `cuda_shim.sh` | Other engines on the same replay |
| `convert_autoround.py` | 4-bit checkpoints |

## Working on kernels

- bf16 values travel as raw `u16`; math happens in f32. Where the reference materializes a
  bf16 tensor, round (`rb`) at the same place.
- A kernel takes raw device pointers into the engine's arena. Its launch shape must not
  depend on data, or the step cannot be captured as a graph.
- Check shared memory against the smallest device you target (about 100 KiB per block on a
  GB10) and opt in above 48 KiB.
- Integer division by a runtime value is slow enough to show in a profile; so is a 64-bit
  one.
- Measure against `aq-ceiling` before optimizing: a kernel at its bandwidth bound needs fewer
  bytes, not better code.
- Keep the fastest variant and delete the others, unless they differ in correctness.

## Documentation

The pages in `docs/` are an [mdBook](https://rust-lang.github.io/mdBook/): `mdbook serve`
previews them, and a push to `main` publishes them to GitHub Pages. `docs/results.md` is
generated ([benchmarks](benchmarks.md#results)).

## Release

Pushing a tag `vX.Y.Z` builds amd64 and arm64 images with provenance and an SBOM, publishes
`ghcr.io/<owner>/aqwenator:X.Y.Z` and `:latest`, and attaches the binaries and their SHA-256
sums to a GitHub release. Versions follow `0.x`: a minor bump may break.

## Replaying expert reads without a model

`aq-replay` (its own crate, no GPU) takes a trace recorded with `AQ_EXPERT_TRACE` and performs
the reads on a real disk, with the device and host tiers simulated. It is where read policies
and backends are compared: direct reads from a thread pool, io_uring, buffered reads and
memory-mapped files. The readers themselves are the `aq-io` crate, which depends on neither
the engine nor a checkpoint format.

```sh
cargo build -p aq-replay --release
python scripts/expert_layout.py <model dir> > run.trace.layout   # on another machine than the recording
target/release/aq-replay run.trace --probe --backend uring
target/release/aq-replay run.trace --policy ahead --backend uring --dev-gib 3 --ram-gib 4
```

The page cache is the host tier of the `mmap` backend: run it under a memory limit
(`systemd-run --user --scope -p MemoryMax=5G`, or a container) to compare at equal memory.
Its hints (`--madvise random`, `--mmap willneed|populate`, `--mmap-evict`) are there to give
the mapping its best case; which of them helps depends on the machine. `--max-secs` bounds a
run, which the slow configurations need.
