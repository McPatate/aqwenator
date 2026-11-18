//! Offline serving benchmark, shaped like `vllm bench serve --dataset-name random
//! --max-concurrency C --ignore-eos`: a closed loop keeps C requests in flight, each with
//! `input_len` random prompt tokens and exactly `output_len` generated tokens.
//!
//!     aq-bench <model> --input-len 1024 --output-len 256 --concurrency 1,8,32,64,128,256 \
//!              [--num-prompts N] [--max-seqs 256] [--max-tokens 8192] [--out results.json]
//!
//! Reports output tokens/s, TTFT, TPOT and ITL percentiles per concurrency, plus cold start:
//! process start -> weights resident -> first token of a warmup request.
//! No HTTP or tokenization in the loop: this measures the engine alone.

use anyhow::{Context, Result};
use aq_engine::engine::{Engine, Request};
use aq_engine::model::{Limits, Model};
use aq_engine::weights::resolve_model;
use std::collections::HashMap;
use std::time::Instant;

struct Args {
    model: String,
    input_len: usize,
    output_len: usize,
    concurrency: Vec<usize>,
    num_prompts: Option<usize>,
    max_seqs: usize,
    max_tokens: usize,
    mixed_prefill: Option<usize>,
    spec_k: usize,
    prompts_file: Option<String>,
    out: Option<String>,
}

fn parse() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let model = it.next().context("usage: aq-bench <model> [options]")?;
    let mut a = Args {
        model,
        input_len: 1024,
        output_len: 256,
        concurrency: vec![1, 4, 8, 16, 32, 64],
        num_prompts: None,
        max_seqs: 64,
        max_tokens: 8192,
        mixed_prefill: None,
        spec_k: 0,
        prompts_file: None,
        out: None,
    };
    while let Some(k) = it.next() {
        let v = it.next().with_context(|| format!("missing value for {k}"))?;
        match k.as_str() {
            "--input-len" => a.input_len = v.parse()?,
            "--output-len" => a.output_len = v.parse()?,
            "--concurrency" => a.concurrency = v.split(',').map(|x| x.parse()).collect::<Result<_, _>>()?,
            "--num-prompts" => a.num_prompts = Some(v.parse()?),
            "--max-seqs" => a.max_seqs = v.parse()?,
            "--max-tokens" => a.max_tokens = v.parse()?,
            "--out" => a.out = Some(v),
            "--mixed-prefill" => a.mixed_prefill = Some(v.parse()?),
            "--spec-k" => a.spec_k = v.parse()?,
            "--prompts-file" => a.prompts_file = Some(v),
            _ => anyhow::bail!("unknown option {k}"),
        }
    }
    Ok(a)
}

/// xorshift64*, so prompts are reproducible without a dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let i = ((p / 100.0) * (v.len() - 1) as f64).round() as usize;
    v[i]
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn main() -> Result<()> {
    let process_start = Instant::now();
    let a = parse()?;
    let dir = resolve_model(&a.model)?;
    let limits = Limits {
        max_tokens: a.max_tokens,
        max_seqs: a.max_seqs,
        max_ctx: (a.input_len + a.output_len).next_multiple_of(16),
        block_size: 16,
        kv_budget_bytes: 0,
        kv_blocks: 0,
        cache_slots: 0,
    };
    let model = Model::load(&dir, limits)?;
    let loaded_s = process_start.elapsed().as_secs_f64();
    let load = (model.load.open_s, model.load.upload_s, model.load.weight_bytes);
    let vocab = model.cfg.vocab_size as u64;
    let mut eng = Engine::new(model, vec![]);
    if let Some(m) = a.mixed_prefill {
        eng.mixed_prefill_budget = m;
    }
    eng.spec_k = a.spec_k;

    // Cold start: first token of a short request right after load (includes module load,
    // cuBLASLt plan creation, first-touch of every kernel).
    eng.add(Request { id: u64::MAX, prompt: vec![9707, 11, 1879, 0], max_new_tokens: 1, ignore_eos: true })?;
    while eng.has_work() {
        eng.step()?;
    }
    let first_token_s = process_start.elapsed().as_secs_f64();
    println!(
        "cold start: weights resident at {loaded_s:.2}s (open {:.3}s, upload {:.2}s, {:.2} GiB), first token at {first_token_s:.2}s",
        load.0,
        load.1,
        load.2 as f64 / (1u64 << 30) as f64
    );

    let mut rng = Rng(0x5eed_1234_abcd_0001);
    // Real prompts (token ids), cycled; otherwise random tokens of --input-len.
    let file_prompts: Vec<Vec<u32>> = match &a.prompts_file {
        Some(f) => serde_json::from_str(&std::fs::read_to_string(f)?)?,
        None => Vec::new(),
    };
    let mut results = Vec::new();
    for &c in &a.concurrency {
        let n = a.num_prompts.unwrap_or((16 * c).max(200));
        let mut next_id = 0u64;
        let mut arrive: HashMap<u64, Instant> = HashMap::new();
        let mut last_tok: HashMap<u64, Instant> = HashMap::new();
        let mut ttft = Vec::new();
        let mut itl = Vec::new();
        let mut tpot = Vec::new();
        let mut e2el = Vec::new();
        let mut first: HashMap<u64, f64> = HashMap::new();
        let mut out_tokens = 0usize;
        let mut steps = 0usize;
        let spec0 = eng.spec;
        let mut host_us = 0u64;
        let start = Instant::now();
        let mut submit = |eng: &mut Engine, next_id: &mut u64, arrive: &mut HashMap<u64, Instant>| -> Result<()> {
            let prompt = if file_prompts.is_empty() {
                (0..a.input_len).map(|_| (1000 + rng.next() % (vocab - 2000)) as u32).collect()
            } else {
                file_prompts[*next_id as usize % file_prompts.len()].clone()
            };
            eng.add(Request { id: *next_id, prompt, max_new_tokens: a.output_len, ignore_eos: true })?;
            arrive.insert(*next_id, Instant::now());
            *next_id += 1;
            Ok(())
        };
        while (next_id as usize) < n.min(c) {
            submit(&mut eng, &mut next_id, &mut arrive)?;
        }
        while eng.has_work() {
            let evs = eng.step()?;
            steps += 1;
            host_us += eng.last.host_us;
            let now = Instant::now();
            for e in evs {
                out_tokens += 1;
                if e.index == 0 {
                    let t = (now - arrive[&e.id]).as_secs_f64();
                    ttft.push(t * 1e3);
                    first.insert(e.id, t);
                } else {
                    itl.push((now - last_tok[&e.id]).as_secs_f64() * 1e3);
                }
                last_tok.insert(e.id, now);
                if e.finished {
                    let e2e = (now - arrive[&e.id]).as_secs_f64();
                    e2el.push(e2e * 1e3);
                    if a.output_len > 1 {
                        tpot.push((e2e - first[&e.id]) / (a.output_len - 1) as f64 * 1e3);
                    }
                    if (next_id as usize) < n {
                        submit(&mut eng, &mut next_id, &mut arrive)?;
                    }
                }
            }
        }
        let dur = start.elapsed().as_secs_f64();
        let mut r = serde_json::json!({
            "engine": "aqwenator",
            "model": a.model,
            "input_len": a.input_len,
            "output_len": a.output_len,
            "concurrency": c,
            "num_prompts": n,
            "max_tokens_per_step": a.max_tokens,
            "mixed_prefill_budget": a.mixed_prefill,
            "duration_s": dur,
            "output_throughput": out_tokens as f64 / dur,
            "total_token_throughput": (out_tokens + n * a.input_len) as f64 / dur,
            "steps": steps,
            "host_us_per_step": host_us as f64 / steps.max(1) as f64,
            "cold_start_loaded_s": loaded_s,
            "cold_start_first_token_s": first_token_s,
            "spec_k": a.spec_k,
            "spec_drafted": eng.spec.drafted - spec0.drafted,
            "spec_accepted": eng.spec.accepted - spec0.accepted,
            "spec_tokens_per_decode_step": (eng.spec.emitted - spec0.emitted) as f64 / (eng.spec.decode_entries - spec0.decode_entries).max(1) as f64,
        });
        for (name, v) in [("ttft", &mut ttft), ("tpot", &mut tpot), ("itl", &mut itl), ("e2el", &mut e2el)] {
            r[format!("mean_{name}_ms")] = serde_json::json!(mean(v));
            for (label, q) in [("median", 50.0), ("p90", 90.0), ("p99", 99.0), ("p99.9", 99.9)] {
                r[format!("{label}_{name}_ms")] = serde_json::json!(pct(v, q));
            }
        }
        let g = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
        println!(
            "c={c:<3} out_tok/s={:8.1} ttft p50/p99/p999={:7.1}/{:7.1}/{:7.1}ms itl p50/p99/p999={:6.2}/{:6.2}/{:6.2}ms tpot p99={:6.2}ms host/step={:.0}us",
            g("output_throughput"),
            g("median_ttft_ms"),
            g("p99_ttft_ms"),
            g("p99.9_ttft_ms"),
            g("median_itl_ms"),
            g("p99_itl_ms"),
            g("p99.9_itl_ms"),
            g("p99_tpot_ms"),
            g("host_us_per_step"),
        );
        if a.spec_k > 0 {
            println!(
                "      spec: drafted {} accepted {} ({:.0}%), {:.2} tokens per decode step",
                r["spec_drafted"], r["spec_accepted"],
                100.0 * r["spec_accepted"].as_f64().unwrap_or(0.0) / r["spec_drafted"].as_f64().unwrap_or(1.0).max(1.0),
                g("spec_tokens_per_decode_step")
            );
        }
        if let Some(p) = eng.model.prof.as_mut() {
            println!("{}", p.report());
        }
        results.push(r);
    }
    if let Some(path) = a.out {
        std::fs::write(&path, serde_json::to_string_pretty(&results)?)?;
        println!("wrote {path}");
    }
    Ok(())
}
