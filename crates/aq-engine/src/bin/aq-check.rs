//! Correctness check against transformers reference data (scripts/golden.py).
//!
//!     aq-check <model id or dir> <golden.safetensors>
//!
//! 1. Per-layer residual stream after prefill (when the golden has `hidden.*`).
//! 2. Teacher-forced logits: prefill the prompt, then feed the reference's greedy tokens one at
//!    a time through the decode path and compare every logits row with the reference. A top-1
//!    disagreement counts as a near-tie when the reference's own top-2 margin is below 0.25
//!    (one bf16 ulp at typical logit magnitudes is ~0.06-0.125); anything else is a real miss.
//! 3. Free-running greedy tokens, single sequence and 3 sequences batched through the scheduler.
//!
//! Exit status is non-zero on any real miss.

use anyhow::{Context, Result};
use aq_engine::engine::{Control, Engine, Request};
use aq_engine::sample::{Sampler, Sampling, by_id, distribution, uniform};
use aq_engine::model::{Limits, Model};
use aq_engine::weights::resolve_model;
use std::time::Instant;

const NEAR_TIE: f32 = 0.25;
/// Tokens generated per request in the speculative-decoding identity test.
const SPEC_TEST_LEN: usize = 256;

/// KL(ref || ours) in nats between the softmax distributions of two logits rows.
fn kl(refr: &[f32], ours: &[f32]) -> f64 {
    let lse = |v: &[f32]| {
        let m = v.iter().copied().fold(f32::MIN, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    };
    let (zr, zo) = (lse(refr), lse(ours));
    refr.iter().zip(ours).map(|(&r, &o)| (r as f64 - zr).exp() * ((r as f64 - zr) - (o as f64 - zo))).sum()
}

fn rel_err(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut num = 0f64;
    let mut den = 0f64;
    let mut max = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (x - y) as f64;
        num += d * d;
        den += (y as f64) * (y as f64);
        max = max.max(d.abs());
    }
    ((num / den.max(1e-30)).sqrt(), max)
}

/// Decode-path vs prefill-path logits along a plain run's own tokens, at several distances
/// before token `same`: gradual growth is numeric drift, a jump is a bug.
fn drift(eng: &mut Engine, prompt: &[u32], off: &[u32], same: usize) -> Result<()> {
    for back in [128usize, 64, 32, 16, 8, 4, 2, 1, 0] {
        if back > same {
            continue;
        }
        let l = same - back;
        eng.add(Request { id: 398, prompt: prompt.to_vec(), max_new_tokens: l + 1, ignore_eos: true })?;
        let mut got = Vec::new();
        while eng.has_work() {
            for e in eng.step()? {
                got.push(e.token);
            }
        }
        let dec = eng.model.logits()?;
        let p: Vec<u32> = prompt.iter().chain(&off[..l]).copied().collect();
        eng.add(Request { id: 397, prompt: p, max_new_tokens: 1, ignore_eos: true })?;
        while eng.has_work() {
            eng.step()?;
        }
        let pre = eng.model.logits()?;
        let (dt, da, db) = top2(&dec);
        let (pt, pa, pb) = top2(&pre);
        let maxd = dec.iter().zip(&pre).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        let num: f32 = dec.iter().zip(&pre).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = pre.iter().map(|y| y * y).sum();
        println!(
            "    drift at token {l}: single-decode replay {} the plain run; decode top-1 {dt} (margin {:.3}), prefill top-1 {pt} (margin {:.3}), max |dlogit| {maxd:.3}, rel_l2 {:.2e}",
            if got[..] == off[..l + 1] { "matches" } else { "DIFFERS from" },
            da - db,
            pa - pb,
            (num / den).sqrt()
        );
    }
    Ok(())
}

fn top2(x: &[f32]) -> (usize, f32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in x.iter().enumerate() {
        if v > v1 {
            v2 = v1;
            v1 = v;
            i1 = i;
        } else if v > v2 {
            v2 = v;
        }
    }
    (i1, v1, v2)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_spec = args.get(1).context("usage: aq-check <model> <golden.safetensors>")?;
    let golden_path = args.get(2).context("usage: aq-check <model> <golden.safetensors>")?;
    let dir = resolve_model(model_spec)?;

    let gbytes = std::fs::read(golden_path)?;
    let g = safetensors::SafeTensors::deserialize(&gbytes)?;
    let as_f32 = |name: &str| -> Result<Vec<f32>> {
        let v = g.tensor(name)?;
        Ok(v.data().chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let as_u32 = |name: &str| -> Result<Vec<u32>> {
        let v = g.tensor(name)?;
        Ok(v.data().chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
    };
    let ids = as_u32("input_ids")?;
    let greedy_ref = as_u32("greedy_ids")?;
    let has_hidden = g.names().iter().any(|n| n.starts_with("hidden."));
    let has_tf = g.names().contains(&"tf_logits");

    let ctx = (ids.len() + greedy_ref.len().max(SPEC_TEST_LEN) + 64).next_multiple_of(256);
    // Prefill chunk per step (AQ_CHECK_CHUNK): small, unaligned values exercise sequences whose
    // prefill spans steps at arbitrary positions.
    let chunk: usize = std::env::var("AQ_CHECK_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(8192)
        .min(std::env::var("AQ_CHECK_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(8192));
    let t0 = Instant::now();
    let budget: usize = std::env::var("AQ_CHECK_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);
    // AQ_CHECK_CACHE_SLOTS: run every leg with extra cache-only state slots.
    let cache_slots: usize = std::env::var("AQ_CHECK_CACHE_SLOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut m = Model::load(&dir, Limits { max_tokens: budget, max_seqs: 8, max_ctx: ctx, block_size: 16, kv_budget_bytes: 0, kv_blocks: 0, cache_slots })?;
    println!(
        "load: {:.2}s, {:.2} GiB on device; prompt {} tokens, {} reference tokens",
        t0.elapsed().as_secs_f64(),
        m.load.weight_bytes as f64 / (1u64 << 30) as f64,
        ids.len(),
        greedy_ref.len()
    );
    let n_layers = m.cfg.num_hidden_layers;
    let vocab = m.cfg.vocab_size;
    let mut failed = false;

    // ---- 1. prefill (chunked by the step budget) + per-layer comparison
    m.reset_slot(0)?;
    let mut dump = Vec::new();
    let mut pos = 0;
    let mut first = 0;
    while pos < ids.len() {
        let n = (ids.len() - pos).min(chunk);
        let want_dump = has_hidden && n == ids.len();
        first = m.step(&m.single(&ids[pos..pos + n], pos), if want_dump { Some(&mut dump) } else { None })?[0];
        pos += n;
    }
    if has_hidden && !dump.is_empty() {
        let mut worst = (0f64, 0usize);
        for (l, ours) in dump.iter().enumerate().take(n_layers - 1) {
            let (rel, _) = rel_err(ours, &as_f32(&format!("hidden.{}", l + 1))?);
            if rel > worst.0 {
                worst = (rel, l);
            }
        }
        println!("layers: worst rel_l2 {:.2e} (layer {})", worst.0, worst.1);
    }
    let lg = m.logits()?;
    let (rel, _) = rel_err(&lg, &as_f32("logits_last")?);
    println!("prefill logits rel_l2 {rel:.2e}");

    // ---- 2. teacher-forced decode
    let tf = if has_tf { as_f32("tf_logits")? } else { Vec::new() };
    // Reference margin (top1 - top2) at generated position k, and its own top-1.
    let ref_at = |k: usize| -> Option<(usize, f32)> {
        (has_tf && k < greedy_ref.len()).then(|| {
            let (i, a, b) = top2(&tf[k * vocab..(k + 1) * vocab]);
            (i, a - b)
        })
    };
    if has_tf {
        let inconsistent: Vec<usize> =
            (0..greedy_ref.len()).filter(|&k| ref_at(k).unwrap().0 != greedy_ref[k] as usize).collect();
        if !inconsistent.is_empty() {
            let k = inconsistent[0];
            println!(
                "reference self-check: transformers generate() and its own full forward disagree at {} positions (first k={k}, margin {:.3}); free-running comparisons after that point are against generate()",
                inconsistent.len(),
                ref_at(k).unwrap().1
            );
        }
    }
    if has_tf {
        let (mut agree, mut ties, mut misses, mut worst_rel) = (0, 0, 0, 0f64);
        let (mut kl_sum, mut kl_max) = (0f64, 0f64);
        let mut pos = ids.len();
        for k in 0..greedy_ref.len() {
            if k > 0 {
                m.step(&m.single(&[greedy_ref[k - 1]], pos), None)?;
                pos += 1;
            }
            let ours = if k == 0 { lg.clone() } else { m.logits()? };
            let refr = &tf[k * vocab..(k + 1) * vocab];
            let (rel, _) = rel_err(&ours, refr);
            worst_rel = worst_rel.max(rel);
            let d = kl(refr, &ours);
            kl_sum += d;
            kl_max = kl_max.max(d);
            let (ro, _, _) = top2(&ours);
            let (rr, r1, r2) = top2(refr);
            if ro == rr {
                agree += 1;
            } else if r1 - r2 < NEAR_TIE {
                ties += 1;
            } else {
                misses += 1;
                println!("  MISS at k={k}: ours {ro} ref {rr} (ref margin {:.3})", r1 - r2);
            }
        }
        println!(
            "teacher-forced: top-1 {agree}/{n} agree, {ties} near-ties, {misses} real misses; worst logits rel_l2 {worst_rel:.2e}; KL(ref||ours) mean {:.2e} max {kl_max:.2e} nats",
            kl_sum / greedy_ref.len() as f64,
            n = greedy_ref.len()
        );
        if std::env::var("AQ_CHECK_TF_ONLY").is_ok() {
            return Ok(());
        }
        failed |= misses > 0;
    }

    // ---- 3. free-running greedy: single, then 3 sequences batched. Teacher forcing advanced
    // slot 0's recurrent state, so start over from a fresh prefill.
    m.reset_slot(0)?;
    let mut pos = 0;
    while pos < ids.len() {
        let n = (ids.len() - pos).min(chunk);
        first = m.step(&m.single(&ids[pos..pos + n], pos), None)?[0];
        pos += n;
    }
    // Our own top-2 margin at each free-running position: the right yardstick for comparing
    // other runs of *our* engine against this one once it has left the reference trajectory.
    let margin_of = |l: &[f32]| {
        let (_, a, b) = top2(l);
        a - b
    };
    let mut our_margin = vec![margin_of(&m.logits()?)];
    let mut out = vec![first];
    while out.len() < greedy_ref.len() {
        let next = m.step(&m.single(&[*out.last().unwrap()], pos), None)?[0];
        our_margin.push(margin_of(&m.logits()?));
        pos += 1;
        out.push(next);
    }
    // A divergence from a sequence that matched up to k is acceptable only at a near-tie of the
    // reference (teacher-forced margin at k below NEAR_TIE).
    let classify = |name: &str, seq: &[u32], against: &[u32], margins: &dyn Fn(usize) -> Option<f32>| -> bool {
        match seq.iter().zip(against).position(|(a, b)| a != b) {
            None => {
                println!("{name}: {}/{} identical", seq.len().min(against.len()), against.len());
                true
            }
            Some(k) => match margins(k).map(|m| ((), m)) {
                Some((_, margin)) if margin < NEAR_TIE => {
                    println!("{name}: identical for {k}/{} tokens, then diverges at a near-tie (margin {margin:.3}): ok", against.len());
                    true
                }
                Some((_, margin)) => {
                    println!("{name}: diverges at k={k} where the margin is {margin:.3}: REAL MISMATCH");
                    false
                }
                None => {
                    println!("{name}: diverges at k={k} (no reference logits to classify): FAIL");
                    false
                }
            },
        }
    };
    let ref_margin = |k: usize| ref_at(k).map(|(_, m)| m);
    let own_margin = |k: usize| our_margin.get(k).copied();
    failed |= !classify("greedy vs reference", &out, &greedy_ref, &ref_margin);
    let single = out.clone();

    let mut eng = Engine::new(m, vec![]);
    if std::env::var("AQ_CHECK_NOCACHE").is_ok() {
        eng.session_cache = false;
    }
    // AQ_SPEC_K: run the engine-level checks below with speculative decoding on.
    let spec_env: usize = std::env::var("AQ_SPEC_K").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    eng.spec_k = spec_env;
    // AQ_CHECK_POLICY=mru,pin,load,srpt: run the engine-level tests under these policies.
    if let Ok(p) = std::env::var("AQ_CHECK_POLICY") {
        for f in p.split(',') {
            match f {
                "mru" => eng.policy.evict = aq_engine::engine::Evict::Mru,
                "idle" => eng.policy.evict = aq_engine::engine::Evict::Idle,
                "pin" => eng.policy.pin_snapshots = true,
                "load" => eng.policy.load_control = true,
                "srpt" => eng.policy.srpt = true,
                "q4" => eng.policy.quantum = 4,
                _ => {}
            }
        }
        println!("policy: {:?}", eng.policy);
    }
    for id in 0..3u64 {
        eng.add(Request { id, prompt: ids.clone(), max_new_tokens: greedy_ref.len(), ignore_eos: true })?;
    }
    let mut got: Vec<Vec<u32>> = vec![Vec::new(); 3];
    while eng.has_work() {
        for e in eng.step()? {
            got[e.id as usize].push(e.token);
        }
    }
    for (i, gseq) in got.iter().enumerate() {
        // Margins from our own single run: same context as the batched run up to the divergence.
        failed |= !classify(&format!("batched seq {i} vs single"), gseq, &single, &own_margin);
    }

    // ---- 3b. staggered admission: one sequence is decoding when two more arrive and prefill,
    // so decode steps and prefill chunks share the GPU (mixed steps, or concurrent lanes with
    // AQ_COEXEC=1). All three must match the single-sequence run.
    let overlap_before = eng.overlap_steps;
    let cache_was = eng.session_cache;
    eng.session_cache = false; // full prefills, so the late arrivals really overlap with decode
    eng.add(Request { id: 300, prompt: ids.clone(), max_new_tokens: greedy_ref.len(), ignore_eos: true })?;
    let mut st: Vec<Vec<u32>> = vec![Vec::new(); 3];
    let mut added = false;
    while eng.has_work() {
        for e in eng.step()? {
            st[(e.id - 300) as usize].push(e.token);
        }
        if !added && st[0].len() >= 2 {
            for id in 301..303u64 {
                eng.add(Request { id, prompt: ids.clone(), max_new_tokens: greedy_ref.len(), ignore_eos: true })?;
            }
            added = true;
        }
    }
    eng.session_cache = cache_was;
    for (i, gseq) in st.iter().enumerate() {
        failed |= !classify(&format!("staggered seq {i} vs single"), gseq, &single, &own_margin);
    }
    if eng.coexec && eng.spec_k == 0 {
        let overlap = eng.overlap_steps - overlap_before;
        println!("co-execution: {overlap} decode steps ran while a prefill batch was in flight");
        failed |= overlap == 0;
    } else if eng.coexec {
        println!("co-execution: off while speculating (spec_k = {}), normal path used", eng.spec_k);
    }

    // ---- 4. session cache: a request that extends a finished one must resume from the parked
    // state and produce the same tokens as a cold engine (session cache off) on the same prompt.
    let turn2: Vec<u32> = ids.iter().copied().chain(single[..8].iter().copied()).chain(ids[..ids.len().min(37)].iter().copied()).collect();
    let mut run_turns = |eng: &mut Engine, cached: bool| -> Result<(Vec<u32>, u64)> {
        eng.session_cache = cached;
        let mut out = Vec::new();
        for (id, prompt, n) in [(100u64, ids.clone(), 9usize), (101, turn2.clone(), 16)] {
            eng.add(Request { id, prompt, max_new_tokens: n, ignore_eos: true })?;
            while eng.has_work() {
                for e in eng.step()? {
                    if e.id == 101 {
                        out.push(e.token);
                    }
                }
            }
        }
        Ok((out, eng.cache.reused_tokens))
    };
    let (warm, reused) = run_turns(&mut eng, true)?;
    let (cold, _) = run_turns(&mut eng, false)?;
    let same = warm.iter().zip(&cold).take_while(|(a, b)| a == b).count();
    println!("session cache: turn 2 resumed from {reused} cached tokens; {same}/{} tokens identical to a cold prefill", cold.len());
    failed |= reused == 0 || same != cold.len();

    // ---- 4b. prompt checkpoint: a request that shares all but the last token of the previous
    // *prompt* and continues with tokens other than the ones this engine generated (what a
    // chat template re-render does) must resume from the checkpoint taken just before the end
    // of that prompt, and match a cold engine.
    {
        let cut = ids.len() - 1;
        let ckpt = ids.len().saturating_sub(aq_engine::engine::CKPT_BACKOFF) / 16 * 16;
        let turn2: Vec<u32> = ids[..cut].iter().copied().chain(ids[..ids.len().min(41)].iter().map(|&t| t ^ 5)).collect();
        // `overlap`: the second request arrives while the first is still generating, so the
        // checkpoint's last (partly filled) page is shared with a running sequence and must be
        // copied before the second request writes to it.
        let run = |eng: &mut Engine, cached: bool, overlap: bool| -> Result<(Vec<Vec<u32>>, u64)> {
            eng.session_cache = cached;
            eng.clear_cache();
            let before = eng.cache.reused_tokens;
            let mut out = vec![Vec::new(); 2];
            eng.add(Request { id: 110, prompt: ids.clone(), max_new_tokens: 24, ignore_eos: true })?;
            let mut added = false;
            while eng.has_work() || !added {
                for e in eng.step()? {
                    out[(e.id - 110) as usize].push(e.token);
                }
                let first_done = out[0].len() >= if overlap { 3 } else { 24 };
                if !added && first_done {
                    eng.add(Request { id: 111, prompt: turn2.clone(), max_new_tokens: 16, ignore_eos: true })?;
                    added = true;
                }
            }
            Ok((out, eng.cache.reused_tokens - before))
        };
        for overlap in [false, true] {
            let (warm, reused) = run(&mut eng, true, overlap)?;
            let (cold, _) = run(&mut eng, false, overlap)?;
            // The checkpoint sits at the last page boundary before the backoff, or (taken inside
            // the prefill kernel) up to a 64-token chunk earlier.
            let mut ok = if ckpt == 0 { reused == 0 } else { reused > 0 && reused as usize <= ckpt && ckpt - reused as usize <= 64 };
            let mut same_total = 0;
            for k in 0..2 {
                let same = warm[k].iter().zip(&cold[k]).take_while(|(a, b)| a == b).count();
                same_total += same;
                if same == cold[k].len() && warm[k].len() == cold[k].len() {
                    continue;
                }
                // Resuming changes how the prompt is chunked, which may flip a near-tie:
                // classify the divergence by the cold logits' top-2 margin at that position.
                let base = if k == 0 { &ids } else { &turn2 };
                let p: Vec<u32> = base.iter().chain(&cold[k][..same]).copied().collect();
                eng.clear_cache();
                eng.add(Request { id: 119, prompt: p, max_new_tokens: 1, ignore_eos: true })?;
                while eng.has_work() {
                    eng.step()?;
                }
                let (_, a, b) = top2(&eng.model.logits()?);
                if a - b < NEAR_TIE {
                    println!("  checkpoint turn {}: identical for {same} tokens, then diverges at a near-tie (margin {:.3}): ok", k + 1, a - b);
                } else {
                    ok = false;
                    println!("  checkpoint turn {}: diverges after {same} tokens where the margin is {:.3}: REAL MISMATCH", k + 1, a - b);
                }
            }
            println!(
                "prompt checkpoint{}: turn 2 resumed from {reused} of {} shared prompt tokens; {same_total}/40 tokens identical to a cold engine",
                if overlap { " (first turn still running)" } else { "" },
                cut
            );
            failed |= !ok;
        }
        eng.session_cache = true;
    }

    // ---- 4c. steady state: every state slot is held by a parked session, as on a server that
    // has been up for a while. A new conversation must still get its prompt checkpoint, and a
    // prefix shared between conversations its snapshot (each evicts a parked entry), also when
    // a snapshot of a shorter common prefix exists already; the resumed turns must match a
    // cold engine.
    {
        // (What this leg checks does not grow with the prompt: 2304 tokens are enough for a
        // second snapshot behind the 1040-token one.)
        let ids = &ids[..ids.len().min(2304)];
        let slots = (eng.limits().max_seqs + eng.limits().cache_slots) as u64;
        let ckpt = ids.len().saturating_sub(aq_engine::engine::CKPT_BACKOFF) / 16 * 16;
        let turn2: Vec<u32> = ids[..ids.len() - 1].iter().copied().chain(ids[..ids.len().min(41)].iter().map(|&t| t ^ 6)).collect();
        let pre = (ids.len().saturating_sub(64)) / 16 * 16;
        let tail = |k: u32| -> Vec<u32> { ids[..pre].iter().copied().chain(ids[pre..].iter().map(|&t| t ^ k)).collect() };
        let shared = ids.len() > 1100;
        let mut run = |eng: &mut Engine, cached: bool| -> Result<(Vec<Vec<u32>>, Vec<usize>)> {
            eng.session_cache = cached;
            eng.clear_cache();
            let mut reqs: Vec<(Vec<u32>, usize)> =
                (0..slots as u32).map(|k| (ids[..ids.len().min(40)].iter().map(|&t| t ^ (k + 9)).collect(), 2)).collect();
            if shared {
                // Two short prompts that share 1040 tokens with each other and with the long
                // ones: the second snapshots that much. The first long prompt forks it, the
                // second forks it too and snapshots the long prefix, the third forks that.
                let short = |k: u32| -> Vec<u32> { ids[..1040].iter().copied().chain(ids[..200].iter().map(|&t| t ^ k)).collect() };
                reqs.extend([(short(7), 2), (short(8), 2), (tail(1), 4), (tail(2), 4), (tail(3), 12)]);
            }
            reqs.extend([(ids.to_vec(), 8), (turn2.clone(), 12)]);
            let (mut out, mut reused) = (Vec::new(), Vec::new());
            let last = reqs.len() - 1;
            for (k, (prompt, n)) in reqs.into_iter().enumerate() {
                // The cold engine only runs the two requests it is compared on.
                if !cached && k != last && k != last - 2 {
                    out.push(Vec::new());
                    reused.push(0);
                    continue;
                }
                eng.add(Request { id: 300 + k as u64, prompt, max_new_tokens: n, ignore_eos: true })?;
                let (mut toks, mut hit) = (Vec::new(), 0);
                while eng.has_work() {
                    for e in eng.step()? {
                        toks.push(e.token);
                        hit = e.cached;
                    }
                }
                out.push(toks);
                reused.push(hit);
            }
            Ok((out, reused))
        };
        let (warm, reused) = run(&mut eng, true)?;
        let (cold, _) = run(&mut eng, false)?;
        let s = warm.len() - 2;
        let same = |k: usize| warm[k].iter().zip(&cold[k]).take_while(|(a, b)| a == b).count();
        let mut ok = (ckpt == 0 || reused[s + 1] > 0) && same(s + 1) == cold[s + 1].len();
        println!(
            "steady state (all {slots} slots parked): turn 2 resumed from {} of {} shared prompt tokens; {}/{} tokens identical to a cold engine",
            reused[s + 1],
            ids.len() - 1,
            same(s + 1),
            cold[s + 1].len()
        );
        if shared {
            // (A second snapshot is taken only where it saves `SNAPSHOT_MIN_TOKENS` more.)
            let want = if pre >= 1040 + aq_engine::engine::SNAPSHOT_MIN_TOKENS { pre } else { 1024 };
            ok &= reused[s - 1] >= want && same(s - 1) == cold[s - 1].len();
            println!(
                "steady state shared prefix ({pre} tokens, behind a 1040-token snapshot): conversations reused {:?}; {}/{} tokens identical to a cold engine",
                &reused[s - 3..s],
                same(s - 1),
                cold[s - 1].len()
            );
        }
        failed |= !ok;
        eng.session_cache = true;
        eng.clear_cache();
    }

    // ---- 5. shared-prefix snapshots: three prompts share a long prefix with different tails.
    // The second one snapshots the prefix, the third forks from it; the third's tokens must
    // match a cold engine's.
    if ids.len() > 1100 {
        let pre = (ids.len() - 64) / 16 * 16;
        let tail = |k: u32| -> Vec<u32> { ids[..pre].iter().copied().chain(ids[pre..].iter().map(|&t| t ^ k)).collect() };
        let prompts = [tail(0), tail(1), tail(2)];
        let mut run_shared = |eng: &mut Engine, cached: bool| -> Result<Vec<u32>> {
            eng.session_cache = cached;
            let mut out = Vec::new();
            for (k, p) in prompts.iter().enumerate() {
                eng.add(Request { id: 200 + k as u64, prompt: p.clone(), max_new_tokens: 12, ignore_eos: true })?;
                if k == 0 {
                    // Let the first one get going so the second sees it as a known sequence.
                    eng.step()?;
                }
            }
            while eng.has_work() {
                for e in eng.step()? {
                    if e.id == 202 {
                        out.push(e.token);
                    }
                }
            }
            Ok(out)
        };
        let before = eng.cache;
        let warm = run_shared(&mut eng, true)?;
        let after = eng.cache;
        let cold = run_shared(&mut eng, false)?;
        let same = warm.iter().zip(&cold).take_while(|(a, b)| a == b).count();
        println!(
            "shared prefix ({pre} tokens): {} snapshot(s), {} fork(s); forked request {same}/{} tokens identical to a cold prefill",
            after.snapshots - before.snapshots,
            after.forks - before.forks,
            cold.len()
        );
        failed |= after.forks == before.forks || same != cold.len();
    }

    // ---- 6. speculative decoding: with fixed decode-attention slices, greedy output with
    // speculation on must be token-identical to speculation off, single and batched.
    {
        let k = if spec_env > 0 { spec_env } else { 4 };
        let prompts: Vec<Vec<u32>> = vec![ids.clone(), ids[..ids.len() * 3 / 4].to_vec(), ids[ids.len() / 4..].to_vec()];
        let saved = (eng.session_cache, eng.spec_k, eng.model.gpu.gemm_min_m);
        // The bit-identity baseline (spec off) must take the same single-stream path as the
        // speculative run: co-execution is not used while speculating, and its decode lane
        // runs GEMMs under a different row cap (so different plans).
        let coexec_saved = eng.coexec;
        eng.coexec = false;
        // Same for background prefill: which sequences share a batch would depend on timing.
        let bg_saved = std::mem::replace(&mut eng.bg_prefill, 0);
        eng.session_cache = false;
        eng.clear_cache();
        // One GEMM plan for every decode-sized GEMM: rows then compute identically whatever
        // else shares the GEMM, so speculative and plain decoding must match bit for bit.
        // With a small step budget, prefill and decode share steps; pin every GEMM to the full
        // row capacity so no row's plan depends on what else is in the step.
        eng.model.gpu.gemm_min_m = if budget < 8192 { eng.model.limits.max_rows() } else { 64 };
        // AQ_CHECK_ROWDEP=N: N copies of one prompt decoded in the same steps. Every row sees
        // the same inputs, so any difference between rows is the kernels (or the GEMM plan)
        // rounding a row differently depending on where it sits in the batch.
        if let Some(nrow) = std::env::var("AQ_CHECK_ROWDEP").ok().and_then(|v| v.parse::<usize>().ok()) {
            eng.spec_k = 0;
            for len in [1usize, 8, 48] {
                for r in 0..nrow {
                    eng.add(Request { id: 500 + r as u64, prompt: prompts[1].clone(), max_new_tokens: len, ignore_eos: true })?;
                }
                let mut out = vec![Vec::new(); nrow];
                while eng.has_work() {
                    for e in eng.step()? {
                        out[(e.id - 500) as usize].push(e.token);
                    }
                }
                let lg = eng.model.logits_rows(nrow)?;
                let v = lg.len() / nrow;
                let maxd = (1..nrow).map(|r| lg[..v].iter().zip(&lg[r * v..(r + 1) * v]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)).collect::<Vec<_>>();
                let differ = (1..nrow).filter(|&r| out[r] != out[0]).count();
                println!("row dependence after {len} tokens: {differ}/{} rows emit different tokens than row 0; max |dlogit| vs row 0 per row {maxd:?}", nrow - 1);
            }
            if std::env::var("AQ_CHECK_ROWDEP_ONLY").is_ok() {
                return Ok(());
            }
        }
        let mut run = |eng: &mut Engine, spec: usize, which: &[usize]| -> Result<Vec<Vec<u32>>> {
            eng.spec_k = spec;
            for &w in which {
                eng.add(Request { id: 300 + w as u64, prompt: prompts[w].clone(), max_new_tokens: SPEC_TEST_LEN, ignore_eos: true })?;
            }
            let mut out = vec![Vec::new(); 3];
            while eng.has_work() {
                for e in eng.step()? {
                    out[(e.id - 300) as usize].push(e.token);
                }
            }
            Ok(out)
        };
        // AQ_CHECK_SPEC_REPEAT=N: repeat the pair N times (drafts vary between runs through
        // hash order, so rare speculation bugs need many runs to show).
        let repeat: usize = std::env::var("AQ_CHECK_SPEC_REPEAT").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        let cases = [("single", vec![0usize]), ("batched x3", vec![0, 1, 2])];
        let mut first_off: Vec<(&str, Vec<Vec<u32>>)> = Vec::new();
        let mut drift_done = false;
        for (name, which) in (0..repeat).flat_map(|_| cases.iter().cloned()) {
            let off = run(&mut eng, 0, &which)?;
            // The non-speculative run must itself be reproducible within a process.
            match first_off.iter().find(|f| f.0 == name) {
                Some((_, f)) => {
                    for &w in &which {
                        let same = f[w].iter().zip(&off[w]).take_while(|(a, b)| a == b).count();
                        if same < SPEC_TEST_LEN {
                            println!("  plain {name} seq {w}: differs from the first repeat's plain run after {same} tokens ({} vs {})", f[w][same], off[w][same]);
                        }
                    }
                }
                None => first_off.push((name, off.clone())),
            }
            // AQ_CHECK_DRIFT_AT=seq:token: run the drift diagnostic there unconditionally.
            if let (false, Some((w, t))) = (drift_done, std::env::var("AQ_CHECK_DRIFT_AT").ok().and_then(|v| v.split_once(':').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())))) {
                if which.contains(&w) && which.len() > 1 {
                    drift_done = true;
                    drift(&mut eng, &prompts[w], &off[w], t)?;
                }
            }
            let before = eng.spec;
            let t = Instant::now();
            let on = run(&mut eng, k, &which)?;
            let dt = t.elapsed().as_secs_f64();
            let st = eng.spec;
            let (drafted, accepted) = (st.drafted - before.drafted, st.accepted - before.accepted);
            let (entries, emitted) = (st.decode_entries - before.decode_entries, st.emitted - before.emitted);
            let mut ok = true;
            let mut ties = 0;
            for &w in &which {
                let same = on[w].iter().zip(&off[w]).take_while(|(a, b)| a == b).count();
                if same == SPEC_TEST_LEN && on[w].len() == SPEC_TEST_LEN {
                    continue;
                }
                // Classify the divergence by the non-speculative logits at that position: run
                // prompt + common prefix alone and look at its top-2 margin.
                eng.spec_k = 0;
                let p: Vec<u32> = prompts[w].iter().chain(&off[w][..same]).copied().collect();
                eng.add(Request { id: 399, prompt: p, max_new_tokens: 1, ignore_eos: true })?;
                while eng.has_work() {
                    eng.step()?;
                }
                let lg = eng.model.logits()?;
                let (top, a, b) = top2(&lg);
                if same < on[w].len() {
                    println!("    at the divergence: prefill-path top-1 {top} ({a:.3}), plain token {} ({:.3}), spec token {} ({:.3})", off[w][same], lg[off[w][same] as usize], on[w][same], lg[on[w][same] as usize]);
                }
                // A margin only means something relative to how far our own exact paths
                // disagree at this position. Replay the plain run alone (single-token decode)
                // and compare the gap between the two disputed tokens under both plain paths,
                // decode and prefill: where the model sits at a bifurcation, GEMM rounding
                // alone moves that gap by logits, and neither token is "the" greedy answer.
                let mut ambiguous = None;
                if a - b >= NEAR_TIE && same < on[w].len() {
                    eng.add(Request { id: 398, prompt: prompts[w].clone(), max_new_tokens: same + 1, ignore_eos: true })?;
                    let mut got = Vec::new();
                    while eng.has_work() {
                        for e in eng.step()? {
                            got.push(e.token);
                        }
                    }
                    let (pt, st) = (off[w][same] as usize, on[w][same] as usize);
                    let gap_pre = lg[pt] - lg[st];
                    if got[..same] != off[w][..same] {
                        let at = got.iter().zip(&off[w]).take_while(|(a, b)| a == b).count();
                        ambiguous = Some(format!("decoded alone, the plain run already leaves this path at token {at}"));
                    } else {
                        let dec = eng.model.logits()?;
                        let gap_dec = dec[pt] - dec[st];
                        let noise = (gap_pre - gap_dec).abs();
                        if gap_pre.min(gap_dec) <= NEAR_TIE + noise {
                            ambiguous = Some(format!("plain-over-spec gap is {gap_pre:.3} by prefill and {gap_dec:.3} by decode, the two plain paths disagree by {noise:.3}"));
                        }
                    }
                }
                if a - b < NEAR_TIE {
                    ties += 1;
                    println!("  spec {name} seq {w}: identical for {same}/{SPEC_TEST_LEN} tokens, then diverges at a near-tie (margin {:.3}): ok", a - b);
                } else if let Some(why) = ambiguous {
                    ties += 1;
                    println!("  spec {name} seq {w}: identical for {same}/{SPEC_TEST_LEN} tokens, then diverges at a numerically ambiguous position ({why}): ok");
                } else {
                    ok = false;
                    println!("  spec {name} seq {w}: diverges after {same}/{SPEC_TEST_LEN} tokens where the margin is {:.3}: REAL MISMATCH", a - b);
                    let lo = same.saturating_sub(12);
                    println!("    plain {:?}", &off[w][lo..(same + 6).min(off[w].len())]);
                    println!("    spec  {:?}", &on[w][lo..(same + 6).min(on[w].len())]);
                    if std::env::var("AQ_CHECK_DRIFT").is_ok() && !drift_done {
                        drift_done = true;
                        // Decode-path vs prefill-path logits along the plain run's own tokens:
                        // gradual growth is numeric drift, a jump is a bug.
                        drift(&mut eng, &prompts[w], &off[w], same)?;
                    }
                }
            }
            println!(
                "speculative ({name}, k={k}): {} identical to non-speculative; drafted {drafted}, accepted {accepted} ({:.0}%), {:.2} tokens per decode step, {:.2}s{}",
                if !ok { "NOT" } else if ties > 0 { "up to near-ties" } else { "all tokens" },
                100.0 * accepted as f64 / drafted.max(1) as f64,
                emitted as f64 / entries.max(1) as f64,
                dt,
                if st.mtp_drafted > before.mtp_drafted {
                    format!("; MTP first-draft acceptance {}/{}", st.mtp_accepted - before.mtp_accepted, st.mtp_drafted - before.mtp_drafted)
                } else {
                    String::new()
                }
            );
            failed |= !ok;
        }
        // ---- 6b. stop tokens under speculation. A request must end exactly at its stop token
        // when that token sits inside an accepted draft, and the session it parks must hold
        // the state of the tokens it emitted, nothing past them: a follow-up that extends the
        // stopped output has to continue like a cold run of the same prompt.
        if let Some((_, plain)) = first_off.iter().find(|f| f.0 == "single") {
            let plain = &plain[0];
            eng.session_cache = true;
            let saved_eos = std::mem::take(&mut eng.eos);
            let before = eng.spec.stop_in_draft;
            let run1 = |eng: &mut Engine, id: u64, prompt: Vec<u32>, n: usize, cache: bool, spec: usize| -> Result<(Vec<u32>, bool)> {
                (eng.session_cache, eng.spec_k) = (cache, spec);
                eng.add(Request { id, prompt, max_new_tokens: n, ignore_eos: false })?;
                let (mut out, mut stop) = (Vec::new(), false);
                while eng.has_work() {
                    for e in eng.step()? {
                        out.push(e.token);
                        stop = e.stop;
                    }
                }
                Ok((out, stop))
            };
            let (mut tested, mut ok) = (0, true);
            // Stop tokens: output positions whose token has not occurred earlier in the output.
            for j in (1..plain.len().min(120)).filter(|&j| !plain[..j].contains(&plain[j])).take(12) {
                tested += 1;
                eng.eos = vec![plain[j]];
                let (got, stop) = run1(&mut eng, 600, prompts[0].clone(), SPEC_TEST_LEN, true, k)?;
                // (A speculative run may leave the plain run's path at a near-tie, see above:
                // the stop token then comes elsewhere, or not at all.)
                let at = got.iter().position(|&t| t == plain[j]);
                if at.map(|a| a + 1) != stop.then_some(got.len()) {
                    ok = false;
                    println!("  stop at token {j}: emitted {} tokens (stop {stop}), first stop token at {at:?}", got.len());
                    continue;
                }
                if !stop {
                    continue;
                }
                eng.eos.clear();
                let next: Vec<u32> = prompts[0].iter().chain(&got).copied().chain([plain[0]]).collect();
                // (A cold run prefills the whole prompt: every fourth position is enough.)
                if tested % 4 != 1 {
                    continue;
                }
                let (warm, _) = run1(&mut eng, 601, next.clone(), 8, true, 0)?;
                let (cold, _) = run1(&mut eng, 602, next.clone(), 8, false, 0)?;
                let same = warm.iter().zip(&cold).take_while(|(a, b)| a == b).count();
                if same < cold.len() {
                    let p: Vec<u32> = next.iter().chain(&cold[..same]).copied().collect();
                    run1(&mut eng, 603, p, 1, false, 0)?;
                    let (_, a, b) = top2(&eng.model.logits()?);
                    if a - b >= NEAR_TIE {
                        ok = false;
                        println!("  stop at token {j}: the follow-up diverges from a cold run after {same} tokens where the margin is {:.3}: REAL MISMATCH", a - b);
                    }
                }
            }
            eng.clear_cache();
            println!(
                "stop under speculation (k={k}): {tested} stop positions, {} inside a fully accepted draft; {}",
                eng.spec.stop_in_draft - before,
                if ok { "all end on the stop token and resume like a cold run" } else { "FAIL" }
            );
            failed |= !ok;
            eng.eos = saved_eos;
        }
        // ---- 6c. sampling. (1) A seeded request is reproducible, and speculation does not
        // change it: the token at each position depends on that position's logits and the
        // seed only, so a sequence decoded alone gives the same tokens with drafts on and off
        // (same GEMM plan pinning as above; batched, logits move by an ulp with what shares
        // the step, which sampling turns into a different token now and then). (2) Batched
        // with drafts, temperature 0 through the host path follows the device's greedy tokens,
        // so every request reads its own logits rows. (3) The first sampled token follows the
        // reference distribution: frequencies over many seeds against
        // softmax(reference logits / T) cut to top-k and top-p.
        if !eng.coexec && eng.bg_prefill == 0 {
            let sp = Sampling { temperature: 0.8, top_p: 0.95, top_k: 40, seed: 1234 };
            // `copies` requests with seeds from `sp.seed` up; `distinct`: one prompt each.
            let mut sample = |eng: &mut Engine, spec: usize, sp: Sampling, n: usize, copies: u64, distinct: bool| -> Result<Vec<Vec<u32>>> {
                eng.spec_k = spec;
                for c in 0..copies {
                    let ctl = Control { pick: Some(Box::new(Sampler::new(Sampling { seed: sp.seed + c, ..sp }))), stop_ids: Vec::new() };
                    // (One token: the frequency check below, against the reference logits of
                    // the whole prompt. Else the first 1024 tokens do.)
                    let p = &prompts[if distinct { c as usize } else { 0 }];
                    let prompt = p[..if n == 1 { p.len() } else { p.len().min(1024) }].to_vec();
                    eng.add_with(Request { id: 700 + c, prompt, max_new_tokens: n, ignore_eos: true }, ctl)?;
                }
                let mut out = vec![Vec::new(); copies as usize];
                while eng.has_work() {
                    for e in eng.step()? {
                        out[(e.id - 700) as usize].push(e.token);
                    }
                }
                Ok(out)
            };
            let cached = std::mem::replace(&mut eng.session_cache, false);
            let before = eng.spec;
            let on = sample(&mut eng, k, sp, 96, 1, false)?;
            let st = eng.spec;
            let again = sample(&mut eng, k, sp, 96, 1, false)?;
            let off = sample(&mut eng, 0, sp, 96, 1, false)?;
            let other = sample(&mut eng, k, Sampling { seed: 99, ..sp }, 96, 1, false)?;
            // Two runs with one seed are identical while their logits are. Where rounding
            // moves a logit (as between the plain and speculative greedy runs above), the
            // same draw can land on the neighbouring token: accepted when the draw sits
            // within 0.1 of a boundary of the distribution at the first difference.
            let mut agree = |eng: &mut Engine, a: &[u32], b: &[u32]| -> Result<String> {
                let p = a.iter().zip(b).take_while(|(x, y)| x == y).count();
                if p == a.len() && p == b.len() {
                    return Ok("identical".into());
                }
                eng.spec_k = 0;
                let prompt: Vec<u32> = prompts[0][..prompts[0].len().min(1024)].iter().chain(&a[..p]).copied().collect();
                eng.add(Request { id: 399, prompt, max_new_tokens: 1, ignore_eos: true })?;
                while eng.has_work() {
                    eng.step()?;
                }
                let (u, mut cum, mut near) = (uniform(sp.seed, p as u64), 0.0, f64::MAX);
                let mut d = distribution(&sp, &eng.model.logits()?);
                by_id(&mut d);
                for (_, pr) in d {
                    cum += pr;
                    near = near.min((u - cum).abs());
                }
                Ok(format!("identical for {p}/96 tokens, then a draw {near:.3} from a boundary: {}", if near < 0.1 { "ok" } else { "DIFFERS" }))
            };
            let (rerun, vs_plain) = (agree(&mut eng, &on[0], &again[0])?, agree(&mut eng, &on[0], &off[0])?);
            // Temperature 0 through the host path, three prompts batched with drafts: every
            // pick must equal the device's greedy token for its row.
            let b0 = eng.spec;
            sample(&mut eng, k, Sampling { temperature: 0.0, ..sp }, 96, 3, true)?;
            let (picked, repicked) = (eng.spec.picked - b0.picked, eng.spec.repicked - b0.repicked);
            let ok = !rerun.ends_with("DIFFERS") && !vs_plain.ends_with("DIFFERS") && on != other && picked >= 288 && repicked == 0;
            println!(
                "sampling (T={} top_p={} top_k={}, k={k}): same seed again {rerun}; speculative vs plain {vs_plain}; other seed {}; drafted {} accepted {} ({:.0}%); temperature 0 batched x3 with drafts: {repicked}/{picked} host picks differ from the device token",
                sp.temperature,
                sp.top_p,
                sp.top_k,
                if on != other { "differs" } else { "IDENTICAL" },
                st.drafted - before.drafted,
                st.accepted - before.accepted,
                100.0 * (st.accepted - before.accepted) as f64 / (st.drafted - before.drafted).max(1) as f64
            );
            failed |= !ok;

            // Frequencies of the first token over many seeds, against the reference logits
            // (transformers, `logits_last`) when the golden has them, else our own.
            let sp = Sampling { temperature: 1.5, top_p: 0.9, top_k: 12, seed: 0 };
            let refl = as_f32("logits_last").unwrap_or_default();
            let want = distribution(&sp, if refl.len() == vocab { &refl } else { &lg });
            const DRAWS: u64 = 1024;
            // (With the session cache: after the first batch a draw prefills only the prompt's
            // last tokens, from the prompt checkpoint.)
            eng.session_cache = true;
            let mut got: std::collections::HashMap<u32, f64> = std::collections::HashMap::new();
            let mut seed = 0;
            while seed < DRAWS {
                for o in sample(&mut eng, k, Sampling { seed, ..sp }, 1, 8, false)? {
                    *got.entry(o[0]).or_default() += 1.0 / DRAWS as f64;
                }
                seed += 8;
            }
            // Total variation distance; sampling noise alone gives about 0.04 at 1024 draws
            // over a dozen tokens, and bf16 logits against the f32 reference add a little.
            let tv = 0.5 * (want.iter().map(|(t, p)| (p - got.get(t).copied().unwrap_or(0.0)).abs()).sum::<f64>()
                + got.iter().filter(|(t, _)| !want.iter().any(|w| w.0 == **t)).map(|(_, f)| f).sum::<f64>());
            let outside = got.keys().filter(|t| !want.iter().any(|w| w.0 == **t)).count();
            println!(
                "sampling frequencies (T={} top_p={} top_k={}, {DRAWS} seeds): {} candidate tokens, top p {:.3}, total variation vs reference {tv:.3}, {outside} token(s) outside the reference set: {}",
                sp.temperature,
                sp.top_p,
                sp.top_k,
                want.len(),
                want.first().map_or(0.0, |w| w.1),
                if tv < 0.1 { "ok" } else { "FAIL" }
            );
            failed |= tv >= 0.1;
            eng.clear_cache();
            eng.session_cache = cached;
        }
        (eng.session_cache, eng.spec_k, eng.model.gpu.gemm_min_m) = saved;
        eng.coexec = coexec_saved;
        eng.bg_prefill = bg_saved;
    }

    // ---- 7. memory pressure: shrink the page pool to about two sessions, then interleave
    // two-turn sessions of three different prompts so admissions evict parked sessions and
    // snapshots and later re-admit them. Every request must match a cold run (no cache).
    if ids.len() > 1100 {
        let bs = eng.limits().block_size;
        let per = (ids.len() + 64 + 64).div_ceil(bs);
        eng.evict_all_parked();
        let free_now = eng.free_pages();
        eng.withhold_pages(free_now.saturating_sub(per * 2 + per / 2));
        let pre = (ids.len() - 64) / 16 * 16;
        let prompt = |k: u32| -> Vec<u32> { ids[..pre].iter().copied().chain(ids[pre..].iter().map(|&t| t ^ k)).collect() };
        let run = |eng: &mut Engine, cached: bool| -> Result<(Vec<Vec<u32>>, Vec<Vec<u32>>)> {
            eng.session_cache = cached;
            let mut outs: Vec<Vec<u32>> = vec![Vec::new(); 6];
            let mut prompts: Vec<Vec<u32>> = vec![Vec::new(); 6];
            // Turn 1 for all three, two at a time; turn 2 extends each with its own output.
            for (k, round) in [[0u32, 1], [2, 0], [1, 2]].iter().enumerate() {
                for &a in round.iter() {
                    let first = outs[a as usize].is_empty();
                    let p = if first {
                        prompt(a + 1)
                    } else {
                        let mut p = prompt(a + 1);
                        p.extend(&outs[a as usize][..7]);
                        p.extend(ids[..40].iter().map(|&t| t ^ (a + 7)));
                        p
                    };
                    let id = if first { a as u64 } else { 3 + a as u64 };
                    prompts[id as usize] = p.clone();
                    eng.add(Request { id: 300 + id, prompt: p, max_new_tokens: 8, ignore_eos: true })?;
                }
                let _ = k;
                while eng.has_work() {
                    for e in eng.step()? {
                        outs[(e.id - 300) as usize].push(e.token);
                    }
                }
            }
            Ok((outs, prompts))
        };
        let before = eng.cache;
        let (warm, _) = run(&mut eng, true)?;
        let after = eng.cache;
        let (cold, prompts) = run(&mut eng, false)?;
        let mut ok = true;
        for k in 0..6 {
            let same = warm[k].iter().zip(&cold[k]).take_while(|(a, b)| a == b).count();
            if same == cold[k].len() && warm[k].len() == cold[k].len() {
                continue;
            }
            // Turn-2 prompts quote the session's own turn-1 output: if turn 1 already diverged
            // (classified above), the two runs were given different turn-2 prompts.
            if k >= 3 && warm[k - 3][..7] != cold[k - 3][..7] {
                println!("  memory pressure request {k}: not comparable (its turn 1 diverged at a near-tie)");
                continue;
            }
            // Classify the divergence by the cold logits' top-2 margin at that position.
            let p: Vec<u32> = prompts[k].iter().chain(&cold[k][..same]).copied().collect();
            eng.add(Request { id: 399, prompt: p, max_new_tokens: 1, ignore_eos: true })?;
            while eng.has_work() {
                eng.step()?;
            }
            let (_, a, b) = top2(&eng.model.logits()?);
            if a - b < NEAR_TIE {
                println!("  memory pressure request {k}: identical for {same} tokens, then diverges at a near-tie (margin {:.3}): ok", a - b);
            } else {
                ok = false;
                println!("  memory pressure request {k}: diverges after {same} tokens where the margin is {:.3}: REAL MISMATCH", a - b);
            }
        }
        println!(
            "memory pressure: {} evictions, {} hits during the run; 6 requests {}",
            after.evictions - before.evictions,
            after.hits - before.hits,
            if ok { "identical to a cold run" } else { "DIFFER from a cold run" }
        );
        failed |= !ok;
    }

    // ---- 8. KV reservation (`Engine::kv_reserve`): a request that reserves only part of its
    // token limit takes pages as it generates. With room in the pool the output is the same
    // as with a full reservation. In a pool that cannot hold what two such requests ask for,
    // both are admitted and run together (a full reservation would run them one after the
    // other) until the pages run out; then one waits, the other ends as if at its limit when
    // it runs dry too, its pages let the first go on to its real limit, and nothing fails.
    {
        let saved = (eng.session_cache, eng.spec_k, eng.kv_reserve);
        // (Page accounting does not depend on the prompt: its first 512 tokens do.)
        let ids = &ids[..ids.len().min(512)];
        (eng.session_cache, eng.spec_k) = (false, 0);
        eng.clear_cache();
        let bs = eng.limits().block_size;
        let mut run = |eng: &mut Engine, reserve: usize, copies: u64| -> Result<(Vec<Vec<u32>>, usize)> {
            eng.kv_reserve = reserve;
            for c in 0..copies {
                eng.add(Request { id: 800 + c, prompt: ids.to_vec(), max_new_tokens: 96, ignore_eos: true })?;
            }
            let (mut out, mut together) = (vec![Vec::new(); copies as usize], 0);
            while eng.has_work() {
                let ev = eng.step()?;
                together = together.max(eng.num_running());
                for e in ev {
                    out[(e.id - 800) as usize].push(e.token);
                }
            }
            Ok((out, together))
        };
        let (full, _) = run(&mut eng, usize::MAX, 1)?;
        let (lazy, _) = run(&mut eng, 16, 1)?;
        // Room for both prompts, their 16 reserved tokens, and about 64 tokens of growth.
        let target = 2 * (ids.len() + 16).div_ceil(bs) + 4;
        eng.withhold_pages(eng.free_pages().saturating_sub(target));
        let before = eng.cache.cut;
        let (_, serial) = run(&mut eng, usize::MAX, 2)?;
        let (cut, together) = run(&mut eng, 16, 2)?;
        let cuts = eng.cache.cut - before;
        // (Two together need not match the single run token for token: near-ties, see leg 6.)
        let ok = lazy == full && serial == 1 && together == 2 && cuts == 1 && cut.iter().all(|o| !o.is_empty()) && cut.iter().any(|o| o.len() < 96) && cut.iter().any(|o| o.len() == 96);
        println!(
            "kv reservation: growing {} a full reservation ({} tokens); pool of {target} pages, 2 requests of {} + 96 tokens: {serial} at a time reserved in full, {together} with 16 reserved, {cuts} ended early after {:?} tokens: {}",
            if lazy == full { "identical to" } else { "DIFFERS from" },
            full[0].len(),
            ids.len(),
            cut.iter().map(|o| o.len()).collect::<Vec<_>>(),
            if ok { "ok" } else { "FAIL" }
        );
        failed |= !ok;
        (eng.session_cache, eng.spec_k, eng.kv_reserve) = saved;
    }

    if failed {
        anyhow::bail!("correctness check FAILED");
    }
    println!("PASS");
    Ok(())
}
