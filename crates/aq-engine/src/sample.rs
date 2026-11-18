//! Host-side token selection for requests that are not plain greedy: sampling
//! (temperature, top-k, top-p, seed) and, through [`Picker`], anything else that needs the
//! logits row (constrained decoding in the server).
//!
//! Speculative decoding stays exact for these requests. The token at output position `i` is a
//! function of that position's logits and of randomness keyed by `(seed, i)` only, never of
//! the draft: a draft token is accepted when it equals the token picked at its row, and the
//! rows after a mismatch are discarded, as in the greedy case. The output is therefore the
//! same with drafts on or off, and distributed as plain sampling. What drafts cost a sampled
//! request is acceptance: drafts predict the most likely token, which sampling picks less
//! often as the temperature rises.

/// Picks the next token of one request from its logits row. Called once per emitted token,
/// in output order; it may keep state (a grammar) and may edit the row (mask it).
pub trait Picker: Send {
    fn pick(&mut self, logits: &mut [f32]) -> u32;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// 0 = greedy (argmax; lowest id on ties).
    pub temperature: f32,
    /// Nucleus: the smallest set of most likely tokens whose probability reaches `top_p`
    /// (1 = off). Applied after `top_k`.
    pub top_p: f32,
    /// Keep the `top_k` most likely tokens (0 = off).
    pub top_k: usize,
    pub seed: u64,
}

/// The sampler: one per request. Same seed and same logits give the same tokens.
#[derive(Debug, Clone)]
pub struct Sampler {
    pub p: Sampling,
    index: u64,
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Uniform in [0, 1) for output position `index` of a request with `seed`.
pub fn uniform(seed: u64, index: u64) -> f64 {
    (splitmix64(splitmix64(seed) ^ index.wrapping_mul(0xD6E8_FEB8_6659_FD93)) >> 11) as f64 / (1u64 << 53) as f64
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] || logits[best].is_nan() {
            best = i;
        }
    }
    best as u32
}

/// The tokens `p` can pick from `logits` with their probabilities, most likely first.
/// Rows masked with `-inf` are never candidates.
pub fn distribution(p: &Sampling, logits: &[f32]) -> Vec<(u32, f64)> {
    let desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
    let mut c: Vec<(f32, u32)> = logits.iter().enumerate().filter(|(_, l)| l.is_finite()).map(|(i, &l)| (l, i as u32)).collect();
    if c.is_empty() {
        return Vec::new();
    }
    if p.top_k > 0 && p.top_k < c.len() {
        c.select_nth_unstable_by(p.top_k - 1, desc);
        c.truncate(p.top_k);
    }
    let max = c.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
    let t = p.temperature.max(1e-6) as f64;
    let w = |l: f32| ((l - max) as f64 / t).exp();
    let total: f64 = c.iter().map(|x| w(x.0)).sum();
    if p.top_p < 1.0 {
        // The nucleus is a short prefix of the sorted row: sort a growing head, not the vocabulary.
        let need = p.top_p.max(0.0) as f64 * total;
        let mut n = 64.min(c.len());
        loop {
            if n < c.len() {
                c.select_nth_unstable_by(n - 1, desc);
            }
            if n == c.len() || c[..n].iter().map(|x| w(x.0)).sum::<f64>() >= need {
                break;
            }
            n = (n * 8).min(c.len());
        }
        c[..n].sort_unstable_by(desc);
        let mut cum = 0.0;
        let keep = c[..n].iter().position(|x| { cum += w(x.0); cum >= need }).map_or(n, |k| k + 1);
        c.truncate(keep.max(1));
    } else {
        c.sort_unstable_by(desc);
    }
    let kept: f64 = c.iter().map(|x| w(x.0)).sum();
    c.into_iter().map(|(l, i)| (i, w(l) / kept)).collect()
}

/// The order in which a draw walks the candidates: by token id, not by probability. Logits
/// move by a rounding step with what else shares a batch; in probability order that could swap
/// two near-equal candidates and hand the same draw a different token, in id order it only
/// moves a boundary by as little.
pub fn by_id(d: &mut [(u32, f64)]) {
    d.sort_unstable_by_key(|x| x.0);
}

impl Sampler {
    pub fn new(p: Sampling) -> Self {
        Self { p, index: 0 }
    }
}

impl Picker for Sampler {
    fn pick(&mut self, logits: &mut [f32]) -> u32 {
        let u = uniform(self.p.seed, self.index);
        self.index += 1;
        if self.p.temperature <= 0.0 {
            return argmax(logits);
        }
        let mut d = distribution(&self.p, logits);
        by_id(&mut d);
        let mut cum = 0.0;
        for &(tok, pr) in &d {
            cum += pr;
            if u < cum {
                return tok;
            }
        }
        d.last().map_or_else(|| argmax(logits), |x| x.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row() -> Vec<f32> {
        // Probabilities 0.5, 0.25, 0.125, 0.0625, 0.0625 at ids 3, 0, 4, 1, 2.
        vec![2f32.ln() * 2.0, 0.0, 0.0, 2f32.ln() * 3.0, 2f32.ln()]
    }

    fn freq(p: Sampling, n: u64) -> Vec<f64> {
        let mut f = vec![0f64; 5];
        let mut s = Sampler::new(p);
        for _ in 0..n {
            f[s.pick(&mut row()) as usize] += 1.0 / n as f64;
        }
        f
    }

    #[test]
    fn greedy_at_zero_temperature() {
        let mut s = Sampler::new(Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 7 });
        assert!((0..50).all(|_| s.pick(&mut row()) == 3));
    }

    #[test]
    fn same_seed_same_tokens() {
        let p = Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 42 };
        let run = |p| { let mut s = Sampler::new(p); (0..200).map(|_| s.pick(&mut row())).collect::<Vec<_>>() };
        assert_eq!(run(p), run(p));
        assert_ne!(run(p), run(Sampling { seed: 43, ..p }));
    }

    #[test]
    fn frequencies_follow_the_softmax() {
        let f = freq(Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 1 }, 40_000);
        for (got, want) in f.iter().zip([0.25, 0.0625, 0.0625, 0.5, 0.125]) {
            assert!((got - want).abs() < 0.01, "{f:?}");
        }
        // Temperature 0.5 squares the odds: 64, 16, 4, 1, 1 over 86.
        let f = freq(Sampling { temperature: 0.5, top_p: 1.0, top_k: 0, seed: 1 }, 40_000);
        for (got, want) in f.iter().zip([16.0 / 86.0, 1.0 / 86.0, 1.0 / 86.0, 64.0 / 86.0, 4.0 / 86.0]) {
            assert!((got - want).abs() < 0.01, "{f:?}");
        }
    }

    #[test]
    fn top_k_and_top_p_cut_the_tail() {
        let d = distribution(&Sampling { temperature: 1.0, top_p: 1.0, top_k: 2, seed: 0 }, &row());
        assert_eq!(d.iter().map(|x| x.0).collect::<Vec<_>>(), [3, 0]);
        assert!((d[0].1 - 2.0 / 3.0).abs() < 1e-6);
        // 0.5 + 0.25 reaches 0.75; 0.8 needs the third token as well.
        let ids = |top_p| distribution(&Sampling { temperature: 1.0, top_p, top_k: 0, seed: 0 }, &row()).iter().map(|x| x.0).collect::<Vec<_>>();
        assert_eq!(ids(0.75), [3, 0]);
        assert_eq!(ids(0.8), [3, 0, 4]);
        assert_eq!(ids(0.0), [3]);
        // A masked row is never picked.
        let mut masked = row();
        masked[3] = f32::NEG_INFINITY;
        assert_eq!(distribution(&Sampling { temperature: 1.0, top_p: 0.1, top_k: 0, seed: 0 }, &masked)[0].0, 0);
    }

    #[test]
    fn a_rounding_step_between_near_equal_candidates_keeps_the_pick() {
        // Ids 1 and 2 are near-equal; which of them is the more likely one must not matter.
        let p = Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 11 };
        let (a, b) = (vec![0.0, 1.0, 1.001, 0.5], vec![0.0, 1.001, 1.0, 0.5]);
        let run = |row: &Vec<f32>| { let mut s = Sampler::new(p); (0..500).map(|_| s.pick(&mut row.clone())).collect::<Vec<_>>() };
        let differ = run(&a).iter().zip(&run(&b)).filter(|(x, y)| x != y).count();
        assert!(differ <= 3, "{differ} of 500 picks changed");
    }

    #[test]
    fn nucleus_over_a_large_row_matches_a_full_sort() {
        let logits: Vec<f32> = (0..50_000u64).map(|i| (uniform(9, i) * 12.0) as f32).collect();
        let p = Sampling { temperature: 0.7, top_p: 0.9, top_k: 0, seed: 0 };
        let d = distribution(&p, &logits);
        let mut all: Vec<(u32, f64)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, (l as f64 / 0.7).exp())).collect();
        all.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let total: f64 = all.iter().map(|x| x.1).sum();
        let mut cum = 0.0;
        let keep = all.iter().position(|x| { cum += x.1; cum >= 0.9 * total }).unwrap() + 1;
        assert_eq!(d.len(), keep);
        assert!(d.iter().zip(&all).all(|(a, b)| a.0 == b.0));
    }
}
