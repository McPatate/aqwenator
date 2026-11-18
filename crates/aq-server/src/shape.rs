//! Per-request changes to the logits row before the token is picked: `logit_bias`, the
//! presence, frequency and repetition penalties, `min_tokens`; and the log-probabilities a
//! request asked for (`logprobs`, `top_logprobs`).
//!
//! A [`Shaper`] wraps the request's picker (the sampler, or the grammar around it). The
//! engine calls a picker once per emitted token, in output order, so the tokens it returned
//! are the output so far, and the log-probabilities it queues line up with the token events.
//!
//! Log-probabilities are those of the model's distribution after bias, penalties and
//! `min_tokens`, before temperature, `top_k`, `top_p` and any grammar mask: what the model
//! thought of each token, not what the sampler was allowed to draw.

use aq_engine::sample::Picker;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

/// Log-probability of an emitted token and of the most likely tokens at its position.
pub struct LogProb {
    pub logprob: f32,
    pub top: Vec<(u32, f32)>,
}

pub type Queue = Arc<Mutex<VecDeque<LogProb>>>;

/// What vLLM reports for a token the distribution excludes.
const NEVER: f32 = -9999.0;

#[derive(Default)]
pub struct Shape {
    /// (token id, added to its logit).
    pub bias: Vec<(u32, f32)>,
    /// Subtracted from the logit of every token generated so far.
    pub presence: f32,
    /// Subtracted once per time the token was generated.
    pub frequency: f32,
    /// Logits of the tokens of the prompt and of the output are divided by this when
    /// positive, multiplied when negative (1 = off).
    pub repetition: f32,
    /// The prompt's tokens (for `repetition`); one that is generated moves to the counts.
    pub prompt: HashSet<u32>,
    /// The tokens in `hold` (end of sequence, stop tokens) cannot come before this many.
    pub min_tokens: usize,
    pub hold: Vec<u32>,
    /// Most likely tokens to report per position, and where to.
    pub top: Option<(usize, Queue)>,
}

pub struct Shaper {
    shape: Shape,
    counts: HashMap<u32, u32>,
    inner: Box<dyn Picker>,
}

impl Shaper {
    pub fn new(shape: Shape, inner: Box<dyn Picker>) -> Self {
        Self { shape, counts: HashMap::new(), inner }
    }
}

impl Picker for Shaper {
    fn pick(&mut self, logits: &mut [f32]) -> u32 {
        let s = &self.shape;
        let n: u32 = self.counts.values().sum();
        // (Ids beyond the row, from a client's `logit_bias`, are ignored.)
        if s.repetition != 1.0 && s.repetition > 0.0 {
            for &t in s.prompt.iter().chain(self.counts.keys()) {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = if *l > 0.0 { *l / s.repetition } else { *l * s.repetition };
                }
            }
        }
        for (&t, &c) in &self.counts {
            if let Some(l) = logits.get_mut(t as usize) {
                *l -= s.presence + s.frequency * c as f32;
            }
        }
        for &(t, b) in &s.bias {
            if let Some(l) = logits.get_mut(t as usize) {
                *l += b;
            }
        }
        if (n as usize) < s.min_tokens {
            for &t in &s.hold {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        // log p(t) = logit(t) - max - ln(sum exp(logit - max))
        let norm = s.top.as_ref().map(|(k, _)| {
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + logits.iter().map(|&l| ((l - max) as f64).exp()).sum::<f64>().ln() as f32;
            let mut top: Vec<(u32, f32)> = Vec::new();
            if *k > 0 {
                let mut c: Vec<(f32, u32)> = logits.iter().enumerate().filter(|(_, l)| l.is_finite()).map(|(i, &l)| (l, i as u32)).collect();
                let desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1));
                if *k < c.len() {
                    c.select_nth_unstable_by(*k - 1, desc);
                    c.truncate(*k);
                }
                c.sort_unstable_by(desc);
                top = c.into_iter().map(|(l, i)| (i, l - lse)).collect();
            }
            (lse, top)
        });
        let shaped = norm.as_ref().map(|_| logits.to_vec());
        let t = self.inner.pick(logits);
        if let (Some((lse, top)), Some(row), Some((_, q))) = (norm, shaped, &s.top) {
            let l = row.get(t as usize).copied().unwrap_or(f32::NEG_INFINITY);
            q.lock().unwrap().push_back(LogProb { logprob: if l.is_finite() { l - lse } else { NEVER }, top });
        }
        *self.counts.entry(t).or_insert(0) += 1;
        self.shape.prompt.remove(&t);
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aq_engine::sample::{Sampler, Sampling};

    fn greedy(shape: Shape) -> Shaper {
        Shaper::new(shape, Box::new(Sampler::new(Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0 })))
    }

    fn picks(p: &mut Shaper, n: usize) -> Vec<u32> {
        (0..n).map(|_| p.pick(&mut [3.0, 2.5, 2.0, 0.5])).collect()
    }

    #[test]
    fn bias_moves_and_bans() {
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, ..Default::default() }), 2), [0, 0]);
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, bias: vec![(2, 1.5)], ..Default::default() }), 1), [2]);
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, bias: vec![(0, -100.0), (1, -100.0), (99, 5.0)], ..Default::default() }), 1), [2]);
    }

    #[test]
    fn penalties_follow_the_output() {
        // Presence: once generated, token 0 (3.0) drops below 1 (2.5), then 1 below 2, then
        // the three are level again and the order restarts.
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, presence: 1.1, ..Default::default() }), 4), [0, 1, 2, 0]);
        // Frequency grows with each use.
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, frequency: 0.3, ..Default::default() }), 4), [0, 0, 1, 0]);
        // Repetition: divides the positive logits of prompt and output tokens.
        let prompt: HashSet<u32> = [0].into_iter().collect();
        assert_eq!(picks(&mut greedy(Shape { repetition: 2.0, prompt, ..Default::default() }), 3), [1, 2, 0]);
    }

    #[test]
    fn min_tokens_holds_the_end_back() {
        assert_eq!(picks(&mut greedy(Shape { repetition: 1.0, min_tokens: 2, hold: vec![0], ..Default::default() }), 3), [1, 1, 0]);
    }

    #[test]
    fn logprobs_line_up_with_the_picks() {
        let q = Queue::default();
        let mut p = greedy(Shape { repetition: 1.0, top: Some((2, q.clone())), bias: vec![(3, -100.0)], ..Default::default() });
        assert_eq!(picks(&mut p, 2), [0, 0]);
        let got: Vec<LogProb> = q.lock().unwrap().drain(..).collect();
        assert_eq!(got.len(), 2);
        let z = (3f64.exp() + 2.5f64.exp() + 2f64.exp() + (-99.5f64).exp()).ln() as f32;
        assert!((got[0].logprob - (3.0 - z)).abs() < 1e-5, "{}", got[0].logprob);
        assert_eq!(got[0].top.iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        assert!((got[0].top[1].1 - (2.5 - z)).abs() < 1e-5);
        assert!(got[0].top.iter().map(|x| (x.1 as f64).exp()).sum::<f64>() < 1.0);
    }
}
