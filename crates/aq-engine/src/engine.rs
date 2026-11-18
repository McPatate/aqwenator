//! Continuous-batching scheduler over [`Model::step`], with a session cache.
//!
//! Every step packs, in order: one decode token for each running sequence that has finished
//! its prompt, then prefill chunks (oldest first) until the token budget is spent.
//!
//! Admission reserves every KV page a request can need (prompt + max new tokens) up front, so
//! a running sequence never has to be preempted.
//!
//! Session cache: a finished sequence is *parked* instead of freed. Its state slot (DeltaNet
//! recurrent + conv state) and KV pages stay as they are, keyed by the exact tokens that were
//! fed through the model. A new request whose prompt strictly extends a parked token list
//! resumes from it and prefills only the new suffix, which is exact: the state is the state
//! after those tokens. Parked sessions are evicted least-recently-used when a new request
//! needs their slot or pages.
//!
//! Shared prefixes: when a new request shares a long page-aligned prefix with a known sequence
//! (e.g. the system prompt every agent starts with), its prefill pauses at that boundary and
//! the engine snapshots the recurrent state into a spare slot, pinning the prefix's KV pages
//! with reference counts. Later requests with that prefix fork from the snapshot: a state-slot
//! copy plus shared read-only pages. Exact, because no sequence ever writes below its own
//! `computed` position, so shared pages are never written.

use crate::draft::{DraftIndex, NgramModel};
use crate::model::{Batch, GDN_TC_CHUNK, Limits, MAX_DECODE_ROWS_PER_SEQ, Model};
use crate::sample::Picker;
use anyhow::{Result, bail};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    pub ignore_eos: bool,
}

/// What a request asks for beyond greedy decoding to its token limit (`Engine::add_with`).
#[derive(Default)]
pub struct Control {
    /// Picks each token from its logits row on the host (sampling, constrained decoding).
    /// `None`: the greedy token, computed on the device.
    pub pick: Option<Box<dyn Picker>>,
    /// Tokens that end this request, besides the engine's end-of-sequence tokens. They apply
    /// even with `ignore_eos`.
    pub stop_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct TokenEvent {
    pub id: u64,
    pub token: u32,
    pub index: usize,
    pub finished: bool,
    /// The request ended on an end-of-sequence token (rather than its token limit).
    pub stop: bool,
    /// Prompt tokens this request did not have to prefill (session cache or shared prefix).
    pub cached: usize,
}

struct Seq {
    req: Request,
    out: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    /// Tokens already fed through the model.
    computed: usize,
    /// Pause prefill at this position and snapshot the state there (shared prefix).
    snapshot_at: Option<usize>,
    /// Spare state slot a speculative step writes into (swapped in on full acceptance).
    spare: Option<u32>,
    /// Part of the prefill batch currently in flight on lane 0 (co-execution).
    in_prefill: bool,
    /// Cancelled while its prefill batch was in flight; retired when that batch is collected.
    cancelled: bool,
    /// Prompt-lookup index over the known tokens, built on the first speculative step.
    draft: Option<DraftIndex>,
    /// Turns served since this session last became resident (cache hits keep counting).
    turns: u32,
    /// Session id: parked entries left by the same chain of turns share it.
    sid: u64,
    /// Draft tokens this sequence may carry next step: `spec_k` normally, doubled (up to the
    /// row limit) after a step that accepted a full draft, because copy runs are long and a
    /// longer draft only pays while they last.
    k_cur: usize,
    /// The MTP head's predictions for the tokens after the last known one (the first from the
    /// main pass, the rest chained), with the number of known tokens they were made at.
    mtp_next: Option<(usize, Vec<u32>)>,
    /// When the request was submitted and admitted, and the steps it took part in before its
    /// first token (AQ_TRACE request log).
    t_add: Instant,
    t_admit: Instant,
    steps: u32,
    reused: usize,
    /// Out of KV pages (`kv_reserve`): decoding waits until another sequence frees some.
    stalled: bool,
    /// Pause prefill here and take the prompt checkpoint (page aligned, a few tokens before
    /// the prompt's end); `None` once taken, or when there is nothing to checkpoint.
    ckpt_at: Option<usize>,
}

/// A prefill batch launched on lane 0 and not collected yet (co-execution). Sequences are
/// named by request id: running indices shift while decode steps retire sequences.
struct Inflight {
    /// (request id, tokens fed) per sequence, in batch order.
    plan: Vec<(u64, usize)>,
    /// Request ids that sample a token from this batch, in sample-row order.
    sampled: Vec<u64>,
    ns: usize,
    tokens: usize,
    /// Lane the batch runs on (0: co-execution's prefill lane; 1: the background prefill lane).
    lane: usize,
    /// Checkpoints tapped inside the batch: (request id, position, slot).
    taps: Vec<(u64, usize, u32)>,
    launched: Instant,
}

impl Seq {
    fn known(&self) -> usize {
        self.req.prompt.len() + self.out.len()
    }
    fn token_at(&self, i: usize) -> u32 {
        let p = self.req.prompt.len();
        if i < p { self.req.prompt[i] } else { self.out[i - p] }
    }
}

/// `t` is a strict prefix of `prompt`. Cheap rejects first (last and middle token), so the
/// full comparison usually runs only for the request's own session.
#[inline]
fn is_strict_prefix(t: &[u32], prompt: &[u32]) -> bool {
    let n = t.len();
    n < prompt.len() && (n == 0 || (t[n - 1] == prompt[n - 1] && t[n / 2] == prompt[n / 2])) && prompt[..n] == t[..]
}

/// Which parked session to evict when pages or slots run out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evict {
    /// Least recently parked first (classic LRU).
    Lru,
    /// Most recently parked first. With bounded tool think time, the session parked longest
    /// ago is the one most likely to come back next, so LRU evicts exactly the wrong one.
    Mru,
    /// Idle-aware: sessions idle longer than `dead_after_ms` are presumed finished and go
    /// first (oldest first); among the live ones, most recently parked first.
    Idle,
}

/// Memory and scheduling policy knobs (all default to the original behavior).
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub evict: Evict,
    /// Never evict shared-prefix snapshots to make room (only `max_snapshots` replacement),
    /// and do not create a snapshot identical to an existing one.
    pub pin_snapshots: bool,
    /// Working-set load control: a request that is not a cache hit may not evict parked
    /// sessions; it waits for free pages unless nothing runs or it has waited `max_wait_ms`.
    pub load_control: bool,
    /// Shortest-remaining-uncached-work first, for admission and prefill chunk order, with
    /// aging past `max_wait_ms`.
    pub srpt: bool,
    pub max_wait_ms: u64,
    /// `Evict::Idle`: idle time after which a parked session is presumed finished.
    pub dead_after_ms: u64,
    /// Load control residency quantum: a waiting non-hit request may evict a live parked
    /// session once that session has been served this many turns since it became resident
    /// (0 = never; only finished sessions).
    pub quantum: u32,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            evict: Evict::Lru,
            pin_snapshots: false,
            load_control: false,
            srpt: false,
            max_wait_ms: 2000,
            dead_after_ms: 6000,
            quantum: 0,
        }
    }
}

/// Recurrent state + KV pages at a shared prefix boundary (page aligned).
struct Snapshot {
    tokens: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    last_used: u64,
    forks: u64,
}

/// Tokens between a prompt checkpoint and the end of its prompt, at least (see
/// `Engine::prompt_checkpoints`).
pub const CKPT_BACKOFF: usize = 4;

/// Prefills of at most this many tokens are not held back by the mixed-step prefill budget.
pub const SMALL_PREFILL: usize = 256;

/// Shared prefixes shorter than this are not worth a snapshot slot.
pub const SNAPSHOT_MIN_TOKENS: usize = 1024;

/// A finished sequence kept for reuse.
struct Parked {
    /// Tokens fed through the model; the slot's state and the KV pages cover exactly these.
    tokens: Vec<u32>,
    slot: u32,
    blocks: Vec<u32>,
    last_used: u64,
    parked_at: Instant,
    turns: u32,
    sid: u64,
    /// A prompt checkpoint (state before the end of a prompt) rather than the end of a turn.
    ckpt: bool,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct CacheStats {
    pub admitted: u64,
    pub hits: u64,
    pub forks: u64,
    pub snapshots: u64,
    pub reused_tokens: u64,
    pub prefilled_tokens: u64,
    pub evictions: u64,
    pub checkpoints: u64,
    pub snapshot_evictions: u64,
    pub deferred: u64,
    /// Scheduler host time (admission + batch building) and steps, cumulative.
    pub host_us: u64,
    pub steps: u64,
    /// Requests ended early because the KV pool had no page left for them (`kv_reserve`),
    /// and times a request waited for pages instead.
    pub cut: u64,
    pub stalled: u64,
}

/// Speculative decoding counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    /// Decode entries that carried at least one draft token.
    pub spec_entries: u64,
    pub drafted: u64,
    pub accepted: u64,
    /// Decode entries that emitted tokens (with or without drafts) and tokens they emitted.
    pub decode_entries: u64,
    pub emitted: u64,
    /// Draft tokens that came from the cross-session model (the rest: prompt lookup).
    pub global_drafted: u64,
    pub global_accepted: u64,
    /// Drafts whose first token came from the MTP head, and how many of those were accepted.
    pub mtp_drafted: u64,
    pub mtp_accepted: u64,
    /// Requests that stopped on a token inside a fully accepted draft (the verified state
    /// then covers tokens past the stop and is discarded).
    pub stop_in_draft: u64,
    /// Tokens picked on the host (`Control::pick`), and how many of them differ from the
    /// device's greedy token for the same row.
    pub picked: u64,
    pub repicked: u64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct StepStats {
    pub tokens: usize,
    pub decode_seqs: usize,
    pub prefill_tokens: usize,
    pub seqs: usize,
    pub host_us: u64,
    pub gpu_us: u64,
}

pub struct Engine {
    pub model: Model,
    waiting: VecDeque<(Request, Instant)>,
    running: Vec<Seq>,
    free_slots: Vec<u32>,
    free_blocks: Vec<u32>,
    parked: Vec<Parked>,
    snapshots: Vec<Snapshot>,
    /// Reference count per KV page (owners: running/parked sequences and snapshots).
    refcnt: Vec<u32>,
    /// Max snapshot slots kept at once.
    pub max_snapshots: usize,
    clock: u64,
    /// Tokens that end a request (unless it sets `ignore_eos`).
    pub eos: Vec<u32>,
    /// Per-request controls, by request id (absent: plain greedy, engine stop tokens only).
    controls: HashMap<u64, Control>,
    pub last: StepStats,
    pub cache: CacheStats,
    /// Keep finished sequences for prefix reuse (on by default).
    pub session_cache: bool,
    /// Max sequences running at once; state slots beyond this hold parked sessions.
    pub max_running: usize,
    /// Speculative decoding: max draft tokens per decode step (0 = off). Greedy requests only.
    /// Exact: drafts are verified through the normal decode-path kernels, the state is written
    /// to a spare slot, and on a rejection the accepted tokens are simply fed again.
    pub spec_k: usize,
    /// Prompt-lookup drafting: longest suffix n-gram (down to `spec_min_ngram`) matched against
    /// the sequence's own earlier tokens.
    pub spec_max_ngram: usize,
    pub spec_min_ngram: usize,
    pub spec: SpecStats,
    /// Cross-session draft model: extends (or supplies) drafts with what the server's own
    /// earlier generations most often produced after the same few tokens. On unless
    /// AQ_NO_GLOBAL_DRAFT is set.
    pub global_draft: bool,
    global: NgramModel,
    /// Grow a sequence's draft length while its drafts are fully accepted (off with
    /// AQ_NO_SPEC_ADAPT).
    pub spec_adaptive: bool,
    /// Draft tokens taken from the MTP head per step: the first comes with the main pass,
    /// the rest from chained passes of the head alone (AQ_MTP_K, default 3). Chaining paid
    /// off once decode attention read K/V once per sequence instead of once per row: on the
    /// DGX Spark's real traces 3.37 -> 3.97 tokens per step and +10% turns per minute; it is
    /// neutral on the B200.
    pub mtp_k: usize,
    /// Burst prefill (AQ_BURST_PREFILL=tokens, 0 = off) and the minimum number of steps
    /// between two bursts (AQ_BURST_EVERY, default 3).
    pub burst_prefill: usize,
    pub burst_every: usize,
    since_burst: usize,
    /// AQ_SPEC_TRACE: log every decode entry's draft, targets and acceptance.
    spec_trace: bool,
    /// AQ_SPEC_FAULT=N: test hook for the correctness check itself. Every Nth rejected draft
    /// is emitted as if it had been accepted, which the check must report as a mismatch.
    spec_fault: u64,
    spec_rejects: u64,
    /// Max prefill tokens in a step that also carries decode tokens. Bounds how long a running
    /// sequence waits for its next token while new prompts are prefilled (ITL tail), at some
    /// cost to TTFT. `usize::MAX` = no separate cap (only the step budget).
    pub mixed_prefill_budget: usize,
    /// Decode/prefill co-execution (AQ_COEXEC=1): prefill batches run asynchronously on lane 0
    /// while decode steps for the other sequences run on a high-priority lane 1.
    pub coexec: bool,
    inflight: Option<Inflight>,
    /// Decode steps that ran while a prefill batch was in flight (co-execution).
    pub overlap_steps: u64,
    /// Minimum time between decode steps while a prefill batch is in flight (AQ_COEXEC_PACE_US):
    /// leaves the GPU to prefill between decode steps, trading inter-token latency for TTFT.
    pub coexec_pace: std::time::Duration,
    last_decode: Option<Instant>,
    /// (larger budget, pending-prefill threshold in tokens): switch mixed steps to the larger
    /// prefill budget while more than `threshold` admitted prompt tokens await prefill.
    pub mixed_prefill_adaptive: Option<(usize, usize)>,
    /// Prompt checkpoints: keep a copy of the state at the last page boundary at least
    /// `CKPT_BACKOFF` tokens before a request's prompt ends (a parked entry sharing the
    /// prompt's KV pages). A later request that extends the *prompt* then resumes from it even
    /// if it does not extend the tokens this engine generated, which is what agent harnesses
    /// send: the previous prompt plus the assistant turn as the chat template renders it.
    /// The backoff is there because templates end a prompt with a generation header whose
    /// last token re-tokenizes once the turn's text follows it (Qwen3.5: `<think>\n` becomes
    /// `<think>\n\n`), so consecutive prompts diverge just before the previous one's end.
    pub prompt_checkpoints: bool,
    /// AQ_TRACE=1: log one line per step (`[step]`: composition and GPU time) and one per
    /// request at its first token (`[req]`: queue wait, steps, time to first token).
    pub trace: bool,
    /// Background prefill (AQ_BG_PREFILL=tokens, 0 = off): a prompt with at least this many
    /// tokens left to prefill runs as one asynchronous batch on a second lane, while steps for
    /// everyone else (decode, speculation, short prefills) continue. Keeps a long tool result
    /// from stalling the sequences that are generating. Not combined with co-execution.
    pub bg_prefill: usize,
    pub policy: Policy,
    /// Frame pacing: target duration of a step that carries decode tokens (µs).
    pub itl_target_us: Option<f64>,
    pace: Pace,
    /// Set during an admission that may only evict sessions presumed finished.
    dead_only: bool,
    /// Resumes from a turn's end [0] and from a prompt checkpoint [1]: which of a session's
    /// two parked entries clients come back to (see `evict_redundant`).
    kind_hits: [u64; 2],
    cut_ids: Vec<u64>,
    /// Output tokens whose KV pages a request reserves at admission (default: its whole token
    /// limit, so it can never run out). With a smaller value a request reserves
    /// `min(limit, kv_reserve)` and takes further pages as it generates, evicting parked
    /// sessions if needed. When the pool has none left it waits for the pages another running
    /// sequence frees when it ends (`CacheStats::stalled`; nothing new is admitted meanwhile);
    /// if no other sequence can run either, it ends there, as if it had reached its limit
    /// (`CacheStats::cut`). That keeps requests with a large or absent limit from
    /// each holding a full context of pages they will mostly not use.
    pub kv_reserve: usize,
}

/// Online step-cost model for frame pacing.
#[derive(Debug, Clone, Copy)]
struct Pace {
    decode_us: f64,
    prefill_us_per_tok: f64,
}

/// Smallest prefill chunk frame pacing will schedule next to decodes (keeps prefill moving).
const PACE_MIN_CHUNK: usize = 256;

impl Engine {
    pub fn new(model: Model, eos: Vec<u32>) -> Self {
        let lim = model.limits;
        let mut eng = Self {
            model,
            waiting: VecDeque::new(),
            running: Vec::new(),
            // Slot `max_seqs` is the model's dummy slot; the cache-only slots come after it.
            free_slots: (0..(lim.max_seqs + 1 + lim.cache_slots) as u32).filter(|&s| s != lim.max_seqs as u32).rev().collect(),
            free_blocks: (0..lim.num_blocks() as u32).rev().collect(),
            parked: Vec::new(),
            snapshots: Vec::new(),
            refcnt: vec![0; lim.num_blocks()],
            max_snapshots: 4,
            kind_hits: [0; 2],
            cut_ids: Vec::new(),
            clock: 0,
            eos,
            controls: HashMap::new(),
            last: StepStats::default(),
            cache: CacheStats::default(),
            session_cache: true,
            max_running: lim.max_seqs,
            spec_k: 0,
            spec_max_ngram: 4,
            spec_min_ngram: 2,
            spec: SpecStats::default(),
            global_draft: std::env::var("AQ_NO_GLOBAL_DRAFT").is_err(),
            global: NgramModel::new(2, 4),
            spec_adaptive: std::env::var("AQ_NO_SPEC_ADAPT").is_err(),
            mtp_k: std::env::var("AQ_MTP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            burst_prefill: std::env::var("AQ_BURST_PREFILL").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            burst_every: std::env::var("AQ_BURST_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            since_burst: 0,
            spec_trace: std::env::var("AQ_SPEC_TRACE").is_ok(),
            spec_fault: std::env::var("AQ_SPEC_FAULT").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            spec_rejects: 0,
            mixed_prefill_budget: usize::MAX,
            coexec: false,
            inflight: None,
            overlap_steps: 0,
            coexec_pace: std::time::Duration::from_micros(
                std::env::var("AQ_COEXEC_PACE_US").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            ),
            last_decode: None,
            mixed_prefill_adaptive: None,
            prompt_checkpoints: std::env::var("AQ_NO_CKPT").is_err(),
            trace: std::env::var("AQ_TRACE").as_deref() == Ok("1"),
            bg_prefill: 0,
            // AQ_SRPT=1: test hook, so aq-check can run its suite under the SRPT prefill order.
            policy: Policy { srpt: std::env::var("AQ_SRPT").as_deref() == Ok("1"), ..Policy::default() },
            itl_target_us: None,
            pace: Pace { decode_us: 5000.0, prefill_us_per_tok: 30.0 },
            dead_only: false,
            kv_reserve: usize::MAX,
        };
        // The MTP head keeps its own KV over the whole context, so it runs on every step of
        // every lane once it is loaded.
        eng.model.mtp_on = eng.model.has_mtp();
        if let Some(n) = std::env::var("AQ_BG_PREFILL").ok().and_then(|v| v.parse::<usize>().ok()).filter(|&n| n > 0) {
            eng.enable_bg_prefill(n).expect("could not create the background prefill lane");
        } else if std::env::var("AQ_COEXEC").as_deref() == Ok("1") {
            eng.enable_coexec().expect("could not create the decode lane for AQ_COEXEC=1");
        }
        eng
    }

    /// Turn on decode/prefill co-execution (creates the model's decode lane).
    pub fn enable_coexec(&mut self) -> Result<()> {
        self.model.enable_decode_lane()?;
        self.coexec = true;
        Ok(())
    }

    /// Turn on background prefill for prompts with at least `min_tokens` left to prefill.
    pub fn enable_bg_prefill(&mut self, min_tokens: usize) -> Result<()> {
        if self.coexec {
            bail!("background prefill and co-execution both need the second lane");
        }
        self.model.enable_prefill_lane()?;
        self.bg_prefill = min_tokens;
        Ok(())
    }

    /// Start the background prefill of the running sequence with the most prompt left, if it
    /// has at least `bg_prefill` tokens to go and the lane is free.
    fn launch_bg(&mut self) -> Result<()> {
        let lim = self.limits();
        let left = |s: &Seq| s.req.prompt.len().saturating_sub(s.computed);
        let Some(i) = (0..self.running.len())
            .filter(|&i| !self.running[i].in_prefill && left(&self.running[i]) >= self.bg_prefill)
            .max_by_key(|&i| left(&self.running[i]))
        else {
            return Ok(());
        };
        let s = &self.running[i];
        let (start, rem) = (s.computed, left(s));
        let mut n = rem.min(lim.max_tokens);
        if let Some(at) = s.snapshot_at.filter(|&at| at > start) {
            n = n.min(at - start);
        }
        let mut tap = None;
        if let Some(cp) = s.ckpt_at.filter(|&at| at > start) {
            let c = start + (s.req.prompt.len() - CKPT_BACKOFF - start) / GDN_TC_CHUNK * GDN_TC_CHUNK;
            match if self.model.supports_tap() && n == rem && c > start { self.cache_slot() } else { None } {
                Some(slot) => tap = Some((c, slot)),
                None => n = n.min(cp - start),
            }
        }
        let (mut b, sampled) = self.build_batch(&[(i, n)])?;
        if let Some((c, slot)) = tap {
            b.seq_tap = vec![(((c - start) / GDN_TC_CHUNK) as u32, slot)];
        }
        // Whatever the main lane still has queued for this sequence (slot reset or copy, a
        // page copy) must have run before the other lane reads it.
        self.model.gpu.sync()?;
        // AQ_BG_LANE=0 (debug): run the batch on the main lane through the same launch/collect path.
        let lane: usize = std::env::var("AQ_BG_LANE").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        self.model.set_lane(lane);
        let ns = self.model.launch(&b);
        self.model.set_lane(0);
        let ns = ns?;
        self.running[i].in_prefill = true;
        let id = self.running[i].req.id;
        self.inflight = Some(Inflight {
            plan: vec![(id, n)],
            sampled: sampled.iter().map(|&k| self.running[k].req.id).collect(),
            ns,
            tokens: n,
            lane,
            taps: tap.map(|(c, slot)| (id, c, slot)).into_iter().collect(),
            launched: Instant::now(),
        });
        Ok(())
    }

    /// Park the checkpoint a state tap wrote into `slot`: the state of running sequence `i`
    /// at position `c`, with shared references to the KV pages below it.
    fn park_tap(&mut self, i: usize, c: usize, slot: u32) {
        let bs = self.limits().block_size;
        let s = &mut self.running[i];
        s.ckpt_at = None;
        let blocks: Vec<u32> = s.blocks[..c.div_ceil(bs)].to_vec();
        let (tokens, sid, turns) = (s.req.prompt[..c].to_vec(), s.sid, s.turns);
        for &b in &blocks {
            self.refcnt[b as usize] += 1;
        }
        self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns, sid, ckpt: true });
        self.cache.checkpoints += 1;
    }

    pub fn limits(&self) -> Limits {
        self.model.limits
    }

    pub fn add(&mut self, req: Request) -> Result<()> {
        let lim = self.limits();
        if req.prompt.is_empty() || req.prompt.len() + req.max_new_tokens > lim.max_ctx {
            bail!("request {} needs {} tokens of context (max {})", req.id, req.prompt.len() + req.max_new_tokens, lim.max_ctx);
        }
        // A request that needs more KV pages than the whole pool can never be admitted; left
        // in the queue it would block every request behind it forever.
        let need = self.reserved_tokens(&req).div_ceil(lim.block_size);
        if need > lim.num_blocks() {
            bail!(
                "request {} needs {} KV pages ({} tokens) but the pool has {}; lower max_tokens or raise the KV budget",
                req.id,
                need,
                self.reserved_tokens(&req),
                lim.num_blocks()
            );
        }
        self.model.mtp_note(req.prompt.iter().copied());
        self.waiting.push_back((req, Instant::now()));
        Ok(())
    }

    /// Tokens whose KV pages `req` holds from admission on: its prompt and the reserved part
    /// of its output (at least one step's worth of rows).
    fn reserved_tokens(&self, req: &Request) -> usize {
        req.prompt.len() + req.max_new_tokens.min(self.kv_reserve.max(MAX_DECODE_ROWS_PER_SEQ))
    }

    /// Make sure running sequence `i` owns the pages its next step can write (its known tokens
    /// plus one step of rows, within its limit). False when the pool has no page left.
    fn grow(&mut self, i: usize) -> bool {
        let bs = self.limits().block_size;
        let s = &self.running[i];
        let want = (s.known() + MAX_DECODE_ROWS_PER_SEQ).min(s.req.prompt.len() + s.req.max_new_tokens).div_ceil(bs);
        while self.running[i].blocks.len() < want {
            match self.take_blocks(1) {
                Some(b) => self.running[i].blocks.extend(b),
                None => return false,
            }
        }
        true
    }

    /// After a step: sequences that go on generating take the pages for their next step (`at`:
    /// running index and the index of the event of the token it just emitted). One the pool
    /// cannot serve waits, as long as another sequence still runs and will free pages when it
    /// ends; when none does, it ends at the token it just emitted, as if at its limit, and
    /// its pages let the waiting ones go on.
    fn grow_or_cut(&mut self, at: &[(usize, usize)], events: &mut [TokenEvent], done: &mut Vec<usize>) {
        if self.kv_reserve == usize::MAX {
            return;
        }
        for &(i, e) in at {
            if events[e].finished || self.grow(i) {
                continue;
            }
            if self.running.iter().enumerate().any(|(k, s)| k != i && !s.stalled && !done.contains(&k)) {
                self.running[i].stalled = true;
                self.cache.stalled += 1;
            } else {
                events[e].finished = true;
                self.cache.cut += 1;
                done.push(i);
            }
        }
    }

    /// Start of a step: sequences waiting for KV pages take them if some came back. If every
    /// running sequence waits (the one that ran was cancelled and its pages are still shared),
    /// the first ends there; `take_cut` reports it, since it has no token to carry the news.
    fn unstall(&mut self) {
        for i in 0..self.running.len() {
            if self.running[i].stalled && self.grow(i) {
                self.running[i].stalled = false;
            }
        }
        if !self.running.is_empty() && self.running.iter().all(|s| s.stalled) {
            self.cut_ids.push(self.running[0].req.id);
            self.cache.cut += 1;
            self.retire(vec![0]);
        }
    }

    /// Requests ended since the last call without a final token event: they waited for KV
    /// pages that never came. To the client they finished at their token limit.
    pub fn take_cut(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.cut_ids)
    }

    /// `add` with per-request controls. Request ids must be unique among live requests.
    pub fn add_with(&mut self, req: Request, ctl: Control) -> Result<()> {
        // Co-execution and background prefill collect a batch on a second lane, steps after
        // the one that launched it, and take its tokens from the device (`apply_sampled`);
        // a picker would need that batch's logits rows, which the lane's later launches
        // overwrite. Both are study modes (environment only, off by default).
        if ctl.pick.is_some() && (self.coexec || self.bg_prefill > 0) {
            bail!("with AQ_COEXEC or AQ_BG_PREFILL tokens are picked on the device: only greedy requests (temperature 0) without constraints, logit_bias, penalties, min_tokens or logprobs");
        }
        let id = req.id;
        self.add(req)?;
        self.controls.insert(id, ctl);
        Ok(())
    }

    /// Stop a request early, e.g. because its client went away. A waiting request is dropped.
    /// A running one is retired as if it had finished (parked for reuse when the session cache
    /// is on), or, if its prefill batch is in flight, as soon as that batch is collected.
    /// Returns false if the request is not known (already finished).
    pub fn cancel(&mut self, id: u64) -> bool {
        if let Some(i) = self.waiting.iter().position(|(r, _)| r.id == id) {
            self.waiting.remove(i);
            self.controls.remove(&id);
            return true;
        }
        let Some(i) = self.running.iter().position(|s| s.req.id == id) else { return false };
        if self.running[i].in_prefill {
            self.running[i].cancelled = true;
        } else {
            self.retire(vec![i]);
        }
        true
    }

    pub fn has_work(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Drop every parked session and snapshot (their slots and pages return to the pool).
    pub fn clear_cache(&mut self) {
        while self.evict_lru() {}
    }

    pub fn num_parked(&self) -> usize {
        self.parked.len()
    }

    /// Drop one owner reference to each page; pages with no owner left go back to the pool.
    fn release(&mut self, blocks: Vec<u32>) {
        for b in blocks {
            let r = &mut self.refcnt[b as usize];
            *r -= 1;
            if *r == 0 {
                self.free_blocks.push(b);
            }
        }
    }

    /// A parked session holds up to two entries, its prompt checkpoint and its turn's end, and
    /// a client comes back to one kind only: the end if it sends back the tokens this engine
    /// generated, the checkpoint if it renders the conversation again (chat templates). Drop
    /// the least-recently-used entry of the kind that has been resumed less, among sessions
    /// that hold both, so that an idle conversation costs one state slot rather than two.
    fn evict_redundant(&mut self) -> bool {
        let [end, ckpt] = self.kind_hits;
        if end == ckpt {
            return false;
        }
        let loser = ckpt < end;
        let Some(i) = (0..self.parked.len())
            .filter(|&i| self.parked[i].ckpt == loser && self.parked.iter().any(|q| q.sid == self.parked[i].sid && q.ckpt != loser))
            .min_by_key(|&i| self.parked[i].last_used)
        else {
            return false;
        };
        let p = self.parked.swap_remove(i);
        self.free_slots.push(p.slot);
        self.release(p.blocks);
        self.cache.evictions += 1;
        true
    }

    /// A state slot for a checkpoint or a snapshot: a free one, else a parked entry's (a
    /// redundant one first, then the least recently used; never a snapshot's). On a server
    /// that has been up for a while every slot is parked, and a checkpoint that only took
    /// free slots would never be made: the newest prompt is worth more than the oldest.
    /// Under load control only redundant entries go (parked sessions are protected there).
    fn cache_slot(&mut self) -> Option<u32> {
        // AQ_NO_CACHE_EVICT=1 (debug): free slots only, the behavior before this function.
        if self.free_slots.is_empty() && std::env::var_os("AQ_NO_CACHE_EVICT").is_none() && !self.evict_redundant() && !self.policy.load_control {
            self.evict_lru_parked();
        }
        self.free_slots.pop()
    }

    /// Drop the least-recently-used parked session only (never a snapshot).
    fn evict_lru_parked(&mut self) -> bool {
        if self.evict_redundant() {
            return true;
        }
        let Some(i) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used) else { return false };
        let p = self.parked.swap_remove(i);
        self.free_slots.push(p.slot);
        self.release(p.blocks);
        self.cache.evictions += 1;
        true
    }

    fn is_dead(&self, i: usize, now: Instant) -> bool {
        now.duration_since(self.parked[i].parked_at).as_millis() as u64 >= self.policy.dead_after_ms
    }

    /// May a restricted admission (load control, non-hit) evict parked session `i`? Sessions
    /// presumed finished always; with a residency quantum, also sessions that have been served
    /// `quantum` turns since they last became resident (working-set rotation).
    fn evictable_restricted(&self, i: usize, now: Instant) -> bool {
        self.is_dead(i, now) || (self.policy.quantum > 0 && self.parked[i].turns >= self.policy.quantum)
    }

    /// Evict one parked session chosen by the policy (or, if none and snapshots are not
    /// pinned, the least-recently-used snapshot), returning its slot and pages.
    /// With `self.dead_only`, only sessions presumed finished may go.
    fn evict_lru(&mut self) -> bool {
        if self.dead_only {
            // Restricted: finished sessions first (oldest first), then the sessions that have
            // used up their residency quantum (most served first).
            let now = Instant::now();
            let dead = (0..self.parked.len()).filter(|&i| self.is_dead(i, now)).min_by_key(|&i| self.parked[i].parked_at);
            let Some(i) = dead.or_else(|| {
                (0..self.parked.len()).filter(|&i| self.evictable_restricted(i, now)).max_by_key(|&i| self.parked[i].turns)
            }) else {
                return false;
            };
            let p = self.parked.swap_remove(i);
            self.free_slots.push(p.slot);
            self.release(p.blocks);
            self.cache.evictions += 1;
            return true;
        }
        if self.evict_redundant() {
            return true;
        }
        let victim = match self.policy.evict {
            Evict::Lru => (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used),
            Evict::Mru => (0..self.parked.len()).max_by_key(|&i| self.parked[i].last_used),
            Evict::Idle => {
                let now = Instant::now();
                let dead = |i: usize| now.duration_since(self.parked[i].parked_at).as_millis() as u64 >= self.policy.dead_after_ms;
                (0..self.parked.len())
                    .filter(|&i| dead(i))
                    .min_by_key(|&i| self.parked[i].parked_at)
                    .or_else(|| (0..self.parked.len()).max_by_key(|&i| self.parked[i].parked_at))
            }
        };
        if let Some(i) = victim {
            let p = self.parked.swap_remove(i);
            self.free_slots.push(p.slot);
            self.release(p.blocks);
            self.cache.evictions += 1;
            return true;
        }
        if self.policy.pin_snapshots {
            return false;
        }
        if let Some(i) = (0..self.snapshots.len()).min_by_key(|&i| self.snapshots[i].last_used) {
            let sn = self.snapshots.swap_remove(i);
            self.free_slots.push(sn.slot);
            self.release(sn.blocks);
            self.cache.evictions += 1;
            self.cache.snapshot_evictions += 1;
            return true;
        }
        false
    }

    /// Test hook: permanently remove up to `n` free pages from the pool (a smaller GPU).
    pub fn withhold_pages(&mut self, n: usize) {
        let keep = self.free_blocks.len() - n.min(self.free_blocks.len());
        self.free_blocks.truncate(keep);
    }

    pub fn free_pages(&self) -> usize {
        self.free_blocks.len()
    }

    /// Test hook: evict every parked session.
    pub fn evict_all_parked(&mut self) {
        while !self.parked.is_empty() {
            self.evict_lru();
        }
    }

    /// Take `n` free KV pages (refcount 1), evicting cached sessions as needed.
    fn take_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        while self.free_blocks.len() < n {
            if !self.evict_lru() {
                return None;
            }
        }
        let out = self.free_blocks.split_off(self.free_blocks.len() - n);
        for &b in &out {
            self.refcnt[b as usize] = 1;
        }
        Some(out)
    }

    /// Longest page-aligned prefix that `prompt` shares with any running or parked sequence.
    fn shared_prefix(&self, prompt: &[u32]) -> usize {
        let bs = self.limits().block_size;
        let lcp = |t: &mut dyn Iterator<Item = u32>| prompt.iter().zip(t).take_while(|(a, b)| **a == *b).count();
        let mut best = 0;
        for p in &self.parked {
            best = best.max(lcp(&mut p.tokens.iter().copied()));
        }
        for s in &self.running {
            best = best.max(lcp(&mut (0..s.known()).map(|k| s.token_at(k))));
        }
        // Keep at least one token to feed, and align down to a page.
        (best.min(prompt.len().saturating_sub(1)) / bs) * bs
    }

    /// Prompt checkpoint of running sequence `i` at its current (page-aligned) position: a
    /// parked entry with a copy of the state and shared references to the KV pages below it.
    /// Skipped when no state slot is free.
    fn checkpoint(&mut self, i: usize) -> Result<()> {
        let Some(slot) = self.cache_slot() else { return Ok(()) };
        let bs = self.limits().block_size;
        let s = &self.running[i];
        let n = s.computed;
        let blocks: Vec<u32> = s.blocks[..n.div_ceil(bs)].to_vec();
        let (src, sid, turns) = (s.slot as usize, s.sid, s.turns);
        let tokens = s.req.prompt[..n].to_vec();
        self.model.copy_slot(slot as usize, src)?;
        for &b in &blocks {
            self.refcnt[b as usize] += 1;
        }
        self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns, sid, ckpt: true });
        self.cache.checkpoints += 1;
        Ok(())
    }

    /// Take checkpoints and shared-prefix snapshots that became due for these running sequences.
    fn after_step(&mut self, idx: impl Iterator<Item = usize>) -> Result<()> {
        for i in idx {
            let s = &self.running[i];
            if s.snapshot_at == Some(s.computed) {
                self.running[i].snapshot_at = None;
                self.snapshot(i)?;
            }
            if self.running[i].ckpt_at == Some(self.running[i].computed) {
                self.running[i].ckpt_at = None;
                self.checkpoint(i)?;
            }
        }
        Ok(())
    }

    /// Drop every parked entry of session `sid` (stale once one of them has been resumed).
    fn drop_session(&mut self, sid: u64) {
        let mut k = 0;
        while k < self.parked.len() {
            if self.parked[k].sid == sid {
                let p = self.parked.swap_remove(k);
                self.free_slots.push(p.slot);
                self.release(p.blocks);
            } else {
                k += 1;
            }
        }
    }

    /// Take a snapshot of running sequence `i` at its current position (page aligned).
    fn snapshot(&mut self, i: usize) -> Result<()> {
        if self.policy.pin_snapshots {
            let s = &self.running[i];
            let n = s.computed;
            if self.snapshots.iter().any(|sn| sn.tokens.len() == n && (0..n).all(|k| sn.tokens[k] == s.token_at(k))) {
                return Ok(());
            }
        }
        if self.snapshots.len() >= self.max_snapshots {
            if let Some(j) = (0..self.snapshots.len()).min_by_key(|&j| self.snapshots[j].last_used) {
                let sn = self.snapshots.swap_remove(j);
                self.free_slots.push(sn.slot);
                self.release(sn.blocks);
            }
        }
        let Some(slot) = self.cache_slot() else { return Ok(()) };
        let bs = self.limits().block_size;
        let s = &self.running[i];
        let n = s.computed;
        let tokens: Vec<u32> = (0..n).map(|k| s.token_at(k)).collect();
        let blocks: Vec<u32> = s.blocks[..n / bs].to_vec();
        let src = s.slot as usize;
        self.model.copy_slot(slot as usize, src)?;
        for &b in &blocks {
            self.refcnt[b as usize] += 1;
        }
        self.snapshots.push(Snapshot { tokens, slot, blocks, last_used: self.clock, forks: 0 });
        self.cache.snapshots += 1;
        Ok(())
    }

    fn find_parked(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.parked.len())
            .filter(|&i| is_strict_prefix(&self.parked[i].tokens, prompt))
            .max_by_key(|&i| self.parked[i].tokens.len())
    }

    fn find_snapshot(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.snapshots.len())
            .filter(|&i| is_strict_prefix(&self.snapshots[i].tokens, prompt))
            .max_by_key(|&i| self.snapshots[i].tokens.len())
    }

    /// Tokens of `prompt` that already have state (parked session or snapshot).
    fn cached_len(&self, prompt: &[u32]) -> usize {
        let p = self.find_parked(prompt).map(|i| self.parked[i].tokens.len()).unwrap_or(0);
        let s = self.find_snapshot(prompt).map(|i| self.snapshots[i].tokens.len()).unwrap_or(0);
        p.max(s)
    }

    /// Admit waiting requests while there is room, per the policy.
    /// Keep a share of the state slots free for checkpoints, taps and forks, which are all
    /// skipped when no slot is free. With a KV pool large enough that nothing is evicted for
    /// memory, parked entries otherwise take every slot; a turn then consumes the checkpoint
    /// it resumes from without leaving a new one, and the session's next turn re-reads most
    /// of its prompt (32 agents on a B200, 128 slots: 6.1M prompt tokens re-read against
    /// 3.1M, 481 turns per minute against 752).
    fn reserve_slots(&mut self) {
        let want = self.limits().max_seqs / 8;
        while self.free_slots.len() < want && self.evict_lru_parked() {}
    }

    fn admit(&mut self) -> Result<()> {
        self.reserve_slots();
        // A running sequence waits for KV pages: the pages that come back are its own first.
        if self.running.iter().any(|s| s.stalled) {
            return Ok(());
        }
        let cap = self.max_running.min(self.limits().max_seqs);
        let pol = self.policy;
        if !pol.load_control && !pol.srpt {
            // Original behavior: strict FIFO, stop at the first request that does not fit.
            while !self.waiting.is_empty() && self.running.len() < cap {
                if !self.try_admit(0, true)? {
                    break;
                }
            }
            return Ok(());
        }
        let now = Instant::now();
        let aged = |t: Instant| now.duration_since(t).as_millis() as u64 >= pol.max_wait_ms;
        // Candidate order: aged requests first, then by remaining uncached work (SRPT) or FIFO.
        let mut order: Vec<(bool, usize, usize, u64)> = self
            .waiting
            .iter()
            .enumerate()
            .map(|(i, (r, t))| {
                let rem = if pol.srpt { r.prompt.len() - self.cached_len(&r.prompt) } else { 0 };
                (!aged(*t), rem, i, r.id)
            })
            .collect();
        order.sort();
        for (not_aged, _, _, id) in order {
            if self.running.len() >= cap {
                break;
            }
            let Some(wi) = self.waiting.iter().position(|(r, _)| r.id == id) else { continue };
            let hit = self.find_parked(&self.waiting[wi].0.prompt).is_some();
            // Hits may evict freely unless a residency quantum is set (then everyone rotates
            // through the same restricted victims); anyone waiting past max_wait may.
            let hit_ok = hit && pol.quantum == 0;
            let allow_evict = !pol.load_control || hit_ok || !not_aged || self.running.is_empty();
            // Without permission to evict live sessions, sessions presumed finished may still go.
            self.dead_only = !allow_evict;
            let ok = self.try_admit(wi, allow_evict);
            self.dead_only = false;
            if !ok? {
                self.cache.deferred += 1;
            }
        }
        Ok(())
    }

    /// Admit waiting request `wi`: resume a parked session that is a strict prefix of its
    /// prompt, else fork a shared-prefix snapshot, else start fresh. With `allow_evict` false
    /// it only uses free slots and pages plus those of sessions presumed finished (the caller
    /// sets `dead_only` to match). Checks feasibility first and returns false, changing
    /// nothing, when the request cannot fit even after the evictions it is allowed.
    fn try_admit(&mut self, wi: usize, allow_evict: bool) -> Result<bool> {
        let bs = self.limits().block_size;
        let req = &self.waiting[wi].0;
        let need = self.reserved_tokens(req).div_ceil(bs);
        let best = self.find_parked(&req.prompt);
        let fork = if best.is_none() { self.find_snapshot(&req.prompt) } else { None };
        {
            // Feasibility before touching anything: free pages, plus (if eviction is allowed)
            // the pages evicting other parked sessions would actually return (refcount 1).
            // Without this, a request that cannot fit evicts the whole cache and then fails.
            let have = match (best, fork) {
                (Some(i), _) => self.parked[i].blocks.len().min(need),
                (None, Some(i)) => self.snapshots[i].blocks.len().min(need),
                _ => 0,
            };
            let mut avail = self.free_blocks.len();
            let mut slots = self.free_slots.len();
            let now = Instant::now();
            // A page comes back when every reference to it is held by an evictable parked
            // entry (a session's checkpoint and its generation end share the prompt's pages).
            // Counted only when the free pages alone do not cover the request.
            let short = avail < need - have + 1;
            let mut held: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for (k, p) in self.parked.iter().enumerate() {
                if Some(k) == best || (!allow_evict && !self.evictable_restricted(k, now)) {
                    continue;
                }
                if short {
                    for &b in &p.blocks {
                        *held.entry(b).or_insert(0) += 1;
                    }
                }
                slots += 1;
            }
            avail += held.iter().filter(|&(&b, &n)| self.refcnt[b as usize] == n).count();
            let slot_ok = best.is_some() || slots > 0;
            // One more page if the resumed entry ends inside a shared page (copied below).
            let cow = match best {
                Some(i) => {
                    let p = &self.parked[i];
                    let n = p.tokens.len();
                    (n % bs != 0 && self.refcnt[p.blocks[n / bs] as usize] > 1) as usize
                }
                None => 0,
            };
            if !slot_ok || avail < need - have + cow {
                return Ok(false);
            }
        }
        // The prefix this request shares with another sequence, when it is worth a snapshot:
        // at least `SNAPSHOT_MIN_TOKENS` beyond the snapshot it can fork. (A snapshot of a
        // short common prefix, say the tool schemas alone, must not keep a longer one, the
        // system prompt after them, from ever being taken.)
        let deeper = if best.is_none() && self.session_cache {
            let have = fork.map_or(0, |i| self.snapshots[i].tokens.len());
            Some(self.shared_prefix(&self.waiting[wi].0.prompt)).filter(|&s| s >= have + SNAPSHOT_MIN_TOKENS)
        } else {
            None
        };
        // A running sequence is about to snapshot that prefix: wait for the snapshot and fork
        // it, rather than prefill the same prefix again (agents that start together would
        // otherwise each compute the system prompt).
        if deeper.is_some() && self.running.iter().any(|s| s.snapshot_at == deeper) {
            return Ok(false);
        }
        let mut snapshot_at = None;
        let mut forked = false;
        let mut resident_turns = 0;
        let mut sid = self.waiting[wi].0.id;
        let best_ckpt = best.is_some_and(|i| self.parked[i].ckpt);
        if best.is_some() {
            self.kind_hits[best_ckpt as usize] += 1;
        }
        let (slot, mut blocks, computed) = if let Some(i) = best {
            resident_turns = self.parked[i].turns;
            sid = self.parked[i].sid;
            let n = self.parked[i].tokens.len();
            // A turn too short to produce a checkpoint of its own must not consume the one it
            // resumes from, or the session's next turn would find nothing: fork it instead
            // (copy the state, share the pages) when a slot is free.
            let prompt_len = self.waiting[wi].0.prompt.len();
            let no_new_ckpt = self.prompt_checkpoints && prompt_len.saturating_sub(CKPT_BACKOFF) / bs * bs <= n;
            if no_new_ckpt && !self.free_slots.is_empty() {
                let slot = self.free_slots.pop().unwrap();
                let (src, blocks) = (self.parked[i].slot as usize, self.parked[i].blocks.clone());
                self.parked[i].last_used = self.clock;
                self.model.copy_slot(slot as usize, src)?;
                for &b in &blocks {
                    self.refcnt[b as usize] += 1;
                }
                // Drop the session's other entries, keep this one.
                let keep = std::mem::replace(&mut self.parked[i].sid, u64::MAX);
                self.drop_session(keep);
                let k = self.parked.iter().position(|p| p.sid == u64::MAX).unwrap();
                self.parked[k].sid = keep;
                (slot, blocks, n)
            } else {
                let p = self.parked.swap_remove(i);
                // The session's other entries (its checkpoint or its generation end) are stale now.
                self.drop_session(sid);
                (p.slot, p.blocks, n)
            }
        } else {
            let slot = match self.free_slots.pop() {
                Some(s) => s,
                None => {
                    if !self.evict_lru() {
                        return Ok(false);
                    }
                    match self.free_slots.pop() {
                        Some(s) => s,
                        None => return Ok(false),
                    }
                }
            };
            // Eviction may have removed or reordered snapshots: look the fork up again by
            // content, never by a stale index.
            let fork = if fork.is_some() { self.find_snapshot(&self.waiting[wi].0.prompt) } else { None };
            if let Some(i) = fork {
                // Fork from the shared-prefix snapshot: copy its state, share its pages.
                let clock = self.clock;
                let sn = &mut self.snapshots[i];
                sn.last_used = clock;
                sn.forks += 1;
                let (src, blocks, n) = (sn.slot as usize, sn.blocks.clone(), sn.tokens.len());
                self.model.copy_slot(slot as usize, src)?;
                for &b in &blocks {
                    self.refcnt[b as usize] += 1;
                }
                forked = true;
                snapshot_at = deeper;
                (slot, blocks, n)
            } else {
                self.model.reset_slot(slot as usize)?;
                snapshot_at = deeper;
                (slot, Vec::new(), 0)
            }
        };
        if blocks.len() > need {
            let extra = blocks.split_off(need);
            self.release(extra);
        }
        // Copy-on-write: the sequence will write the rest of the page that holds position
        // `computed`. If another owner still references that page, continue on a private copy.
        let cow = computed % bs != 0 && self.refcnt[blocks[computed / bs] as usize] > 1;
        let more = need - blocks.len() + cow as usize;
        match self.take_blocks(more) {
            Some(mut b) => {
                if cow {
                    let fresh = b.pop().unwrap();
                    let old = std::mem::replace(&mut blocks[computed / bs], fresh);
                    self.model.copy_page(fresh as usize, old as usize)?;
                    self.release(vec![old]);
                }
                blocks.extend(b)
            }
            None => {
                // Not enough pages even after evicting everything parked: put the session
                // back (or release the fresh slot) and wait for running sequences to finish.
                if best.is_some() {
                    let tokens = self.waiting[wi].0.prompt[..computed].to_vec();
                    self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns: resident_turns, sid, ckpt: best_ckpt });
                } else {
                    self.free_slots.push(slot);
                    self.release(blocks);
                }
                return Ok(false);
            }
        }
        let (req, t_add) = self.waiting.remove(wi).unwrap();
        let ckpt_at = if self.session_cache && self.prompt_checkpoints {
            Some(req.prompt.len().saturating_sub(CKPT_BACKOFF) / bs * bs).filter(|&at| at > computed)
        } else {
            None
        };
        self.cache.admitted += 1;
        if forked {
            self.cache.forks += 1;
        }
        if computed > 0 {
            self.cache.hits += 1;
            self.cache.reused_tokens += computed as u64;
        }
        self.cache.prefilled_tokens += (req.prompt.len() - computed) as u64;
        self.running.push(Seq { req, out: Vec::new(), slot, blocks, computed, snapshot_at, spare: None, in_prefill: false, cancelled: false, draft: None, turns: resident_turns, sid, ckpt_at, t_add, t_admit: Instant::now(), steps: 0, reused: computed, k_cur: 0, mtp_next: None, stalled: false });
        Ok(true)
    }

    /// Prompt-lookup draft for running sequence `i`: what followed the most recent earlier
    /// occurrence of the longest suffix n-gram of its known tokens (see [`DraftIndex`]).
    /// Returns the draft and how many of its leading tokens came from the prompt lookup; the
    /// rest were added by the cross-session model.
    fn draft(&mut self, i: usize, k: usize) -> (Vec<u32>, usize, bool) {
        let (min_n, max_n) = (self.spec_min_ngram, self.spec_max_ngram);
        let s = &mut self.running[i];
        let mut idx = match s.draft.take() {
            Some(idx) if idx.matches(min_n, max_n) => idx,
            _ => DraftIndex::new(min_n, max_n),
        };
        for p in idx.len()..s.known() {
            idx.push(s.token_at(p));
        }
        let mut d = idx.draft(k);
        s.draft = Some(idx);
        let mut local = d.len();
        // The MTP head's prediction for the next token outranks a lookup that disagrees with
        // it: the head has seen the whole context, the lookup only the last few tokens.
        let n_known = s.known();
        let mtp = s.mtp_next.as_ref().filter(|(at, t)| *at == n_known && !t.is_empty()).map(|(_, t)| t);
        if let Some(t) = mtp {
            if d.first() != Some(&t[0]) {
                d.clear();
                d.extend_from_slice(&t[..t.len().min(k)]);
                local = 0;
            }
        }
        let from_mtp = mtp.is_some();
        if self.global_draft && d.len() < k {
            let n = s.known();
            let tail: Vec<u32> = (n.saturating_sub(4)..n).map(|p| s.token_at(p)).collect();
            self.global.extend(&tail, &mut d, k);
        }
        (d, local, from_mtp)
    }

    fn ensure_blocks(&mut self, si: usize, upto: usize) -> Result<()> {
        let bs = self.limits().block_size;
        while self.running[si].blocks.len() * bs < upto {
            let Some(b) = self.free_blocks.pop() else { bail!("KV page pool exhausted") };
            self.refcnt[b as usize] = 1;
            self.running[si].blocks.push(b);
        }
        Ok(())
    }

    /// Build the batch for `plan` (running index, tokens to feed). Returns the batch and the
    /// running indices that sample a token from it, in sample-row order.
    fn build_batch(&mut self, plan: &[(usize, usize)]) -> Result<(Batch, Vec<usize>)> {
        let mb = self.limits().max_blocks_per_seq();
        let mut b = Batch::default();
        let mut sampled = Vec::new();
        for (k, &(i, n)) in plan.iter().enumerate() {
            let upto = self.running[i].computed + n;
            self.ensure_blocks(i, upto)?;
            let s = &self.running[i];
            b.seq_start.push(b.tokens.len() as u32);
            b.seq_len.push(n as u32);
            b.seq_slot.push(s.slot);
            for p in s.computed..upto {
                b.tokens.push(s.token_at(p));
                b.tok_pos.push(p as u32);
                b.tok_seq.push(k as u32);
            }
            let mut row = s.blocks.clone();
            row.resize(mb, 0);
            b.block_tables.extend_from_slice(&row);
            b.next_tok_last.push(if upto < s.known() { s.token_at(upto) } else { u32::MAX });
            if upto == s.known() {
                b.sample_rows.push(b.tokens.len() as u32 - 1);
                sampled.push(i);
            }
        }
        Ok((b, sampled))
    }

    /// Append sampled tokens, emit events; returns running indices that finished.
    fn apply_sampled(&mut self, sampled: &[usize], toks: &[u32], events: &mut Vec<TokenEvent>) -> Vec<usize> {
        let mut done = Vec::new();
        let at: Vec<(usize, usize)> = sampled.iter().enumerate().map(|(k, &i)| (i, events.len() + k)).collect();
        for (&i, &tok) in sampled.iter().zip(toks) {
            let s = &mut self.running[i];
            s.out.push(tok);
            if self.trace && s.out.len() == 1 {
                eprintln!(
                    "[req] id {} prompt {} reused {} wait_ms {:.1} ttft_ms {:.1} steps {}",
                    s.req.id,
                    s.req.prompt.len(),
                    s.reused,
                    s.t_admit.duration_since(s.t_add).as_secs_f64() * 1e3,
                    s.t_add.elapsed().as_secs_f64() * 1e3,
                    s.steps
                );
            }
            let stop = (!s.req.ignore_eos && self.eos.contains(&tok)) || self.controls.get(&s.req.id).is_some_and(|c| c.stop_ids.contains(&tok));
            let finished = stop || s.out.len() >= s.req.max_new_tokens;
            events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop, cached: s.reused });
            if finished {
                done.push(i);
            }
        }
        self.grow_or_cut(&at, events, &mut done);
        done
    }

    /// Park (session cache) or free finished sequences, given their running indices.
    fn retire(&mut self, mut done: Vec<usize>) {
        let bs = self.limits().block_size;
        done.sort_unstable_by(|a, b| b.cmp(a));
        done.dedup();
        for i in done {
            let mut s = self.running.swap_remove(i);
            self.controls.remove(&s.req.id);
            if let Some(sp) = s.spare {
                self.free_slots.push(sp);
            }
            // Nothing computed (cancelled before its first chunk): nothing worth keeping.
            if self.session_cache && s.computed > 0 {
                let tokens = (0..s.computed).map(|k| s.token_at(k)).collect();
                // Admission reserved pages for the whole token limit. Keep only the pages that
                // hold computed tokens: a request that stops early (EOS well before a large
                // max_tokens) must not pin the unused reservation while it is parked.
                let keep = s.computed.div_ceil(bs).min(s.blocks.len());
                let extra = s.blocks.split_off(keep);
                self.release(extra);
                self.parked.push(Parked { tokens, slot: s.slot, blocks: s.blocks, last_used: self.clock, parked_at: Instant::now(), turns: s.turns + 1, sid: s.sid, ckpt: false });
            } else {
                self.free_slots.push(s.slot);
                self.release(s.blocks);
            }
        }
    }

    fn index_of(&self, id: u64) -> usize {
        self.running.iter().position(|s| s.req.id == id).expect("in-flight sequence missing")
    }

    /// Co-execution step: keep one prefill batch in flight on lane 0 and run a decode step for
    /// every other ready sequence on the high-priority lane 1. Decode tokens never wait for a
    /// prefill chunk; a prefill batch is collected once its lane-0 event completes (or waited
    /// for when nothing is decoding).
    ///
    /// Ordering: admission (slot reset/copy) and snapshots are enqueued on lane 0, the lane of
    /// every sequence until its prefill completes; a sequence moves to lane 1 only after the
    /// host has observed its prefill batch complete, and back to lane 0 (a later turn) only
    /// after it finished decoding, which is synchronous. KV pages and state slots are per
    /// sequence, so the two lanes never write the same memory.
    fn step_coexec(&mut self) -> Result<Vec<TokenEvent>> {
        let t0 = Instant::now();
        let lim = self.limits();
        self.model.set_lane(0);
        self.clock += 1;
        self.unstall();
        self.admit()?;
        let mut events = Vec::new();

        // 1. Launch a prefill batch on lane 0 if none is in flight.
        let mut prefill_tokens = 0;
        if self.inflight.is_none() {
            let mut budget = lim.max_tokens;
            let mut plan = Vec::new();
            for (i, s) in self.running.iter().enumerate() {
                if budget == 0 {
                    break;
                }
                if !s.in_prefill && s.computed < s.req.prompt.len() {
                    let mut n = (s.req.prompt.len() - s.computed).min(budget);
                    for at in [s.snapshot_at, s.ckpt_at].into_iter().flatten().filter(|&at| at > s.computed) {
                        n = n.min(at - s.computed);
                    }
                    plan.push((i, n));
                    budget -= n;
                }
            }
            if !plan.is_empty() {
                let (b, sampled) = self.build_batch(&plan)?;
                let ns = self.model.launch(&b)?;
                prefill_tokens = b.tokens.len();
                for &(i, _) in &plan {
                    self.running[i].in_prefill = true;
                }
                self.inflight = Some(Inflight {
                    plan: plan.iter().map(|&(i, n)| (self.running[i].req.id, n)).collect(),
                    sampled: sampled.iter().map(|&i| self.running[i].req.id).collect(),
                    ns,
                    tokens: b.tokens.len(),
                    lane: 0,
                    taps: Vec::new(),
                    launched: Instant::now(),
                });
            }
        }

        // 2. One decode step on lane 1 for every ready sequence outside the in-flight batch.
        let dplan: Vec<(usize, usize)> = self
            .running
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.in_prefill && !s.stalled && s.computed >= s.req.prompt.len() && s.known() > s.computed)
            // Normally one pending token; more after a rejected speculative step (spec_k
            // switched off since), which the decode path also handles.
            .map(|(i, s)| (i, (s.known() - s.computed).min(MAX_DECODE_ROWS_PER_SEQ)))
            .collect();
        let decode_seqs = dplan.len();
        if decode_seqs > 0 && self.inflight.is_some() && !self.coexec_pace.is_zero() {
            // Pacing: give the in-flight prefill the GPU until the next decode step is due.
            if let Some(last) = self.last_decode {
                let due = last + self.coexec_pace;
                while Instant::now() < due {
                    if self.model.lane_done(0)? {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
            }
        }
        let t1 = Instant::now();
        if decode_seqs > 0 {
            self.last_decode = Some(t1);
            if self.inflight.is_some() {
                self.overlap_steps += 1;
            }
            let (mut b, sampled) = self.build_batch(&dplan)?;
            b.n_decode = decode_seqs;
            self.model.set_lane(1);
            let toks = self.model.step(&b, None);
            self.model.set_lane(0);
            let toks = toks?;
            for &(i, n) in &dplan {
                self.running[i].computed += n;
            }
            let done = self.apply_sampled(&sampled, &toks, &mut events);
            self.retire(done);
        }

        // 3. Collect the prefill batch when it is done; wait for it if nothing else can run.
        if self.inflight.is_some() && (decode_seqs == 0 || self.model.lane_done(0)?) {
            prefill_tokens = prefill_tokens.max(self.collect_inflight(&mut events)?);
        }
        self.last = StepStats {
            tokens: decode_seqs + prefill_tokens,
            decode_seqs,
            prefill_tokens,
            seqs: decode_seqs,
            host_us: t1.duration_since(t0).as_micros() as u64,
            gpu_us: t1.elapsed().as_micros() as u64,
        };
        Ok(events)
    }

    /// Wait for the in-flight prefill batch (co-execution), apply it and emit its sampled tokens.
    /// Returns the batch's token count (0 if none was in flight).
    fn collect_inflight(&mut self, events: &mut Vec<TokenEvent>) -> Result<usize> {
        let Some(inf) = self.inflight.take() else { return Ok(0) };
        let back = self.model.lane();
        self.model.set_lane(inf.lane);
        let toks = self.model.collect(inf.ns);
        self.model.set_lane(back);
        let toks = toks?;
        if self.trace {
            eprintln!("[bg] lane {} tokens {} ms {:.1}", inf.lane, inf.tokens, inf.launched.elapsed().as_secs_f64() * 1e3);
        }
        for &(id, n) in &inf.plan {
            let i = self.index_of(id);
            self.running[i].computed += n;
            self.running[i].in_prefill = false;
            self.running[i].steps += 1;
        }
        for &(id, c, slot) in &inf.taps {
            let i = self.index_of(id);
            self.park_tap(i, c, slot);
        }
        let idx: Vec<usize> = inf.plan.iter().map(|&(id, _)| self.index_of(id)).collect();
        self.after_step(idx.into_iter())?;
        let sampled: Vec<usize> = inf.sampled.iter().map(|&id| self.index_of(id)).collect();
        let mut done = self.apply_sampled(&sampled, &toks, events);
        done.extend(inf.plan.iter().map(|&(id, _)| self.index_of(id)).filter(|&i| self.running[i].cancelled));
        self.retire(done);
        Ok(inf.tokens)
    }

    /// Run one step. Returns one event per sampled token.
    ///
    /// Co-execution and speculative decoding do not combine: with `spec_k > 0` the engine
    /// takes the normal (single-stream, lane 0) path, after draining any prefill batch still
    /// in flight from an earlier co-execution step.
    pub fn step(&mut self) -> Result<Vec<TokenEvent>> {
        if self.coexec {
            if self.spec_k == 0 {
                return self.step_coexec();
            }
            if self.inflight.is_some() {
                let mut events = Vec::new();
                self.collect_inflight(&mut events)?;
                if !events.is_empty() {
                    return Ok(events);
                }
            }
        }
        let t0 = Instant::now();
        let lim = self.limits();
        let mut budget = lim.max_tokens;

        // Admit waiting requests while there is room to run them.
        self.clock += 1;
        self.unstall();
        self.admit()?;

        // Background prefill: collect a finished batch, start the next long prompt.
        let mut bg_events = Vec::new();
        if self.bg_prefill > 0 {
            if self.inflight.as_ref().is_some_and(|inf| inf.lane == 0) || (self.inflight.is_some() && self.model.lane_done(1)?) {
                self.collect_inflight(&mut bg_events)?;
            }
            if self.inflight.is_none() {
                self.launch_bg()?;
                // AQ_BG_SYNC=1 (debug): wait for the batch right away, so the two lanes never overlap.
                if self.inflight.is_some() && std::env::var("AQ_BG_SYNC").is_ok() {
                    self.collect_inflight(&mut bg_events)?;
                }
            }
        }
        let bg_min = if self.bg_prefill > 0 { self.bg_prefill } else { usize::MAX };

        // Decode-path entries first (pending tokens + optional drafts), then prefill chunks in
        // admission order.
        let mut plan: Vec<(usize, usize)> = Vec::new(); // (running index, n tokens)
        let mut drafts: Vec<Vec<u32>> = Vec::new(); // per decode entry
        let mut draft_local: Vec<usize> = Vec::new(); // leading tokens from the prompt lookup
        let mut draft_mtp: Vec<bool> = Vec::new(); // first token backed by the MTP head
        for i in 0..self.running.len() {
            let s = &self.running[i];
            if s.computed < s.req.prompt.len() || budget == 0 || s.stalled {
                continue;
            }
            let pending = s.known() - s.computed;
            let rem = s.req.max_new_tokens - s.out.len();
            let mut d = Vec::new();
            let mut local = 0;
            let mut from_mtp = false;
            if self.spec_k > 0 && rem > 1 {
                let k_cur = s.k_cur.max(self.spec_k);
                let kmax = k_cur.min(rem - 1).min(MAX_DECODE_ROWS_PER_SEQ.saturating_sub(pending));
                if kmax > 0 {
                    (d, local, from_mtp) = self.draft(i, kmax);
                }
            }
            if !d.is_empty() && self.running[i].spare.is_none() {
                // A spare slot for the speculative state: a free one, else the least-recently
                // used parked session's (never a shared-prefix snapshot's).
                let slot = match self.free_slots.pop() {
                    Some(x) => Some(x),
                    None => {
                        if self.evict_lru_parked() { self.free_slots.pop() } else { None }
                    }
                };
                self.running[i].spare = slot;
            }
            if self.running[i].spare.is_none() {
                d.clear();
            }
            // A decode entry costs the step budget what plain decoding would (one token);
            // re-fed tokens and drafts ride along in the step's row headroom
            // (`Limits::max_rows`), so speculation never changes how much prefill fits.
            let n = pending + d.len();
            plan.push((i, n));
            draft_local.push(local.min(d.len()));
            draft_mtp.push(from_mtp && !d.is_empty());
            drafts.push(d);
            budget -= 1;
        }
        let decode_seqs = plan.len();
        if decode_seqs > 0 {
            // Adaptive: while admitted prefill work is backed up past the threshold, use the
            // larger budget (throughput and TTFT under overload); otherwise the small one
            // (inter-token latency when prefill is light).
            let mixed = match self.mixed_prefill_adaptive {
                Some((hi, threshold)) => {
                    let pending: usize = self
                        .running
                        .iter()
                        .map(|s| s.req.prompt.len().saturating_sub(s.computed))
                        .sum();
                    if pending > threshold { hi } else { self.mixed_prefill_budget }
                }
                None => self.mixed_prefill_budget,
            };
            budget = budget.min(mixed);
            // Frame pacing: size the prefill part so the whole step fits the ITL target,
            // from online estimates of the decode cost and the per-token prefill cost.
            if let Some(target) = self.itl_target_us {
                let room = target - self.pace.decode_us.min(target * 0.8);
                let tokens = (room / self.pace.prefill_us_per_tok.max(1e-3)) as usize;
                budget = budget.min(tokens.max(PACE_MIN_CHUNK));
            }
        }
        // Prefill order: admission order, or shortest remaining prefill first (SRPT).
        // Prompts long enough for the background lane wait for it instead of taking main steps.
        let mut prefill: Vec<usize> = (0..self.running.len())
            .filter(|&i| {
                let s = &self.running[i];
                let left = s.req.prompt.len().saturating_sub(s.computed);
                left > 0 && left < bg_min && !s.in_prefill
            })
            .collect();
        // With SRPT the longest-waiting prompt would starve behind a stream of shorter ones, so
        // it goes first with a guaranteed share: half the budget, or more when the shorter
        // prompts cannot use their half.
        let mut elder: Option<(usize, usize)> = None;
        if self.policy.srpt {
            let left = |i: usize| self.running[i].req.prompt.len() - self.running[i].computed;
            prefill.sort_by_key(|&i| left(i));
            if let Some(&e) = prefill.iter().min_by_key(|&&i| self.running[i].t_add) {
                if e != prefill[0] && left(e) > SMALL_PREFILL {
                    let others: usize = prefill.iter().filter(|&&i| i != e && left(i) > SMALL_PREFILL).map(|&i| left(i)).sum();
                    elder = Some((e, (budget / 2).max(budget.saturating_sub(others))));
                    prefill.retain(|&i| i != e);
                    prefill.insert(0, e);
                }
            }
        }
        // Burst prefill: a long prompt fed 1280 tokens at a time pays the step's fixed cost
        // (and the decode rows) once per slice, and shares the slices with other prompts. Every
        // `burst_every` steps the longest-waiting prompt that needs more than the budget gets a
        // step's whole prefill allowance up to `burst_prefill` tokens instead: fewer, longer
        // steps for it, at the price of one long inter-token gap for the decoding sequences.
        let mut burst = false;
        if self.burst_prefill > 0 && decode_seqs > 0 && self.since_burst >= self.burst_every {
            let left = |i: usize| self.running[i].req.prompt.len() - self.running[i].computed;
            if let Some(&e) = prefill.iter().filter(|&&i| left(i) > budget).min_by_key(|&&i| self.running[i].t_add) {
                elder = Some((e, self.burst_prefill.max(budget)));
                prefill.retain(|&i| i != e);
                prefill.insert(0, e);
                burst = true;
            }
        }
        self.since_burst = if burst { 0 } else { self.since_burst.saturating_add(1) };
        // Checkpoints taken inside this step's prefill: (plan index, position, slot).
        let mut taps: Vec<(usize, usize, u32)> = Vec::new();
        let can_tap = self.model.supports_tap();
        // Hard cap on prefill tokens (the step's row capacity); `budget` is the soft one.
        let mut room = lim.max_tokens - decode_seqs;
        for i in prefill {
            let s = &self.running[i];
            let start = s.computed;
            let rem = s.req.prompt.len() - start;
            // Small prefills ride along whatever the mixed-step budget says: next to the
            // weights a step streams anyway they cost almost nothing, and making a short
            // turn (or the last few tokens after a checkpoint) wait behind a long prompt is
            // what the time-to-first-token tail is made of.
            let cap = match elder {
                Some((e, share)) if e == i => share,
                _ => budget,
            };
            let mut n = if rem <= SMALL_PREFILL { rem.min(room) } else { rem.min(cap).min(room) };
            if n == 0 {
                continue;
            }
            if let Some(at) = s.snapshot_at.filter(|&at| at > start) {
                n = n.min(at - start);
            }
            if let Some(cp) = s.ckpt_at.filter(|&at| at > start) {
                // The checkpoint is free when this step reaches the end of the prompt: tap
                // the state at the last 64-token chunk boundary before the backoff. Otherwise
                // stop at the page-aligned checkpoint position and take it between steps.
                let c = start + (s.req.prompt.len() - CKPT_BACKOFF - start) / GDN_TC_CHUNK * GDN_TC_CHUNK;
                match if can_tap && n == rem && c > start { self.cache_slot() } else { None } {
                    Some(slot) => taps.push((plan.len(), c, slot)),
                    None => n = n.min(cp - start),
                }
            }
            plan.push((i, n));
            budget = budget.saturating_sub(n);
            room -= n;
        }
        if plan.is_empty() {
            // Nothing for the main lane: wait for the background batch rather than spin.
            if bg_events.is_empty() && self.inflight.is_some() {
                self.collect_inflight(&mut bg_events)?;
            }
            return Ok(bg_events);
        }

        let mb = lim.max_blocks_per_seq();
        let mut b = Batch::default();
        // Per plan entry: (first sampled row index in `toks`, number of sampled rows).
        let mut samples: Vec<(usize, usize)> = Vec::new();
        for (k, &(i, n)) in plan.iter().enumerate() {
            let d: &[u32] = if k < decode_seqs { &drafts[k] } else { &[] };
            let real = n - d.len(); // tokens already known
            let upto = self.running[i].computed + n;
            self.ensure_blocks(i, upto)?;
            let s = &self.running[i];
            b.seq_start.push(b.tokens.len() as u32);
            b.seq_len.push(n as u32);
            b.seq_slot.push(s.slot);
            b.seq_slot_out.push(if d.is_empty() { s.slot } else { s.spare.unwrap() });
            b.seq_tap.push(match taps.iter().find(|t| t.0 == k) {
                Some(&(_, c, slot)) => (((c - s.computed) / GDN_TC_CHUNK) as u32, slot),
                None => (0, 0),
            });
            for p in s.computed..s.computed + real {
                b.tokens.push(s.token_at(p));
                b.tok_pos.push(p as u32);
                b.tok_seq.push(k as u32);
            }
            for (j, &t) in d.iter().enumerate() {
                b.tokens.push(t);
                b.tok_pos.push((s.computed + real + j) as u32);
                b.tok_seq.push(k as u32);
            }
            let mut row = s.blocks.clone();
            row.resize(mb, 0);
            b.block_tables.extend_from_slice(&row);
            b.next_tok_last.push(if upto < s.known() { s.token_at(upto) } else { u32::MAX });
            if s.computed + real == s.known() {
                // Sample at the last known token and at every draft.
                let first = b.tokens.len() - d.len() - 1;
                samples.push((b.sample_rows.len(), d.len() + 1));
                for r in first..b.tokens.len() {
                    b.sample_rows.push(r as u32);
                }
            } else {
                samples.push((0, 0));
            }
        }
        b.n_decode = decode_seqs;
        let host_us = t0.elapsed().as_micros() as u64;
        self.cache.host_us += host_us;
        self.cache.steps += 1;
        let t1 = Instant::now();
        let toks = self.model.step(&b, None)?;
        let gpu_us = t1.elapsed().as_micros() as u64;
        if self.trace {
            // Decode rows and the KV keys they scan; prefill tokens and the keys they scan.
            let (mut rows, mut dkeys, mut ptok, mut pkeys) = (0usize, 0usize, 0usize, 0usize);
            for (k, &(i, n)) in plan.iter().enumerate() {
                let c = self.running[i].computed;
                if k < decode_seqs {
                    rows += n;
                    dkeys += n * c;
                } else {
                    ptok += n;
                    pkeys += n * (c + n / 2);
                }
            }
            eprintln!(
                "[step] dec {decode_seqs} rows {rows} dkeys {dkeys} pseqs {} ptok {ptok} pkeys {pkeys} taps {} gpu_us {gpu_us} host_us {host_us}",
                plan.len() - decode_seqs,
                taps.len()
            );
        }
        for &(i, _) in &plan {
            self.running[i].steps += 1;
        }
        // Update the pacing model: pure-decode steps give the decode cost, mixed steps give
        // the per-token prefill cost on top of it.
        let prefill_tokens = b.tokens.len() - decode_seqs;
        if prefill_tokens == 0 {
            self.pace.decode_us = 0.9 * self.pace.decode_us + 0.1 * gpu_us as f64;
        } else if prefill_tokens >= 64 {
            let base = if decode_seqs > 0 { self.pace.decode_us } else { 0.0 };
            let per = ((gpu_us as f64 - base).max(0.0)) / prefill_tokens as f64;
            self.pace.prefill_us_per_tok = 0.8 * self.pace.prefill_us_per_tok + 0.2 * per;
        }

        // Tokens from a collected background batch come first: this step may already have
        // decoded the next token of the same sequence.
        let mut events = bg_events;
        let mut done = Vec::new();
        // (running index, sampled row) of sequences to chain MTP drafts for.
        let mut chain: Vec<(usize, usize)> = Vec::new();
        let eos = self.eos.clone();
        let mut st = self.spec;
        // (running index, index of its last event) of the sequences that emitted this step.
        let mut emitted_at: Vec<(usize, usize)> = Vec::new();
        for (k, &(i, n)) in plan.iter().enumerate() {
            let d: &[u32] = if k < decode_seqs { &drafts[k] } else { &[] };
            let (r0, nr) = samples[k];
            let mut targets = toks[r0..r0 + nr].to_vec();
            // Accept drafts while they equal the target's greedy choice at the previous row.
            let mut m = d.iter().zip(&targets).take_while(|(a, b)| a == b).count();
            if self.spec_fault > 0 && m < d.len() {
                self.spec_rejects += 1;
                if self.spec_rejects % self.spec_fault == 0 {
                    targets[m] = d[m];
                }
            }
            let s = &mut self.running[i];
            let ignore_eos = s.req.ignore_eos;
            let stop_ids: Vec<u32> = self.controls.get(&s.req.id).map(|c| c.stop_ids.clone()).unwrap_or_default();
            let stops = |t: &u32| (!ignore_eos && eos.contains(t)) || stop_ids.contains(t);
            // Tokens this entry emits: the accepted targets up to and including the first one
            // that ends the request (the token limit cannot fall inside a draft: see `kmax`).
            let mut emit = if nr == 0 { 0 } else { targets[..m + 1].iter().position(&stops).map_or(m + 1, |p| p + 1) };
            let mut repicked = false;
            if let (Some(pick), true) = (self.controls.get_mut(&s.req.id).and_then(|c| c.pick.as_mut()), nr > 0) {
                // Host-side choice, row by row and only for tokens that are emitted (the picker
                // advances with each): the token picked at a row decides whether the draft fed
                // at the next row stands, exactly as the greedy token does above.
                (m, emit) = (0, 0);
                for j in 0..nr {
                    let mut row = self.model.logits_row(r0 + j)?;
                    let t = pick.pick(&mut row);
                    repicked = t != targets[j];
                    st.picked += 1;
                    st.repicked += repicked as u64;
                    targets[j] = t;
                    emit = j + 1;
                    if stops(&t) || d.get(j) != Some(&t) {
                        break;
                    }
                    m = j + 1;
                }
            }
            if k < decode_seqs && self.spec_trace {
                eprintln!(
                    "[spec] id {} out {} computed {} known {} n {n} slot {} spare {:?} local {} draft {d:?} targets {targets:?} m {m}",
                    s.req.id,
                    s.out.len(),
                    s.computed,
                    s.known(),
                    s.slot,
                    s.spare,
                    draft_local[k]
                );
            }
            // The spare slot holds the state after the pending tokens and all m accepted
            // drafts. A stop before the last accepted draft leaves that state ahead of the
            // tokens the request emitted: it is then dropped like a rejection, so `computed`
            // never passes the known tokens and the session parks the state of what it emitted.
            let overrun = !d.is_empty() && m == d.len() && emit < m;
            if overrun {
                st.stop_in_draft += 1;
            }
            if d.is_empty() {
                s.computed += n;
            } else if m == d.len() && !overrun {
                // Every draft accepted: the spare slot holds the state after all n tokens.
                s.k_cur = if self.spec_adaptive && d.len() >= s.k_cur.max(self.spec_k) { (2 * d.len()).min(MAX_DECODE_ROWS_PER_SEQ - 1) } else { 0 };
                s.computed += n;
                let old = s.slot;
                s.slot = s.spare.unwrap();
                s.spare = Some(old);
            }
            // else: rejection. The old slot still holds the state before this step; the
            // accepted tokens become pending and are fed again next step.
            if !d.is_empty() && (m < d.len() || overrun) {
                s.k_cur = 0;
            }
            if k < decode_seqs && nr > 0 {
                st.decode_entries += 1;
                if !d.is_empty() {
                    st.spec_entries += 1;
                    st.drafted += d.len() as u64;
                    st.accepted += m as u64;
                    st.global_drafted += (d.len() - draft_local[k]) as u64;
                    st.global_accepted += m.saturating_sub(draft_local[k]) as u64;
                    if draft_mtp[k] {
                        st.mtp_drafted += 1;
                        st.mtp_accepted += (m >= 1) as u64;
                    }
                }
            }
            // The MTP head's prediction at the row whose sample is the last token emitted.
            // (The head continues the device's greedy token: no use after a different pick.)
            let mtp_pred = if nr > 0 && !repicked { self.model.last_mtp.get(r0 + m).copied() } else { None };
            let mut finished = false;
            for &tok in &targets[..emit] {
                s.out.push(tok);
                self.model.mtp_note([tok]);
                if self.global_draft {
                    let n = s.known() - 1;
                    let ctx: Vec<u32> = (n.saturating_sub(4)..n).map(|p| s.token_at(p)).collect();
                    self.global.observe(&ctx, tok);
                }
                if self.trace && s.out.len() == 1 {
                    eprintln!(
                        "[req] id {} prompt {} reused {} wait_ms {:.1} ttft_ms {:.1} steps {}",
                        s.req.id,
                        s.req.prompt.len(),
                        s.reused,
                        s.t_admit.duration_since(s.t_add).as_secs_f64() * 1e3,
                        s.t_add.elapsed().as_secs_f64() * 1e3,
                        s.steps
                    );
                }
                let stop = stops(&tok);
                finished = stop || s.out.len() >= s.req.max_new_tokens;
                events.push(TokenEvent { id: s.req.id, token: tok, index: s.out.len() - 1, finished, stop, cached: s.reused });
                if k < decode_seqs {
                    st.emitted += 1;
                }
                if finished {
                    break;
                }
            }
            debug_assert!(s.computed <= s.known());
            if emit > 0 {
                emitted_at.push((i, events.len() - 1));
            }
            s.mtp_next = mtp_pred.map(|t| (s.known(), vec![t]));
            if finished {
                done.push(i);
            } else if mtp_pred.is_some() {
                chain.push((i, r0 + m));
            }
        }
        self.spec = st;
        self.grow_or_cut(&emitted_at, &mut events, &mut done);
        chain.retain(|c| !done.contains(&c.0) && !self.running[c.0].stalled);
        // Chained MTP drafts for the sequences that keep decoding: as many as the draft
        // budget and the KV pages they already own allow (chained rows write to the head's
        // cache at the positions the next pass overwrites).
        if self.spec_k > 1 && self.mtp_k > 1 && !chain.is_empty() {
            let bs = self.limits().block_size;
            let want = self.mtp_k.min(self.spec_k) - 1;
            let rows: Vec<(usize, u32, usize, &[u32])> = chain
                .iter()
                .map(|&(i, sample)| {
                    let s = &self.running[i];
                    let pos = s.known() - 1;
                    let rem = (s.req.max_new_tokens - s.out.len()).saturating_sub(2);
                    (sample, pos as u32, want.min(rem).min((s.blocks.len() * bs).saturating_sub(pos)), &s.blocks[..])
                })
                .collect();
            let more = self.model.mtp_chain(&rows)?;
            for (&(i, _), toks) in chain.iter().zip(more) {
                if let Some((_, t)) = self.running[i].mtp_next.as_mut() {
                    t.extend(toks);
                }
            }
        }
        // Tapped checkpoints: the tap slot holds the state at `c`; park it with the pages below.
        for &(k, c, slot) in &taps {
            self.park_tap(plan[k].0, c, slot);
        }
        self.after_step(plan.iter().map(|&(i, _)| i))?;
        self.retire(done);
        self.last = StepStats {
            tokens: b.tokens.len(),
            decode_seqs,
            prefill_tokens: b.tokens.len() - decode_seqs,
            seqs: plan.len(),
            host_us,
            gpu_us,
        };
        Ok(events)
    }
}
