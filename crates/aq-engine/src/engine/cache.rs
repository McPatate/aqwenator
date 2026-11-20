use super::*;

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
pub(super) struct Snapshot {
    pub(super) tokens: Vec<u32>,
    pub(super) slot: u32,
    pub(super) blocks: Vec<u32>,
    pub(super) last_used: u64,
    pub(super) forks: u64,
}

/// Tokens between a prompt checkpoint and the end of its prompt, at least (see
/// `Engine::prompt_checkpoints`).
pub const CKPT_BACKOFF: usize = 4;

/// Shared prefixes shorter than this are not worth a snapshot slot.
pub const SNAPSHOT_MIN_TOKENS: usize = 1024;

/// A finished sequence kept for reuse.
pub(super) struct Parked {
    /// Tokens fed through the model; the slot's state and the KV pages cover exactly these.
    pub(super) tokens: Vec<u32>,
    pub(super) slot: u32,
    pub(super) blocks: Vec<u32>,
    pub(super) last_used: u64,
    pub(super) parked_at: Instant,
    pub(super) turns: u32,
    pub(super) sid: u64,
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
}

impl Engine {
    /// Park a checkpoint of running sequence `i` at prompt position `n`: `slot` already holds
    /// the state there, and the entry shares the KV pages below it.
    pub(super) fn park_prefix(&mut self, i: usize, n: usize, slot: u32) {
        let bs = self.limits().block_size;
        let s = &mut self.running[i];
        s.ckpt_at = None;
        let blocks = s.blocks[..n.div_ceil(bs)].to_vec();
        let (tokens, sid, turns) = (s.req.prompt[..n].to_vec(), s.sid, s.turns);
        self.pool.share(&blocks);
        self.parked.push(Parked { tokens, slot, blocks, last_used: self.clock, parked_at: Instant::now(), turns, sid });
        self.cache.checkpoints += 1;
    }

    fn drop_parked(&mut self, i: usize) {
        let p = self.parked.swap_remove(i);
        self.pool.slots.push(p.slot);
        self.pool.release(p.blocks);
    }

    fn drop_snapshot(&mut self, i: usize) {
        let sn = self.snapshots.swap_remove(i);
        self.pool.slots.push(sn.slot);
        self.pool.release(sn.blocks);
    }

    /// Drop every parked session and snapshot, pinned or not.
    pub fn clear_cache(&mut self) {
        while !self.parked.is_empty() {
            self.drop_parked(0);
        }
        while !self.snapshots.is_empty() {
            self.drop_snapshot(0);
        }
    }

    pub fn num_parked(&self) -> usize {
        self.parked.len()
    }

    /// Evict the least-recently-used parked session (never a snapshot).
    pub(super) fn evict_lru_parked(&mut self) -> bool {
        let Some(i) = (0..self.parked.len()).min_by_key(|&i| self.parked[i].last_used) else { return false };
        self.drop_parked(i);
        self.cache.evictions += 1;
        true
    }

    pub(super) fn is_dead(&self, i: usize, now: Instant) -> bool {
        now.duration_since(self.parked[i].parked_at).as_millis() as u64 >= self.policy.dead_after_ms
    }

    /// May a restricted admission (load control, non-hit) evict parked session `i`? Sessions
    /// presumed finished always; with a residency quantum, also sessions that have been served
    /// `quantum` turns since they last became resident (working-set rotation).
    pub(super) fn evictable_restricted(&self, i: usize, now: Instant) -> bool {
        self.is_dead(i, now) || (self.policy.quantum > 0 && self.parked[i].turns >= self.policy.quantum)
    }

    /// Evict one parked session chosen by the policy, or, if none and snapshots are not
    /// pinned, the least-recently-used snapshot. With `self.dead_only`, only sessions that
    /// `evictable_restricted` allows: finished ones first, then the most served.
    pub(super) fn evict_lru(&mut self) -> bool {
        let now = Instant::now();
        let parked = 0..self.parked.len();
        let oldest_dead = parked.clone().filter(|&i| self.is_dead(i, now)).min_by_key(|&i| self.parked[i].parked_at);
        let victim = if self.dead_only {
            oldest_dead.or_else(|| parked.filter(|&i| self.evictable_restricted(i, now)).max_by_key(|&i| self.parked[i].turns))
        } else {
            match self.policy.evict {
                Evict::Lru => parked.min_by_key(|&i| self.parked[i].last_used),
                Evict::Idle => oldest_dead.or_else(|| parked.max_by_key(|&i| self.parked[i].parked_at)),
            }
        };
        if let Some(i) = victim {
            self.drop_parked(i);
            self.cache.evictions += 1;
            return true;
        }
        if self.dead_only || self.policy.pin_snapshots {
            return false;
        }
        let Some(i) = (0..self.snapshots.len()).min_by_key(|&i| self.snapshots[i].last_used) else { return false };
        self.drop_snapshot(i);
        self.cache.evictions += 1;
        self.cache.snapshot_evictions += 1;
        true
    }

    /// Test hook: permanently remove up to `n` free pages from the pool (a smaller GPU).
    pub fn withhold_pages(&mut self, n: usize) {
        let keep = self.pool.pages.len() - n.min(self.pool.pages.len());
        self.pool.pages.truncate(keep);
    }

    pub fn free_pages(&self) -> usize {
        self.pool.pages.len()
    }

    /// Test hook: evict every parked session.
    pub fn evict_all_parked(&mut self) {
        while !self.parked.is_empty() {
            self.evict_lru();
        }
    }

    /// Take `n` free KV pages (refcount 1), evicting cached sessions as needed.
    pub(super) fn take_blocks(&mut self, n: usize) -> Option<Vec<u32>> {
        while self.pool.pages.len() < n {
            if !self.evict_lru() {
                return None;
            }
        }
        self.pool.take_pages(n)
    }

    /// Longest page-aligned prefix that `prompt` shares with any running or parked sequence.
    pub(super) fn shared_prefix(&self, prompt: &[u32]) -> usize {
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

    /// A state slot for a checkpoint or a snapshot: a free one, else that of the least
    /// recently used parked session. Load control protects parked sessions: free slots only.
    fn cache_slot(&mut self) -> Option<u32> {
        if self.pool.slots.is_empty() && !self.policy.load_control {
            self.evict_lru_parked();
        }
        self.pool.slots.pop()
    }

    /// Prompt checkpoint of running sequence `i` at its current (page-aligned) position.
    /// Skipped when no state slot can be had.
    pub(super) fn checkpoint(&mut self, i: usize) -> Result<()> {
        let Some(slot) = self.cache_slot() else { return Ok(()) };
        self.model.copy_slot(slot as usize, self.running[i].slot as usize)?;
        self.park_prefix(i, self.running[i].computed, slot);
        Ok(())
    }

    /// Take checkpoints and shared-prefix snapshots that became due for these running sequences.
    pub(super) fn after_step(&mut self, idx: impl Iterator<Item = usize>) -> Result<()> {
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
    pub(super) fn drop_session(&mut self, sid: u64) {
        while let Some(k) = self.parked.iter().position(|p| p.sid == sid) {
            self.drop_parked(k);
        }
    }

    /// Take a snapshot of running sequence `i` at its current position (page aligned).
    pub(super) fn snapshot(&mut self, i: usize) -> Result<()> {
        if self.policy.pin_snapshots {
            let s = &self.running[i];
            let n = s.computed;
            if self.snapshots.iter().any(|sn| sn.tokens.len() == n && (0..n).all(|k| sn.tokens[k] == s.token_at(k))) {
                return Ok(());
            }
        }
        if self.snapshots.len() >= self.max_snapshots {
            if let Some(j) = (0..self.snapshots.len()).min_by_key(|&j| self.snapshots[j].last_used) {
                self.drop_snapshot(j);
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
        self.pool.share(&blocks);
        self.snapshots.push(Snapshot { tokens, slot, blocks, last_used: self.clock, forks: 0 });
        self.cache.snapshots += 1;
        Ok(())
    }

    pub(super) fn find_parked(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.parked.len())
            .filter(|&i| is_strict_prefix(&self.parked[i].tokens, prompt))
            .max_by_key(|&i| self.parked[i].tokens.len())
    }

    pub(super) fn find_snapshot(&self, prompt: &[u32]) -> Option<usize> {
        if !self.session_cache {
            return None;
        }
        (0..self.snapshots.len())
            .filter(|&i| is_strict_prefix(&self.snapshots[i].tokens, prompt))
            .max_by_key(|&i| self.snapshots[i].tokens.len())
    }

    /// Tokens of `prompt` that already have state (parked session or snapshot).
    pub(super) fn cached_len(&self, prompt: &[u32]) -> usize {
        let p = self.find_parked(prompt).map(|i| self.parked[i].tokens.len()).unwrap_or(0);
        let s = self.find_snapshot(prompt).map(|i| self.snapshots[i].tokens.len()).unwrap_or(0);
        p.max(s)
    }

    /// Keep an eighth of the state slots free: checkpoints, taps and forks are skipped when
    /// none is, and with a KV pool large enough that nothing is evicted for memory, parked
    /// entries would take every slot.
    pub(super) fn reserve_slots(&mut self) {
        let want = self.limits().max_seqs / 8;
        while self.pool.slots.len() < want && self.evict_lru_parked() {}
    }

    /// Park (session cache) or free finished sequences, given their running indices.
    pub(super) fn retire(&mut self, mut done: Vec<usize>) {
        let bs = self.limits().block_size;
        done.sort_unstable_by(|a, b| b.cmp(a));
        done.dedup();
        for i in done {
            let mut s = self.running.swap_remove(i);
            if let Some(sp) = s.spare {
                self.pool.slots.push(sp);
            }
            // Nothing computed (cancelled before its first chunk): nothing worth keeping.
            if self.session_cache && s.computed > 0 {
                let tokens = (0..s.computed).map(|k| s.token_at(k)).collect();
                // Admission reserved pages for the whole token limit. Keep only the pages that
                // hold computed tokens: a request that stops early (EOS well before a large
                // max_tokens) must not pin the unused reservation while it is parked.
                let keep = s.computed.div_ceil(bs).min(s.blocks.len());
                let extra = s.blocks.split_off(keep);
                self.pool.release(extra);
                self.parked.push(Parked { tokens, slot: s.slot, blocks: s.blocks, last_used: self.clock, parked_at: Instant::now(), turns: s.turns + 1, sid: s.sid });
            } else {
                self.pool.slots.push(s.slot);
                self.pool.release(s.blocks);
            }
        }
    }
}
