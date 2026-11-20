//! Qwen3.5 (`qwen3_5`) dense hybrid model: weights on device + a batched forward step.
//!
//! One `step` runs a mixed batch: any number of sequences, each contributing a chunk of
//! tokens (a prefill chunk or a single decode token). Full-attention layers read a paged KV
//! cache; gated-delta-net layers keep a fixed-size recurrent state per sequence slot.

use crate::config::{LayerKind, TextConfig};
use crate::experts::ExpertPool;
use crate::ngram::Ngram;
use crate::cublaslt::CUDA_R_16BF;
use crate::gpu::{Fmt, Gpu, Prof, Weight, cfg, cfg_smem, de};
use cuda_core::LaunchConfig;
use crate::weights::Checkpoint;
use anyhow::Result;
use half::bf16;
use std::path::Path;
use std::time::Instant;

const P: &str = "model.language_model.";

/// A gated residual: the streams' norm weights, the low-rank gate that mixes them into a
/// block's input, and (0 for the final one) the weights that spread the block's output.
#[derive(Clone, Copy, Default)]
struct Hc {
    norm: u64,
    down: u64,
    up: u64,
    inject: u64,
}

/// The n-gram embedding one layer adds to the streams before its attention block.
struct Ple {
    layer: usize,
    table: Ngram,
    key: u64,
    value: u64,
    norm_key: u64,
    norm_query: u64,
    norm_conv: u64,
    conv_w: u64,
    /// Per state slot: the conv's 9 last inputs, and the two tokens before the sequence's
    /// next one.
    state: u64,
    before: Vec<[u32; 2]>,
    rows: Vec<u8>,
}

/// Row stride of a gated residual's injection logits (one per stream, padding).
const HC_INJ_LD: usize = 16;

pub struct Layer {
    pub kind: LayerKind,
    in_norm: u64,
    post_norm: u64,
    // with several residual streams, in place of the two norms
    hc_attn: Hc,
    hc_mlp: Hc,
    // gated delta net
    gdn_proj: Weight,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    gnorm: u64,
    out_proj: Weight,
    // full attention
    qkv: Weight,
    q_norm: u64,
    k_norm: u64,
    o_proj: Weight,
    // mlp
    gate_up: Weight,
    down: Weight,
    // mixture of experts (0 for a dense MLP; gate_up and down are then the shared expert's):
    // router rows then the shared expert's gate row, and where `moe_gemm` finds the experts:
    // expert e's weights are `ex_slots[e] * ex_stride` bytes past these
    router: u64,
    ex_gate_up: Weight,
    ex_down: Weight,
    ex_slots: u64,
    ex_stride: [u32; 2],
    ex_base: u32,
    // per-slot state pools
    delta_state: u64,
    conv_state: u64,
    k_cache: u64,
    v_cache: u64,
}

pub struct Scratch {
    amax_v: u64,
    amax_i: u64,
    // decode attention partials (one per row, head and slice)
    attn_part: u64,
    attn_ml: u64,
    // decode attention: this step's tiles
    dtile_start: u64,
    dtile_len: u64,
    dec_tiles: usize,
    /// Warps per `attn_decode_tq` block this step (sized for the longest decode sequence).
    dec_warps: usize,
    // chunked GDN + tiled attention (prefill)
    chunk_start: u64,
    chunk_len: u64,
    seq_chunk0: u64,
    seq_nchunk: u64,
    tile_start: u64,
    tile_len: u64,
    gu: u64,
    gw: u64,
    gqp: u64,
    gkp: u64,
    gintra: u64,
    gdecay: u64,
    go: u64,
    // tensor-core chunked GDN (per chunk-head: Qn, Kn, U, W, G, h snapshot, Vn^T)
    tqn: u64,
    tkn: u64,
    tu: u64,
    tw: u64,
    tg: u64,
    tht: u64,
    tvnt: u64,
    ids: u64,
    tok_pos: u64,
    tok_seq: u64,
    seq_start: u64,
    seq_len: u64,
    seq_slot: u64,
    seq_slot_out: u64,
    // state taps (prompt checkpoints inside a prefill chunk)
    seq_tap_n: u64,
    seq_tap_len: u64,
    seq_tap_slot: u64,
    rows: u64,
    tables: u64,
    hfinal: u64,
    x: u64,
    // gated residual: normed streams, gate (low rank, then per stream), injection logits and weights
    xn: u64,
    hc_a: u64,
    hc_b: u64,
    hc_c: u64,
    inj: u64,
    /// This step's n-gram embeddings.
    ple_e: u64,
    h: u64,
    proj: u64,
    conv: u64,
    mix: u64,
    delta: u64,
    // mixture of experts: router logits, the 8 experts and weights per token, rows grouped
    // by expert, jobs, and the activation and output of each (token, expert) row
    moe_logits: u64,
    /// Each token's selections (16 u16), then room for guesses of the next two layers'.
    moe_idx: u64,
    moe_w: u64,
    moe_list: u64,
    moe_jobs: u64,
    moe_act: u64,
    moe_out: u64,
    q: u64,
    logits: u64,
    tok: u64,
    // MTP head (stubs unless the head is loaded)
    next: u64,
    mtp_hfinal: u64,
    mtp_logits: u64,
    mtp_tok: u64,
}

/// The multi-token-prediction head: one full-attention layer over
/// `fc([norm_e(embed(next token)), norm_h(final-normed hidden)])`, then its own norm and the
/// shared output head. At position p it predicts token p + 2.
pub struct Mtp {
    layer: Layer,
    fc: u64,
    norm: u64,
    pre_e: u64,
    pre_h: u64,
    /// The head's argmax runs over `short_n` gathered output-embedding rows (0 = the whole
    /// vocabulary): reading the full table costs more than the rest of the head, and a wrong
    /// shortlist only costs a rejected draft.
    short_n: usize,
    /// Token id in each slot, and each token's slot (`u32::MAX` = not shortlisted).
    short_ids: Vec<u32>,
    short_pos: Vec<u32>,
    /// Next slot to recycle (the lower half keeps the lowest token ids for good).
    short_ring: usize,
    /// Slots whose row has to be gathered again before the next step.
    short_dirty: Vec<u32>,
    g_short_ids: u64,
    g_short_slots: u64,
    short_w: u64,
    /// With the embedding table kept in host memory: the shortlisted tokens' embedding rows
    /// and each token's slot, on the device for the draft head (0 otherwise).
    short_e: u64,
    g_short_pos: u64,
}

impl Mtp {
    /// Width of the head's logits rows.
    fn head_width(&self, cfg: &TextConfig) -> usize {
        if self.short_n > 0 { self.short_n } else { cfg.vocab_size }
    }
}

pub struct LoadStats {
    pub open_s: f64,
    pub upload_s: f64,
    pub weight_bytes: usize,
}

/// Chunk length of the CUDA-core chunked gated delta rule (must match the kernels).
pub const GDN_CHUNK: usize = 32;
/// Chunk length of the tensor-core chunked gated delta rule (must match the kernels).
pub const GDN_TC_CHUNK: usize = 64;
/// Dynamic shared memory of the tensor-core GDN kernels (bytes; must match the kernels).
const GDN_TC_PREP_SMEM: u32 = 70672;
const GDN_TC_SCAN_SMEM: u32 = 93696;
const GDN_TC_OUT_SMEM: u32 = 88832;
/// Value columns per `gdn_tc_scan` block (must match the kernel).
const GDN_TC_SCAN_VB: u32 = 32;
/// Query rows per tile of the prefill attention kernel (must match the kernels).
pub const ATTN_QTILE: usize = 16;
/// Blocks per row in the two-phase argmax.
pub const ARGMAX_PARTS: usize = 32;
/// Capacity of the decode-attention partials in (row x slice) units.
fn attn_part_rows(max_seqs: usize) -> usize {
    max_seqs * MAX_DECODE_ROWS_PER_SEQ * DEC_TC_SLICES
}

/// Max tokens one decode-path sequence may feed in a step (speculative verification).
pub const MAX_DECODE_ROWS_PER_SEQ: usize = 9;
/// Slices of the decode attention's fixed bounds table.
pub const DEC_TC_SLICES: usize = 64;

/// Engine sizing, fixed at load time (every buffer is preallocated).
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// Max tokens in one step (prefill chunks + decode tokens).
    pub max_tokens: usize,
    /// Max concurrent sequences (recurrent-state slots).
    pub max_seqs: usize,
    /// Max context length per sequence.
    pub max_ctx: usize,
    /// KV cache page size in tokens.
    pub block_size: usize,
    /// KV page pool size in bytes; 0 = enough for every slot at `max_ctx`. With a budget,
    /// slots can outnumber what fits at full context (idle sessions share the pool).
    pub kv_budget_bytes: usize,
    /// Pages in the pool, resolved at load time from the budget.
    pub kv_blocks: usize,
}

impl Limits {
    /// Token rows a step may hold: the token budget plus headroom for speculative draft rows
    /// (drafts do not count against the budget).
    pub fn max_rows(&self) -> usize {
        self.max_tokens + self.max_seqs * (MAX_DECODE_ROWS_PER_SEQ - 1)
    }
    pub fn max_chunks(&self) -> usize {
        self.max_tokens / GDN_CHUNK + self.max_seqs
    }
    /// Upper bound on 64-token chunks in one step: full chunks plus one partial per sequence.
    pub fn max_tc_chunks(&self) -> usize {
        self.max_tokens / GDN_TC_CHUNK + self.max_seqs.min(self.max_tokens)
    }
    pub fn max_tiles(&self) -> usize {
        self.max_tokens / ATTN_QTILE + self.max_seqs
    }
    pub fn max_blocks_per_seq(&self) -> usize {
        self.max_ctx.div_ceil(self.block_size)
    }
    pub fn num_blocks(&self) -> usize {
        if self.kv_blocks > 0 { self.kv_blocks } else { self.max_seqs * self.max_blocks_per_seq() }
    }
}

/// One step's work. Token-level arrays have one entry per token; sequence-level arrays one
/// entry per sequence in the batch. `sample_rows` lists the token rows whose logits are
/// sampled (the last row of each sequence that finishes its chunk of input this step).
#[derive(Debug, Default, Clone)]
pub struct Batch {
    pub tokens: Vec<u32>,
    pub tok_pos: Vec<u32>,
    pub tok_seq: Vec<u32>,
    pub seq_start: Vec<u32>,
    pub seq_len: Vec<u32>,
    pub seq_slot: Vec<u32>,
    pub block_tables: Vec<u32>,
    pub sample_rows: Vec<u32>,
    /// Sequences `0..n_decode` take the decode path: DeltaNet layers run them token by token
    /// with the state in registers, attention runs one split-K row per token. Normally one
    /// token each; a speculative step feeds pending tokens plus drafts.
    pub n_decode: usize,
    /// Slot each sequence's updated state is written to (empty = same as `seq_slot`). A
    /// speculative step writes to a spare slot so the old state survives a rejection.
    pub seq_slot_out: Vec<u32>,
    /// State taps, one `(chunks, slot)` per sequence (empty = none): after `chunks` full
    /// 64-token chunks of a prefill sequence, its recurrent and conv state are also written to
    /// `slot`. Only with the tensor-core chunked DeltaNet path (`Model::supports_tap`).
    pub seq_tap: Vec<(u32, u32)>,
    /// Per sequence, the token that follows its last row if it is already known (a prefill
    /// chunk that does not end the prompt), else `u32::MAX`. Only the MTP head reads it; empty
    /// = unknown for every sequence.
    pub next_tok_last: Vec<u32>,
}

pub struct Model {
    pub cfg: TextConfig,
    pub gpu: Gpu,
    /// The embedding table on the device, or 0 when it stays in host memory (`embed_host`).
    embed: u64,
    /// The mapped checkpoint and the embedding tensor's name, and the buffer a step's rows
    /// are gathered into.
    embed_host: Option<(Checkpoint, String)>,
    embed_rows: Vec<u8>,
    final_norm: u64,
    lm_head: Weight,
    /// The bf16 output head, which the MTP shortlist gathers from (0 when only its 8-bit
    /// form in `lm_head` is kept).
    head_table: u64,
    pub layers: Vec<Layer>,
    /// MTP draft head (when the checkpoint has one, unless AQ_MTP=0).
    mtp: Option<Mtp>,
    /// Run the MTP head this step (set by the engine; needs `has_mtp`).
    pub mtp_on: bool,
    /// MTP predictions of the last step, one per sampled row: the token expected after the
    /// one that row sampled. Empty when the head did not run.
    pub last_mtp: Vec<u32>,
    /// For each entry of `last_mtp`, its row in the MTP head's GPU buffers (a graph step
    /// samples every row and picks the caller's afterwards).
    last_mtp_rows: Vec<u32>,
    /// The forward pass being enqueued is a chained MTP pass (no backbone).
    chain_mode: bool,
    s: Scratch,
    pub limits: Limits,
    pub load: LoadStats,
    pub prof: Option<Prof>,
    /// Captured decode graphs, keyed by (padded batch size, attention splits), with the
    /// profile mark names recorded inside them (when profiling).
    graphs: std::collections::HashMap<(usize, usize), (u64, Vec<&'static str>)>,
    /// Split-K slices for this step's decode attention (set by `upload`).
    cur_splits: usize,
    /// Decode-path token rows of this step (set by `upload`).
    cur_nd_rows: usize,
    /// This step has at least one state tap (set by `upload`).
    cur_tap: bool,
    /// Logits buffer of the last step that sampled, whichever lane ran it.
    last_logits: u64,
    pub use_graphs: bool,
    /// AQ_MOE_ROUTES=<file> (debug): every forward appends, per mixture-of-experts layer,
    /// `[layer u32][tokens u32][experts per token u32]` then each token's experts as u16.
    routes: Option<std::fs::File>,
    experts: ExpertPool,
    ple: Option<Ple>,
    /// The gated residual that mixes the streams into the output head's input.
    final_hc: Hc,
    /// Prefill work lists of the step being run (set by `upload`).
    cur_chunks: usize,
    cur_tiles: usize,
    /// A/B switches back to the reference kernels (AQ_GDN_RECURRENT=1, AQ_ATTN_NAIVE=1,
    /// AQ_GDN_CHUNK_CC=1 for the CUDA-core chunked GDN).
    ref_gdn: bool,
    ref_attn: bool,
    /// AQ_ATTN_TILE=1: use the CUDA-core tiled prefill attention instead of tensor cores.
    tile_attn: bool,
    /// Dynamic shared memory of `attention_tc` for this model's GQA group size.
    attn_tc_smem: u32,
    attn_nbuf: u32,
    /// Keys and values are cached in 8 bits (AQ_KV=int8).
    kv8: bool,
    /// Slice bounds of `attn_decode_tq` (keys, DEC_TC_SLICES + 1 entries; device copy and host
    /// copy): fine slices first, coarser ones for long contexts, the last one open-ended.
    dec_bounds: u64,
    dec_bounds_h: Vec<u32>,
    /// 16-token sub-tiles per `attention_tc` block (AQ_ATTN_QT, default 2): each K/V tile a
    /// block loads serves `16 * attn_qt` query tokens.
    attn_qt: usize,
    cc_gdn: bool,
    /// Rows of the active lane's scratch: the GEMM row cap and the per-step token limit.
    s_cap: usize,
    /// The inactive lane's per-step state when a decode lane exists (co-execution); swapped
    /// with the active fields by `set_lane`.
    alt: Option<LaneState>,
    /// One completion event per lane (co-execution).
    lane_events: Vec<cuda_core::Event>,
}

/// Everything that is per step and per lane: scratch and metadata buffers, the work lists of
/// the step being run, and the CUDA graphs captured against this lane's buffers.
struct LaneState {
    s: Scratch,
    cap: usize,
    graphs: std::collections::HashMap<(usize, usize), (u64, Vec<&'static str>)>,
    cur_chunks: usize,
    cur_tiles: usize,
    cur_splits: usize,
    cur_nd_rows: usize,
}

/// Allocate one lane's scratch for `lim.max_rows()` token rows (and `lim.max_seqs` sequences).
fn alloc_scratch(gpu: &mut Gpu, cfg: &TextConfig, lim: &Limits, cc_gdn: bool, mtp: Option<usize>) -> Result<Scratch> {
    // `mtp`: width of the MTP head's logits (shortlist size or the vocabulary), if loaded.
    let mtp_width = mtp.unwrap_or(0);
    let mtp = mtp.is_some();
    let mrows = if mtp { lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ } else { 0 };
    let hd = cfg.hidden_size;
    let wide = cfg.gdn_proj_ld().max(cfg.attn_proj_dim()).max(2 * cfg.intermediate_size);
    let mixw = cfg.value_dim().max(cfg.num_attention_heads * cfg.head_dim).max(cfg.intermediate_size);
    let t = lim.max_rows();
    let mt = if cfg.moe() { t } else { 0 };
    // Residual streams, and the same or 0 for buffers only a gated residual needs.
    let (hcn, hcx) = (cfg.hc_count, if cfg.hc_count > 1 { cfg.hc_count } else { 0 });
    let ek = cfg.num_experts_per_tok;
    let sq = lim.max_seqs;
    let hvd = cfg.linear_num_value_heads * 128;
    let hq = cfg.num_attention_heads;
    // Only the selected chunked-GDN path gets real scratch; the other gets a stub.
    let (cct, tcc) = if cc_gdn { (t, 0) } else { (0, lim.max_tc_chunks() * cfg.linear_num_value_heads) };
    Ok(Scratch {
        amax_v: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * ARGMAX_PARTS * 4)?,
        amax_i: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * ARGMAX_PARTS * 4)?,
        attn_part: gpu.alloc(attn_part_rows(sq) * hq * cfg.head_dim * 4)?,
        attn_ml: gpu.alloc(attn_part_rows(sq) * hq * 2 * 4)?,
        dtile_start: gpu.alloc((sq * MAX_DECODE_ROWS_PER_SEQ + 8) * 4)?,
        dtile_len: gpu.alloc((sq * MAX_DECODE_ROWS_PER_SEQ + 8) * 4)?,
        dec_tiles: 0,
        dec_warps: 1,
        chunk_start: gpu.alloc(lim.max_chunks() * 4)?,
        chunk_len: gpu.alloc(lim.max_chunks() * 4)?,
        seq_chunk0: gpu.alloc(sq * 4)?,
        seq_nchunk: gpu.alloc(sq * 4)?,
        tile_start: gpu.alloc(lim.max_tiles() * 4)?,
        tile_len: gpu.alloc(lim.max_tiles() * 4)?,
        gu: gpu.alloc(cct * hvd * 4)?,
        gw: gpu.alloc(cct * hvd * 4)?,
        gqp: gpu.alloc(cct * hvd * 4)?,
        gkp: gpu.alloc(cct * hvd * 4)?,
        gintra: gpu.alloc(if cc_gdn { lim.max_chunks() * cfg.linear_num_value_heads * GDN_CHUNK * GDN_CHUNK * 4 } else { 0 })?,
        gdecay: gpu.alloc(lim.max_chunks() * cfg.linear_num_value_heads * 4)?,
        go: gpu.alloc(cct * hvd * 2)?,
        tqn: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tkn: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tu: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tw: gpu.alloc(tcc * GDN_TC_CHUNK * 128 * 2)?,
        tg: gpu.alloc(tcc * GDN_TC_CHUNK * 4)?,
        tht: gpu.alloc(tcc * 128 * 128 * 2)?,
        tvnt: gpu.alloc(tcc * 128 * GDN_TC_CHUNK * 2)?,
        ids: gpu.alloc(t * 4)?,
        tok_pos: gpu.alloc(t * 4)?,
        tok_seq: gpu.alloc(t * 4)?,
        seq_start: gpu.alloc((sq + 1) * 4)?,
        seq_len: gpu.alloc((sq + 1) * 4)?,
        seq_slot: gpu.alloc((sq + 1) * 4)?,
        seq_slot_out: gpu.alloc((sq + 1) * 4)?,
        seq_tap_n: gpu.alloc((sq + 1) * 4)?,
        seq_tap_len: gpu.alloc((sq + 1) * 4)?,
        seq_tap_slot: gpu.alloc((sq + 1) * 4)?,
        rows: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * 4)?,
        tables: gpu.alloc((sq + 1) * lim.max_blocks_per_seq() * 4)?,
        hfinal: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * hd * 2)?,
        x: gpu.alloc(t * hd * hcn * 2)?,
        xn: gpu.alloc(t * hd * hcx * 2)?,
        hc_a: gpu.alloc(t * cfg.hc_lowrank * 2)?,
        hc_b: gpu.alloc(t * hd * hcx * 2)?,
        hc_c: gpu.alloc(t * HC_INJ_LD * (hcx > 0) as usize * 2)?,
        inj: gpu.alloc(t * hcx * 4)?,
        ple_e: gpu.alloc(if cfg.ple_layer_ids.is_empty() { 0 } else { t * hd * 2 })?,
        h: gpu.alloc(t * hd * 2)?,
        proj: gpu.alloc(t * wide * 2)?,
        conv: gpu.alloc(t * cfg.conv_dim() * 2)?,
        mix: gpu.alloc(t * mixw * 2)?,
        delta: gpu.alloc(t * hd * 2)?,
        moe_logits: gpu.alloc(mt * moe_router_ld(cfg) * 2)?,
        moe_idx: gpu.alloc(mt * 96)?,
        moe_w: gpu.alloc(mt * ek * 4)?,
        moe_list: gpu.alloc(mt * ek * 4)?,
        moe_jobs: gpu.alloc(if mt > 0 { moe_max_jobs(mt, cfg) * 8 } else { 0 })?,
        moe_act: gpu.alloc(mt * ek * cfg.moe_intermediate_size * 2)?,
        moe_out: gpu.alloc(mt * (ek + 1) * hd * 2)?,
        q: gpu.alloc(t * cfg.num_attention_heads * cfg.head_dim * 2)?,
        logits: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * cfg.vocab_size * 2)?,
        tok: gpu.alloc(sq * MAX_DECODE_ROWS_PER_SEQ * 4)?,
        next: gpu.alloc(if mtp { t * 4 } else { 0 })?,
        mtp_hfinal: gpu.alloc(mrows * hd * 2)?,
        mtp_logits: gpu.alloc(mrows * mtp_width * 2)?,
        mtp_tok: gpu.alloc(mrows * 4)?,
    })
}

/// Shared-memory bytes of `attention_tc`: Q rows (16 per sub-tile and query head of the group)
/// plus `nbuf` pairs of 32-key K and V tiles, all 256-wide bf16, plus with the 8-bit cache
/// the staging area for one pair as stored.
fn attention_tc_smem(group: usize, qt: usize, nbuf: usize, kv8: bool) -> u32 {
    ((qt * group * 16 + nbuf * 2 * 32) * 256 * 2 + if kv8 { 2 * 32 * KV8_ROW } else { 0 }) as u32
}

/// Shared-memory bytes of `attn_decode_tq`: 8 query slots and a 16-key P tile per warp plus one
/// K and one V tile of 16 keys.
fn attn_decode_tq_smem(warps: usize, kv8: bool) -> u32 {
    ((8 * warps + 2 * 16) * 512 + 256 * warps + if kv8 { 2 * 16 * KV8_ROW } else { 0 }) as u32
}

/// Bytes of one key or value row in the 8-bit KV cache (256 int8 and an f32 scale; must match
/// the kernels).
const KV8_ROW: usize = 272;

/// Padded batch sizes that get a captured decode graph.
const DECODE_BUCKETS: &[usize] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512];

/// Row stride of the router logits: the experts, the shared expert's gate, padding.
fn moe_router_ld(cfg: &TextConfig) -> usize {
    (cfg.num_experts + 1).next_multiple_of(8)
}
/// Shared memory of a `moe_gemm` block.
const MOE_GEMM_SMEM: u32 = 3 * (27648 + 4608) + 256;

/// Most jobs `moe_bucket` can produce for `t` tokens: full ones and a partial one per expert.
fn moe_max_jobs(t: usize, cfg: &TextConfig) -> usize {
    (cfg.num_experts_per_tok * t).div_ceil(64) + cfg.num_experts
}

/// The MLP weights of layer `n(..)`: gate/up and down (the shared expert's in a mixture of
/// experts), then router and stacked experts (0 for a dense MLP).
#[allow(clippy::type_complexity)]
/// A layer's MLP weights and, for a mixture, its router and experts.
struct Mlp {
    gate_up: Weight,
    down: Weight,
    router: u64,
    ex_gate_up: Weight,
    ex_down: Weight,
    ex_slots: u64,
    ex_stride: [u32; 2],
    ex_base: u32,
}

fn load_hc(gpu: &mut Gpu, ck: &Checkpoint, cfg: &TextConfig, prefix: &str, inject: bool) -> Result<Hc> {
    let w = |s: &str| format!("{prefix}.{s}.weight");
    let pad = (HC_INJ_LD - cfg.hc_count) * cfg.hc_count * cfg.hidden_size * 2;
    Ok(Hc {
        norm: concat_bf16(gpu, ck, &[w("hc_norm")])?,
        down: concat_bf16(gpu, ck, &[w("input_mix_weight_down")])?,
        up: concat_bf16(gpu, ck, &[w("input_mix_weight_up")])?,
        inject: if inject { concat_bf16_padded(gpu, ck, &[w("block_inject_weight")], pad)? } else { 0 },
    })
}

fn mlp_weights(gpu: &mut Gpu, ck: &Checkpoint, cfg: &TextConfig, pool: &mut ExpertPool, n: &dyn Fn(&str) -> String, fmt: (Fmt, Fmt)) -> Result<Mlp> {
    let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
    let mlp = |s: &str| n(&format!("{}.{s}", if cfg.moe() { "mlp.shared_expert" } else { "mlp" }));
    let gate_up = gemm_weight(gpu, ck, &[mlp("gate_proj.weight"), mlp("up_proj.weight")], 0, h, fmt.0)?;
    let down = gemm_weight(gpu, ck, &[mlp("down_proj.weight")], 0, inter, fmt.1)?;
    let mut m = Mlp { gate_up, down, router: 0, ex_gate_up: Weight::default(), ex_down: Weight::default(), ex_slots: 0, ex_stride: [0; 2], ex_base: 0 };
    if !cfg.moe() {
        return Ok(m);
    }
    let (ne, ei) = (cfg.num_experts, cfg.moe_intermediate_size);
    m.router = concat_bf16_padded(gpu, ck, &[n("mlp.gate.weight"), n("mlp.shared_expert_gate.weight")], (moe_router_ld(cfg) - ne - 1) * h * 2)?;
    let slots: Vec<u32> = if ck.has(&n("mlp.experts.q4r")) {
        let at = |i: usize| pool.p + pool.off[i] as u64;
        m.ex_gate_up = Weight { p: at(0), scale: at(1), fmt: Fmt::Int4 };
        m.ex_down = Weight { p: at(2), scale: at(3), fmt: Fmt::Int4 };
        m.ex_stride = [pool.off[4] as u32; 2];
        m.ex_base = pool.load(gpu, ck, ne, &n("mlp.experts"))?;
        // When paging, `ExpertPool::page` writes the table before each use.
        (m.ex_base..m.ex_base + ne as u32).collect()
    } else {
        m.ex_gate_up = gemm_weight(gpu, ck, &[n("mlp.experts.gate_up_proj")], 0, h, Fmt::Bf16)?;
        m.ex_down = gemm_weight(gpu, ck, &[n("mlp.experts.down_proj")], 0, ei, Fmt::Bf16)?;
        m.ex_stride = [(2 * ei * h * 2) as u32, (h * ei * 2) as u32];
        (0..ne as u32).collect()
    };
    m.ex_slots = gpu.alloc_upload(&slots)?;
    Ok(m)
}

/// Upload several bf16 matrices concatenated along rows into one buffer.
fn concat_bf16(gpu: &mut Gpu, ck: &Checkpoint, names: &[String]) -> Result<u64> {
    concat_bf16_padded(gpu, ck, names, 0)
}

/// Same, with `pad_bytes` of zeros after the last matrix (zero rows that round the GEMM's
/// N up to a tile-friendly size).
fn concat_bf16_padded(gpu: &mut Gpu, ck: &Checkpoint, names: &[String], pad_bytes: usize) -> Result<u64> {
    let views = names.iter().map(|n| ck.bf16(n)).collect::<Result<Vec<_>>>()?;
    let total: usize = views.iter().map(|v| v.bytes.len()).sum::<usize>() + pad_bytes;
    // Not zeroed: the uploads overwrite everything but the padding.
    let p = gpu.alloc_uninit(total)?;
    let mut off = 0u64;
    for v in views {
        // SAFETY: `Model::load` flushes the queue while the checkpoint is still open.
        unsafe { gpu.queue_upload(p + off, v.bytes, v.direct)? };
        off += v.bytes.len() as u64;
    }
    if pad_bytes > 0 {
        gpu.zero(p + off, pad_bytes)?;
        gpu.sync()?;
    }
    Ok(p)
}

/// A GEMM weight `[n, k]`: the named matrices concatenated along rows, then `pad_bytes` of
/// zero rows. Unless `fmt` is bf16 it is quantized as soon as its upload has landed and the
/// bf16 copy freed, so the load never holds much more than the final footprint.
fn gemm_weight(gpu: &mut Gpu, ck: &Checkpoint, names: &[String], pad_bytes: usize, k: usize, fmt: Fmt) -> Result<Weight> {
    // A pre-quantized checkpoint (scripts/convert_autoround.py) has `<name>.q4` codes and
    // `<name>.q4s` scales in place of the bf16 matrix.
    if names.iter().all(|n| ck.has(&format!("{n}.q4"))) {
        anyhow::ensure!(k % 128 == 0, "4-bit weights need K to be a multiple of 128");
        let pad_rows = pad_bytes / 2 / k;
        let (mut qv, mut sv) = (Vec::new(), Vec::new());
        for n in names {
            let (q, sc) = (ck.get(&format!("{n}.q4"))?, ck.get(&format!("{n}.q4s"))?);
            anyhow::ensure!(q.dtype == safetensors::Dtype::U8 && sc.dtype == safetensors::Dtype::F32 && q.bytes.len() / (k / 2) == sc.bytes.len() / (k / 128 * 4), "{n}: unexpected 4-bit layout");
            qv.push(q);
            sv.push(sc);
        }
        let (qb, sb): (usize, usize) = (qv.iter().map(|v| v.bytes.len()).sum(), sv.iter().map(|v| v.bytes.len()).sum());
        let (p, scale) = (gpu.alloc_uninit(qb + pad_rows * k / 2)?, gpu.alloc_uninit(sb + pad_rows * k / 128 * 4)?);
        let (mut qo, mut so) = (0u64, 0u64);
        for (q, sc) in qv.into_iter().zip(sv) {
            // SAFETY: `Model::load` flushes the queue while the checkpoint is still open.
            unsafe {
                gpu.queue_upload(p + qo, q.bytes, q.direct)?;
                gpu.queue_upload(scale + so, sc.bytes, sc.direct)?;
            }
            qo += q.bytes.len() as u64;
            so += sc.bytes.len() as u64;
        }
        if pad_rows > 0 {
            // Zero scales make the padding rows zero whatever their codes are.
            gpu.zero(p + qo, pad_rows * k / 2)?;
            gpu.zero(scale + so, pad_rows * k / 128 * 4)?;
            gpu.sync()?;
        }
        return Ok(Weight { p, scale, fmt: Fmt::Int4 });
    }
    let p = concat_bf16_padded(gpu, ck, names, pad_bytes)?;
    let mut bytes = pad_bytes;
    for n in names {
        bytes += ck.bf16(n)?.bytes.len();
    }
    if gpu.qsim.0 > 0 {
        gpu.drain_uploads()?;
        let mode = if gpu.qsim.0 == 3 && k % 2048 != 0 { 2 } else { gpu.qsim.0 };
        unsafe { gpu.k.qsim(&gpu.stream, cfg(((bytes / 2 / k) as u32, 1, 1), 256), p as *mut u16, k as u32, mode) }.map_err(de)?;
    }
    if fmt == Fmt::Bf16 || (fmt == Fmt::Int8 && k % 32 != 0) {
        return Ok(Weight::bf16(p));
    }
    gpu.drain_uploads()?;
    gpu.quantize_weight(p, bytes / 2 / k, k, fmt, false)
}

impl Model {
    pub fn load(dir: &Path, limits: Limits) -> Result<Self> {
        // The tensor-core attention kernels fetch K/V in 16-key tiles that must not straddle pages.
        anyhow::ensure!(limits.block_size % 16 == 0, "KV page size must be a multiple of 16 tokens");
        let t0 = Instant::now();
        let cfg = TextConfig::load(dir)?;
        let ck = Checkpoint::open(dir)?;
        let open_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let mut gpu = Gpu::new(0)?;
        let one = |n: &str| vec![format!("{P}{n}")];

        // A 4-bit checkpoint is for small devices, and its embedding table and output head
        // are still bf16: the head is stored in 8 bits and the table stays in host memory (a
        // step uploads its tokens' rows; the draft head keeps its shortlist's).
        let small_tables = ["gate_proj.weight.q4", "experts.q4r"].iter().any(|n| ck.has(&format!("{P}layers.0.mlp.{n}"))) && cfg.hidden_size % 32 == 0;
        let tied = cfg.tied() || !ck.has("lm_head.weight");
        let embed_name = format!("{P}embed_tokens.weight");
        let mut small_head = None;
        let (embed, head_table) = if small_tables {
            ck.bf16(&embed_name)?;
            let t = concat_bf16(&mut gpu, &ck, &[if tied { embed_name.clone() } else { "lm_head.weight".to_string() }])?;
            gpu.drain_uploads()?;
            small_head = Some(gpu.quantize_weight(t, cfg.vocab_size, cfg.hidden_size, Fmt::Int8, false)?);
            (0, 0)
        } else {
            let e = concat_bf16(&mut gpu, &ck, &[embed_name.clone()])?;
            (e, if tied { e } else { concat_bf16(&mut gpu, &ck, &["lm_head.weight".to_string()])? })
        };
        let hc = cfg.hc_count > 1;
        anyhow::ensure!(!hc || small_tables, "a model with several residual streams needs a 4-bit checkpoint for now");
        let final_norm = if hc { 0 } else { concat_bf16(&mut gpu, &ck, &one("norm.weight"))? };
        let final_hc = if hc { load_hc(&mut gpu, &ck, &cfg, &format!("{P}hyper_connection_mixer"), false)? } else { Hc::default() };
        // AQ_QUANT (not exact) stores the layers' GEMM weights in 8 bits:
        //   fp8   E4M3 weights and activations;
        //   int8  int8 weights with a scale per 32 values, bf16 activations.
        // AQ_QUANT_HEAD=1: the output head too. AQ_QUANT_SET: which layer GEMMs (default all):
        // g = GDN input projection, o = GDN output projection, q = attention q/k/v,
        // a = attention output, u = MLP gate/up, d = MLP down.
        let quant = match std::env::var("AQ_QUANT").as_deref() {
            Ok("fp8") => Fmt::Fp8,
            Ok("int8") => Fmt::Int8,
            Ok(other) if !other.is_empty() => anyhow::bail!("AQ_QUANT={other}: expected fp8 or int8"),
            _ => Fmt::Bf16,
        };
        anyhow::ensure!(quant != Fmt::Fp8 || !cfg.moe(), "AQ_QUANT=fp8 does not cover mixture-of-experts models");
        let set = std::env::var("AQ_QUANT_SET").unwrap_or_else(|_| "goqaud".into());
        let q = |c: char| if set.contains(c) { quant } else { Fmt::Bf16 };
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);

        let hv = cfg.linear_num_value_heads;
        // AQ_KV=int8 (not exact): keys and values are cached as int8 with a scale per row,
        // 272 bytes instead of 512, expanded to bf16 a tile at a time by the attention kernels.
        let kv8 = match std::env::var("AQ_KV").as_deref() {
            Ok("int8") => true,
            Ok(other) if !other.is_empty() && other != "bf16" => anyhow::bail!("AQ_KV={other}: expected int8 or bf16"),
            _ => false,
        };
        anyhow::ensure!(!kv8 || cfg.head_dim == 256, "the 8-bit KV cache needs 256-wide heads");
        let kv_row = if kv8 { KV8_ROW } else { cfg.head_dim * 2 };
        let mut lim = limits;
        // The MTP draft head is used whenever the checkpoint has one (AQ_MTP=0 turns it off).
        // It adds one attention layer's KV to every page.
        let want_mtp = !hc && std::env::var("AQ_MTP").as_deref() != Ok("0") && ck.has("mtp.fc.weight") && ["", ".q4"].iter().any(|q| ck.has(&format!("mtp.layers.0.self_attn.q_proj.weight{q}")));
        if lim.kv_budget_bytes > 0 {
            let n_full = (0..cfg.num_hidden_layers).filter(|&l| cfg.kind(l) == LayerKind::Full).count() + want_mtp as usize;
            let page_bytes = n_full * 2 * lim.block_size * cfg.num_key_value_heads * kv_row;
            lim.kv_blocks = lim.kv_budget_bytes / page_bytes;
        }
        let kv_bytes = (lim.num_blocks() + 1) * lim.block_size * cfg.num_key_value_heads * kv_row;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let pooled: Vec<String> = (0..cfg.num_hidden_layers).map(|l| format!("{P}layers.{l}.")).chain(want_mtp.then(|| "mtp.layers.0.".into())).collect();
        let mut pool = ExpertPool::new(&mut gpu, &ck, &cfg, &pooled)?;
        // Seconds spent allocating and zeroing recurrent state and KV pages (not weights).
        let mut state_s = 0.0f64;
        for l in 0..cfg.num_hidden_layers {
            let n = |s: &str| format!("{P}layers.{l}.{s}");
            let kind = cfg.kind(l);
            let m = mlp_weights(&mut gpu, &ck, &cfg, &mut pool, &n, (q('u'), q('d')))?;
            let mut layer = Layer {
                kind,
                in_norm: if hc { 0 } else { concat_bf16(&mut gpu, &ck, &[n("input_layernorm.weight")])? },
                post_norm: if hc { 0 } else { concat_bf16(&mut gpu, &ck, &[n("post_attention_layernorm.weight")])? },
                hc_attn: if hc { load_hc(&mut gpu, &ck, &cfg, &n("attn_hyper_connection"), true)? } else { Hc::default() },
                hc_mlp: if hc { load_hc(&mut gpu, &ck, &cfg, &n("mlp_hyper_connection"), true)? } else { Hc::default() },
                gdn_proj: Weight::default(),
                conv_w: 0,
                a_log: 0,
                dt_bias: 0,
                gnorm: 0,
                out_proj: Weight::default(),
                qkv: Weight::default(),
                q_norm: 0,
                k_norm: 0,
                o_proj: Weight::default(),
                gate_up: m.gate_up,
                down: m.down,
                router: m.router,
                ex_gate_up: m.ex_gate_up,
                ex_down: m.ex_down,
                ex_slots: m.ex_slots,
                ex_stride: m.ex_stride,
                ex_base: m.ex_base,
                delta_state: 0,
                conv_state: 0,
                k_cache: 0,
                v_cache: 0,
            };
            match kind {
                LayerKind::Linear => {
                    let la = |s: &str| n(&format!("linear_attn.{s}"));
                    let pad = (cfg.gdn_proj_ld() - cfg.gdn_proj_dim()) * cfg.hidden_size * 2;
                    layer.gdn_proj = gemm_weight(
                        &mut gpu,
                        &ck,
                        &[la("in_proj_qkv.weight"), la("in_proj_z.weight"), la("in_proj_b.weight"), la("in_proj_a.weight")],
                        pad,
                        h,
                        q('g'),
                    )?;
                    layer.conv_w = concat_bf16(&mut gpu, &ck, &[la("conv1d.weight")])?;
                    layer.a_log = gpu.alloc_upload(&ck.f32_vec(&la("A_log"))?)?;
                    layer.dt_bias = gpu.alloc_upload(&ck.f32_vec(&la("dt_bias"))?)?;
                    layer.gnorm = gpu.alloc_upload(&ck.f32_vec(&la("norm.weight"))?)?;
                    layer.out_proj = gemm_weight(&mut gpu, &ck, &[la("out_proj.weight")], 0, cfg.value_dim(), q('o'))?;
                    let t = Instant::now();
                    layer.delta_state = gpu.alloc((lim.max_seqs + 1) * hv * 128 * 128 * 4)?;
                    layer.conv_state = gpu.alloc((lim.max_seqs + 1) * cfg.conv_dim() * 3 * 2)?;
                    gpu.sync()?;
                    state_s += t.elapsed().as_secs_f64();
                }
                LayerKind::Full => {
                    let sa = |s: &str| n(&format!("self_attn.{s}"));
                    layer.qkv = gemm_weight(&mut gpu, &ck, &[sa("q_proj.weight"), sa("k_proj.weight"), sa("v_proj.weight")], 0, h, q('q'))?;
                    layer.q_norm = gpu.alloc_upload(&ck.bf16_vec(&sa("q_norm.weight"))?)?;
                    layer.k_norm = gpu.alloc_upload(&ck.bf16_vec(&sa("k_norm.weight"))?)?;
                    layer.o_proj = gemm_weight(&mut gpu, &ck, &[sa("o_proj.weight")], 0, cfg.num_attention_heads * cfg.head_dim, q('a'))?;
                    let t = Instant::now();
                    layer.k_cache = gpu.alloc(kv_bytes)?;
                    layer.v_cache = gpu.alloc(kv_bytes)?;
                    gpu.sync()?;
                    state_s += t.elapsed().as_secs_f64();
                }
            }
            layers.push(layer);
        }
        let ple = match cfg.ple_layer_ids.first().filter(|&&l| l <= cfg.num_hidden_layers) {
            Some(&l) => {
                let n = |s: &str| format!("{P}layers.{}.ple.{s}", l - 1);
                let table = Ngram::open(&ck, &cfg, n("ple_embedding."))?;
                Some(Ple {
                    layer: l - 1,
                    before: vec![table.start(); lim.max_seqs + 1],
                    table,
                    key: concat_bf16(&mut gpu, &ck, &[n("key_proj.weight")])?,
                    value: concat_bf16(&mut gpu, &ck, &[n("value_proj.weight")])?,
                    norm_key: concat_bf16(&mut gpu, &ck, &[n("norm_key.weight")])?,
                    norm_query: concat_bf16(&mut gpu, &ck, &[n("norm_query.weight")])?,
                    norm_conv: concat_bf16(&mut gpu, &ck, &[n("norm_conv.weight")])?,
                    conv_w: concat_bf16(&mut gpu, &ck, &[n("conv1d.weight")])?,
                    state: gpu.alloc((lim.max_seqs + 1) * 9 * cfg.hc_count * cfg.hidden_size * 2)?,
                    rows: Vec::new(),
                })
            }
            None => None,
        };
        let mtp = if want_mtp {
            let n = |s: &str| format!("mtp.layers.0.{s}");
            let sa = |s: &str| n(&format!("self_attn.{s}"));
            let m = mlp_weights(&mut gpu, &ck, &cfg, &mut pool, &n, (Fmt::Bf16, Fmt::Bf16))?;
            let layer = Layer {
                kind: LayerKind::Full,
                in_norm: if hc { 0 } else { concat_bf16(&mut gpu, &ck, &[n("input_layernorm.weight")])? },
                post_norm: if hc { 0 } else { concat_bf16(&mut gpu, &ck, &[n("post_attention_layernorm.weight")])? },
                hc_attn: if hc { load_hc(&mut gpu, &ck, &cfg, &n("attn_hyper_connection"), true)? } else { Hc::default() },
                hc_mlp: if hc { load_hc(&mut gpu, &ck, &cfg, &n("mlp_hyper_connection"), true)? } else { Hc::default() },
                gdn_proj: Weight::default(),
                conv_w: 0,
                a_log: 0,
                dt_bias: 0,
                gnorm: 0,
                out_proj: Weight::default(),
                qkv: gemm_weight(&mut gpu, &ck, &[sa("q_proj.weight"), sa("k_proj.weight"), sa("v_proj.weight")], 0, h, Fmt::Bf16)?,
                q_norm: gpu.alloc_upload(&ck.bf16_vec(&sa("q_norm.weight"))?)?,
                k_norm: gpu.alloc_upload(&ck.bf16_vec(&sa("k_norm.weight"))?)?,
                o_proj: gemm_weight(&mut gpu, &ck, &[sa("o_proj.weight")], 0, cfg.num_attention_heads * cfg.head_dim, Fmt::Bf16)?,
                gate_up: m.gate_up,
                down: m.down,
                router: m.router,
                ex_gate_up: m.ex_gate_up,
                ex_down: m.ex_down,
                ex_slots: m.ex_slots,
                ex_stride: m.ex_stride,
                ex_base: m.ex_base,
                delta_state: 0,
                conv_state: 0,
                k_cache: gpu.alloc(kv_bytes)?,
                v_cache: gpu.alloc(kv_bytes)?,
            };
            // AQ_MTP_SHORT=N: shortlist size (default 32768; 0 = argmax over the whole vocabulary).
            let mut short_n: usize = std::env::var("AQ_MTP_SHORT").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);
            if short_n >= cfg.vocab_size || short_n % 8 != 0 {
                short_n = 0;
            }
            let short_ids: Vec<u32> = (0..short_n as u32).collect();
            let mut short_pos = vec![u32::MAX; cfg.vocab_size];
            for (slot, &t) in short_ids.iter().enumerate() {
                short_pos[t as usize] = slot as u32;
            }
            Some(Mtp {
                short_n,
                short_dirty: short_ids.clone(),
                short_ring: short_n / 2,
                g_short_ids: gpu.alloc(short_n * 4)?,
                g_short_slots: gpu.alloc(short_n * 4)?,
                short_w: gpu.alloc(short_n * cfg.hidden_size * 2)?,
                short_e: if small_tables { gpu.alloc(short_n * cfg.hidden_size * 2)? } else { 0 },
                g_short_pos: if small_tables { gpu.alloc(cfg.vocab_size * 4)? } else { 0 },
                short_ids,
                short_pos,
                layer,
                fc: concat_bf16(&mut gpu, &ck, &["mtp.fc.weight".to_string()])?,
                norm: concat_bf16(&mut gpu, &ck, &["mtp.norm.weight".to_string()])?,
                pre_e: concat_bf16(&mut gpu, &ck, &["mtp.pre_fc_norm_embedding.weight".to_string()])?,
                pre_h: concat_bf16(&mut gpu, &ck, &["mtp.pre_fc_norm_hidden.weight".to_string()])?,
            })
        } else {
            None
        };
        if mtp.is_some() {
            eprintln!("MTP draft head loaded");
        }
        gpu.sync()?;
        gpu.flush_uploads()?;
        let mut lm_head = small_head.unwrap_or(Weight::bf16(head_table));
        if layers.iter().any(|l| l.gate_up.fmt == Fmt::Int4 || l.ex_gate_up.fmt == Fmt::Int4) {
            anyhow::ensure!(quant != Fmt::Fp8, "AQ_QUANT=fp8 does not apply to a pre-quantized checkpoint");
            gpu.enable_int8(inter.max(h).max(cfg.value_dim()).max(cfg.num_attention_heads * cfg.head_dim))?;
            anyhow::ensure!(mtp.as_ref().is_none_or(|m| m.short_n > 0), "the embedding table is in host memory: the draft head needs its shortlist (AQ_MTP_SHORT)");
            eprintln!("4-bit weights from the checkpoint (one scale per 128), 8-bit head, embedding table in host memory: not exact");
        }
        if quant != Fmt::Bf16 {
            let head = std::env::var("AQ_QUANT_HEAD").as_deref() == Ok("1") && cfg.vocab_size % 8 == 0 && h % 32 == 0;
            match quant {
                Fmt::Fp8 => {
                    let rows = lim.max_tokens.max(lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ) + 512;
                    gpu.enable_fp8(rows, inter.max(2 * h).max(cfg.value_dim()).max(cfg.num_attention_heads * cfg.head_dim))?;
                }
                _ => {
                    gpu.enable_int8(inter.max(h).max(cfg.value_dim()).max(cfg.num_attention_heads * cfg.head_dim))?;
                }
            }
            // The bf16 head stays: the embedding lookup (tied weights) and the MTP shortlist read it.
            if head {
                lm_head = gpu.quantize_weight(head_table, cfg.vocab_size, h, quant, true)?;
            }
            eprintln!("{quant:?} weights (set {set}, head {head}): not exact");
        }
        let weight_bytes = gpu.allocated_bytes;
        let layers_s = t1.elapsed().as_secs_f64();
        eprintln!(
            "load: open {open_s:.2}s, weights {:.2}s, state+KV alloc {state_s:.2}s",
            layers_s - state_s
        );
        eprintln!(
            "memory: {:.2} GiB weights+state+KV ({} KV pages = {} tokens, {} state slots of {:.1} MiB)",
            weight_bytes as f64 / (1u64 << 30) as f64,
            lim.num_blocks(),
            lim.num_blocks() * lim.block_size,
            lim.max_seqs,
            cfg.state_bytes_per_seq() as f64 / (1 << 20) as f64
        );

        let cc_gdn = std::env::var("AQ_GDN_CHUNK_CC").as_deref() == Ok("1");
        let s = alloc_scratch(&mut gpu, &cfg, &lim, cc_gdn, mtp.as_ref().map(|m| m.head_width(&cfg)))?;
        gpu.sync()?;
        let lane_events = vec![gpu.ctx.new_event().map_err(de)?];
        let grp = cfg.num_attention_heads / cfg.num_key_value_heads / cfg.kv_share();
        // Sub-tiles per block: capped by 1024 threads and 100 KiB of shared memory (the most a
        // block may ask for on the GB10; 4 sub-tiles of a 4-head group need 160 KiB).
        let smem_max = gpu.max_smem_optin()? as u32;
        let mut attn_qt: usize = std::env::var("AQ_ATTN_QT").ok().and_then(|v| v.parse().ok()).unwrap_or(2).clamp(1, 4);
        while attn_qt > 1 && (32 * grp * attn_qt > 1024 || attention_tc_smem(grp, attn_qt, 1, kv8) > smem_max.min(100 << 10)) {
            attn_qt -= 1;
        }
        // A second K/V buffer pair (the next tile loads while this one is used) where the
        // device has the shared memory for it; AQ_ATTN_PIPE=0|1 forces it off or on.
        let attn_nbuf: usize = match std::env::var("AQ_ATTN_PIPE").as_deref() {
            Ok("0") => 1,
            _ if !kv8 && attention_tc_smem(grp, attn_qt, 2, false) <= smem_max => 2,
            _ => 1,
        };
        let attn_tc_smem = attention_tc_smem(grp, attn_qt, attn_nbuf, kv8);
        eprintln!("prefill attention: {attn_qt} sub-tiles, {attn_nbuf} K/V buffer(s), {} KiB shared memory of {} KiB", attn_tc_smem >> 10, smem_max >> 10);
        gpu.opt_in_dynamic_smem(
            <aq_kernels::kernels::__attention_tc_CudaKernel as cuda_host::CudaKernel>::PTX_NAME,
            attn_tc_smem,
        )?;
        gpu.opt_in_dynamic_smem(
            <aq_kernels::kernels::__attn_decode_tq_CudaKernel as cuda_host::CudaKernel>::PTX_NAME,
            attn_decode_tq_smem(8, kv8),
        )?;
        // Fixed slices: 32 fine ones (AQ_DEC_TC_SLICE keys, default 128), then 32 coarse ones
        // sized to reach max_ctx; the last is open-ended. Bounds are multiples of the KV page.
        let fine = std::env::var("AQ_DEC_TC_SLICE").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(128).next_multiple_of(32.max(lim.block_size));
        let coarse = lim.max_ctx.saturating_sub(32 * fine).div_ceil(32).next_multiple_of(32.max(lim.block_size)).max(fine);
        let mut dec_bounds_h: Vec<u32> = (0..=DEC_TC_SLICES).map(|j| (if j <= 32 { j * fine } else { 32 * fine + (j - 32) * coarse }) as u32).collect();
        dec_bounds_h[DEC_TC_SLICES] = u32::MAX;
        let dec_bounds = gpu.alloc_upload(&dec_bounds_h)?;
        gpu.set_max_dynamic_smem("gdn_tc_prep", GDN_TC_PREP_SMEM)?;
        gpu.set_max_dynamic_smem("gdn_tc_scan", GDN_TC_SCAN_SMEM)?;
        gpu.set_max_dynamic_smem("gdn_tc_out", GDN_TC_OUT_SMEM)?;
        if cfg.moe() {
            gpu.set_max_dynamic_smem("moe_gemm", MOE_GEMM_SMEM)?;
        }
        let upload_s = t1.elapsed().as_secs_f64();
        eprintln!(
            "load: scratch + kernel setup {:.2}s, {:.2} GiB of scratch ({:.2} GiB allocated in all)",
            upload_s - layers_s,
            (gpu.allocated_bytes - weight_bytes) as f64 / (1u64 << 30) as f64,
            gpu.allocated_bytes as f64 / (1u64 << 30) as f64
        );
        if kv8 {
            anyhow::ensure!(
                std::env::var("AQ_ATTN_NAIVE").is_err() && std::env::var("AQ_ATTN_TILE").is_err(),
                "the reference attention kernels do not read the 8-bit KV cache"
            );
            eprintln!("8-bit KV cache: not exact");
        }
        Ok(Self {
            cfg,
            gpu,
            embed,
            embed_host: small_tables.then_some((ck, embed_name)),
            embed_rows: Vec::new(),
            final_norm,
            lm_head,
            head_table,
            layers,
            mtp,
            mtp_on: false,
            last_mtp: Vec::new(),
            last_mtp_rows: Vec::new(),
            chain_mode: false,
            s,
            limits: lim,
            load: LoadStats { open_s, upload_s, weight_bytes },
            prof: Prof::from_env(),
            graphs: Default::default(),
            use_graphs: std::env::var("AQ_NO_GRAPHS").is_err() && std::env::var("AQ_MOE_ROUTES").is_err() && !pool.paged(),
            experts: pool,
            ple,
            final_hc,
            routes: std::env::var("AQ_MOE_ROUTES").ok().map(std::fs::File::create).transpose()?,
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
            cur_tap: false,
            last_logits: 0,
            ref_gdn: std::env::var("AQ_GDN_RECURRENT").as_deref() == Ok("1"),
            ref_attn: std::env::var("AQ_ATTN_NAIVE").as_deref() == Ok("1"),
            tile_attn: std::env::var("AQ_ATTN_TILE").as_deref() == Ok("1"),
            attn_tc_smem,
            attn_nbuf: attn_nbuf as u32,
            kv8,
            dec_bounds,
            dec_bounds_h,
            attn_qt,
            cc_gdn,
            s_cap: lim.max_rows(),
            alt: None,
            lane_events,
        })
    }

    /// Create the decode lane for decode/prefill co-execution: a high-priority stream with its
    /// own cuBLASLt workspace, plus scratch and metadata buffers sized for one decode token per
    /// sequence (plus the usual decode-row headroom). Lane 0 keeps the full-size scratch for
    /// prefill.
    pub fn enable_decode_lane(&mut self) -> Result<()> {
        if self.alt.is_some() {
            return Ok(());
        }
        self.set_lane_raw(0);
        // AQ_COEXEC_PRIO: stream priority of the decode lane (default: the device's highest;
        // 0 = same priority as prefill).
        let prio = match std::env::var("AQ_COEXEC_PRIO").ok().and_then(|v| v.parse::<i32>().ok()) {
            Some(p) => p,
            None => self.gpu.highest_priority()?,
        };
        let lane = self.gpu.add_lane(prio)?;
        anyhow::ensure!(lane == 1, "decode lane must be lane 1");
        let mut l2 = self.limits;
        l2.max_tokens = l2.max_seqs;
        let s = alloc_scratch(&mut self.gpu, &self.cfg, &l2, self.cc_gdn, self.mtp.as_ref().map(|m| m.head_width(&self.cfg)))?;
        self.gpu.sync()?;
        self.lane_events.push(self.gpu.ctx.new_event().map_err(de)?);
        self.alt = Some(LaneState {
            s,
            cap: l2.max_rows(),
            graphs: Default::default(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
        });
        Ok(())
    }

    /// Create a background prefill lane: a second, low-priority stream with its own cuBLASLt
    /// workspace and full-size scratch, so a long prompt can be prefilled while steps for the
    /// other sequences keep running on lane 0.
    pub fn enable_prefill_lane(&mut self) -> Result<()> {
        if self.alt.is_some() {
            return Ok(());
        }
        self.set_lane_raw(0);
        let prio = self.gpu.lowest_priority()?;
        let lane = self.gpu.add_lane(prio)?;
        anyhow::ensure!(lane == 1, "prefill lane must be lane 1");
        let lim = self.limits;
        let s = alloc_scratch(&mut self.gpu, &self.cfg, &lim, self.cc_gdn, self.mtp.as_ref().map(|m| m.head_width(&self.cfg)))?;
        self.gpu.sync()?;
        self.lane_events.push(self.gpu.ctx.new_event().map_err(de)?);
        self.alt = Some(LaneState {
            s,
            cap: lim.max_rows(),
            graphs: Default::default(),
            cur_chunks: 0,
            cur_tiles: 0,
            cur_splits: 1,
            cur_nd_rows: 0,
        });
        Ok(())
    }

    /// Which lane is active (0 without co-execution).
    pub fn lane(&self) -> usize {
        self.gpu.lane
    }

    fn set_lane_raw(&mut self, lane: usize) {
        self.gpu.set_lane(lane);
    }

    /// Make `lane` active: swap in its scratch, work lists and graphs, and point the GPU at its
    /// stream and workspace. Lane 1 exists only after `enable_decode_lane`.
    pub fn set_lane(&mut self, lane: usize) {
        if lane == self.gpu.lane {
            return;
        }
        let alt = self.alt.as_mut().expect("set_lane(1) without enable_decode_lane");
        std::mem::swap(&mut self.s, &mut alt.s);
        std::mem::swap(&mut self.s_cap, &mut alt.cap);
        std::mem::swap(&mut self.graphs, &mut alt.graphs);
        std::mem::swap(&mut self.cur_chunks, &mut alt.cur_chunks);
        std::mem::swap(&mut self.cur_tiles, &mut alt.cur_tiles);
        std::mem::swap(&mut self.cur_splits, &mut alt.cur_splits);
        std::mem::swap(&mut self.cur_nd_rows, &mut alt.cur_nd_rows);
        self.gpu.set_lane(lane);
    }

    /// Enqueue a whole step on the active lane without waiting for it: upload, forward, argmax.
    /// Returns the number of sampled rows; collect them with `collect` once `lane_done`.
    pub fn launch(&mut self, b: &Batch) -> Result<usize> {
        let nseq = b.seq_start.len();
        let ns = b.sample_rows.len();
        anyhow::ensure!(!b.tokens.is_empty() && b.tokens.len() <= self.s_cap, "{} tokens exceeds lane capacity", b.tokens.len());
        anyhow::ensure!(nseq <= self.limits.max_seqs, "{nseq} sequences exceeds max_seqs");
        self.upload(b)?;
        self.forward(b.tokens.len(), nseq, ns, b.n_decode, None)?;
        let lane = self.gpu.lane;
        self.lane_events[lane].record(&self.gpu.stream).map_err(de)?;
        Ok(ns)
    }

    /// Whether everything enqueued on `lane` up to its last `launch` has finished.
    pub fn lane_done(&self, lane: usize) -> Result<bool> {
        self.lane_events[lane].query().map_err(de)
    }

    /// Sampled tokens of the active lane's last `launch` (waits for it).
    pub fn collect(&mut self, ns: usize) -> Result<Vec<u32>> {
        if ns == 0 {
            self.gpu.sync()?;
            return Ok(Vec::new());
        }
        self.gpu.download(self.s.tok, ns)
    }

    /// Create and run once, on scratch buffers, every cuBLASLt plan the engine can hit, so no
    /// step pays for a plan's first use. Returns the number of row counts tried.
    pub fn prewarm_gemms(&mut self) -> Result<usize> {
        let c = self.cfg.clone();
        let lim = self.limits;
        let mut ms: Vec<usize> = (1..=16).map(|i| i * 16).collect();
        ms.extend((3..=16).map(|i| i * 128));
        ms.extend((5..=16).map(|i| i * 512));
        ms.retain(|&m| m <= lim.max_tokens);
        let (s, g) = (&self.s, &mut self.gpu);
        let mut shapes: Vec<(usize, usize, u64, u64)> = Vec::new(); // (n, k, x, y)
        let mut seen = std::collections::HashSet::new();
        for l in &self.layers {
            let w = match l.kind {
                LayerKind::Linear => vec![(c.gdn_proj_ld(), c.hidden_size, s.h, s.proj, l.gdn_proj), (c.hidden_size, c.value_dim(), s.mix, s.delta, l.out_proj)],
                LayerKind::Full => vec![
                    (c.attn_proj_dim(), c.hidden_size, s.h, s.proj, l.qkv),
                    (c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, s.delta, l.o_proj),
                ],
            };
            let mlp = [(2 * c.intermediate_size, c.hidden_size, s.h, s.proj, l.gate_up), (c.hidden_size, c.intermediate_size, s.mix, s.delta, l.down)];
            let router = (l.router != 0).then_some((moe_router_ld(&c), c.hidden_size, s.h, s.moe_logits, Weight::bf16(l.router)));
            for (n, k, x, y, wt) in w.into_iter().chain(mlp).chain(router) {
                if seen.insert((n, k)) {
                    shapes.push((n, k, x, y));
                    for &m in &ms {
                        g.gemm(m, lim.max_tokens, n, k, x, wt, y, CUDA_R_16BF)?;
                    }
                }
            }
        }
        for m in 1..=lim.max_seqs {
            g.gemm(m, lim.max_seqs, c.vocab_size, c.hidden_size, s.hfinal, self.lm_head, s.logits, CUDA_R_16BF)?;
        }
        g.sync()?;
        Ok(ms.len())
    }

    /// Clear a slot's recurrent state before a new sequence uses it (KV pages are written
    /// before they are read and need no reset).
    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        if let Some(p) = self.ple.as_mut() {
            p.before[slot] = p.table.start();
            let sz = 9 * self.cfg.hc_count * self.cfg.hidden_size * 2;
            self.gpu.zero(p.state + (slot * sz) as u64, sz)?;
        }
        let dsz = self.cfg.linear_num_value_heads * 128 * 128 * 4;
        let csz = self.cfg.conv_dim() * 3 * 2;
        for l in &self.layers {
            if l.kind == LayerKind::Linear {
                self.gpu.zero(l.delta_state + (slot * dsz) as u64, dsz)?;
                self.gpu.zero(l.conv_state + (slot * csz) as u64, csz)?;
            }
        }
        Ok(())
    }

    /// Whether prefill sequences can carry a state tap (`Batch::seq_tap`).
    pub fn supports_tap(&self) -> bool {
        !self.cc_gdn && !self.ref_gdn
    }

    /// Copy one KV page (every full-attention layer's K and V rows) into another, on the stream.
    pub fn copy_page(&mut self, dst: usize, src: usize) -> Result<()> {
        let psz = self.limits.block_size * self.cfg.num_key_value_heads * if self.kv8 { KV8_ROW } else { self.cfg.head_dim * 2 };
        let st = self.gpu.stream.cu_stream();
        for l in &self.layers {
            if l.kind == LayerKind::Full {
                for base in [l.k_cache, l.v_cache] {
                    unsafe { cuda_core::memory::memcpy_dtod_async(base + (dst * psz) as u64, base + (src * psz) as u64, psz, st) }.map_err(de)?;
                }
            }
        }
        Ok(())
    }

    /// Copy one slot's recurrent state (DeltaNet + conv) into another, on the stream.
    pub fn copy_slot(&mut self, dst: usize, src: usize) -> Result<()> {
        if let Some(p) = self.ple.as_mut() {
            p.before[dst] = p.before[src];
            let sz = 9 * self.cfg.hc_count * self.cfg.hidden_size * 2;
            unsafe { cuda_core::memory::memcpy_dtod_async(p.state + (dst * sz) as u64, p.state + (src * sz) as u64, sz, self.gpu.stream.cu_stream()) }.map_err(de)?;
        }
        let dsz = self.cfg.linear_num_value_heads * 128 * 128 * 4;
        let csz = self.cfg.conv_dim() * 3 * 2;
        let st = self.gpu.stream.cu_stream();
        for l in &self.layers {
            if l.kind == LayerKind::Linear {
                unsafe {
                    cuda_core::memory::memcpy_dtod_async(l.delta_state + (dst * dsz) as u64, l.delta_state + (src * dsz) as u64, dsz, st)
                        .map_err(de)?;
                    cuda_core::memory::memcpy_dtod_async(l.conv_state + (dst * csz) as u64, l.conv_state + (src * csz) as u64, csz, st)
                        .map_err(de)?;
                }
            }
        }
        Ok(())
    }

    /// Convenience for a single sequence in slot 0 with an identity block table.
    pub fn single(&self, ids: &[u32], pos: usize) -> Batch {
        let n = ids.len() as u32;
        Batch {
            tokens: ids.to_vec(),
            tok_pos: (0..n).map(|i| pos as u32 + i).collect(),
            tok_seq: vec![0; ids.len()],
            seq_start: vec![0],
            seq_len: vec![n],
            seq_slot: vec![0],
            block_tables: (0..self.limits.max_blocks_per_seq() as u32).collect(),
            sample_rows: vec![n - 1],
            n_decode: (n == 1) as usize,
            seq_slot_out: Vec::new(),
            seq_tap: Vec::new(),
            next_tok_last: Vec::new(),
        }
    }

    /// Run one batch and return the greedy token for each entry of `b.sample_rows`.
    /// With `dump`, also return the residual stream after every layer (as f32).
    pub fn step(&mut self, b: &Batch, dump: Option<&mut Vec<Vec<f32>>>) -> Result<Vec<u32>> {
        let nseq = b.seq_start.len();
        let ns = b.sample_rows.len();
        let lim = self.limits;
        anyhow::ensure!(!b.tokens.is_empty() && b.tokens.len() <= self.s_cap, "{} tokens exceeds the lane's capacity", b.tokens.len());
        anyhow::ensure!(nseq <= lim.max_seqs, "{nseq} sequences exceeds max_seqs");
        anyhow::ensure!(ns <= lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ, "too many sampled rows");
        self.last_mtp.clear();
        self.last_mtp_rows.clear();
        let pure_decode = nseq > 0 && b.n_decode == nseq && ns == nseq && b.tokens.len() == nseq;
        if self.use_graphs && pure_decode && dump.is_none() {
            return self.step_graph(b);
        }
        let decode_only = nseq > 0 && b.n_decode == nseq && b.tokens.len() > nseq;
        if self.use_graphs && decode_only && dump.is_none() {
            return self.step_graph_spec(b);
        }
        self.upload(b)?;
        self.forward(b.tokens.len(), nseq, ns, b.n_decode, dump)?;
        if let Some(p) = self.prof.as_mut() {
            p.finish()?;
        }
        if ns == 0 {
            return Ok(Vec::new());
        }
        if self.mtp_active() {
            self.last_mtp = self.gpu.download(self.s.mtp_tok, ns)?;
            self.last_mtp_rows = (0..ns as u32).collect();
        }
        self.gpu.download(self.s.tok, ns)
    }

    /// Pure-decode step through a captured CUDA graph. The batch is padded to a bucket size
    /// with rows that use the reserved dummy state slot (`max_seqs`) and dummy KV page
    /// (`num_blocks`), so padding never touches a live sequence.
    fn step_graph(&mut self, b: &Batch) -> Result<Vec<u32>> {
        let lim = self.limits;
        let n = b.seq_start.len();
        let bucket = DECODE_BUCKETS.iter().copied().find(|&x| x >= n).unwrap_or(n).min(lim.max_seqs).max(n);
        let mut p = b.clone();
        let mb = lim.max_blocks_per_seq();
        for k in n..bucket {
            p.tokens.push(0);
            p.tok_pos.push(0);
            p.tok_seq.push(k as u32);
            p.seq_start.push(k as u32);
            p.seq_len.push(1);
            p.seq_slot.push(lim.max_seqs as u32);
            if !p.seq_slot_out.is_empty() {
                p.seq_slot_out.push(lim.max_seqs as u32);
            }
            p.block_tables.extend(std::iter::repeat_n(lim.num_blocks() as u32, mb));
            p.sample_rows.push(k as u32);
        }
        p.n_decode = bucket;
        self.upload(&p)?;
        let key = (bucket + self.mtp_active() as usize * 50_000_000, self.cur_splits + self.s.dec_tiles * 1000);
        if let Some((exec, names)) = self.graphs.get(&key) {
            if let Some(p) = self.prof.as_mut() {
                // The graph replays the event records captured with it.
                p.active = p.filter != 2;
                p.names = names.clone();
            }
            self.gpu.launch_graph(*exec)?;
            if let Some(p) = self.prof.as_mut() {
                p.finish()?;
            }
        } else {
            // Eager run first: produces this step's tokens and tunes every GEMM plan the
            // graph will use (tuning synchronizes, which capture forbids).
            self.forward(bucket, bucket, bucket, bucket, None)?;
            if let Some(p) = self.prof.as_mut() {
                p.finish()?;
            }
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.forward(bucket, bucket, bucket, bucket, None);
            let exec = self.gpu.end_capture()?;
            r?;
            let names = self.prof.as_mut().map(|p| std::mem::take(&mut p.names)).unwrap_or_default();
            self.graphs.insert(key, (exec, names));
        }
        if self.mtp_active() {
            self.last_mtp = self.gpu.download(self.s.mtp_tok, bucket)?;
            self.last_mtp.truncate(n);
            self.last_mtp_rows = (0..n as u32).collect();
        }
        let mut toks = self.gpu.download(self.s.tok, bucket)?;
        toks.truncate(n);
        Ok(toks)
    }

    /// Decode-only step where sequences may feed several tokens (speculative verification),
    /// through a captured graph: sequences are padded to a bucket and rows to a multiple of
    /// 16, the padding going to the dummy slot and dummy KV page. All rows are sampled and the
    /// caller's are picked out afterwards. Padding the slice count is exact because attention
    /// slices are fixed (empty slices add nothing).
    fn step_graph_spec(&mut self, b: &Batch) -> Result<Vec<u32>> {
        let lim = self.limits;
        let n = b.seq_start.len();
        let w = *b.seq_len.iter().max().unwrap() as usize;
        let bucket = DECODE_BUCKETS.iter().copied().find(|&x| x >= n).unwrap_or(n).min(lim.max_seqs).max(n);
        // Rows padded to a multiple of 16 (not bucket * W): sequences keep their own lengths,
        // so the padding is only the round-up, bounded by `max_rows`.
        let rows = b
            .tokens
            .len()
            .next_multiple_of(16)
            .max(bucket)
            .min(self.s_cap)
            .min(lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ)
            .max(b.tokens.len());
        let _ = w;
        let used = b.tokens.len();
        let mb = lim.max_blocks_per_seq();
        let dummy_slot = lim.max_seqs as u32;
        let dummy_page = lim.num_blocks() as u32;
        let mut p = b.clone();
        if p.seq_slot_out.is_empty() {
            p.seq_slot_out = p.seq_slot.clone();
        }
        // Empty dummy sequences up to the bucket, then the padding sequence (index `bucket`).
        for _ in n..=bucket {
            p.seq_start.push(used as u32);
            p.seq_len.push(0);
            p.seq_slot.push(dummy_slot);
            p.seq_slot_out.push(dummy_slot);
            p.block_tables.extend(std::iter::repeat_n(dummy_page, mb));
        }
        for r in used..rows {
            p.tokens.push(0);
            p.tok_pos.push(((r - used) % lim.block_size) as u32);
            p.tok_seq.push(bucket as u32);
        }
        p.sample_rows = (0..rows as u32).collect();
        p.n_decode = bucket + 1;
        self.upload(&p)?;
        // Every row runs through attention (padding rows read only the dummy page), and the
        // slice count is rounded up so a few graphs cover all context lengths.
        self.cur_nd_rows = rows;
        anyhow::ensure!(rows * self.cur_splits <= attn_part_rows(lim.max_seqs), "decode attention partials overflow");
        // With tensor-core decode attention the tile list is graph data (uploaded per step), but
        // the tile count is the grid: it joins the key.
        let key = (
            bucket * 100_000 + rows * 10 + 1 + self.mtp_active() as usize * 50_000_000,
            self.cur_splits + self.s.dec_tiles * 1000 + self.s.dec_warps * 100_000_000,
        );
        if let Some((exec, names)) = self.graphs.get(&key) {
            if let Some(p) = self.prof.as_mut() {
                // The graph replays the event records captured with it.
                p.active = p.filter != 2;
                p.names = names.clone();
            }
            self.gpu.launch_graph(*exec)?;
            if let Some(p) = self.prof.as_mut() {
                p.finish()?;
            }
        } else {
            self.forward(rows, bucket + 1, rows, bucket + 1, None)?;
            if let Some(pr) = self.prof.as_mut() {
                pr.finish()?;
            }
            self.gpu.sync()?;
            self.gpu.begin_capture()?;
            let r = self.forward(rows, bucket + 1, rows, bucket + 1, None);
            let exec = self.gpu.end_capture()?;
            r?;
            let names = self.prof.as_mut().map(|pr| std::mem::take(&mut pr.names)).unwrap_or_default();
            self.graphs.insert(key, (exec, names));
        }
        if self.mtp_active() {
            let all: Vec<u32> = self.gpu.download(self.s.mtp_tok, rows)?;
            self.last_mtp = b.sample_rows.iter().map(|&r| all[r as usize]).collect();
            self.last_mtp_rows = b.sample_rows.clone();
        }
        let all = self.gpu.download(self.s.tok, rows)?;
        Ok(b.sample_rows.iter().map(|&r| all[r as usize]).collect())
    }

    /// Whether the checkpoint's MTP head is loaded.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    fn mtp_active(&self) -> bool {
        self.mtp_on && self.mtp.is_some()
    }

    /// Chained MTP drafts after a step that ran the head. Each entry of `rows` continues from
    /// one of that step's sampled rows: `(index into last_mtp, position of the first chained
    /// row, how many drafts to chain, block table)`. Pass j feeds draft j and the previous
    /// pass's output at the next position and yields draft j + 1; the KV rows it writes to
    /// the head's cache sit at positions the next real pass overwrites. Returns the chained
    /// drafts per entry. Must be called before the next step on this lane.
    pub fn mtp_chain(&mut self, rows: &[(usize, u32, usize, &[u32])]) -> Result<Vec<Vec<u32>>> {
        let mut out = vec![Vec::new(); rows.len()];
        if !self.mtp_active() || rows.is_empty() {
            return Ok(out);
        }
        let mb = self.limits.max_blocks_per_seq();
        // `alive[i]`: (entry, its row in the previous pass's GPU buffers).
        let mut alive: Vec<(usize, u32)> = rows.iter().enumerate().map(|(e, r)| (e, self.last_mtp_rows[r.0])).collect();
        let kmax = rows.iter().map(|r| r.2).max().unwrap_or(0);
        self.chain_mode = true;
        let mut run = || -> Result<()> {
            for j in 0..kmax {
                alive.retain(|&(e, _)| rows[e].2 > j);
                if alive.is_empty() {
                    break;
                }
                let n = alive.len();
                let mut b = Batch { n_decode: n, ..Batch::default() };
                for (i, &(e, _)) in alive.iter().enumerate() {
                    b.tokens.push(0);
                    b.tok_pos.push(rows[e].1 + j as u32);
                    b.tok_seq.push(i as u32);
                    b.seq_start.push(i as u32);
                    b.seq_len.push(1);
                    b.seq_slot.push(self.limits.max_seqs as u32);
                    let mut row = rows[e].3.to_vec();
                    row.resize(mb, 0);
                    b.block_tables.extend_from_slice(&row);
                    b.sample_rows.push(i as u32);
                }
                self.upload(&b)?;
                let sel: Vec<u32> = alive.iter().map(|&(_, r)| r).collect();
                self.gpu.upload(self.s.next, &sel)?;
                self.forward(n, n, n, n, None)?;
                if let Some(p) = self.prof.as_mut() {
                    p.finish()?;
                }
                let toks: Vec<u32> = self.gpu.download(self.s.mtp_tok, n)?;
                for (i, a) in alive.iter_mut().enumerate() {
                    out[a.0].push(toks[i]);
                    a.1 = i as u32;
                }
            }
            Ok(())
        };
        let r = run();
        self.chain_mode = false;
        r.map(|_| out)
    }

    /// Tell the MTP head's shortlist about tokens that are in play (prompt and generated
    /// tokens): each one not yet shortlisted takes over the oldest recyclable slot.
    pub fn mtp_note(&mut self, toks: impl IntoIterator<Item = u32>) {
        let Some(m) = self.mtp.as_mut().filter(|m| m.short_n > 0) else { return };
        for t in toks {
            if m.short_pos.get(t as usize).copied() != Some(u32::MAX) {
                continue;
            }
            let slot = m.short_ring;
            m.short_ring = if slot + 1 == m.short_n { m.short_n / 2 } else { slot + 1 };
            m.short_pos[m.short_ids[slot] as usize] = u32::MAX;
            m.short_ids[slot] = t;
            m.short_pos[t as usize] = slot as u32;
            m.short_dirty.push(slot as u32);
        }
    }

    fn upload(&mut self, b: &Batch) -> Result<()> {
        // Prefill work lists: GDN chunks and attention query tiles for sequences n_decode..
        let (mut cs, mut cl, mut c0, mut cn, mut ts, mut tl) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        let chunk = if self.cc_gdn { GDN_CHUNK } else { GDN_TC_CHUNK };
        for k in b.n_decode..b.seq_start.len() {
            let (start, len) = (b.seq_start[k] as usize, b.seq_len[k] as usize);
            c0.push(cs.len() as u32);
            let mut o = 0;
            while o < len {
                cs.push((start + o) as u32);
                cl.push((len - o).min(chunk) as u32);
                o += chunk;
            }
            cn.push(cs.len() as u32 - *c0.last().unwrap());
            let qtile = if self.tile_attn || self.ref_attn { ATTN_QTILE } else { ATTN_QTILE * self.attn_qt };
            let mut o = 0;
            while o < len {
                ts.push((start + o) as u32);
                tl.push((len - o).min(qtile) as u32);
                o += qtile;
            }
        }
        self.cur_chunks = cs.len();
        self.cur_tiles = ts.len();
        // Decode attention tiles: rows of one sequence at consecutive positions (padding rows
        // too in a decode-only batch). The slice count is rounded up to a power of two so a
        // few graphs cover every context length.
        let nd_seqs = b.n_decode;
        let nd = if nd_seqs == 0 { 0 } else { (b.seq_start[nd_seqs - 1] + b.seq_len[nd_seqs - 1]) as usize };
        self.cur_nd_rows = nd;
        self.cur_splits = 1;
        self.s.dec_tiles = 0;
        let (mut ds, mut dl): (Vec<u32>, Vec<u32>) = (vec![], vec![]);
        if nd > 0 && !self.ref_attn {
            let rows = if b.n_decode == b.seq_start.len() { b.tokens.len() } else { nd };
            let max_keys = b.tok_pos[..rows].iter().copied().max().unwrap_or(0) + 1;
            let need = self.dec_bounds_h.iter().position(|&x| x >= max_keys).unwrap_or(DEC_TC_SLICES).max(1);
            let splits = need.next_power_of_two().min(DEC_TC_SLICES);
            // Warps per block: enough query slots (8 per warp) for the longest real decode
            // sequence, at most 16 rows; longer runs are cut into more tiles.
            let rpw = 8 / (self.cfg.num_attention_heads / self.cfg.num_key_value_heads / self.cfg.kv_share());
            let longest = b.seq_len[..nd_seqs].iter().copied().max().unwrap_or(1).max(1) as usize;
            // At most 8 warps a block (8 query heads per KV head: one row per warp).
            let warps = longest.div_ceil(rpw).min(16 / rpw).min(8);
            let cap = (warps * rpw) as u32;
            anyhow::ensure!(
                rows * splits <= attn_part_rows(self.limits.max_seqs) && rows <= self.limits.max_seqs * MAX_DECODE_ROWS_PER_SEQ,
                "decode attention partials overflow ({rows} rows x {splits} slices)"
            );
            for r in 0..rows {
                let new = r == 0 || *dl.last().unwrap() == cap || b.tok_seq[r] != b.tok_seq[r - 1] || b.tok_pos[r] != b.tok_pos[r - 1] + 1;
                if new {
                    ds.push(r as u32);
                    dl.push(1);
                } else {
                    *dl.last_mut().unwrap() += 1;
                }
            }
            // Empty tiles round the count up to a multiple of 8: the count is the launch grid,
            // so it is part of a captured graph's shape.
            while ds.len() % 8 != 0 {
                ds.push(0);
                dl.push(0);
            }
            self.s.dec_tiles = ds.len();
            self.s.dec_warps = warps;
            self.cur_splits = splits;
        }
        // State taps (dummy slot and zero chunks where a sequence has none).
        let ns_all = b.seq_start.len();
        let dummy = self.limits.max_seqs as u32;
        let (mut tn, mut tlen, mut tslot) = (vec![0u32; ns_all], vec![0u32; ns_all], vec![dummy; ns_all]);
        for (k, &(n, slot)) in b.seq_tap.iter().enumerate() {
            if n > 0 {
                anyhow::ensure!(k >= b.n_decode && (n as usize * GDN_TC_CHUNK) <= b.seq_len[k] as usize && self.supports_tap(), "invalid state tap");
                (tn[k], tlen[k], tslot[k]) = (n, n * GDN_TC_CHUNK as u32, slot);
            }
        }
        self.cur_tap = tn.iter().any(|&n| n > 0);
        let (s, g) = (&self.s, &self.gpu);
        if !cs.is_empty() {
            g.upload(s.seq_tap_n, &tn)?;
            g.upload(s.seq_tap_len, &tlen)?;
            g.upload(s.seq_tap_slot, &tslot)?;
            g.upload(s.chunk_start, &cs)?;
            g.upload(s.chunk_len, &cl)?;
            g.upload(s.seq_chunk0, &c0)?;
            g.upload(s.seq_nchunk, &cn)?;
            g.upload(s.tile_start, &ts)?;
            g.upload(s.tile_len, &tl)?;
        }
        if !ds.is_empty() {
            g.upload(s.dtile_start, &ds)?;
            g.upload(s.dtile_len, &dl)?;
        }
        g.upload(s.ids, &b.tokens)?;
        if let Some((ck, name)) = self.embed_host.as_ref().filter(|_| !self.chain_mode) {
            // The embedding table is in host memory: this step's rows go straight to `x`.
            let (table, hb) = (ck.get(name)?.bytes, self.cfg.hidden_size * 2);
            self.embed_rows.clear();
            for &tok in &b.tokens {
                for _ in 0..self.cfg.hc_count {
                    self.embed_rows.extend_from_slice(&table[tok as usize * hb..(tok as usize + 1) * hb]);
                }
            }
            g.upload(s.x, &self.embed_rows)?;
            if let Some(p) = self.ple.as_mut() {
                p.rows.clear();
                for (q, (&start, &len)) in b.seq_start.iter().zip(&b.seq_len).enumerate() {
                    let toks = &b.tokens[start as usize..(start + len) as usize];
                    let before = p.before[b.seq_slot[q] as usize];
                    p.table.gather(ck, before, toks, &mut p.rows)?;
                    // The two tokens before the next one, after `n` of this step's.
                    let after = |n: usize| [1, 0].map(|k| if n > k { toks[n - 1 - k] } else { before[1 - (k - n)] });
                    if let Some(&(chunks, slot)) = b.seq_tap.get(q).filter(|t| t.0 > 0) {
                        p.before[slot as usize] = after(chunks as usize * 64);
                    }
                    p.before[*b.seq_slot_out.get(q).unwrap_or(&b.seq_slot[q]) as usize] = after(toks.len());
                }
                g.upload(s.ple_e, &p.rows)?;
            }
        }
        g.upload(s.tok_pos, &b.tok_pos)?;
        g.upload(s.tok_seq, &b.tok_seq)?;
        g.upload(s.seq_start, &b.seq_start)?;
        g.upload(s.seq_len, &b.seq_len)?;
        g.upload(s.seq_slot, &b.seq_slot)?;
        g.upload(s.seq_slot_out, if b.seq_slot_out.is_empty() { &b.seq_slot } else { &b.seq_slot_out })?;
        g.upload(s.tables, &b.block_tables)?;
        g.upload(s.rows, &b.sample_rows)?;
        let mtp_on = self.mtp_on;
        if let Some(m) = self.mtp.as_mut().filter(|m| mtp_on && !m.short_dirty.is_empty()) {
            let st = g.stream.clone();
            // Refresh the shortlisted output rows that changed since the last step.
            if m.short_dirty.len() > m.short_n {
                m.short_dirty = (0..m.short_n as u32).collect();
            }
            g.upload(m.g_short_ids, &m.short_ids)?;
            g.upload(m.g_short_slots, &m.short_dirty)?;
            unsafe {
                g.k.gather_rows(
                    &st,
                    cfg((m.short_dirty.len() as u32, 1, 1), 256),
                    (if self.head_table != 0 { self.head_table } else { self.lm_head.p }) as *const u16,
                    self.lm_head.scale as *const u16,
                    (self.head_table == 0) as u32,
                    m.g_short_ids as *const u32,
                    m.g_short_slots as *const u32,
                    m.short_w as *mut u16,
                    self.cfg.hidden_size as u32,
                )
            }
            .map_err(de)?;
            if let Some((ck, name)) = self.embed_host.as_ref() {
                // The same slots' embedding rows, from host memory, and the token -> slot map.
                let (table, hb) = (ck.get(name)?.bytes, self.cfg.hidden_size * 2);
                let row = |slot: u32| { let tok = m.short_ids[slot as usize] as usize; &table[tok * hb..(tok + 1) * hb] };
                if m.short_dirty.len() == m.short_n {
                    self.embed_rows.clear();
                    for slot in 0..m.short_n as u32 {
                        self.embed_rows.extend_from_slice(row(slot));
                    }
                    g.upload(m.short_e, &self.embed_rows)?;
                } else {
                    for &slot in &m.short_dirty {
                        g.upload(m.short_e + (slot as usize * hb) as u64, row(slot))?;
                    }
                }
                g.upload(m.g_short_pos, &m.short_pos)?;
            }
            m.short_dirty.clear();
        }
        if self.mtp_on && self.mtp.is_some() {
            // The token after each row: the next row of its sequence, the token it samples
            // (top bit + sample index, resolved on the GPU), or what the caller says follows.
            let mut next = vec![0u32; b.tokens.len()];
            for (k, (&st, &len)) in b.seq_start.iter().zip(&b.seq_len).enumerate() {
                let (st, len) = (st as usize, len as usize);
                for r in st..st + len {
                    next[r] = if r + 1 < st + len { b.tokens[r + 1] } else { b.next_tok_last.get(k).copied().filter(|&t| t != u32::MAX).unwrap_or(0) };
                }
            }
            for (j, &r) in b.sample_rows.iter().enumerate() {
                next[r as usize] = 0x8000_0000 | j as u32;
            }
            g.upload(s.next, &next)?;
        }
        Ok(())
    }

    /// Enqueue the whole forward pass for `t` tokens in `nseq` sequences (`nd` of them pure
    /// decode, listed first), ending with argmax for `ns` sampled rows. No host sync unless
    /// profiling or dumping, so it can be captured into a graph.
    fn forward(&mut self, t: usize, nseq: usize, ns: usize, nd_total: usize, mut dump: Option<&mut Vec<Vec<f32>>>) -> Result<()> {
        let lim = self.limits;
        // Row capacity of the active lane's scratch (GEMM M-bucket cap).
        let cap = self.s_cap;
        let c = self.cfg.clone();
        let hd = c.hidden_size as u32;
        let eps = c.rms_norm_eps as f32;
        let gdn_gate = (c.output_gate_type.as_deref() == Some("sigmoid")) as u32;
        let (s, g) = (&self.s, &mut self.gpu);
        let prof = &mut self.prof;
        macro_rules! mark {
            ($name:literal) => {
                if let Some(p) = prof.as_mut() {
                    p.mark(g, $name)?;
                }
            };
        }
        if let Some(p) = prof.as_mut() {
            let pure_decode = nd_total == nseq;
            p.active = match p.filter {
                1 => pure_decode,
                2 => !pure_decode,
                _ => true,
            };
            p.start(g)?;
        }
        let st = g.stream.clone();
        let mb = lim.max_blocks_per_seq() as u32;
        let bs = lim.block_size as u32;
        let chain = self.chain_mode;
        if !chain && self.embed != 0 {
            unsafe {
                g.k.embed(&st, cfg((t as u32, 1, 1), 256), s.ids as *const u32, self.embed as *const u16, s.x as *mut u16, hd)
            }
            .map_err(de)?;
            mark!("embed");
        }

        let hcn = c.hc_count as u32;
        // Gated residual before a block: inject the previous block's output into the streams
        // (`$add`), then mix them into the block's input `h` and keep the injection weights.
        macro_rules! hc_gate {
            ($hc:expr, $add:expr) => {{
                let (hw, lr, tu) = (c.hidden_size * c.hc_count, c.hc_lowrank, t as u32);
                unsafe { g.k.hc_norm(&st, cfg((tu, hcn, 1), 256), s.x as *mut u16, s.delta as *const u16, s.inj as *const f32, $hc.norm as *const u16, s.xn as *mut u16, hd, hcn, eps, $add as u32) }.map_err(de)?;
                g.gemm(t, cap, lr, hw, s.xn, Weight::bf16($hc.down), s.hc_a, CUDA_R_16BF)?;
                unsafe { g.k.silu_scaled(&st, cfg(((tu * lr as u32).div_ceil(256), 1, 1), 256), s.hc_a as *mut u16, tu * lr as u32, 1.0 / hcn as f32) }.map_err(de)?;
                g.gemm(t, cap, hw, lr, s.hc_a, Weight::bf16($hc.up), s.hc_b, CUDA_R_16BF)?;
                if $hc.inject != 0 {
                    g.gemm(t, cap, HC_INJ_LD, hw, s.xn, Weight::bf16($hc.inject), s.hc_c, CUDA_R_16BF)?;
                }
                let c_ptr = if $hc.inject != 0 { s.hc_c } else { 0 };
                unsafe { g.k.hc_mix(&st, cfg((tu, 1, 1), 256), s.xn as *const u16, s.hc_b as *const u16, c_ptr as *const u16, HC_INJ_LD as u32, s.h as *mut u16, s.inj as *mut f32, hd, hcn) }.map_err(de)?;
            }};
        }
        // Final norm, output head and argmax for the sampled rows, from the residual stream
        // (x + delta) as the last layer left it.
        macro_rules! head {
            ($norm:expr, $hfinal:expr, $logits:expr, $tok:expr, $w:expr, $width:expr) => {{
                if hcn > 1 {
                    // No final norm: the last gated residual mixes the streams into the head's input.
                    hc_gate!(self.final_hc, true);
                    unsafe { g.k.take_rows(&st, cfg((ns as u32, 1, 1), 256), s.h as *const u16, s.rows as *const u32, $hfinal as *mut u16, hd) }.map_err(de)?;
                } else {
                    unsafe {
                        g.k.final_norm_rows(
                            &st,
                            cfg((ns as u32, 1, 1), 256),
                            s.x as *const u16,
                            s.delta as *const u16,
                            $norm as *const u16,
                            $hfinal as *mut u16,
                            s.rows as *const u32,
                            hd,
                            eps,
                        )
                    }
                    .map_err(de)?;
                }
                mark!("final_norm");
                g.gemm(ns, lim.max_seqs * MAX_DECODE_ROWS_PER_SEQ, $width, c.hidden_size, $hfinal, $w, $logits, CUDA_R_16BF)?;
                mark!("gemm_lm_head");
                unsafe {
                    g.k.argmax_part(
                        &st,
                        cfg((ns as u32, ARGMAX_PARTS as u32, 1), 256),
                        $logits as *const u16,
                        $width as u32,
                        s.amax_v as *mut f32,
                        s.amax_i as *mut u32,
                    )
                }
                .map_err(de)?;
                unsafe {
                    g.k.argmax_final(&st, cfg((ns as u32, 1, 1), 32), s.amax_v as *const f32, s.amax_i as *const u32, ARGMAX_PARTS as u32, $tok as *mut u32)
                }
                .map_err(de)?;
                mark!("argmax");
            }};
        }
        // A chained MTP pass runs the head alone, fed by the previous pass's output.
        let backbone = if chain { &self.layers[..0] } else { &self.layers[..] };
        let nl = backbone.len();
        let mtp = if self.mtp_on { self.mtp.as_ref() } else { None };
        for (li, l) in backbone.iter().chain(mtp.map(|m| &m.layer)).enumerate() {
            if li == nl && chain {
                let m = mtp.unwrap();
                unsafe {
                    g.k.mtp_chain_prep(
                        &st,
                        cfg((t as u32, 1, 1), 256),
                        s.mtp_hfinal as *const u16,
                        s.mtp_tok as *const u32,
                        s.next as *const u32,
                        m.pre_h as *const u16,
                        m.pre_e as *const u16,
                        (if self.embed != 0 { self.embed } else { m.short_e }) as *const u16,
                        m.g_short_pos as *const u32,
                        s.proj as *mut u16,
                        hd,
                        eps,
                    )
                }
                .map_err(de)?;
                g.gemm(t, cap, c.hidden_size, 2 * c.hidden_size, s.proj, Weight::bf16(m.fc), s.x, CUDA_R_16BF)?;
            } else if li == nl {
                // The backbone is done: sample, then start the MTP head's residual stream.
                let m = mtp.unwrap();
                if ns > 0 {
                    self.last_logits = s.logits;
                    head!(self.final_norm, s.hfinal, s.logits, s.tok, self.lm_head, c.vocab_size);
                }
                unsafe {
                    g.k.mtp_prep(
                        &st,
                        cfg((t as u32, 1, 1), 256),
                        s.x as *const u16,
                        s.delta as *const u16,
                        self.final_norm as *const u16,
                        m.pre_h as *const u16,
                        m.pre_e as *const u16,
                        (if self.embed != 0 { self.embed } else { m.short_e }) as *const u16,
                        m.g_short_pos as *const u32,
                        s.next as *const u32,
                        s.tok as *const u32,
                        s.proj as *mut u16,
                        hd,
                        eps,
                    )
                }
                .map_err(de)?;
                g.gemm(t, cap, c.hidden_size, 2 * c.hidden_size, s.proj, Weight::bf16(m.fc), s.x, CUDA_R_16BF)?;
                mark!("mtp_in");
            }
            let ple = self.ple.as_ref().filter(|p| p.layer == li && li < nl);
            if let Some(p) = ple {
                // The streams with the previous layer's output in, normed as the query; then
                // the n-gram embedding, gated per stream, is added to them.
                let (hw, tu) = (c.hidden_size * c.hc_count, t as u32);
                let norm = |g: &Gpu, x: u64, w: u64, out: u64, add: bool| unsafe { g.k.hc_norm(&st, cfg((tu, hcn, 1), 256), x as *mut u16, s.delta as *const u16, s.inj as *const f32, w as *const u16, out as *mut u16, hd, hcn, eps, add as u32) }.map_err(de);
                norm(g, s.x, p.norm_query, s.xn, li > 0)?;
                g.gemm(t, cap, hw, c.hidden_size, s.ple_e, Weight::bf16(p.key), s.hc_b, CUDA_R_16BF)?;
                norm(g, s.hc_b, p.norm_key, s.hc_b, false)?;
                g.gemm(t, cap, c.hidden_size, c.hidden_size, s.ple_e, Weight::bf16(p.value), s.h, CUDA_R_16BF)?;
                unsafe {
                    g.k.ple_gate(&st, cfg((tu, hcn, 1), 256), s.hc_b as *const u16, s.xn as *mut u16, s.h as *const u16, hd, hcn).map_err(de)?;
                }
                norm(g, s.xn, p.norm_conv, s.hc_b, false)?;
                unsafe {
                    g.k.ple_conv(&st, cfg(((hw as u32).div_ceil(256), tu, 1), 256), s.hc_b as *const u16, s.xn as *const u16, p.conv_w as *const u16, s.x as *mut u16, p.state as *const u16, s.tok_seq as *const u32, s.seq_start as *const u32, s.seq_slot as *const u32, hw as u32).map_err(de)?;
                    let keep = cfg(((hw as u32).div_ceil(256), nseq as u32, 1), 256);
                    if self.cur_tap {
                        g.k.ple_state(&st, keep, s.hc_b as *const u16, p.state as *mut u16, s.seq_start as *const u32, s.seq_tap_len as *const u32, s.seq_slot as *const u32, s.seq_tap_slot as *const u32, hw as u32).map_err(de)?;
                    }
                    g.k.ple_state(&st, keep, s.hc_b as *const u16, p.state as *mut u16, s.seq_start as *const u32, s.seq_len as *const u32, s.seq_slot as *const u32, s.seq_slot_out as *const u32, hw as u32).map_err(de)?;
                }
                mark!("ple");
            }
            if hcn > 1 {
                hc_gate!(l.hc_attn, li > 0 && li != nl && ple.is_none());
            } else {
                unsafe {
                    g.k.add_rmsnorm(
                        &st,
                        cfg((t as u32, 1, 1), 256),
                        s.x as *mut u16,
                        s.delta as *const u16,
                        l.in_norm as *const u16,
                        s.h as *mut u16,
                        hd,
                        eps,
                        (li > 0 && li != nl) as u32,
                    )
                }
                .map_err(de)?;
            }

            match l.kind {
                LayerKind::Linear => {
                    let n = c.gdn_proj_ld();
                    mark!("norm");
                    g.gemm(t, cap, n, c.hidden_size, s.h, l.gdn_proj, s.proj, CUDA_R_16BF)?;
                    mark!("gemm_gdn_in");
                    let cd = c.conv_dim() as u32;
                    unsafe {
                        g.k.conv1d_tok(
                            &st,
                            cfg((cd.div_ceil(256), t as u32, 1), 256),
                            s.proj as *const u16,
                            n as u32,
                            l.conv_state as *const u16,
                            l.conv_w as *const u16,
                            s.conv as *mut u16,
                            s.tok_seq as *const u32,
                            s.seq_start as *const u32,
                            s.seq_slot as *const u32,
                            cd,
                        )
                    }
                    .map_err(de)?;
                    if self.cur_tap {
                        // Conv state at each tap: the last 3 inputs before it, into the tap slot.
                        unsafe {
                            g.k.conv_state_update(
                                &st,
                                cfg((cd.div_ceil(256), nseq as u32, 1), 256),
                                s.proj as *const u16,
                                n as u32,
                                l.conv_state as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_tap_len as *const u32,
                                s.seq_slot as *const u32,
                                s.seq_tap_slot as *const u32,
                                cd,
                            )
                        }
                        .map_err(de)?;
                    }
                    unsafe {
                        g.k.conv_state_update(
                            &st,
                            cfg((cd.div_ceil(256), nseq as u32, 1), 256),
                            s.proj as *const u16,
                            n as u32,
                            l.conv_state as *mut u16,
                            s.seq_start as *const u32,
                            s.seq_len as *const u32,
                            s.seq_slot as *const u32,
                            s.seq_slot_out as *const u32,
                            cd,
                        )
                    }
                    .map_err(de)?;
                    mark!("conv1d");
                    let z_off = c.conv_dim() as u32;
                    let b_off = z_off + c.value_dim() as u32;
                    let a_off = b_off + c.linear_num_value_heads as u32;
                    let (hv_u, hk_u) = (c.linear_num_value_heads as u32, c.linear_num_key_heads as u32);
                    let nd = nd_total as u32;
                    if nd > 0 {
                        unsafe {
                            g.k.gdn_decode(
                                &st,
                                cfg((hv_u, nd, 1), 128),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                l.gnorm as *const f32,
                                l.delta_state as *mut f32,
                                s.mix as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_len as *const u32,
                                s.seq_slot as *const u32,
                                s.seq_slot_out as *const u32,
                                hk_u,
                                hv_u,
                                eps,
                                gdn_gate,
                            )
                        }
                        .map_err(de)?;
                    }
                    if (nseq as u32) > nd && self.ref_gdn {
                        unsafe {
                            g.k.gdn_recurrent(
                                &st,
                                cfg((hv_u, nseq as u32 - nd, 1), 512),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                l.gnorm as *const f32,
                                l.delta_state as *mut f32,
                                s.mix as *mut u16,
                                s.seq_start as *const u32,
                                s.seq_len as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                hk_u,
                                hv_u,
                                eps,
                                gdn_gate,
                            )
                        }
                        .map_err(de)?;
                    } else if (nseq as u32) > nd && !self.cc_gdn {
                        let nc = self.cur_chunks as u32;
                        unsafe {
                            g.k.gdn_tc_prep(
                                &st,
                                cfg_smem((nc, hv_u, 1), 128, GDN_TC_PREP_SMEM),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.tqn as *mut u16,
                                s.tkn as *mut u16,
                                s.tu as *mut u16,
                                s.tw as *mut u16,
                                s.tg as *mut f32,
                                hk_u,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_prep");
                        unsafe {
                            g.k.gdn_tc_scan(
                                &st,
                                cfg_smem((128 / GDN_TC_SCAN_VB, hv_u, nseq as u32 - nd), 256, GDN_TC_SCAN_SMEM),
                                s.tkn as *const u16,
                                s.tu as *const u16,
                                s.tw as *const u16,
                                s.tg as *const f32,
                                s.seq_chunk0 as *const u32,
                                s.seq_nchunk as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                l.delta_state as *mut f32,
                                s.tht as *mut u16,
                                s.tvnt as *mut u16,
                                hv_u,
                                s.seq_tap_n as *const u32,
                                s.seq_tap_slot as *const u32,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_scan");
                        unsafe {
                            g.k.gdn_tc_out(
                                &st,
                                cfg_smem((nc, hv_u, 1), 128, GDN_TC_OUT_SMEM),
                                s.tqn as *const u16,
                                s.tkn as *const u16,
                                s.tg as *const f32,
                                s.tht as *const u16,
                                s.tvnt as *const u16,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                l.gnorm as *const f32,
                                s.mix as *mut u16,
                                hv_u,
                                eps,
                                gdn_gate,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_out");
                    } else if (nseq as u32) > nd {
                        let nc = self.cur_chunks as u32;
                        unsafe {
                            g.k.gdn_chunk_prep(
                                &st,
                                cfg((nc, hv_u, 1), 128),
                                s.conv as *const u16,
                                cd,
                                s.proj as *const u16,
                                n as u32,
                                b_off,
                                a_off,
                                l.a_log as *const f32,
                                l.dt_bias as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.gu as *mut f32,
                                s.gw as *mut f32,
                                s.gqp as *mut f32,
                                s.gkp as *mut f32,
                                s.gintra as *mut f32,
                                s.gdecay as *mut f32,
                                hk_u,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_prep");
                        unsafe {
                            g.k.gdn_chunk_scan(
                                &st,
                                cfg((hv_u, nseq as u32 - nd, 1), 128),
                                s.gu as *const f32,
                                s.gw as *const f32,
                                s.gqp as *const f32,
                                s.gkp as *const f32,
                                s.gintra as *const f32,
                                s.gdecay as *const f32,
                                s.chunk_start as *const u32,
                                s.chunk_len as *const u32,
                                s.seq_chunk0 as *const u32,
                                s.seq_nchunk as *const u32,
                                s.seq_slot as *const u32,
                                nd,
                                l.delta_state as *mut f32,
                                s.go as *mut u16,
                                hv_u,
                            )
                        }
                        .map_err(de)?;
                        mark!("gdn_scan");
                        unsafe {
                            g.k.gated_norm(
                                &st,
                                cfg((t as u32, hv_u, 1), 128),
                                s.go as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                z_off,
                                l.gnorm as *const f32,
                                s.mix as *mut u16,
                                s.tok_seq as *const u32,
                                nd,
                                hv_u,
                                eps,
                                gdn_gate,
                            )
                        }
                        .map_err(de)?;
                    }
                    mark!("gdn");
                    g.gemm(t, cap, c.hidden_size, c.value_dim(), s.mix, l.out_proj, s.delta, CUDA_R_16BF)?;
                    mark!("gemm_gdn_out");
                }
                LayerKind::Full => {
                    let n = c.attn_proj_dim();
                    let (h, hkv) = (c.num_attention_heads as u32, c.num_key_value_heads as u32);
                    // KV heads as the tensor-core kernels count them (see `TextConfig::kv_share`).
                    let (share, hkv_tc) = (c.kv_share() as u32, c.num_key_value_heads as u32 * c.kv_share() as u32);
                    mark!("norm");
                    g.gemm(t, cap, n, c.hidden_size, s.h, l.qkv, s.proj, CUDA_R_16BF)?;
                    mark!("gemm_qkv");
                    unsafe {
                        g.k.attn_prep(
                            &st,
                            cfg((t as u32, h + hkv, 1), c.head_dim as u32),
                            s.proj as *const u16,
                            n as u32,
                            l.q_norm as *const u16,
                            l.k_norm as *const u16,
                            s.q as *mut u16,
                            l.k_cache as *mut u16,
                            l.v_cache as *mut u16,
                            s.tok_pos as *const u32,
                            s.tok_seq as *const u32,
                            s.tables as *const u32,
                            mb,
                            bs,
                            h,
                            hkv,
                            c.rot_dim() as u32,
                            c.rope_parameters.rope_theta as f32,
                            eps,
                            self.kv8 as u32,
                        )
                    }
                    .map_err(de)?;
                    mark!("attn_prep");
                    let scale = 1.0 / (c.head_dim as f32).sqrt();
                    if self.ref_attn {
                        unsafe {
                            g.k.attention(
                                &st,
                                cfg((t as u32, h, 1), 256),
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                            )
                        }
                        .map_err(de)?;
                    } else if s.dec_tiles > 0 {
                        let ns = self.cur_splits as u32;
                        unsafe {
                            g.k.attn_decode_tq(
                                &st,
                                LaunchConfig {
                                    grid_dim: (s.dec_tiles as u32, hkv_tc, ns),
                                    block_dim: (32 * s.dec_warps as u32, 1, 1),
                                    shared_mem_bytes: attn_decode_tq_smem(s.dec_warps, self.kv8),
                                },
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.dtile_start as *const u32,
                                s.dtile_len as *const u32,
                                self.dec_bounds as *const u32,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv_tc,
                                scale,
                                s.attn_part as *mut f32,
                                s.attn_ml as *mut f32,
                                self.kv8 as u32,
                                share,
                            )
                        }
                        .map_err(de)?;
                        mark!("attn_dec");
                        unsafe {
                            g.k.attn_decode_combine(
                                &st,
                                cfg((self.cur_nd_rows as u32, h, 1), 256),
                                s.attn_part as *const f32,
                                s.attn_ml as *const f32,
                                ns,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                h,
                            )
                        }
                        .map_err(de)?;
                        mark!("attn_dec_combine");
                    }
                    if self.cur_tiles > 0 && !self.ref_attn && !self.tile_attn {
                        let grp = h / hkv_tc;
                        let launch = cuda_core::LaunchConfig {
                            grid_dim: (self.cur_tiles as u32, hkv_tc, 1),
                            block_dim: (32 * grp * self.attn_qt as u32, 1, 1),
                            shared_mem_bytes: self.attn_tc_smem,
                        };
                        unsafe {
                            g.k.attention_tc(
                                &st,
                                launch,
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tile_start as *const u32,
                                s.tile_len as *const u32,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv_tc,
                                scale,
                                self.attn_nbuf,
                                self.kv8 as u32,
                                share,
                            )
                        }
                        .map_err(de)?;
                    }
                    if self.cur_tiles > 0 && !self.ref_attn && self.tile_attn {
                        unsafe {
                            g.k.attention_tile(
                                &st,
                                cfg((self.cur_tiles as u32, h, 1), 256),
                                s.q as *const u16,
                                l.k_cache as *const u16,
                                l.v_cache as *const u16,
                                s.proj as *const u16,
                                n as u32,
                                s.mix as *mut u16,
                                s.tile_start as *const u32,
                                s.tile_len as *const u32,
                                s.tok_pos as *const u32,
                                s.tok_seq as *const u32,
                                s.tables as *const u32,
                                mb,
                                bs,
                                h,
                                hkv,
                                scale,
                            )
                        }
                        .map_err(de)?;
                    }
                    mark!("attn_prefill");
                    g.gemm(t, cap, c.hidden_size, c.num_attention_heads * c.head_dim, s.mix, l.o_proj, s.delta, CUDA_R_16BF)?;
                    mark!("gemm_o");
                }
            }

            if hcn > 1 {
                hc_gate!(l.hc_mlp, true);
            } else {
                unsafe {
                    g.k.add_rmsnorm(
                        &st,
                        cfg((t as u32, 1, 1), 256),
                        s.x as *mut u16,
                        s.delta as *const u16,
                        l.post_norm as *const u16,
                        s.h as *mut u16,
                        hd,
                        eps,
                        1,
                    )
                }
                .map_err(de)?;
            }
            mark!("norm");
            let moe = l.router != 0;
            let inter = c.intermediate_size;
            // The shared expert of a mixture is the dense MLP, written after the experts' rows.
            let mlp_out = if moe { s.moe_out + (c.num_experts_per_tok * t * c.hidden_size * 2) as u64 } else { s.delta };
            g.gemm(t, cap, 2 * inter, c.hidden_size, s.h, l.gate_up, s.proj, CUDA_R_16BF)?;
            mark!("gemm_gate_up");
            let nvec = (t * inter / 8) as u32;
            unsafe {
                g.k.silu_mul(&st, cfg((nvec.div_ceil(256), 1, 1), 256), s.proj as *const u16, s.mix as *mut u16, inter as u32, t as u32)
            }
            .map_err(de)?;
            mark!("silu_mul");
            g.gemm(t, cap, c.hidden_size, inter, s.mix, l.down, mlp_out, CUDA_R_16BF)?;
            mark!("gemm_down");
            if moe {
                let ei = c.moe_intermediate_size as u32;
                let jobs = moe_max_jobs(t, &c) as u32;
                let (tu, ld) = (t as u32, moe_router_ld(&c) as u32);
                let (ne, ek) = (c.num_experts as u32, c.num_experts_per_tok as u32);
                g.gemm(t, cap, ld as usize, c.hidden_size, s.h, Weight::bf16(l.router), s.moe_logits, CUDA_R_16BF)?;
                mark!("gemm_router");
                unsafe {
                    g.k.moe_route(&st, cfg((tu.div_ceil(8), 1, 1), 256), s.moe_logits as *const u16, ld, s.moe_idx as *mut u16, s.moe_w as *mut f32, tu, ne, ek).map_err(de)?;
                    g.k.moe_bucket(&st, cfg((1, 1, 1), ne), s.moe_idx as *const u64, tu, s.moe_list as *mut u32, s.moe_jobs as *mut u32, jobs, ne, ek).map_err(de)?;
                }
                mark!("moe_route");
                let paged = self.experts.paged() && l.ex_gate_up.fmt == Fmt::Int4;
                // The next layers' routers on this layer's input name most of what they will select.
                let guessed = |n: usize, k: u32| backbone.get(li + n).filter(|x| paged && k > 0 && x.ex_gate_up.fmt == Fmt::Int4);
                let (next, far) = (guessed(1, self.experts.ahead), guessed(2, self.experts.far));
                for (i, (n, k)) in [(next, self.experts.ahead), (far, self.experts.far)].into_iter().enumerate() {
                    if let Some(n) = n {
                        g.gemm(t, cap, ld as usize, c.hidden_size, s.h, Weight::bf16(n.router), s.proj, CUDA_R_16BF)?;
                        unsafe { g.k.moe_route(&st, cfg((tu.div_ceil(8), 1, 1), 256), s.proj as *const u16, ld, (s.moe_idx + ((i + 1) * t * 32) as u64) as *mut u16, s.q as *mut f32, tu, ne, k).map_err(de)? };
                    }
                }
                // Each token's selections, then 0xffff up to 16; then the guesses, in one read.
                let regions = if far.is_some() { 3 } else if next.is_some() { 2 } else { 1 };
                let idx: Vec<u16> = if paged || self.routes.is_some() { g.download(s.moe_idx, t * 16 * regions)? } else { Vec::new() };
                let part = |i: usize| idx.get(i * t * 16..(i + 1) * t * 16).unwrap_or(&[]);
                if paged {
                    self.experts.page(g, l.ex_base, c.num_experts, part(0), l.ex_slots, next.map(|n| (n.ex_base, part(1))), far.map(|n| (n.ex_base, part(2))))?;
                    mark!("moe_page");
                }
                let idx = part(0);
                if let Some(f) = self.routes.as_mut() {
                    use std::io::Write;
                    let head = [li as u32, tu, ek].into_iter().flat_map(u32::to_le_bytes);
                    let picks = idx.chunks(16).flat_map(|c| &c[..ek as usize]).flat_map(|e| e.to_le_bytes());
                    f.write_all(&head.chain(picks).collect::<Vec<u8>>())?;
                }
                let (list, jl) = (s.moe_list as *const u32, s.moe_jobs as *const u32);
                unsafe {
                    g.k.moe_gemm(&st, cfg_smem((ei / 64, jobs, 1), 256, MOE_GEMM_SMEM), s.h as *const u16, hd, l.ex_gate_up.p as *const u16, s.moe_act as *mut u16, list, jl, 2 * ei, hd, 1, l.ex_gate_up.scale as *const f32, l.ex_slots as *const u32, l.ex_stride[0], ek)
                        .map_err(de)?;
                }
                mark!("moe_gate_up");
                unsafe {
                    g.k.moe_gemm(&st, cfg_smem((hd / 128, jobs, 1), 256, MOE_GEMM_SMEM), s.moe_act as *const u16, ei, l.ex_down.p as *const u16, s.moe_out as *mut u16, list, jl, hd, ei, 0, l.ex_down.scale as *const f32, l.ex_slots as *const u32, l.ex_stride[1], ek)
                        .map_err(de)?;
                }
                mark!("moe_down");
                unsafe {
                    g.k.moe_combine(&st, cfg((tu, 1, 1), 256), s.moe_out as *const u16, s.moe_w as *const f32, s.moe_logits as *const u16, ld, s.delta as *mut u16, hd, tu, ne, ek).map_err(de)?;
                }
                mark!("moe_combine");
            }

            if let Some(d) = dump.as_deref_mut().filter(|_| li < nl) {
                let (hs, nh) = (c.hidden_size, c.hc_count);
                let x: Vec<u16> = g.download(s.x, t * hs * nh)?;
                let dl: Vec<u16> = g.download(s.delta, t * hs)?;
                let inj: Vec<f32> = if nh > 1 { g.download(s.inj, t * nh)? } else { vec![1.0; t] };
                let f = |b: u16| bf16::from_bits(b).to_f32();
                let r = |v: f32| bf16::from_f32(v).to_f32();
                // The stream each layer leaves: the block's output injected with its weights.
                d.push((0..t * hs * nh).map(|i| r(f(x[i]) + if nh > 1 { r(f(dl[i / (hs * nh) * hs + i % hs]) * inj[i / hs]) } else { f(dl[i]) })).collect());
            }
        }
        if self.experts.paged() {
            // Another lane's forward may evict what this one's queued kernels still read.
            g.sync()?;
            g.side_sync()?;
            self.experts.report();
        }

        if ns == 0 {
            return Ok(());
        }
        match mtp {
            Some(m) if m.short_n > 0 => {
                head!(m.norm, s.mtp_hfinal, s.mtp_logits, s.mtp_tok, Weight::bf16(m.short_w), m.short_n);
                unsafe { g.k.remap_ids(&st, cfg(((ns as u32).div_ceil(256), 1, 1), 256), s.mtp_tok as *mut u32, m.g_short_ids as *const u32, ns as u32) }.map_err(de)?;
            }
            Some(m) => head!(m.norm, s.mtp_hfinal, s.mtp_logits, s.mtp_tok, self.lm_head, c.vocab_size),
            None => {
                self.last_logits = s.logits;
                head!(self.final_norm, s.hfinal, s.logits, s.tok, self.lm_head, c.vocab_size);
            }
        }
        Ok(())
    }

    /// Logits of the first sampled row of the last step.
    pub fn logits(&self) -> Result<Vec<f32>> {
        let src = if self.last_logits != 0 { self.last_logits } else { self.s.logits };
        let v: Vec<u16> = self.gpu.download(src, self.cfg.vocab_size)?;
        Ok(v.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }

    /// Logits of the first `n` sampled rows of the last step, row after row.
    pub fn logits_rows(&self, n: usize) -> Result<Vec<f32>> {
        let src = if self.last_logits != 0 { self.last_logits } else { self.s.logits };
        let v: Vec<u16> = self.gpu.download(src, n * self.cfg.vocab_size)?;
        Ok(v.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }

    /// Post-final-norm hidden state of the first sampled row of the last step, as f32.
    pub fn final_hidden(&self) -> Result<Vec<f32>> {
        let h: Vec<u16> = self.gpu.download(self.s.hfinal, self.cfg.hidden_size)?;
        Ok(h.into_iter().map(|b| bf16::from_bits(b).to_f32()).collect())
    }
}
