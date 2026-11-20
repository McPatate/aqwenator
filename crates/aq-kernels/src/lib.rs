//! Device kernels for the Qwen3.5 (`qwen3_5`) hybrid architecture, written in Rust and
//! compiled to PTX by cuda-oxide.
//!
//! Conventions:
//! - bf16 values travel as raw `u16` bits; math happens in f32.
//! - Every kernel takes raw device pointers: the engine owns one preallocated arena and
//!   hands out views into it, which keeps buffer addresses stable for CUDA graph capture.
//! - `rb(x)` rounds an f32 to bf16 and back. It is used where the reference (transformers,
//!   bf16 eager) materializes a bf16 tensor, so intermediate rounding matches.
//!
//! These are correctness-first versions (milestone 1). The fast paths (paged attention,
//! chunked gated delta rule, fused epilogues) come later and get A/B tested against these.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments, non_snake_case)]

use cuda_device::convert::{bf16_to_f32, cvt_f32x2_bf16x2, f32_to_bf16_rne};
use cuda_device::vector::U32x4;
use cuda_device::{thread, warp};

const FULL: u32 = 0xffff_ffff;

#[inline(always)]
unsafe fn ld(p: *const u16, i: usize) -> f32 {
    bf16_to_f32(unsafe { *p.add(i) })
}

#[inline(always)]
unsafe fn st(p: *mut u16, i: usize, v: f32) {
    unsafe { *p.add(i) = f32_to_bf16_rne(v) }
}

#[inline(always)]
fn rb(v: f32) -> f32 {
    bf16_to_f32(f32_to_bf16_rne(v))
}

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + libm::expf(-x))
}

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + libm::expf(-x))
}

/// The gate of a gated RMSNorm: `silu`, or `sigmoid` in models that say so.
#[inline(always)]
fn gate_act(z: f32, sigmoid_gate: u32) -> f32 {
    if sigmoid_gate != 0 { sigmoid(z) } else { silu(z) }
}

/// torch.nn.functional.softplus with beta=1, threshold=20.
#[inline(always)]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { libm::log1pf(libm::expf(x)) }
}

#[inline(always)]
fn warp_sum(mut v: f32) -> f32 {
    v += warp::shuffle_xor_f32_sync(FULL, v, 16);
    v += warp::shuffle_xor_f32_sync(FULL, v, 8);
    v += warp::shuffle_xor_f32_sync(FULL, v, 4);
    v += warp::shuffle_xor_f32_sync(FULL, v, 2);
    v += warp::shuffle_xor_f32_sync(FULL, v, 1);
    v
}

#[allow(dead_code)]
#[inline(always)]
fn warp_max(mut v: f32) -> f32 {
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 16));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 8));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 4));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 2));
    v = v.max(warp::shuffle_xor_f32_sync(FULL, v, 1));
    v
}

/// Sum over the whole block; every thread gets the result. `smem` holds >= 32 f32.
#[inline(always)]
unsafe fn block_sum(v: f32, smem: *mut f32) -> f32 {
    let v = warp_sum(v);
    let tid = thread::threadIdx_x();
    let nw = thread::blockDim_x().div_ceil(32);
    if tid % 32 == 0 {
        unsafe { *smem.add((tid / 32) as usize) = v };
    }
    thread::sync_threads();
    let mut t = 0.0f32;
    let mut i = 0u32;
    while i < nw {
        t += unsafe { *smem.add(i as usize) };
        i += 1;
    }
    thread::sync_threads();
    t
}

/// Maximum over the whole block; every thread gets the result. `smem` holds >= 32 f32.
#[inline(always)]
unsafe fn block_max(v: f32, smem: *mut f32) -> f32 {
    let v = warp_max(v);
    let tid = thread::threadIdx_x();
    let nw = thread::blockDim_x().div_ceil(32);
    if tid % 32 == 0 {
        unsafe { *smem.add((tid / 32) as usize) = v };
    }
    thread::sync_threads();
    let mut t = 0.0f32;
    let mut i = 0u32;
    while i < nw {
        t = t.max(unsafe { *smem.add(i as usize) });
        i += 1;
    }
    thread::sync_threads();
    t
}

/// FP8 E4M3 code of `v`: round to nearest even, saturating at +-448 (NaN saturates too).
/// E4M3 is sign, 4 exponent bits (bias 7), 3 mantissa bits; subnormals are multiples of 2^-9.
#[inline(always)]
fn e4m3(v: f32) -> u32 {
    let b = v.to_bits();
    let sign = (b >> 24) & 0x80;
    let a = b & 0x7fff_ffff;
    if a >= 0x43e0_0000 {
        return sign | 0x7e; // |v| >= 448
    }
    if a < 0x3c80_0000 {
        // |v| < 2^-6
        let x = f32::from_bits(a) * 512.0;
        let f = x as u32;
        let r = x - f as f32;
        let up = r > 0.5 || (r == 0.5 && (f & 1) == 1);
        return sign | (f + up as u32);
    }
    let m = a & 0x7f_ffff;
    let keep = m >> 20;
    let rem = m & 0xf_ffff;
    let mut code = (((a >> 23) - 120) << 3) | keep;
    if rem > 0x8_0000 || (rem == 0x8_0000 && (keep & 1) == 1) {
        code += 1; // a mantissa carry moves into the exponent field, as it should
    }
    sign | code
}

/// Offset (in bf16 elements) of 16-byte chunk `chunk` of row `row` in a shared tile whose rows
/// are 256 bf16 (32 chunks). Chunks are XOR-swizzled by `row % 8`, so the 8 row addresses an
/// `ldmatrix` reads for one logical chunk land in 8 different bank groups.
#[inline(always)]
fn sw256(row: usize, chunk: usize) -> usize {
    row * 256 + ((chunk ^ (row & 7)) * 8)
}

// ---------------------------------------------------------------------------------------
// Warp MMA helpers (mma.sync.m16n8k16 bf16 -> f32; sm_80+, portable to sm_120/121).
//
// Every product is written as C = X . Y^T with X [M][K] and Y [N][K] both row-major bf16
// in shared memory (K contiguous), so A and B fragments are plain u32 pair loads. Rows are
// padded by 8 bf16 so the 8 rows a fragment touches land in distinct banks.
// ---------------------------------------------------------------------------------------

#[inline(always)]
unsafe fn lds32(p: *const u16, idx: usize) -> u32 {
    unsafe { *(p.add(idx) as *const u32) }
}

/// A fragment of the 16x16 tile at (r0, k0) of row-major X (row stride `ld`).
#[inline(always)]
unsafe fn frag_a(p: *const u16, ld: usize, r0: usize, k0: usize, g: usize, t: usize) -> [u32; 4] {
    unsafe {
        [
            lds32(p, (r0 + g) * ld + k0 + 2 * t),
            lds32(p, (r0 + g + 8) * ld + k0 + 2 * t),
            lds32(p, (r0 + g) * ld + k0 + 2 * t + 8),
            lds32(p, (r0 + g + 8) * ld + k0 + 2 * t + 8),
        ]
    }
}

/// B fragment (k16 x n8) for Y stored [N][K] row-major: rows n0..n0+8, cols k0..k0+16.
#[inline(always)]
unsafe fn frag_b(p: *const u16, ld: usize, n0: usize, k0: usize, g: usize, t: usize) -> [u32; 2] {
    unsafe { [lds32(p, (n0 + g) * ld + k0 + 2 * t), lds32(p, (n0 + g) * ld + k0 + 2 * t + 8)] }
}

/// Two f32 -> packed bf16x2 (round to nearest even), low half first.
#[inline(always)]
fn pk(lo: f32, hi: f32) -> u32 {
    (f32_to_bf16_rne(lo) as u32) | ((f32_to_bf16_rne(hi) as u32) << 16)
}

#[inline(always)]
unsafe fn mma(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4] {
    unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c, a, b) }
}

/// `cp.async` one paged K tile and V tile of `kt` keys starting at key `kb` (256 bf16 each, keys
/// `>= k1` zero-filled) into XOR-swizzled shared tiles, by `nthr` threads. The caller commits.
///
/// The tile must lie inside one KV page (`kb % kt == 0` and `bs % kt == 0`): the page lookup
/// and its integer division then happen once per thread, not once per 16-byte copy.
#[inline(always)]
unsafe fn kv_tile_cp_async(
    ks: *mut u16,
    vs: *mut u16,
    k_cache: *const u16,
    v_cache: *const u16,
    table: *const u32,
    bs: usize,
    hkv: usize,
    kv: usize,
    kb: usize,
    k1: usize,
    kt: usize,
    tid: usize,
    nthr: usize,
) {
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_cg_zfill_16};
    let page = (kb as u32 / bs as u32) as usize;
    let blk = unsafe { *table.add(page) } as usize;
    // bf16 offset of the tile's first key, this KV head.
    let base = ((blk * bs + (kb - page * bs)) * hkv + kv) * 256;
    let valid = if k1 > kb { k1 - kb } else { 0 };
    let mut i = tid;
    while i < kt * 32 {
        let row = i / 32;
        let c = i % 32;
        let dk = unsafe { ks.add(sw256(row, c)) } as *mut u32;
        let dv = unsafe { vs.add(sw256(row, c)) } as *mut u32;
        if row < valid {
            let off = base + row * hkv * 256 + c * 8;
            unsafe {
                cp_async_cg_16(dk, k_cache.add(off) as *const u32);
                cp_async_cg_16(dv, v_cache.add(off) as *const u32);
            }
        } else {
            unsafe {
                cp_async_cg_zfill_16(dk, k_cache as *const u8, 0);
                cp_async_cg_zfill_16(dv, v_cache as *const u8, 0);
            }
        }
        i += nthr;
    }
}

/// 8-bit KV cache: a key's or a value's row is 272 bytes, 256 int8 then its f32 scale
/// (value = code * scale), so a row is still a whole number of 16-byte pieces.
const KV8_ROW: usize = 272;

/// `kv_tile_cp_async` for the 8-bit cache: request `kt` keys' rows as stored, K rows to `sk`
/// and V rows to `sv` (zeros past `k1`). Like it, the keys must lie within one KV page.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn kv_tile8_cp_async(
    sk: *mut u8,
    sv: *mut u8,
    k_cache: *const u8,
    v_cache: *const u8,
    table: *const u32,
    bs: usize,
    hkv: usize,
    kv: usize,
    kb: usize,
    k1: usize,
    kt: usize,
    tid: usize,
    nthr: usize,
) {
    use cuda_device::async_copy::{cp_async_cg_16, cp_async_cg_zfill_16};
    let page = (kb as u32 / bs as u32) as usize;
    let blk = unsafe { *table.add(page) } as usize;
    let base = ((blk * bs + (kb - page * bs)) * hkv + kv) * KV8_ROW;
    let valid = if k1 > kb { k1 - kb } else { 0 };
    let mut i = tid;
    while i < kt * 17 {
        let row = i / 17;
        let c = i % 17;
        let dk = unsafe { sk.add(row * KV8_ROW + c * 16) } as *mut u32;
        let dv = unsafe { sv.add(row * KV8_ROW + c * 16) } as *mut u32;
        if row < valid {
            let off = base + row * hkv * KV8_ROW + c * 16;
            unsafe {
                cp_async_cg_16(dk, k_cache.add(off) as *const u32);
                cp_async_cg_16(dv, v_cache.add(off) as *const u32);
            }
        } else {
            unsafe {
                cp_async_cg_zfill_16(dk, k_cache, 0);
                cp_async_cg_zfill_16(dv, v_cache, 0);
            }
        }
        i += nthr;
    }
}

/// Expand a staged 8-bit tile (`kt` K rows, then `kt` V rows) into the swizzled bf16 K and V
/// tiles the MMAs read.
#[inline(always)]
unsafe fn kv_tile8_unpack(stage: *const u8, ks: *mut u16, vs: *mut u16, kt: usize, tid: usize, nthr: usize) {
    let mut i = tid;
    while i < kt * 32 {
        let row = i / 32;
        let c = i % 32;
        let mut which = 0;
        while which < 2 {
            let src = unsafe { stage.add((which * kt + row) * KV8_ROW) };
            let sc = unsafe { *(src.add(256) as *const f32) };
            let v = unsafe { *(src.add(c * 8) as *const u64) };
            let dst = unsafe { (if which == 0 { ks } else { vs }).add(sw256(row, c)) } as *mut u32;
            let mut e = 0;
            while e < 4 {
                let lo = (v >> (16 * e)) as u8 as i8 as f32 * sc;
                let hi = (v >> (16 * e + 8)) as u8 as i8 as f32 * sc;
                unsafe { *dst.add(e) = pk(lo, hi) };
                e += 1;
            }
            which += 1;
        }
        i += nthr;
    }
}

/// Asynchronous (cp.async.cg, 16 bytes per op) copy of a [rows][cols] bf16 tile from global
/// to shared memory by `nthr` threads. `cols` must be a multiple of 8 and both row strides
/// (in bytes) and base addresses 16-byte aligned. The caller commits and waits.
#[inline(always)]
unsafe fn tile_cp_async(dst: *mut u16, ldd: usize, src: *const u16, lds: usize, rows: usize, cols: usize, tid: usize, nthr: usize) {
    let cpr = cols / 8;
    let total = rows * cpr;
    let mut idx = tid;
    while idx < total {
        let r = idx / cpr;
        let c = idx % cpr;
        unsafe {
            cuda_device::async_copy::cp_async_cg_16(dst.add(r * ldd + c * 8) as *mut u32, src.add(r * lds + c * 8) as *const u32)
        };
        idx += nthr;
    }
}

/// Issue (and commit as one group) the cp.async loads of one chunk for `gdn_tc_scan` (256
/// threads), into the stage buffer at `sb`: W and Kn [64][136], this block's 32 U columns
/// [64][40], G [64].
#[inline(always)]
unsafe fn gdn_scan_issue(sb: *mut u8, kn_in: *const u16, u_in: *const u16, w_in: *const u16, g_in: *const f32, ib: usize, vb: usize, tid: usize) {
    const C: usize = 64;
    const D: usize = 128;
    const VB: usize = 32;
    unsafe {
        tile_cp_async(sb.cast::<u16>(), D + 8, w_in.add(ib * C * D), D, C, D, tid, 256);
        tile_cp_async(sb.add(17408).cast::<u16>(), D + 8, kn_in.add(ib * C * D), D, C, D, tid, 256);
        tile_cp_async(sb.add(34816).cast::<u16>(), VB + 8, u_in.add(ib * C * D + VB * vb), D, C, VB, tid, 256);
        if tid < 16 {
            cuda_device::async_copy::cp_async_cg_16(sb.add(39936 + 16 * tid) as *mut u32, g_in.add(ib * C + 4 * tid) as *const u32);
        }
        cuda_device::async_copy::cp_async_commit_group();
    }
}

#[cuda_device::cuda_module]
pub mod kernels {
    use super::*;
    use cuda_device::{DynamicSharedArray, SharedArray, kernel};

    /// out[t, :] = table[ids[t], :], or `rows[ids[t] & 0x7fff_ffff, :]` for an id with the top
    /// bit set (a row computed elsewhere, e.g. an image token). grid = (T), block = (256).
    #[kernel]
    pub unsafe fn embed(ids: *const u32, table: *const u16, rows: *const u16, out: *mut u16, hidden: u32) {
        let t = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let id = unsafe { *ids.add(t) };
        let src = if id >> 31 != 0 { unsafe { rows.add((id & 0x7fff_ffff) as usize * h) } } else { unsafe { table.add(id as usize * h) } };
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            unsafe { *out.add(t * h + i) = *src.add(i) };
            i += thread::blockDim_x() as usize;
        }
    }

    /// Gated residual over `hc` streams of `hidden` (rows of `x` are `[stream][hidden]`), first
    /// half, for one token and stream: if `add` != 0 the previous block's output is injected,
    /// `x = bf16(x + bf16(delta * inj[token, stream]))` in place; then
    /// `out = rmsnorm(x) * (1 + w[stream])`, the norm taken over the stream.
    /// Requires hidden % 8 == 0 and hidden <= 8192. grid = (T, hc), block = (256).
    #[kernel]
    pub unsafe fn hc_norm(x: *mut u16, delta: *const u16, inj: *const f32, w: *const u16, out: *mut u16, hidden: u32, hc: u32, eps: f32, add: u32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let tid = thread::threadIdx_x() as usize;
        let nv = hidden as usize / 8;
        let (tok, sm) = (thread::blockIdx_x() as usize, thread::blockIdx_y() as usize);
        let row = (tok * hc as usize + sm) * nv;
        let xv = unsafe { x.cast::<U32x4>().add(row) };
        let dv = unsafe { delta.cast::<U32x4>().add(tok * nv) };
        let wv = unsafe { w.cast::<U32x4>().add(sm * nv) };
        let ov = unsafe { out.cast::<U32x4>().add(row) };
        let gain = if add != 0 { unsafe { *inj.add(tok * hc as usize + sm) } } else { 0.0 };
        let mut vals = [0.0f32; 32];
        let mut ss = 0.0f32;
        #[unroll]
        for c in 0..4 {
            let v = tid + c * 256;
            if v < nv {
                let xw = unsafe { *xv.add(v) }.to_array();
                let dw = if add != 0 { unsafe { *dv.add(v) }.to_array() } else { [0u32; 4] };
                let mut packed = [0u32; 4];
                #[unroll]
                for e in 0..4 {
                    let (mut a0, mut a1) = cvt_f32x2_bf16x2(xw[e]);
                    if add != 0 {
                        let (d0, d1) = cvt_f32x2_bf16x2(dw[e]);
                        a0 = rb(a0 + rb(d0 * gain));
                        a1 = rb(a1 + rb(d1 * gain));
                        packed[e] = pk(a0, a1);
                    }
                    vals[c * 8 + 2 * e] = a0;
                    vals[c * 8 + 2 * e + 1] = a1;
                    ss += a0 * a0 + a1 * a1;
                }
                if add != 0 {
                    unsafe { *xv.add(v) = U32x4::new(packed) };
                }
            }
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        #[unroll]
        for c in 0..4 {
            let v = tid + c * 256;
            if v < nv {
                let ww = unsafe { *wv.add(v) }.to_array();
                let mut packed = [0u32; 4];
                #[unroll]
                for e in 0..4 {
                    let (w0, w1) = cvt_f32x2_bf16x2(ww[e]);
                    packed[e] = pk(vals[c * 8 + 2 * e] * inv * (1.0 + w0), vals[c * 8 + 2 * e + 1] * inv * (1.0 + w1));
                }
                unsafe { *ov.add(v) = U32x4::new(packed) };
            }
        }
    }

    /// `a[i] = bf16(silu(a[i] * scale))` for `n` values. grid = (ceil(n / 256)), block = (256).
    #[kernel]
    pub unsafe fn silu_scaled(a: *mut u16, n: u32, scale: f32) {
        let i = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
        if i < n as usize {
            unsafe { st(a, i, silu(ld(a, i) * scale)) };
        }
    }

    /// Gated residual, second half, for one token: the block's input
    /// `h[i] = mean over streams of bf16(bf16(sigmoid(b[s, i])) * xn[s, i])`, and, with `c` not
    /// null, how much of the block's output each stream takes:
    /// `inj[s] = 2 * bf16(sigmoid(c[s] / hc))` (`c` rows have stride `ldc`).
    /// grid = (T), block = (256), hc <= 8.
    #[kernel]
    pub unsafe fn hc_mix(xn: *const u16, b: *const u16, c: *const u16, ldc: u32, h: *mut u16, inj: *mut f32, hidden: u32, hc: u32) {
        let tid = thread::threadIdx_x() as usize;
        let tok = thread::blockIdx_x() as usize;
        let (hd, n) = (hidden as usize, hc as usize);
        if !c.is_null() && tid < n {
            unsafe { *inj.add(tok * n + tid) = 2.0 * rb(sigmoid(ld(c, tok * ldc as usize + tid) / hc as f32)) };
        }
        let mut i = tid;
        while i < hd {
            let mut acc = 0.0f32;
            #[unroll]
            for sm in 0..8 {
                if sm < n {
                    let at = (tok * n + sm) * hd + i;
                    acc += rb(rb(sigmoid(unsafe { ld(b, at) })) * unsafe { ld(xn, at) });
                }
            }
            unsafe { st(h, tok * hd + i, acc / hc as f32) };
            i += 256;
        }
    }

    /// n-gram embedding gate for one token and stream: with
    /// `g = sum_i bf16(key[s, i] * q[s, i]) / sqrt(hidden)` and `g' = sign(g) * sqrt(max(|g|, 1e-6))`,
    /// `q[s, i] = bf16(bf16(sigmoid(g')) * value[i])` in place (the stream's query is consumed).
    /// grid = (T, hc), block = (256).
    #[kernel]
    pub unsafe fn ple_gate(key: *const u16, q: *mut u16, value: *const u16, hidden: u32, hc: u32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let tid = thread::threadIdx_x() as usize;
        let (tok, hd) = (thread::blockIdx_x() as usize, hidden as usize);
        let row = (tok * hc as usize + thread::blockIdx_y() as usize) * hd;
        let mut dot = 0.0f32;
        let mut i = tid;
        while i < hd {
            dot += rb(unsafe { ld(key, row + i) * ld(q, row + i) });
            i += 256;
        }
        let g = rb(rb(unsafe { block_sum(dot, red) }) / libm::sqrtf(hidden as f32));
        let mag = rb(libm::sqrtf(if g.abs() > 1.0e-6 { g.abs() } else { 1.0e-6 }));
        let gate = rb(sigmoid(if g > 0.0 { mag } else if g < 0.0 { -mag } else { 0.0 }));
        let mut i = tid;
        while i < hd {
            unsafe { st(q, row + i, gate * ld(value, tok * hd + i)) };
            i += 256;
        }
    }

    /// n-gram embedding epilogue for one token: over each channel `c` of the streams,
    /// `x[c] = bf16(x[c] + bf16(gv[c] + bf16(silu(conv[c]))))` where `conv` is a depthwise
    /// causal convolution of `gvn` over the token's sequence, 4 taps 3 tokens apart
    /// (`w[c][j]` weighs the token `3 * (3 - j)` back). Tokens before the step come from the
    /// sequence's slot of `state`: the 9 rows of `gvn` before it, oldest first.
    /// grid = (ceil(width / 256), T), block = (256).
    #[kernel]
    pub unsafe fn ple_conv(gvn: *const u16, gv: *const u16, w: *const u16, x: *mut u16, state: *const u16, tok_seq: *const u32, seq_start: *const u32, seq_slot: *const u32, width: u32) {
        let c = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
        let tok = thread::blockIdx_y() as usize;
        let wd = width as usize;
        if c >= wd {
            return;
        }
        let seq = unsafe { *tok_seq.add(tok) } as usize;
        let back = tok - unsafe { *seq_start.add(seq) } as usize;
        let past = unsafe { state.add(*seq_slot.add(seq) as usize * 9 * wd) };
        let mut acc = 0.0f32;
        #[unroll]
        for j in 0..4 {
            let d = 3 * (3 - j);
            let v = if d <= back { unsafe { ld(gvn, (tok - d) * wd + c) } } else { unsafe { ld(past, (9 + back - d) * wd + c) } };
            acc += unsafe { ld(w, c * 4 + j) } * v;
        }
        let y = rb(unsafe { ld(gv, tok * wd + c) } + rb(silu(rb(acc))));
        unsafe { st(x, tok * wd + c, ld(x, tok * wd + c) + y) };
    }

    /// Keep the 9 rows of `gvn` that precede position `len[seq]` of each sequence (from its
    /// slot `slot_in` of `state` where the step has fewer) in its slot `slot_out`, oldest
    /// first. A sequence with `len` 0 is skipped.
    /// grid = (ceil(width / 256), n_seq), block = (256).
    #[kernel]
    pub unsafe fn ple_state(gvn: *const u16, state: *mut u16, seq_start: *const u32, len: *const u32, slot_in: *const u32, slot_out: *const u32, width: u32) {
        let c = thread::blockIdx_x() as usize * 256 + thread::threadIdx_x() as usize;
        let seq = thread::blockIdx_y() as usize;
        let wd = width as usize;
        let n = unsafe { *len.add(seq) } as usize;
        if c >= wd || n == 0 {
            return;
        }
        let first = unsafe { *seq_start.add(seq) } as usize;
        let (src, dst) = unsafe { (state.add(*slot_in.add(seq) as usize * 9 * wd), state.add(*slot_out.add(seq) as usize * 9 * wd)) };
        // Row k of the result is n - 9 + k of the step: ascending k never reads a row already written.
        let mut k = 0;
        while k < 9 {
            let v = if n + k >= 9 { unsafe { *gvn.add((first + n + k - 9) * wd + c) } } else { unsafe { *src.add((n + k) * wd + c) } };
            unsafe { *dst.add(k * wd + c) = v };
            k += 1;
        }
    }

    /// `dst[r] = src[rows[r]]`, rows of `hidden` bf16 (a multiple of 8). grid = (R), block = (256).
    #[kernel]
    pub unsafe fn take_rows(src: *const u16, rows: *const u32, dst: *mut u16, hidden: u32) {
        let r = thread::blockIdx_x() as usize;
        let nv = hidden as usize / 8;
        let from = unsafe { src.cast::<U32x4>().add(*rows.add(r) as usize * nv) };
        let mut v = thread::threadIdx_x() as usize;
        while v < nv {
            unsafe { *dst.cast::<U32x4>().add(r * nv + v) = *from.add(v) };
            v += 256;
        }
    }

    /// Zero-centered RMSNorm (`x * rsqrt(mean(x^2) + eps) * (1 + w)`), optionally fused with the
    /// residual add: if `add` != 0, first `x = bf16(x + delta)` in place.
    /// 16-byte loads (8 bf16 per thread per chunk); the row stays in registers between the
    /// sum-of-squares and the scaling pass. Requires hidden % 8 == 0 and hidden <= 8192.
    /// grid = (T), block = (256).
    #[kernel]
    pub unsafe fn add_rmsnorm(
        x: *mut u16,
        delta: *const u16,
        w: *const u16,
        out: *mut u16,
        hidden: u32,
        eps: f32,
        add: u32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let tid = thread::threadIdx_x() as usize;
        let nv = hidden as usize / 8;
        let row = thread::blockIdx_x() as usize * nv;
        let xv = unsafe { x.cast::<U32x4>().add(row) };
        let dv = unsafe { delta.cast::<U32x4>().add(row) };
        let wv = w.cast::<U32x4>();
        let ov = unsafe { out.cast::<U32x4>().add(row) };
        let mut vals = [0.0f32; 32];
        let mut ss = 0.0f32;
        #[unroll]
        for c in 0..4 {
            let v = tid + c * 256;
            if v < nv {
                let xw = unsafe { *xv.add(v) }.to_array();
                let dw = if add != 0 { unsafe { *dv.add(v) }.to_array() } else { [0u32; 4] };
                let mut packed = [0u32; 4];
                #[unroll]
                for e in 0..4 {
                    let (mut a0, mut a1) = cvt_f32x2_bf16x2(xw[e]);
                    if add != 0 {
                        let (d0, d1) = cvt_f32x2_bf16x2(dw[e]);
                        a0 = rb(a0 + d0);
                        a1 = rb(a1 + d1);
                        packed[e] = (f32_to_bf16_rne(a0) as u32) | ((f32_to_bf16_rne(a1) as u32) << 16);
                    }
                    vals[c * 8 + 2 * e] = a0;
                    vals[c * 8 + 2 * e + 1] = a1;
                    ss += a0 * a0 + a1 * a1;
                }
                if add != 0 {
                    unsafe { *xv.add(v) = U32x4::new(packed) };
                }
            }
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        #[unroll]
        for c in 0..4 {
            let v = tid + c * 256;
            if v < nv {
                let ww = unsafe { *wv.add(v) }.to_array();
                let mut packed = [0u32; 4];
                #[unroll]
                for e in 0..4 {
                    let (w0, w1) = cvt_f32x2_bf16x2(ww[e]);
                    let y0 = vals[c * 8 + 2 * e] * inv * (1.0 + w0);
                    let y1 = vals[c * 8 + 2 * e + 1] * inv * (1.0 + w1);
                    packed[e] = (f32_to_bf16_rne(y0) as u32) | ((f32_to_bf16_rne(y1) as u32) << 16);
                }
                unsafe { *ov.add(v) = U32x4::new(packed) };
            }
        }
    }

    /// Per-row FP8 quantization of a bf16 matrix: `out[r, :] = e4m3(x[r, :] / scale[r])`, with
    /// `scale[r]` the power of two that puts the row's largest magnitude in [128, 256) (E4M3
    /// tops out at 448). A power of two keeps the scaling exact, and since E4M3 is a
    /// floating-point format the scale only places its range, it does not cost precision.
    /// Requires k % 8 == 0. grid = (rows), block = (256).
    #[kernel]
    pub unsafe fn quant_e4m3(x: *const u16, out: *mut u8, scale: *mut f32, k: u32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let tid = thread::threadIdx_x() as usize;
        let nv = k as usize / 8;
        let r = thread::blockIdx_x() as usize;
        let xv = unsafe { x.cast::<U32x4>().add(r * nv) };
        // Largest magnitude, compared as bf16 bit patterns (monotone in the magnitude).
        let mut top = 0u32;
        let mut v = tid;
        while v < nv {
            let w = unsafe { *xv.add(v) }.to_array();
            #[unroll]
            for e in 0..4 {
                top = top.max(w[e] & 0x7fff).max((w[e] >> 16) & 0x7fff);
            }
            v += 256;
        }
        let top = unsafe { block_max(top as f32, red) } as u32;
        // Biased exponent of the row maximum; the scale is 2^(exponent - 7).
        let ex = top >> 7;
        let (s, inv) = if ex >= 8 { (f32::from_bits((ex - 7) << 23), f32::from_bits((261 - ex) << 23)) } else { (1.0, 1.0) };
        if tid == 0 {
            unsafe { *scale.add(r) = s };
        }
        let ov = unsafe { out.add(r * k as usize) }.cast::<u64>();
        let mut v = tid;
        while v < nv {
            let w = unsafe { *xv.add(v) }.to_array();
            let mut packed = 0u64;
            #[unroll]
            for e in 0..4 {
                let (a0, a1) = cvt_f32x2_bf16x2(w[e]);
                packed |= ((e4m3(a0 * inv) | (e4m3(a1 * inv) << 8)) as u64) << (16 * e);
            }
            unsafe { *ov.add(v) = packed };
            v += 256;
        }
    }

    /// Diagnostic (AQ_QSIM): round every row of a bf16 matrix through an 8-bit format in
    /// place, to measure what that format costs with everything else unchanged. `mode` 1 is
    /// E4M3 with the row's power-of-two scale (as `quant_e4m3`), 2 is linear int8 scaled to
    /// the row's largest magnitude, 3 is int8 scaled per block of 32 elements. Requires
    /// k % 8 == 0. grid = (rows), block = (256).
    #[kernel]
    pub unsafe fn qsim(x: *mut u16, k: u32, mode: u32) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let tid = thread::threadIdx_x() as usize;
        let nv = k as usize / 8;
        let xv = unsafe { x.cast::<U32x4>().add(thread::blockIdx_x() as usize * nv) };
        let mut top = 0u32;
        let mut v = tid;
        while v < nv {
            let w = unsafe { *xv.add(v) }.to_array();
            #[unroll]
            for e in 0..4 {
                top = top.max(w[e] & 0x7fff).max((w[e] >> 16) & 0x7fff);
            }
            v += 256;
        }
        let top = unsafe { block_max(top as f32, red) } as u32;
        let ex = top >> 7;
        let (s, inv) = if ex >= 8 { (f32::from_bits((ex - 7) << 23), f32::from_bits((261 - ex) << 23)) } else { (1.0, 1.0) };
        let amax = bf16_to_f32(top as u16);
        let mut v = tid;
        while v < nv {
            let w = unsafe { *xv.add(v) }.to_array();
            let mut packed = [0u32; 4];
            #[unroll]
            for e in 0..4 {
                let (a0, a1) = cvt_f32x2_bf16x2(w[e]);
                let mut r = [a0, a1];
                #[unroll]
                for i in 0..2 {
                    if mode == 1 {
                        let c = e4m3(r[i] * inv);
                        let (ef, m) = ((c >> 3) & 15, c & 7);
                        let mag = if ef == 0 { m as f32 * (1.0 / 512.0) } else { f32::from_bits(((ef + 120) << 23) | (m << 20)) };
                        r[i] = if c & 0x80 != 0 { -mag * s } else { mag * s };
                    } else if mode == 2 && amax > 0.0 {
                        r[i] = libm::rintf(r[i] / amax * 127.0) * amax / 127.0;
                    }
                }
                if mode == 3 {
                    // int8 with one scale per 32 elements: the 4 lanes that hold a block
                    // (8 elements each) agree on its largest magnitude. Needs k % 2048 == 0.
                    let mut bm = (w[0] & 0x7fff).max((w[0] >> 16) & 0x7fff);
                    #[unroll]
                    for j in 1..4 {
                        bm = bm.max(w[j] & 0x7fff).max((w[j] >> 16) & 0x7fff);
                    }
                    let mut bf = bm as f32;
                    bf = bf.max(warp::shuffle_xor_f32_sync(FULL, bf, 1));
                    bf = bf.max(warp::shuffle_xor_f32_sync(FULL, bf, 2));
                    let bmax = bf16_to_f32(bf as u32 as u16);
                    if bmax > 0.0 {
                        r[0] = libm::rintf(a0 / bmax * 127.0) * bmax / 127.0;
                        r[1] = libm::rintf(a1 / bmax * 127.0) * bmax / 127.0;
                    }
                }
                packed[e] = pk(r[0], r[1]);
            }
            unsafe { *xv.add(v) = U32x4::new(packed) };
            v += 256;
        }
    }

    /// int8 quantization of a bf16 matrix with one scale per block of 32 elements along a
    /// row: `q = rint(x / bmax * 127)`, `bmax[r, b]` = the block's largest magnitude (bf16).
    /// Requires k % 32 == 0. grid = (rows), block = (256).
    #[kernel]
    pub unsafe fn quant_i8b(x: *const u16, q: *mut u8, bmax: *mut u16, k: u32) {
        let nb = k as usize / 32;
        let r = thread::blockIdx_x() as usize;
        let mut b = thread::threadIdx_x() as usize;
        while b < nb {
            let xv = unsafe { x.cast::<U32x4>().add((r * nb + b) * 4) };
            let mut w = [[0u32; 4]; 4];
            let mut top = 0u32;
            #[unroll]
            for j in 0..4 {
                w[j] = unsafe { *xv.add(j) }.to_array();
                #[unroll]
                for e in 0..4 {
                    top = top.max(w[j][e] & 0x7fff).max((w[j][e] >> 16) & 0x7fff);
                }
            }
            unsafe { *bmax.add(r * nb + b) = top as u16 };
            let m = bf16_to_f32(top as u16);
            let inv = if m > 0.0 { 127.0 / m } else { 0.0 };
            let qv = unsafe { q.add((r * nb + b) * 32) }.cast::<u64>();
            #[unroll]
            for j in 0..4 {
                let mut packed = 0u64;
                #[unroll]
                for e in 0..4 {
                    let (a0, a1) = cvt_f32x2_bf16x2(w[j][e]);
                    let (q0, q1) = (libm::rintf(a0 * inv) as i32 as u32 & 0xff, libm::rintf(a1 * inv) as i32 as u32 & 0xff);
                    packed |= ((q0 | (q1 << 8)) as u64) << (16 * e);
                }
                unsafe { *qv.add(j) = packed };
            }
            b += 256;
        }
    }

    /// Inverse of `quant_i8b` over `blocks` blocks of 32 elements (rows are contiguous, so a
    /// matrix is just `n * k / 32` blocks): `out = bf16(q * bmax / 127)`, one block per thread.
    /// grid = (ceil(blocks / 256)), block = (256).
    #[kernel]
    pub unsafe fn dequant_i8b(q: *const u8, bmax: *const u16, out: *mut u16, blocks: u32) {
        let b = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if b >= blocks as usize {
            return;
        }
        let s = bf16_to_f32(unsafe { *bmax.add(b) }) * (1.0 / 127.0);
        let qv = unsafe { q.add(b * 32) }.cast::<u64>();
        let ov = unsafe { out.cast::<U32x4>().add(b * 4) };
        #[unroll]
        for j in 0..4 {
            let v = unsafe { *qv.add(j) };
            let mut packed = [0u32; 4];
            #[unroll]
            for e in 0..4 {
                let q0 = (v >> (16 * e)) as u8 as i8 as f32;
                let q1 = (v >> (16 * e + 8)) as u8 as i8 as f32;
                packed[e] = pk(q0 * s, q1 * s);
            }
            unsafe { *ov.add(j) = U32x4::new(packed) };
        }
    }

    /// Expand 4-bit weights to bf16: `out = bf16((code - 8) * scale)`, codes two per byte (low
    /// nibble first), one f32 scale per 128 codes along a row. Rows are contiguous and their
    /// length is a multiple of 128, so a matrix is `n * k / 32` blocks of 32 codes and block
    /// `b` uses scale `b / 4`; one block per thread. grid = (ceil(blocks / 256)), block = (256).
    #[kernel]
    pub unsafe fn dequant_i4g(q: *const u8, scale: *const f32, out: *mut u16, blocks: u32) {
        let b = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if b >= blocks as usize {
            return;
        }
        let s = unsafe { *scale.add(b / 4) };
        let qv = unsafe { q.add(b * 16) }.cast::<u32>();
        let ov = unsafe { out.cast::<U32x4>().add(b * 4) };
        #[unroll]
        for j in 0..4 {
            let v = unsafe { *qv.add(j) };
            let mut packed = [0u32; 4];
            #[unroll]
            for e in 0..4 {
                let c0 = ((v >> (8 * e)) & 15) as f32 - 8.0;
                let c1 = ((v >> (8 * e + 4)) & 15) as f32 - 8.0;
                packed[e] = pk(c0 * s, c1 * s);
            }
            unsafe { *ov.add(j) = U32x4::new(packed) };
        }
    }

    /// Split a bf16 matrix into two int8 digit planes for `gemm_i8w`:
    /// `x ~ sx[r, b] * (256 * xh + xl)`, a 16-bit fixed-point form with one scale per block of
    /// 32 elements along a row, `sx = max|block| / 32639` (the bound keeps the high digit of a
    /// balanced split in int8 range). A scale per block, not per row: one very large element
    /// then costs precision only to its 31 neighbours.
    /// Requires k % 32 == 0. grid = (rows), block = (256).
    #[kernel]
    pub unsafe fn quant_x16(x: *const u16, xh: *mut u8, xl: *mut u8, sx: *mut f32, k: u32) {
        let nb = k as usize / 32;
        let r = thread::blockIdx_x() as usize;
        let mut b = thread::threadIdx_x() as usize;
        while b < nb {
            let xv = unsafe { x.cast::<U32x4>().add((r * nb + b) * 4) };
            let mut w = [[0u32; 4]; 4];
            let mut top = 0u32;
            #[unroll]
            for j in 0..4 {
                w[j] = unsafe { *xv.add(j) }.to_array();
                #[unroll]
                for e in 0..4 {
                    top = top.max(w[j][e] & 0x7fff).max((w[j][e] >> 16) & 0x7fff);
                }
            }
            let mx = bf16_to_f32(top as u16);
            unsafe { *sx.add(r * nb + b) = mx * (1.0 / 32639.0) };
            let inv = if mx > 0.0 { 32639.0 / mx } else { 0.0 };
            #[unroll]
            for j in 0..4 {
                let (mut hi, mut lo) = (0u64, 0u64);
                #[unroll]
                for e in 0..4 {
                    let (a0, a1) = cvt_f32x2_bf16x2(w[j][e]);
                    let (i0, i1) = (libm::rintf(a0 * inv) as i32, libm::rintf(a1 * inv) as i32);
                    lo |= (((i0 & 0xff) | ((i1 & 0xff) << 8)) as u64) << (16 * e);
                    hi |= (((((i0 + 128) >> 8) & 0xff) | ((((i1 + 128) >> 8) & 0xff) << 8)) as u64) << (16 * e);
                }
                unsafe {
                    *(xh.add((r * nb + b) * 32 + j * 8) as *mut u64) = hi;
                    *(xl.add((r * nb + b) * 32 + j * 8) as *mut u64) = lo;
                }
            }
            b += 256;
        }
    }

    /// `y[m, n] = x[m, k] . w[n, k]^T` for block-scaled int8 weights (`quant_i8b` layout), on
    /// the integer tensor cores: the weight bytes go from memory into `mma` as they are, read
    /// once. The activations come as the two int8 digit planes of `quant_x16`; each 32-wide
    /// block of k is one `m16n8k32` product per digit, and the block's weight and activation
    /// scales are applied to that partial sum as it is added to the f32 accumulator.
    ///
    /// A block owns 128 weight rows (warp w: rows 16w..16w+16, two n8 tiles) and up to 128
    /// activation rows, and walks K in chunks of 64 through a 4-stage `cp.async` ring, so three
    /// chunks of weights, scales and activations are in flight while one is multiplied.
    /// Operands are read with `ldmatrix`; the four 16-byte pieces of a shared row are stored
    /// XOR-swizzled by `(row >> 1) & 3` so the eight rows of a fragment hit distinct banks.
    ///
    /// Stage layout: W [128][64] i8, weight scales [128] x 2 bf16, activation scales
    /// [mt * 16] x 2 f32, XH then XL [mt * 16][64] i8.
    /// Requires k % 64 == 0 and n % 8 == 0.
    /// grid = (ceil(n / 128), ceil(m / 128)), block = (256),
    /// dynamic shared memory = 4 * (8704 + min(m, 128).div_ceil(16) * 2176) bytes.
    #[kernel]
    pub unsafe fn gemm_i8w(xh: *const u8, xl: *const u8, sx: *const f32, q: *const u8, bmax: *const u16, y: *mut u16, m: u32, n: u32, k: u32, ldy: u32, w4: u32) {
        use cuda_device::async_copy::{cp_async_ca_4, cp_async_ca_8, cp_async_ca_zfill_4, cp_async_ca_zfill_8, cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group};
        use cuda_device::wmma::{ldmatrix_x4, mma_m16n8k32_s32_s8};
        const S: usize = 4;
        // Byte offset of 16-byte piece `c` of shared row `r` (rows of 64 bytes).
        let sw = |r: usize, c: usize| r * 64 + ((c ^ ((r >> 1) & 3)) * 16);
        let sm = DynamicSharedArray::<u8, 16>::get_raw();
        let (m, n, k, ldy) = (m as usize, n as usize, k as usize, ldy as usize);
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let g = lane / 4;
        let t = lane % 4;
        let (lr, lm) = (lane % 8, lane / 8); // ldmatrix: row and matrix this lane addresses
        let nb = thread::blockIdx_x() as usize * 128;
        let m0 = thread::blockIdx_y() as usize * 128;
        let mc = if m - m0 < 128 { m - m0 } else { 128 };
        let mtc = mc.div_ceil(16);
        let plane = mtc * 1024;
        // A ring slot: the weight tile (int8: 128 rows x 64 bytes, swizzled; 4-bit: 128 x 32
        // bytes of codes as they are stored), the rows' scales, the activation rows' scales,
        // the two digit planes. With 4-bit weights the first 8192 bytes of shared memory hold
        // the tile being used, unpacked to int8.
        let w4 = w4 != 0;
        let wsc = if w4 { 4096 } else { 8192 };
        let asc = wsc + 512;
        let xo = asc + mtc * 128;
        let stage = xo + 2 * plane;
        let ring = if w4 { unsafe { sm.add(8192) } } else { sm };
        let kb = k / 32;
        let chunks = k / 64;

        // Issue chunk `c` into ring slot `c % S` and commit it as one group.
        let issue = |c: usize| {
            let sb = unsafe { ring.add((c % S) * stage) };
            let kc = c * 64;
            unsafe {
                if w4 {
                    // Codes: 128 rows x 2 vectors, one per thread.
                    let (r, v) = (tid / 2, tid % 2);
                    let dst = sb.add(r * 32 + v * 16) as *mut u32;
                    if nb + r < n {
                        cp_async_cg_16(dst, q.add((nb + r) * (k / 2) + kc / 2 + v * 16) as *const u32);
                    } else {
                        cp_async_cg_zfill_16(dst, q, 0);
                    }
                } else {
                    // Weights: 128 rows x 4 vectors, two per thread.
                    #[unroll]
                    for u in 0..2 {
                        let idx = tid + 256 * u;
                        let (r, v) = (idx / 4, idx % 4);
                        let dst = sb.add(sw(r, v)) as *mut u32;
                        if nb + r < n {
                            cp_async_cg_16(dst, q.add((nb + r) * k + kc + v * 16) as *const u32);
                        } else {
                            cp_async_cg_zfill_16(dst, q, 0);
                        }
                    }
                }
                // Each row's scales for this chunk: two bf16 block magnitudes, or with 4-bit
                // weights the f32 scale of the 128 columns it is in.
                if tid < 128 {
                    let dst = sb.add(wsc + tid * 4) as *mut u32;
                    if nb + tid < n {
                        if w4 {
                            cp_async_ca_4(dst, (bmax as *const u32).add((nb + tid) * (k / 128) + kc / 128));
                        } else {
                            cp_async_ca_4(dst, bmax.add((nb + tid) * kb + kc / 32) as *const u32);
                        }
                    } else {
                        cp_async_ca_zfill_4(dst, q, 0);
                    }
                }
                // The two block scales of each activation row.
                if tid < mtc * 16 {
                    let dst = sb.add(asc + tid * 8) as *mut u32;
                    if tid < mc {
                        cp_async_ca_8(dst, sx.add((m0 + tid) * kb + kc / 32) as *const u32);
                    } else {
                        cp_async_ca_zfill_8(dst, q, 0);
                    }
                }
                // Activation digits: mtc * 16 rows x (4 high + 4 low) vectors.
                let mut idx = tid;
                while idx < mtc * 128 {
                    let (r, v) = (idx / 8, idx % 8);
                    let dst = sb.add(xo + (v / 4) * plane + sw(r, v % 4)) as *mut u32;
                    if r < mc {
                        let src = if v < 4 { xh } else { xl };
                        cp_async_cg_16(dst, src.add((m0 + r) * k + kc + (v % 4) * 16) as *const u32);
                    } else {
                        cp_async_cg_zfill_16(dst, q, 0);
                    }
                    idx += 256;
                }
                cp_async_commit_group();
            }
        };

        #[unroll]
        for c in 0..3 {
            if c < chunks {
                issue(c);
            } else {
                unsafe { cp_async_commit_group() };
            }
        }
        let mut acc = [[[0.0f32; 4]; 2]; 8];
        let mut c = 0;
        while c < chunks {
            // Chunk c has landed for this thread; after the barrier, for the block, and
            // everyone is done with chunk c - 1, whose slot the next issue reuses.
            unsafe { cp_async_wait_group(2) };
            thread::sync_threads();
            if c + 3 < chunks {
                issue(c + 3);
            } else {
                unsafe { cp_async_commit_group() };
            }
            let sb = unsafe { ring.add((c % S) * stage) };
            if w4 {
                // Unpack the chunk's codes (two per byte, low nibble first, value = code - 8)
                // into int8 operand rows: each thread turns its 16 bytes into two pieces.
                let (r, v) = (tid / 2, tid % 2);
                let src = unsafe { sb.add(r * 32 + v * 16) } as *const u32;
                #[unroll]
                for hh in 0..2 {
                    let mut out = [0u64; 2];
                    #[unroll]
                    for u in 0..2 {
                        let x = unsafe { *src.add(2 * hh + u) };
                        let mut o = 0u64;
                        #[unroll]
                        for jj in 0..4 {
                            let by = ((x >> (8 * jj)) & 0xff) as u64;
                            o |= ((by & 15) | ((by >> 4) << 8)) << (16 * jj);
                        }
                        // code - 8 as int8: flip bit 3, then extend it through the high nibble.
                        let tt = o ^ 0x0808_0808_0808_0808;
                        let s8 = tt & 0x0808_0808_0808_0808;
                        out[u] = tt | (s8 << 1) | (s8 << 2) | (s8 << 3) | (s8 << 4);
                    }
                    let dst = unsafe { sm.add(sw(r, 2 * v + hh)) } as *mut u64;
                    unsafe {
                        *dst = out[0];
                        *dst.add(1) = out[1];
                    }
                }
                thread::sync_threads();
            }
            let wop = if w4 { sm } else { sb };
            #[unroll]
            for blk in 0..2 {
                // Both n8 tiles of this warp: rows lr and 8 + lr, two 16-byte pieces each.
                let b = unsafe { ldmatrix_x4(wop.add(sw(w * 16 + 8 * (lm >> 1) + lr, 2 * blk + (lm & 1))) as *const u32) };
                let mut sc = [[0.0f32; 2]; 2];
                #[unroll]
                for nt in 0..2 {
                    // This lane's two output columns are weight rows 2t and 2t + 1 of the tile.
                    let sv = unsafe { *(sb.add(wsc + (w * 16 + nt * 8 + 2 * t) * 4) as *const u64) };
                    sc[nt] = if w4 {
                        [f32::from_bits(sv as u32), f32::from_bits((sv >> 32) as u32)]
                    } else {
                        [bf16_to_f32((sv >> (16 * blk)) as u16) * (1.0 / 127.0), bf16_to_f32((sv >> (32 + 16 * blk)) as u16) * (1.0 / 127.0)]
                    };
                }
                #[unroll]
                for mt in 0..8 {
                    if mt < mtc {
                        let fx = unsafe { [*(sb.add(asc + (mt * 16 + g) * 8 + blk * 4) as *const f32), *(sb.add(asc + (mt * 16 + g + 8) * 8 + blk * 4) as *const f32)] };
                        let o = sw(mt * 16 + 8 * (lm & 1) + lr, 2 * blk + (lm >> 1));
                        let ah = unsafe { ldmatrix_x4(sb.add(xo + o) as *const u32) };
                        let al = unsafe { ldmatrix_x4(sb.add(xo + plane + o) as *const u32) };
                        #[unroll]
                        for nt in 0..2 {
                            let bn = [b[2 * nt], b[2 * nt + 1]];
                            let p = unsafe { mma_m16n8k32_s32_s8([0; 4], ah, bn) };
                            let p = unsafe { mma_m16n8k32_s32_s8([p[0] << 8, p[1] << 8, p[2] << 8, p[3] << 8], al, bn) };
                            #[unroll]
                            for e in 0..4 {
                                acc[mt][nt][e] += p[e] as f32 * (sc[nt][e & 1] * fx[e >> 1]);
                            }
                        }
                    }
                }
            }
            c += 1;
        }

        #[unroll]
        for mt in 0..8 {
            if mt < mtc {
                let r0 = m0 + mt * 16 + g;
                #[unroll]
                for nt in 0..2 {
                    let col = nb + w * 16 + nt * 8 + 2 * t;
                    if col < n {
                        if r0 < m {
                            unsafe { *(y.add(r0 * ldy + col) as *mut u32) = pk(acc[mt][nt][0], acc[mt][nt][1]) };
                        }
                        if r0 + 8 < m {
                            unsafe { *(y.add((r0 + 8) * ldy + col) as *mut u32) = pk(acc[mt][nt][2], acc[mt][nt][3]) };
                        }
                    }
                }
            }
        }
    }

    /// `y[r, c] *= rs[r] * cs[c]` in place on a bf16 matrix.
    /// Requires n % 8 == 0. grid = (rows), block = (256).
    #[kernel]
    pub unsafe fn scale_outer(y: *mut u16, rs: *const f32, cs: *const f32, n: u32) {
        let nv = n as usize / 8;
        let r = thread::blockIdx_x() as usize;
        let yv = unsafe { y.cast::<U32x4>().add(r * nv) };
        let a = unsafe { *rs.add(r) };
        let mut v = thread::threadIdx_x() as usize;
        while v < nv {
            let w = unsafe { *yv.add(v) }.to_array();
            let mut packed = [0u32; 4];
            #[unroll]
            for e in 0..4 {
                let (y0, y1) = cvt_f32x2_bf16x2(w[e]);
                let (c0, c1) = unsafe { (*cs.add(v * 8 + 2 * e), *cs.add(v * 8 + 2 * e + 1)) };
                packed[e] = pk(y0 * a * c0, y1 * a * c1);
            }
            unsafe { *yv.add(v) = U32x4::new(packed) };
            v += 256;
        }
    }

    /// out[t, i] = silu(gu[t, i]) * gu[t, inter + i], 8 elements per thread with 16-byte
    /// loads. Requires inter % 8 == 0. grid = (ceil(T * inter / 8 / 256)), block = (256).
    #[kernel]
    pub unsafe fn silu_mul(gu: *const u16, out: *mut u16, inter: u32, n_tokens: u32) {
        let nv = inter as usize / 8;
        let idx = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if idx >= n_tokens as usize * nv {
            return;
        }
        let t = idx / nv;
        let v = idx % nv;
        let g = unsafe { *gu.cast::<U32x4>().add(t * 2 * nv + v) }.to_array();
        let u = unsafe { *gu.cast::<U32x4>().add(t * 2 * nv + nv + v) }.to_array();
        let mut packed = [0u32; 4];
        #[unroll]
        for e in 0..4 {
            let (g0, g1) = cvt_f32x2_bf16x2(g[e]);
            let (u0, u1) = cvt_f32x2_bf16x2(u[e]);
            let y0 = rb(silu(g0)) * u0;
            let y1 = rb(silu(g1)) * u1;
            packed[e] = (f32_to_bf16_rne(y0) as u32) | ((f32_to_bf16_rne(y1) as u32) << 16);
        }
        unsafe { *out.cast::<U32x4>().add(idx) = U32x4::new(packed) };
    }

    /// Mixture-of-experts routing for one token per warp: softmax over the `ne` router logits
    /// (f32), the `topk` largest, renormalized to sum to one and rounded to bf16 as the
    /// reference does. They are written in ascending expert order, the order the reference
    /// adds the experts' outputs in: `idx[token]` is 16 u16, the selections lowest first then
    /// 0xffff, and `wout[token * topk + s]` their weights.
    /// Requires ne % 32 == 0, ne <= 512 and topk <= 16. grid = (ceil(T / 8)), block = (256).
    #[kernel]
    pub unsafe fn moe_route(logits: *const u16, ldl: u32, idx: *mut u16, wout: *mut f32, n_tokens: u32, ne: u32, topk: u32) {
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let tok = thread::blockIdx_x() as usize * 8 + tid / 32;
        if tok >= n_tokens as usize {
            return;
        }
        let (per, k) = (ne as usize / 32, topk as usize);
        let mut v = [f32::NEG_INFINITY; 16];
        let mut mx = f32::NEG_INFINITY;
        #[unroll]
        for i in 0..16 {
            if i < per {
                v[i] = unsafe { ld(logits, tok * ldl as usize + lane * per + i) };
                mx = mx.max(v[i]);
            }
        }
        let mx = warp_max(mx);
        let mut sum = 0.0f32;
        #[unroll]
        for i in 0..16 {
            if i < per {
                sum += libm::expf(v[i] - mx);
            }
        }
        let sum = warp_sum(sum);
        let mut se = [0u32; 16];
        let mut sp = [0.0f32; 16];
        #[unroll]
        for r in 0..16 {
            if r < k {
                let mut bv = f32::NEG_INFINITY;
                let mut bi = 0usize;
                #[unroll]
                for i in 0..16 {
                    if v[i] > bv {
                        bv = v[i];
                        bi = i;
                    }
                }
                let top = warp_max(bv);
                // The lowest expert index holding the maximum (indices are exact in f32).
                let cand = if bv == top { (lane * per + bi) as f32 } else { 1.0e9 };
                let e = -warp_max(-cand);
                if cand == e {
                    #[unroll]
                    for i in 0..16 {
                        if i == bi {
                            v[i] = f32::NEG_INFINITY;
                        }
                    }
                }
                se[r] = e as u32;
                sp[r] = libm::expf(top - mx) / sum;
            }
        }
        if lane != 0 {
            return;
        }
        let mut tot = 0.0f32;
        #[unroll]
        for r in 0..16 {
            if r < k {
                tot += sp[r];
            }
        }
        #[unroll]
        for r in 0..16 {
            if r < k {
                // Rank of the selection by expert index: where it goes in the output.
                let mut rank = 0usize;
                #[unroll]
                for o in 0..16 {
                    if o < k && se[o] < se[r] {
                        rank += 1;
                    }
                }
                unsafe {
                    *idx.add(tok * 16 + rank) = se[r] as u16;
                    *wout.add(tok * k + rank) = rb(sp[r] / tot);
                }
            } else {
                unsafe { *idx.add(tok * 16 + r) = 0xffff };
            }
        }
    }

    /// Group the rows of a step's expert GEMMs by expert. Row `token * topk + selection` is
    /// that token through its `selection`-th expert. `list` receives the rows expert after
    /// expert, and `jobs` one entry per up to 64 rows of one expert:
    /// `[offset into list, expert | rows << 16]`. Entries past the last job get zero rows.
    /// Thread `e` scans every token's selections (`moe_route`'s `idx`) for expert `e` (at most
    /// one match).
    /// grid = (1), block = (ne), ne <= 512. `max_jobs` >= ceil(topk * T / 64) + ne.
    #[kernel]
    pub unsafe fn moe_bucket(idx: *const u64, n_tokens: u32, list: *mut u32, jobs: *mut u32, max_jobs: u32, ne: u32, topk: u32) {
        static mut CNT: SharedArray<u32, 512> = SharedArray::UNINIT;
        const LO: u64 = 0x0001_0001_0001_0001;
        let cnt = (&raw mut CNT).cast::<u32>();
        let tid = thread::threadIdx_x();
        let e = tid as usize;
        let n = n_tokens as usize;
        let words = (topk as usize).div_ceil(4);
        let jr = 64u32;
        // 1 + the selection of `tok` that is expert `e`, or 0. Bit 15 of the lowest u16 of a
        // word equal to `e` is set in `z` (bits above it may be set too).
        let find = |tok: usize| {
            let mut sel = 0u32;
            let mut w = 0;
            while w < words {
                let x = unsafe { *idx.add(tok * 4 + w) } ^ (e as u64 * LO);
                let z = x.wrapping_sub(LO) & !x & (LO << 15);
                if z != 0 {
                    // Bits below the lowest set one: 15 + 16 * (position in the word).
                    sel = 1 + 4 * w as u32 + (((z & z.wrapping_neg()) - 1).count_ones() >> 4);
                }
                w += 1;
            }
            sel
        };
        let mut c = 0u32;
        let mut p = 0;
        while p < n {
            c += (find(p) != 0) as u32;
            p += 1;
        }
        unsafe { *cnt.add(e) = c };
        thread::sync_threads();
        let (mut off, mut joff, mut total) = (0u32, 0u32, 0u32);
        let mut o = 0;
        while o < ne as usize {
            let co = unsafe { *cnt.add(o) };
            if o < e {
                off += co;
                joff += co.div_ceil(jr);
            }
            total += co.div_ceil(jr);
            o += 1;
        }
        let mut j = 0u32;
        let mut p = 0;
        while p < n {
            let sel = find(p);
            if sel != 0 {
                unsafe { *list.add((off + j) as usize) = p as u32 * topk + sel - 1 };
                j += 1;
            }
            p += 1;
        }
        let mut i = 0u32;
        while i * jr < c {
            let mc = if c - i * jr < jr { c - i * jr } else { jr };
            unsafe {
                *jobs.add(2 * (joff + i) as usize) = off + i * jr;
                *jobs.add(2 * (joff + i) as usize + 1) = e as u32 | (mc << 16);
            }
            i += 1;
        }
        let mut i = total + tid;
        while i < max_jobs {
            unsafe { *jobs.add(2 * i as usize + 1) = 0 };
            i += ne;
        }
    }

    /// Grouped expert GEMM over the jobs of `moe_bucket`. Expert `e`'s `[n][k]` weights are
    /// `slots[e] * stride` bytes past `w`: bf16, or with `scale` not null the 4-bit codes and
    /// (as far past `scale`) scales `dequant_i4g` reads, staged as codes and expanded to the
    /// same bf16 values before use. A 3-stage `cp.async` ring in shared memory, each stage
    /// holding 64 columns of K for 128 weight rows and the job's (up to 64) gathered rows.
    ///
    /// `fused` != 0, the gate/up product (`n = 2 * inter`): the input row of list entry `p` is
    /// its token's row of `x` (`p / topk`), a block owns 64 of the `inter` outputs (each warp 8 gate rows
    /// and the 8 up rows that go with them), and writes
    /// `y[p, i] = bf16(silu(gate_i)) * up_i` with gate and up rounded to bf16 first, as the
    /// reference does. grid = (inter / 64, max_jobs).
    /// `fused` == 0, the down product: `y[p, :] = x[p, :] . w[expert]^T`, 128 outputs a block.
    /// grid = (n / 128, max_jobs).
    ///
    /// Requires n % 128 == 0 and k % 64 == 0.
    /// block = (256), dynamic shared memory = 3 * (27648 + 4608) + 256 bytes.
    #[kernel]
    pub unsafe fn moe_gemm(x: *const u16, ldx: u32, w: *const u16, y: *mut u16, list: *const u32, jobs: *const u32, n: u32, k: u32, fused: u32, scale: *const f32, slots: *const u32, stride: u32, topk: u32) {
        use cuda_device::async_copy::{cp_async_ca_4, cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_group};
        const S: usize = 3;
        const LD: usize = 72;
        const STAGE: usize = (128 + 64) * LD * 2;
        // 4-bit staging of a chunk: 32 bytes of codes then one scale for each of 128 rows.
        const QSTAGE: usize = 128 * 32 + 128 * 4;
        let job = thread::blockIdx_y() as usize;
        let je = unsafe { *jobs.add(2 * job + 1) };
        let mc = (je >> 16) as usize;
        if mc == 0 {
            return;
        }
        let e = (je & 0xffff) as usize;
        let jo = unsafe { *jobs.add(2 * job) } as usize;
        let sm = DynamicSharedArray::<u8, 16>::get_raw();
        let rid = unsafe { sm.add(S * STAGE) } as *mut u32;
        let (n, k, ldx) = (n as usize, k as usize, ldx as usize);
        let fused = fused != 0;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let wp = tid / 32;
        let g = lane / 4;
        let t = lane % 4;
        let bx = thread::blockIdx_x() as usize;
        let mtc = mc.div_ceil(16);
        let at = unsafe { *slots.add(e) } as usize * stride as usize;
        let w4 = !scale.is_null();
        let qbase = unsafe { w.cast::<u8>().add(at) };
        let (wbase, sbase) = (qbase.cast::<u16>(), unsafe { scale.cast::<u8>().add(at) }.cast::<f32>());
        if tid < 64 {
            unsafe { *rid.add(tid) = if tid < mc { *list.add(jo + tid) } else { 0 } };
        }
        thread::sync_threads();
        let chunks = k / 64;
        let qst = unsafe { sm.add(S * STAGE + 256) };
        // Weight row of shared row r: fused, warp r / 16 has gate rows then up rows of its 8
        // outputs; otherwise the block's 128 rows in order.
        let wrow = |r: usize| if fused { ((r % 16) / 8) * (n / 2) + bx * 64 + (r / 16) * 8 + r % 8 } else { bx * 128 + r };
        let issue = |c: usize| {
            let sb = unsafe { sm.add((c % S) * STAGE) } as *mut u16;
            let kc = c * 64;
            unsafe {
                if w4 {
                    let qs = qst.add((c % S) * QSTAGE);
                    let r = tid / 2;
                    cp_async_cg_16(qs.add(tid * 16) as *mut u32, qbase.add((wrow(r) * k + kc) / 2 + (tid % 2) * 16) as *const u32);
                    if tid < 128 {
                        cp_async_ca_4(qs.add(128 * 32 + tid * 4) as *mut u32, sbase.add(wrow(tid) * (k / 128) + kc / 128) as *const u32);
                    }
                } else {
                    // Weights: 128 rows x 8 pieces, four per thread.
                    #[unroll]
                    for u in 0..4 {
                        let i = tid + 256 * u;
                        let (r, cc) = (i / 8, i % 8);
                        cp_async_cg_16(sb.add(r * LD + cc * 8) as *mut u32, wbase.add(wrow(r) * k + kc + cc * 8) as *const u32);
                    }
                }
                let xs = sb.add(128 * LD);
                let mut i = tid;
                while i < mtc * 128 {
                    let (r, cc) = (i / 8, i % 8);
                    let dst = xs.add(r * LD + cc * 8) as *mut u32;
                    if r < mc {
                        let p = *rid.add(r) as usize;
                        let row = if fused { p / topk as usize } else { p };
                        cp_async_cg_16(dst, x.add(row * ldx + kc + cc * 8) as *const u32);
                    } else {
                        cp_async_cg_zfill_16(dst, x as *const u8, 0);
                    }
                    i += 256;
                }
                cp_async_commit_group();
            }
        };
        #[unroll]
        for c in 0..2 {
            if c < chunks {
                issue(c);
            } else {
                unsafe { cp_async_commit_group() };
            }
        }
        let mut acc = [[[0.0f32; 4]; 2]; 4];
        let mut c = 0;
        while c < chunks {
            // Chunk c has landed; after the barrier nobody reads chunk c - 1, whose slot the
            // next issue reuses.
            unsafe { cp_async_wait_group(1) };
            thread::sync_threads();
            if c + 2 < chunks {
                issue(c + 2);
            } else {
                unsafe { cp_async_commit_group() };
            }
            let ws = unsafe { sm.add((c % S) * STAGE) } as *const u16;
            let xs = unsafe { ws.add(128 * LD) };
            if w4 {
                let qs = unsafe { qst.add((c % S) * QSTAGE) };
                #[unroll]
                for u in 0..4 {
                    let i = tid + 256 * u;
                    let (r, cc) = (i / 8, i % 8);
                    let v = unsafe { *(qs.add(r * 32 + cc * 4) as *const u32) };
                    let sc = unsafe { *(qs.add(128 * 32 + r * 4) as *const f32) };
                    let mut packed = [0u32; 4];
                    #[unroll]
                    for b in 0..4 {
                        let c0 = ((v >> (8 * b)) & 15) as f32 - 8.0;
                        let c1 = ((v >> (8 * b + 4)) & 15) as f32 - 8.0;
                        packed[b] = pk(c0 * sc, c1 * sc);
                    }
                    unsafe { *(ws.add(r * LD + cc * 8) as *mut U32x4) = U32x4::new(packed) };
                }
                thread::sync_threads();
            }
            #[unroll]
            for ks in 0..4 {
                let b0 = unsafe { frag_b(ws, LD, wp * 16, ks * 16, g, t) };
                let b1 = unsafe { frag_b(ws, LD, wp * 16 + 8, ks * 16, g, t) };
                #[unroll]
                for mt in 0..4 {
                    if mt < mtc {
                        let a = unsafe { frag_a(xs, LD, mt * 16, ks * 16, g, t) };
                        acc[mt][0] = unsafe { mma(acc[mt][0], a, b0) };
                        acc[mt][1] = unsafe { mma(acc[mt][1], a, b1) };
                    }
                }
            }
            c += 1;
        }
        #[unroll]
        for mt in 0..4 {
            #[unroll]
            for half in 0..2 {
                let r = mt * 16 + g + 8 * half;
                if r < mc {
                    let row = unsafe { *rid.add(r) } as usize;
                    if fused {
                        let (g0, g1) = (rb(acc[mt][0][2 * half]), rb(acc[mt][0][2 * half + 1]));
                        let (u0, u1) = (rb(acc[mt][1][2 * half]), rb(acc[mt][1][2 * half + 1]));
                        let col = bx * 64 + wp * 8 + 2 * t;
                        unsafe { *(y.add(row * (n / 2) + col) as *mut u32) = pk(rb(silu(g0)) * u0, rb(silu(g1)) * u1) };
                    } else {
                        #[unroll]
                        for nt8 in 0..2 {
                            let col = bx * 128 + wp * 16 + nt8 * 8 + 2 * t;
                            unsafe { *(y.add(row * n + col) as *mut u32) = pk(acc[mt][nt8][2 * half], acc[mt][nt8][2 * half + 1]) };
                        }
                    }
                }
            }
        }
    }

    /// Mixture-of-experts output of one token: the `topk` expert outputs (rows
    /// `token * topk + s` of `po`) scaled by their routing weights and added in order, then the
    /// shared expert's (row `topk * T + token`) scaled by the sigmoid of its gate (column `ne`
    /// of the router logits), rounding to bf16 wherever the reference does.
    /// Requires hidden % 8 == 0 and topk <= 16. grid = (T), block = (256).
    #[kernel]
    pub unsafe fn moe_combine(po: *const u16, rw: *const f32, logits: *const u16, ldl: u32, out: *mut u16, hidden: u32, n_tokens: u32, ne: u32, topk: u32) {
        let nv = hidden as usize / 8;
        let k = topk as usize;
        let tok = thread::blockIdx_x() as usize;
        let gate = rb(sigmoid(unsafe { ld(logits, tok * ldl as usize + ne as usize) }));
        let mut wk = [0.0f32; 16];
        #[unroll]
        for s in 0..16 {
            if s < k {
                wk[s] = unsafe { *rw.add(tok * k + s) };
            }
        }
        let mut v = thread::threadIdx_x() as usize;
        while v < nv {
            let mut acc = [0.0f32; 8];
            #[unroll]
            for s in 0..16 {
                if s < k {
                    let pv = unsafe { *po.cast::<U32x4>().add((tok * k + s) * nv + v) }.to_array();
                    #[unroll]
                    for e in 0..4 {
                        let (a0, a1) = cvt_f32x2_bf16x2(pv[e]);
                        acc[2 * e] = rb(acc[2 * e] + rb(a0 * wk[s]));
                        acc[2 * e + 1] = rb(acc[2 * e + 1] + rb(a1 * wk[s]));
                    }
                }
            }
            let sv = unsafe { *po.cast::<U32x4>().add((k * n_tokens as usize + tok) * nv + v) }.to_array();
            let mut packed = [0u32; 4];
            #[unroll]
            for e in 0..4 {
                let (s0, s1) = cvt_f32x2_bf16x2(sv[e]);
                packed[e] = pk(acc[2 * e] + rb(gate * s0), acc[2 * e + 1] + rb(gate * s1));
            }
            unsafe { *out.cast::<U32x4>().add(tok * nv + v) = U32x4::new(packed) };
            v += 256;
        }
    }

    /// Full-attention prologue for one token and one head.
    ///
    /// `qkv` rows are `[q|gate interleaved per head: H * 2D][k: Hkv * D][v: Hkv * D]`.
    /// Query heads get q_norm + partial RoPE into `q_out[T, H, D]`; key heads get
    /// k_norm + partial RoPE written into the paged cache, and v copied next to it.
    /// Paged layout: `cache[block][slot in block][Hkv][D]`, block ids from the token's
    /// sequence row of `block_tables` (`max_blocks` entries per sequence).
    /// grid = (T, H + Hkv), block = (D).
    #[kernel]
    pub unsafe fn attn_prep(
        qkv: *const u16,
        ld_qkv: u32,
        q_norm: *const u16,
        k_norm: *const u16,
        q_out: *mut u16,
        k_cache: *mut u16,
        v_cache: *mut u16,
        tok_pos: *const u32,
        rope_off: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        rot_dim: u32,
        rope_h: u32,
        rope_w: u32,
        rope_theta: f32,
        eps: f32,
        kv8: u32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut ROW: SharedArray<f32, 512> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let rowbuf = (&raw mut ROW).cast::<f32>();

        let t = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let dim = thread::blockDim_x() as usize;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let row = t * ld_qkv as usize;
        let pos = unsafe { *tok_pos.add(t) } as usize;

        let is_q = head < h;
        let (src, w) = if is_q {
            (row + head * 2 * dim + d, q_norm)
        } else {
            (row + h * 2 * dim + (head - h) * dim + d, k_norm)
        };
        let x = unsafe { ld(qkv, src) };
        let ss = unsafe { block_sum(x * x, red) };
        let inv = 1.0 / libm::sqrtf(ss / dim as f32 + eps);
        // RMSNorm output is materialized in bf16 by the reference.
        let xn = rb(x * inv * (1.0 + unsafe { ld(w, d) }));
        unsafe { *rowbuf.add(d) = xn };
        thread::sync_threads();

        let rot = rot_dim as usize;
        let half = rot / 2;
        let mut y = xn;
        if d < rot {
            let i = d % half;
            let inv_freq = 1.0 / libm::powf(rope_theta, (2 * i) as f32 / rot as f32);
            // Three positions per token (time, row, column), interleaved over the frequencies:
            // `rope_off` holds each one's distance from `pos`, zero for a text token.
            let axis = if i % 3 == 1 && i < rope_h as usize {
                1
            } else if i % 3 == 2 && i < rope_w as usize {
                2
            } else {
                0
            };
            let p = (pos as u32).wrapping_add(unsafe { *rope_off.add(t * 3 + axis) });
            let f = inv_freq * p as f32;
            let c = rb(libm::cosf(f));
            let s = rb(libm::sinf(f));
            // rotate_half: [-x2, x1]
            let other = if d < half {
                -unsafe { *rowbuf.add(d + half) }
            } else {
                unsafe { *rowbuf.add(d - half) }
            };
            y = rb(rb(xn * c) + rb(other * s));
        }

        if is_q {
            unsafe { st(q_out, (t * h + head) * dim + d, y) };
        } else {
            let kv = head - h;
            let seq = unsafe { *tok_seq.add(t) } as usize;
            let bs = block_size as usize;
            let blk = unsafe { *block_tables.add(seq * max_blocks as usize + pos / bs) } as usize;
            let vec = (blk * bs + pos % bs) * hkv + kv;
            let v = unsafe { *qkv.add(row + h * 2 * dim + hkv * dim + kv * dim + d) };
            if kv8 == 0 {
                let off = vec * dim + d;
                unsafe { st(k_cache, off, y) };
                unsafe { *v_cache.add(off) = v };
            } else {
                // 8-bit cache: each row as int8 with its own scale (the largest magnitude / 127).
                let vf = bf16_to_f32(v);
                let mk = unsafe { block_max(y.abs(), red) };
                let mv = unsafe { block_max(vf.abs(), red) };
                let kq = unsafe { (k_cache as *mut u8).add(vec * KV8_ROW) };
                let vq = unsafe { (v_cache as *mut u8).add(vec * KV8_ROW) };
                let ik = if mk > 0.0 { 127.0 / mk } else { 0.0 };
                let iv = if mv > 0.0 { 127.0 / mv } else { 0.0 };
                unsafe {
                    *kq.add(d) = libm::rintf(y * ik) as i32 as u8;
                    *vq.add(d) = libm::rintf(vf * iv) as i32 as u8;
                    if d == 0 {
                        *(kq.add(256) as *mut f32) = mk * (1.0 / 127.0);
                        *(vq.add(256) as *mut f32) = mv * (1.0 / 127.0);
                    }
                }
            }
        }
    }

    /// Causal GQA attention over the paged KV cache, with the output gate fused:
    /// `out[t, h*D + d] = attn[t, h, d] * sigmoid(gate[t, h, d])`.
    ///
    /// grid = (T, H), block = (256) = 8 warps. Each warp walks a strided subset of keys with
    /// an online softmax over its own lane slice of D (D = 256, 8 dims per lane); the 8
    /// partial states are merged through shared memory at the end.
    #[kernel]
    pub unsafe fn attention(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
    ) {
        const D: usize = 256;
        const PER: usize = 8;
        static mut M: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut L: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut ACC: SharedArray<f32, 2048> = SharedArray::UNINIT;
        let sm = (&raw mut M).cast::<f32>();
        let sl = (&raw mut L).cast::<f32>();
        let sacc = (&raw mut ACC).cast::<f32>();

        let t = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let kv = head / (h / hkv);
        let n_keys = unsafe { *tok_pos.add(t) } as usize + 1;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t) as usize * max_blocks as usize) };

        let mut qv = [0.0f32; PER];
        let qbase = (t * h + head) * D + lane * PER;
        let mut i = 0;
        while i < PER {
            qv[i] = unsafe { ld(q, qbase + i) } * scale;
            i += 1;
        }
        let mut acc = [0.0f32; PER];
        let mut m = -1.0e30f32;
        let mut l = 0.0f32;

        let mut j = w;
        while j < n_keys {
            let blk = unsafe { *table.add(j / bs) } as usize;
            let kb = ((blk * bs + j % bs) * hkv + kv) * D + lane * PER;
            let mut dot = 0.0f32;
            let mut i = 0;
            while i < PER {
                dot += qv[i] * unsafe { ld(k_cache, kb + i) };
                i += 1;
            }
            let s = warp_sum(dot);
            let m_new = m.max(s);
            let corr = libm::expf(m - m_new);
            let p = libm::expf(s - m_new);
            l = l * corr + p;
            let mut i = 0;
            while i < PER {
                acc[i] = acc[i] * corr + p * unsafe { ld(v_cache, kb + i) };
                i += 1;
            }
            m = m_new;
            j += 8;
        }

        if lane == 0 {
            unsafe {
                *sm.add(w) = m;
                *sl.add(w) = l;
            }
        }
        let mut i = 0;
        while i < PER {
            unsafe { *sacc.add(w * D + lane * PER + i) = acc[i] };
            i += 1;
        }
        thread::sync_threads();

        // Thread tid now owns output dim d = tid.
        let d = tid;
        let mut mm = -1.0e30f32;
        let mut k = 0;
        while k < 8 {
            mm = mm.max(unsafe { *sm.add(k) });
            k += 1;
        }
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        let mut k = 0;
        while k < 8 {
            let f = libm::expf(unsafe { *sm.add(k) } - mm);
            den += unsafe { *sl.add(k) } * f;
            num += unsafe { *sacc.add(k * D + d) } * f;
            k += 1;
        }
        let o = rb(num / den);
        let g = unsafe { ld(qkv, t * ld_qkv as usize + head * 2 * D + D + d) };
        unsafe { st(out, t * h * D + head * D + d, o * rb(sigmoid(g))) };
    }

    /// Split-K decode attention, phase 2: merge the splits of each (token, head), normalize,
    /// apply the output gate. grid = (n_decode, H), block = (256).
    #[kernel]
    pub unsafe fn attn_decode_combine(
        part_acc: *const f32,
        part_ml: *const f32,
        n_splits: u32,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        n_heads: u32,
    ) {
        const D: usize = 256;
        let t = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let h = n_heads as usize;
        let ns = n_splits as usize;
        let base = (t * h + head) * ns;
        let mut mm = -1.0e30f32;
        let mut s = 0usize;
        while s < ns {
            mm = mm.max(unsafe { *part_ml.add((base + s) * 2) });
            s += 1;
        }
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        let mut s = 0usize;
        while s < ns {
            let f = libm::expf(unsafe { *part_ml.add((base + s) * 2) } - mm);
            den += unsafe { *part_ml.add((base + s) * 2 + 1) } * f;
            num += unsafe { *part_acc.add((base + s) * D + d) } * f;
            s += 1;
        }
        let o = rb(num / den);
        let g = unsafe { ld(qkv, t * ld_qkv as usize + head * 2 * D + D + d) };
        unsafe { st(out, t * h * D + head * D + d, o * rb(sigmoid(g))) };
    }

    /// Depthwise causal conv1d (kernel 4) + SiLU over the first `conv_dim` columns of `x`
    /// (row stride `ld_x`), carrying the last 3 inputs per channel in the sequence's state
    /// slot: `state[slot][c * 3 + j]`. Sequence `s` owns rows `seq_start[s]..+seq_len[s]`.
    /// grid = (ceil(conv_dim / 256), S), block = (256).
    #[kernel]
    pub unsafe fn conv1d_silu(
        x: *const u16,
        ld_x: u32,
        state_base: *mut u16,
        w: *const u16,
        out: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if c >= conv_dim as usize {
            return;
        }
        let sq = thread::blockIdx_y() as usize;
        let cd = conv_dim as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n_tokens = unsafe { *seq_len.add(sq) };
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let w0 = unsafe { ld(w, c * 4) };
        let w1 = unsafe { ld(w, c * 4 + 1) };
        let w2 = unsafe { ld(w, c * 4 + 2) };
        let w3 = unsafe { ld(w, c * 4 + 3) };
        let mut s0 = unsafe { ld(state, c * 3) };
        let mut s1 = unsafe { ld(state, c * 3 + 1) };
        let mut s2 = unsafe { ld(state, c * 3 + 2) };
        let mut t = start;
        while t < start + n_tokens as usize {
            let xt = unsafe { ld(x, t * ld_x as usize + c) };
            let acc = w0 * s0 + w1 * s1 + w2 * s2 + w3 * xt;
            unsafe { st(out, t * cd + c, silu(rb(acc))) };
            s0 = s1;
            s1 = s2;
            s2 = xt;
            t += 1;
        }
        unsafe {
            st(state, c * 3, s0);
            st(state, c * 3 + 1, s1);
            st(state, c * 3 + 2, s2);
        }
    }

    /// Gated delta rule, token-recurrent form, fused with the gated RMSNorm epilogue.
    ///
    /// One block per (value head `h`, sequence) (grid = (Hv, S)), block = (512): thread `(r, c)` with
    /// `c = tid % 128` (value column) and `r = tid / 128` keeps rows `32r..32r+32` of column
    /// `c` of the f32 state `S[h] (Dk = 128 x Dv = 128)` in registers.
    ///
    /// Per token: `S *= exp(g)`, `delta = beta * (v - S^T k)`, `S += k delta^T`, `o = S^T q`,
    /// then `out = rmsnorm(o) * w * silu(z)`. q and k are L2-normalized; q is scaled by
    /// `1/sqrt(Dk)`. `conv` rows are `[q: Hk*Dk][k: Hk*Dk][v: Hv*Dv]`; `proj` rows hold z at
    /// `z_off`, b at `b_off`, a at `a_off`.
    #[kernel]
    pub unsafe fn gdn_recurrent(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const f32,
        state_base: *mut f32,
        out: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        seq_off: u32,
        n_k_heads: u32,
        n_v_heads: u32,
        eps: f32,
        sigmoid_gate: u32,
    ) {
        const DK: usize = 128;
        const DV: usize = 128;
        const R: usize = 32;
        static mut QS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut RED1: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut RED2: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut SC: SharedArray<f32, 8> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<f32>();
        let ks = (&raw mut KS).cast::<f32>();
        let red1 = (&raw mut RED1).cast::<f32>();
        let red2 = (&raw mut RED2).cast::<f32>();
        let sc = (&raw mut SC).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let c = tid % DV;
        let r = tid / DV;
        let lane = tid % 32;
        let warp_id = tid / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let cd = conv_dim as usize;
        let ldp = ld_proj as usize;
        let key_dim = hk * DK;

        let sq = (thread::blockIdx_y() + seq_off) as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n_tokens = unsafe { *seq_len.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * hv * DK * DV) };
        let sbase = h * DK * DV;
        let mut s = [0.0f32; R];
        let mut i = 0;
        while i < R {
            s[i] = unsafe { *state.add(sbase + (r * R + i) * DV + c) };
            i += 1;
        }
        let neg_a = -libm::expf(unsafe { *a_log.add(h) });
        let dtb = unsafe { *dt_bias.add(h) };
        let nw = unsafe { *norm_w.add(c) };
        let q_scale = 1.0 / libm::sqrtf(DK as f32);

        let mut t = start;
        while t < start + n_tokens {
            let crow = t * cd;
            if tid < DK {
                unsafe { *qs.add(tid) = ld(conv, crow + kh * DK + tid) };
            } else if tid < 2 * DK {
                let k = tid - DK;
                unsafe { *ks.add(k) = ld(conv, crow + key_dim + kh * DK + k) };
            }
            thread::sync_threads();
            if warp_id < 2 {
                let src = if warp_id == 0 { qs } else { ks };
                let mut ss = 0.0f32;
                let mut i = 0;
                while i < 4 {
                    let v = unsafe { *src.add(lane * 4 + i) };
                    ss += v * v;
                    i += 1;
                }
                let tot = warp_sum(ss);
                if lane == 0 {
                    unsafe { *sc.add(warp_id) = 1.0 / libm::sqrtf(tot + 1.0e-6) };
                }
            } else if tid == 64 {
                let a = unsafe { ld(proj, t * ldp + a_off as usize + h) };
                let b = unsafe { ld(proj, t * ldp + b_off as usize + h) };
                let g = neg_a * softplus(a + dtb);
                unsafe {
                    *sc.add(2) = libm::expf(g);
                    *sc.add(3) = rb(sigmoid(b));
                }
            }
            thread::sync_threads();
            let inv_q = unsafe { *sc.add(0) } * q_scale;
            let inv_k = unsafe { *sc.add(1) };
            let decay = unsafe { *sc.add(2) };
            let beta = unsafe { *sc.add(3) };

            let mut part = 0.0f32;
            let mut i = 0;
            while i < R {
                s[i] *= decay;
                part += s[i] * unsafe { *ks.add(r * R + i) } * inv_k;
                i += 1;
            }
            unsafe { *red1.add(r * DV + c) = part };
            thread::sync_threads();
            let kv_mem = unsafe {
                *red1.add(c) + *red1.add(DV + c) + *red1.add(2 * DV + c) + *red1.add(3 * DV + c)
            };
            let v = unsafe { ld(conv, crow + 2 * key_dim + h * DV + c) };
            let delta = (v - kv_mem) * beta;
            let mut po = 0.0f32;
            let mut i = 0;
            while i < R {
                s[i] += unsafe { *ks.add(r * R + i) } * inv_k * delta;
                po += s[i] * unsafe { *qs.add(r * R + i) } * inv_q;
                i += 1;
            }
            unsafe { *red2.add(r * DV + c) = po };
            thread::sync_threads();

            if r == 0 {
                let o = rb(unsafe {
                    *red2.add(c) + *red2.add(DV + c) + *red2.add(2 * DV + c) + *red2.add(3 * DV + c)
                });
                let ss = warp_sum(o * o);
                if lane == 0 {
                    unsafe { *red1.add(warp_id) = ss };
                }
                unsafe { *red2.add(c) = o };
            }
            thread::sync_threads();
            if r == 0 {
                let var = unsafe { *red1.add(0) + *red1.add(1) + *red1.add(2) + *red1.add(3) }
                    / DV as f32;
                let o = unsafe { *red2.add(c) };
                let y = nw * rb(o / libm::sqrtf(var + eps));
                let z = unsafe { ld(proj, t * ldp + z_off as usize + h * DV + c) };
                unsafe { st(out, t * hv * DV + h * DV + c, y * gate_act(z, sigmoid_gate)) };
            }
            thread::sync_threads();
            t += 1;
        }

        let mut i = 0;
        while i < R {
            unsafe { *state.add(sbase + (r * R + i) * DV + c) = s[i] };
            i += 1;
        }
    }

    /// Gated delta rule for sequences that feed exactly one token this step (decode).
    ///
    /// grid = (Hv, n_decode), block = (128): thread `j` owns column `j` of the head's f32
    /// state, all 128 rows in registers, so `S^T k` and `S^T q` are thread-local sums and
    /// the state is read and written exactly once. Same math and epilogue as `gdn_recurrent`.
    #[kernel]
    pub unsafe fn gdn_decode(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        norm_w: *const f32,
        state_base: *mut f32,
        out: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        seq_slot_out: *const u32,
        n_k_heads: u32,
        n_v_heads: u32,
        eps: f32,
        sigmoid_gate: u32,
    ) {
        const D: usize = 128;
        static mut QS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut KS: SharedArray<f32, 128> = SharedArray::UNINIT;
        static mut R1: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut R2: SharedArray<f32, 4> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<f32>();
        let ks = (&raw mut KS).cast::<f32>();
        let r1 = (&raw mut R1).cast::<f32>();
        let r2 = (&raw mut R2).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let sq = thread::blockIdx_y() as usize;
        let j = thread::threadIdx_x() as usize;
        let lane = j % 32;
        let w = j / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let t0 = unsafe { *seq_start.add(sq) } as usize;
        let n_tok = unsafe { *seq_len.add(sq) } as usize;
        let slot = unsafe { *seq_slot.add(sq) } as usize;
        let slot_out = unsafe { *seq_slot_out.add(sq) } as usize;
        let sp = unsafe { state_base.add((slot * hv + h) * D * D + j) };
        let sp_out = unsafe { state_base.add((slot_out * hv + h) * D * D + j) };

        // Issue all state loads first: 128 independent coalesced loads per thread.
        let mut s = [0.0f32; 128];
        #[unroll]
        for i in 0..128 {
            s[i] = unsafe { *sp.add(i * D) };
        }

        // Tokens of this sequence are processed in order with the state in registers (the
        // same per-token arithmetic for 1 token or for a speculative run of several).
        let mut tt = 0usize;
        while tt < n_tok {
            let t = t0 + tt;
            let crow = t * conv_dim as usize;
            let key_dim = hk * D;
            let qj = unsafe { ld(conv, crow + kh * D + j) };
            let kj = unsafe { ld(conv, crow + key_dim + kh * D + j) };
            let vj = unsafe { ld(conv, crow + 2 * key_dim + h * D + j) };
            let prow = t * ld_proj as usize;
            let a = unsafe { ld(proj, prow + a_off as usize + h) };
            let b = unsafe { ld(proj, prow + b_off as usize + h) };
            let z = unsafe { ld(proj, prow + z_off as usize + h * D + j) };
            let qq = warp_sum(qj * qj);
            let kk = warp_sum(kj * kj);
            unsafe {
                *qs.add(j) = qj;
                *ks.add(j) = kj;
            }
            if lane == 0 {
                unsafe {
                    *r1.add(w) = qq;
                    *r1.add(4 + w) = kk;
                }
            }
            thread::sync_threads();
            let inv_q = 1.0
                / libm::sqrtf(unsafe { *r1.add(0) + *r1.add(1) + *r1.add(2) + *r1.add(3) } + 1.0e-6)
                / libm::sqrtf(D as f32);
            let inv_k = 1.0 / libm::sqrtf(unsafe { *r1.add(4) + *r1.add(5) + *r1.add(6) + *r1.add(7) } + 1.0e-6);
            let g = -libm::expf(unsafe { *a_log.add(h) }) * softplus(a + unsafe { *dt_bias.add(h) });
            let decay = libm::expf(g);
            let beta = rb(sigmoid(b));

            let mut kv = 0.0f32;
            #[unroll]
            for i in 0..128 {
                s[i] *= decay;
                kv += s[i] * unsafe { *ks.add(i) };
            }
            let delta = (vj - kv * inv_k) * beta;
            let dk = delta * inv_k;
            let mut o = 0.0f32;
            #[unroll]
            for i in 0..128 {
                s[i] += unsafe { *ks.add(i) } * dk;
                o += s[i] * unsafe { *qs.add(i) };
            }

            let ob = rb(o * inv_q);
            let oo = warp_sum(ob * ob);
            if lane == 0 {
                unsafe { *r2.add(w) = oo };
            }
            thread::sync_threads();
            let var = unsafe { *r2.add(0) + *r2.add(1) + *r2.add(2) + *r2.add(3) } / D as f32;
            let y = unsafe { *norm_w.add(j) } * rb(ob / libm::sqrtf(var + eps));
            unsafe { st(out, t * hv * D + h * D + j, y * gate_act(z, sigmoid_gate)) };
            // qs/ks/r1/r2 are rewritten by the next token.
            thread::sync_threads();
            tt += 1;
        }
        #[unroll]
        for i in 0..128 {
            unsafe { *sp_out.add(i * D) = s[i] };
        }

    }

    // ------------------------------------------------------------------------------------
    // Prefill path: token-parallel conv, chunked gated delta rule, tiled attention.
    // ------------------------------------------------------------------------------------

    /// Token-parallel causal conv1d + SiLU: thread (c, t) reads its 4-tap window from the
    /// projection rows of its own sequence, or from the slot's carried state before the
    /// chunk start. State is advanced afterwards by `conv_state_update`.
    /// grid = (ceil(conv_dim / 256), T), block = (256).
    #[kernel]
    pub unsafe fn conv1d_tok(
        x: *const u16,
        ld_x: u32,
        state_base: *const u16,
        w: *const u16,
        out: *mut u16,
        tok_seq: *const u32,
        seq_start: *const u32,
        seq_slot: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let cd = conv_dim as usize;
        if c >= cd {
            return;
        }
        let t = thread::blockIdx_y() as usize;
        let sq = unsafe { *tok_seq.add(t) } as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let i = t - start;
        let ldx = ld_x as usize;
        let mut acc = 0.0f32;
        let mut j = 0usize;
        while j < 4 {
            // window element j holds input at local index i - 3 + j
            let v = if i + j >= 3 {
                unsafe { ld(x, (start + i + j - 3) * ldx + c) }
            } else {
                unsafe { ld(state, c * 3 + i + j) }
            };
            acc += unsafe { ld(w, c * 4 + j) } * v;
            j += 1;
        }
        unsafe { st(out, t * cd + c, silu(rb(acc))) };
    }

    /// Advance each sequence's conv state to its last 3 inputs. grid = (ceil(cd/256), S).
    #[kernel]
    pub unsafe fn conv_state_update(
        x: *const u16,
        ld_x: u32,
        state_base: *mut u16,
        seq_start: *const u32,
        seq_len: *const u32,
        seq_slot: *const u32,
        seq_slot_out: *const u32,
        conv_dim: u32,
    ) {
        let c = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        let cd = conv_dim as usize;
        if c >= cd {
            return;
        }
        let sq = thread::blockIdx_y() as usize;
        let start = unsafe { *seq_start.add(sq) } as usize;
        let n = unsafe { *seq_len.add(sq) } as usize;
        let state = unsafe { state_base.add(*seq_slot.add(sq) as usize * cd * 3) };
        let old = unsafe { [*state.add(c * 3), *state.add(c * 3 + 1), *state.add(c * 3 + 2)] };
        let state = unsafe { state_base.add(*seq_slot_out.add(sq) as usize * cd * 3) };
        let mut j = 0usize;
        while j < 3 {
            // new state j = input at local index n - 3 + j
            let v = if n + j >= 3 {
                unsafe { *x.add((start + n + j - 3) * ld_x as usize + c) }
            } else {
                old[n + j]
            };
            unsafe { *state.add(c * 3 + j) = v };
            j += 1;
        }
    }

    /// Chunked gated delta rule, phase 1 (fully parallel): for one 32-token chunk and one
    /// value head, build everything the sequential scan needs.
    ///
    /// With `Kn`, `Qn` the L2-normalized keys/queries (q also scaled by 1/sqrt(Dk)), `g` the
    /// log decays and `G = cumsum(g)` within the chunk:
    ///   A[i][j]  = beta_i exp(G_i - G_j) Kn_i.Kn_j            (j < i)
    ///   X        = (I + A)^-1                                (forward substitution)
    ///   U        = X (beta V)              W  = X (beta exp(G) Kn)
    ///   Qp       = Qn exp(G)               Kp = Kn exp(G_last - G)
    ///   intra    = exp(G_i - G_j) Qn_i.Kn_j                 (j <= i)
    ///   decay    = exp(G_last)
    /// grid = (n_chunks, Hv), block = (128).
    #[kernel]
    pub unsafe fn gdn_chunk_prep(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        chunk_start: *const u32,
        chunk_len: *const u32,
        u_out: *mut f32,
        w_out: *mut f32,
        qp_out: *mut f32,
        kp_out: *mut f32,
        intra_out: *mut f32,
        decay_out: *mut f32,
        n_k_heads: u32,
        n_v_heads: u32,
    ) {
        const C: usize = 32;
        const D: usize = 128;
        static mut KN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut QN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut AM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut XM: SharedArray<f32, 1024> = SharedArray::UNINIT;
        static mut GB: SharedArray<f32, 96> = SharedArray::UNINIT;
        let kn = (&raw mut KN).cast::<f32>();
        let qn = (&raw mut QN).cast::<f32>();
        let am = (&raw mut AM).cast::<f32>();
        let xm = (&raw mut XM).cast::<f32>();
        let gb = (&raw mut GB).cast::<f32>(); // [0..32) gamma, [32..64) beta, [64..96) scratch

        let ci = thread::blockIdx_x() as usize;
        let h = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let lane = d % 32;
        let wid = d / 32;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let cd = conv_dim as usize;
        let key_dim = hk * D;
        let c0 = unsafe { *chunk_start.add(ci) } as usize;
        let len = unsafe { *chunk_len.add(ci) } as usize;

        // Load raw q/k columns; rows past the chunk end are zero.
        let mut r = 0usize;
        while r < C {
            let (kv, qv) = if r < len {
                let row = (c0 + r) * cd;
                unsafe { (ld(conv, row + key_dim + kh * D + d), ld(conv, row + kh * D + d)) }
            } else {
                (0.0, 0.0)
            };
            unsafe {
                *kn.add(r * D + d) = kv;
                *qn.add(r * D + d) = qv;
            }
            r += 1;
        }
        if d < C {
            let (g, beta) = if d < len {
                let prow = (c0 + d) * ld_proj as usize;
                let a = unsafe { ld(proj, prow + a_off as usize + h) };
                let b = unsafe { ld(proj, prow + b_off as usize + h) };
                let g = -libm::expf(unsafe { *a_log.add(h) }) * softplus(a + unsafe { *dt_bias.add(h) });
                (g, rb(sigmoid(b)))
            } else {
                (0.0, 0.0)
            };
            unsafe {
                *gb.add(64 + d) = g;
                *gb.add(32 + d) = beta;
            }
        }
        thread::sync_threads();

        // L2-normalize rows (warp w handles rows 8w..8w+8); q also gets 1/sqrt(D).
        let mut rr = 0usize;
        while rr < 8 {
            let row = wid * 8 + rr;
            let mut sk = 0.0f32;
            let mut sq = 0.0f32;
            let mut e = 0usize;
            while e < 4 {
                let k = unsafe { *kn.add(row * D + lane * 4 + e) };
                let q = unsafe { *qn.add(row * D + lane * 4 + e) };
                sk += k * k;
                sq += q * q;
                e += 1;
            }
            let ik = 1.0 / libm::sqrtf(warp_sum(sk) + 1.0e-6);
            let iq = 1.0 / libm::sqrtf(warp_sum(sq) + 1.0e-6) / libm::sqrtf(D as f32);
            let mut e = 0usize;
            while e < 4 {
                unsafe {
                    *kn.add(row * D + lane * 4 + e) *= ik;
                    *qn.add(row * D + lane * 4 + e) *= iq;
                }
                e += 1;
            }
            rr += 1;
        }
        if d == 0 {
            let mut acc = 0.0f32;
            let mut i = 0usize;
            while i < C {
                acc += unsafe { *gb.add(64 + i) };
                unsafe { *gb.add(i) = acc };
                i += 1;
            }
        }
        thread::sync_threads();

        // A (strict lower) and intra (lower incl. diagonal). Warp w owns rows 8w..8w+8, lane
        // owns column j = lane; the dim loop is skewed by j so lanes hit distinct banks.
        let ib = ci * hv + h;
        let j = lane;
        let mut e = 0usize;
        while e < 8 {
            let i = wid * 8 + e;
            let idx = i * C + j;
            let mut a = 0.0f32;
            let mut q = 0.0f32;
            if j <= i {
                let f = libm::expf(unsafe { *gb.add(i) - *gb.add(j) });
                let mut kk = 0.0f32;
                let mut qk = 0.0f32;
                let mut x = 0usize;
                while x < D {
                    let xx = (x + j) % D;
                    let kj = unsafe { *kn.add(j * D + xx) };
                    kk += unsafe { *kn.add(i * D + xx) } * kj;
                    qk += unsafe { *qn.add(i * D + xx) } * kj;
                    x += 1;
                }
                if j < i {
                    a = unsafe { *gb.add(32 + i) } * f * kk;
                }
                q = f * qk;
            }
            unsafe {
                *am.add(idx) = a;
                *xm.add(idx) = if i == j { 1.0 } else { 0.0 };
                *intra_out.add(ib * C * C + idx) = q;
            }
            e += 1;
        }
        thread::sync_threads();

        // X = (I + A)^-1, row by row: X[i][m] = -A[i][m] - sum_{m<j<i} A[i][j] X[j][m].
        let mut i = 1usize;
        while i < C {
            if d < i {
                let m = d;
                let mut v = -unsafe { *am.add(i * C + m) };
                let mut j = m + 1;
                while j < i {
                    v -= unsafe { *am.add(i * C + j) * *xm.add(j * C + m) };
                    j += 1;
                }
                unsafe { *xm.add(i * C + m) = v };
            }
            thread::sync_threads();
            i += 1;
        }

        // Per column d: U = X (beta V), W = X (beta e^G Kn), plus decayed q/k rows.
        let mut vb = [0.0f32; 32];
        let mut kb = [0.0f32; 32];
        let glast = unsafe { *gb.add(C - 1) };
        #[unroll]
        for m in 0..32 {
            let beta = unsafe { *gb.add(32 + m) };
            let v = if m < len { unsafe { ld(conv, (c0 + m) * cd + 2 * key_dim + h * D + d) } } else { 0.0 };
            vb[m] = beta * v;
            kb[m] = beta * libm::expf(unsafe { *gb.add(m) }) * unsafe { *kn.add(m * D + d) };
        }
        let mut i = 0usize;
        while i < len {
            let mut u = 0.0f32;
            let mut w = 0.0f32;
            #[unroll]
            for m in 0..32 {
                let x = unsafe { *xm.add(i * C + m) };
                u += x * vb[m];
                w += x * kb[m];
            }
            let row = ((c0 + i) * hv + h) * D + d;
            let gi = unsafe { *gb.add(i) };
            unsafe {
                *u_out.add(row) = u;
                *w_out.add(row) = w;
                *qp_out.add(row) = *qn.add(i * D + d) * libm::expf(gi);
                *kp_out.add(row) = *kn.add(i * D + d) * libm::expf(glast - gi);
            }
            i += 1;
        }
        if d == 0 {
            unsafe { *decay_out.add(ib) = libm::expf(glast) };
        }
    }

    /// Chunked gated delta rule, phase 2: sequential over one sequence's chunks, state in
    /// registers (thread `j` owns state column `j`). Per chunk:
    ///   Vn = U - W S;  O = Qp S + intra Vn;  S = decay S + Kp^T Vn.
    /// Writes bf16 `o` (pre-norm) rows. grid = (Hv, n_prefill_seqs), block = (128).
    #[kernel]
    pub unsafe fn gdn_chunk_scan(
        u_in: *const f32,
        w_in: *const f32,
        qp_in: *const f32,
        kp_in: *const f32,
        intra_in: *const f32,
        decay_in: *const f32,
        chunk_start: *const u32,
        chunk_len: *const u32,
        seq_chunk0: *const u32,
        seq_nchunk: *const u32,
        seq_slot: *const u32,
        seq_off: u32,
        state_base: *mut f32,
        o_out: *mut u16,
        n_v_heads: u32,
    ) {
        const C: usize = 32;
        const D: usize = 128;
        static mut BUF: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut VN: SharedArray<f32, 4096> = SharedArray::UNINIT;
        static mut IN: SharedArray<f32, 1024> = SharedArray::UNINIT;
        let buf = (&raw mut BUF).cast::<f32>();
        let vn = (&raw mut VN).cast::<f32>();
        let intra = (&raw mut IN).cast::<f32>();

        let h = thread::blockIdx_x() as usize;
        let p = thread::blockIdx_y() as usize; // index among prefill sequences
        let j = thread::threadIdx_x() as usize;
        let hv = n_v_heads as usize;
        let slot = unsafe { *seq_slot.add(p + seq_off as usize) } as usize;
        let sp = unsafe { state_base.add((slot * hv + h) * D * D + j) };
        let mut s = [0.0f32; 128];
        #[unroll]
        for k in 0..128 {
            s[k] = unsafe { *sp.add(k * D) };
        }
        let c_first = unsafe { *seq_chunk0.add(p) } as usize;
        let n_chunks = unsafe { *seq_nchunk.add(p) } as usize;

        let mut cc = 0usize;
        while cc < n_chunks {
            let ci = c_first + cc;
            let c0 = unsafe { *chunk_start.add(ci) } as usize;
            let len = unsafe { *chunk_len.add(ci) } as usize;
            let ib = ci * hv + h;
            // W rows -> buf, intra -> smem
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *w_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            let mut e = 0usize;
            while e < 8 {
                unsafe { *intra.add(j * 8 + e) = *intra_in.add(ib * C * C + j * 8 + e) };
                e += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                let mut a = [0.0f32; 4];
                #[unroll]
                for k in 0..128 {
                    a[k % 4] += unsafe { *buf.add(i * D + k) } * s[k];
                }
                let acc = unsafe { *u_in.add(((c0 + i) * hv + h) * D + j) } - ((a[0] + a[1]) + (a[2] + a[3]));
                unsafe { *vn.add(i * D + j) = acc };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *qp_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                let mut a = [0.0f32; 4];
                #[unroll]
                for k in 0..128 {
                    a[k % 4] += unsafe { *buf.add(i * D + k) } * s[k];
                }
                let mut o = (a[0] + a[1]) + (a[2] + a[3]);
                let mut m = 0usize;
                while m <= i {
                    o += unsafe { *intra.add(i * C + m) * *vn.add(m * D + j) };
                    m += 1;
                }
                unsafe { st(o_out, ((c0 + i) * hv + h) * D + j, o) };
                i += 1;
            }
            thread::sync_threads();
            let mut i = 0usize;
            while i < len {
                unsafe { *buf.add(i * D + j) = *kp_in.add(((c0 + i) * hv + h) * D + j) };
                i += 1;
            }
            thread::sync_threads();
            let dec = unsafe { *decay_in.add(ib) };
            #[unroll]
            for k in 0..128 {
                s[k] *= dec;
            }
            let mut i = 0usize;
            while i < len {
                let v = unsafe { *vn.add(i * D + j) };
                #[unroll]
                for k in 0..128 {
                    s[k] += unsafe { *buf.add(i * D + k) } * v;
                }
                i += 1;
            }
            thread::sync_threads();
            cc += 1;
        }
        #[unroll]
        for k in 0..128 {
            unsafe { *sp.add(k * D) = s[k] };
        }
    }

    /// Gated RMSNorm epilogue for prefill tokens: `out = rmsnorm(o) * w * silu(z)` per
    /// (token, value head). Tokens of decode sequences (`tok_seq < n_decode`) are skipped:
    /// `gdn_decode` already wrote them. grid = (T, Hv), block = (128).
    #[kernel]
    pub unsafe fn gated_norm(
        o_in: *const u16,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        norm_w: *const f32,
        out: *mut u16,
        tok_seq: *const u32,
        n_decode: u32,
        n_v_heads: u32,
        eps: f32,
        sigmoid_gate: u32,
    ) {
        const D: usize = 128;
        static mut R: SharedArray<f32, 4> = SharedArray::UNINIT;
        let red = (&raw mut R).cast::<f32>();
        let t = thread::blockIdx_x() as usize;
        if unsafe { *tok_seq.add(t) } < n_decode {
            return;
        }
        let h = thread::blockIdx_y() as usize;
        let j = thread::threadIdx_x() as usize;
        let hv = n_v_heads as usize;
        let o = unsafe { ld(o_in, (t * hv + h) * D + j) };
        let ss = warp_sum(o * o);
        if j % 32 == 0 {
            unsafe { *red.add(j / 32) = ss };
        }
        thread::sync_threads();
        let var = unsafe { *red.add(0) + *red.add(1) + *red.add(2) + *red.add(3) } / D as f32;
        let y = unsafe { *norm_w.add(j) } * rb(o / libm::sqrtf(var + eps));
        let z = unsafe { ld(proj, t * ld_proj as usize + z_off as usize + h * D + j) };
        unsafe { st(out, t * hv * D + h * D + j, y * gate_act(z, sigmoid_gate)) };
    }

    /// Tiled causal GQA prefill attention with the output gate fused. One block per
    /// (tile of up to 16 query rows from one sequence, query head); K/V are streamed in
    /// 32-key tiles through shared memory and reused by all 16 queries.
    /// Thread `d` owns output dim `d` for all 16 queries. grid = (n_tiles, H), block = (256).
    #[kernel]
    pub unsafe fn attention_tile(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        tile_start: *const u32,
        tile_len: *const u32,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
    ) {
        const D: usize = 256;
        const QT: usize = 16;
        const KT: usize = 32;
        static mut QS: SharedArray<u16, 4096> = SharedArray::UNINIT;
        static mut KS: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut VS: SharedArray<u16, 8192> = SharedArray::UNINIT;
        static mut SS: SharedArray<f32, 512> = SharedArray::UNINIT;
        static mut CORR: SharedArray<f32, 16> = SharedArray::UNINIT;
        static mut LSUM: SharedArray<f32, 16> = SharedArray::UNINIT;
        let qs = (&raw mut QS).cast::<u16>();
        let ks = (&raw mut KS).cast::<u16>();
        let vs = (&raw mut VS).cast::<u16>();
        let ss = (&raw mut SS).cast::<f32>();
        let corr = (&raw mut CORR).cast::<f32>();
        let lsum = (&raw mut LSUM).cast::<f32>();

        let tile = thread::blockIdx_x() as usize;
        let head = thread::blockIdx_y() as usize;
        let d = thread::threadIdx_x() as usize;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let kv = head / (h / hkv);
        let t0 = unsafe { *tile_start.add(tile) } as usize;
        let nq = unsafe { *tile_len.add(tile) } as usize;
        let pos0 = unsafe { *tok_pos.add(t0) } as usize;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t0) as usize * max_blocks as usize) };
        let n_keys = pos0 + nq; // keys 0..=last query position

        let mut r = 0usize;
        while r < QT {
            unsafe { *qs.add(r * D + d) = if r < nq { *q.add(((t0 + r) * h + head) * D + d) } else { 0 } };
            r += 1;
        }
        // Row state (m, l) lives in thread r < 16.
        let mut m_row = -1.0e30f32;
        let mut l_row = 0.0f32;
        let mut acc = [0.0f32; 16];

        let mut kb = 0usize;
        while kb < n_keys {
            let nk = (n_keys - kb).min(KT);
            // Load K/V tile: 32 keys x 256 dims; thread d loads dim d of each key.
            let mut k = 0usize;
            while k < KT {
                let (kvv, vvv) = if k < nk {
                    let j = kb + k;
                    let blk = unsafe { *table.add(j / bs) } as usize;
                    let off = ((blk * bs + j % bs) * hkv + kv) * D + d;
                    unsafe { (*k_cache.add(off), *v_cache.add(off)) }
                } else {
                    (0, 0)
                };
                unsafe {
                    *ks.add(k * D + d) = kvv;
                    *vs.add(k * D + d) = vvv;
                }
                k += 1;
            }
            thread::sync_threads();
            // Scores: warp w owns query rows 2w and 2w+1, lane owns key `lane`. The dim loop is
            // skewed by the lane so the 32 lanes hit 32 different banks of the K tile.
            let lane = d % 32;
            let q0 = (d / 32) * 2;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut x = 0usize;
            while x < D {
                let xx = (x + 2 * lane) % D;
                let kv = bf16_to_f32(unsafe { *ks.add(lane * D + xx) });
                s0 += bf16_to_f32(unsafe { *qs.add(q0 * D + xx) }) * kv;
                s1 += bf16_to_f32(unsafe { *qs.add((q0 + 1) * D + xx) }) * kv;
                x += 1;
            }
            let valid = |qi: usize| qi < nq && lane < nk && kb + lane <= pos0 + qi;
            unsafe {
                *ss.add(q0 * KT + lane) = if valid(q0) { s0 * scale } else { -1.0e30 };
                *ss.add((q0 + 1) * KT + lane) = if valid(q0 + 1) { s1 * scale } else { -1.0e30 };
            }
            thread::sync_threads();
            // Online softmax per query row.
            if d < QT {
                let mut mx = m_row;
                let mut k = 0usize;
                while k < KT {
                    mx = mx.max(unsafe { *ss.add(d * KT + k) });
                    k += 1;
                }
                let c = libm::expf(m_row - mx);
                let mut l = l_row * c;
                let mut k = 0usize;
                while k < KT {
                    let sv = unsafe { *ss.add(d * KT + k) };
                    let pv = if sv <= -1.0e29 { 0.0 } else { libm::expf(sv - mx) };
                    unsafe { *ss.add(d * KT + k) = pv };
                    l += pv;
                    k += 1;
                }
                m_row = mx;
                l_row = l;
                unsafe { *corr.add(d) = c };
            }
            thread::sync_threads();
            #[unroll]
            for qi in 0..16 {
                let mut a = acc[qi] * unsafe { *corr.add(qi) };
                let mut k = 0usize;
                while k < nk {
                    a += unsafe { *ss.add(qi * KT + k) } * bf16_to_f32(unsafe { *vs.add(k * D + d) });
                    k += 1;
                }
                acc[qi] = a;
            }
            thread::sync_threads();
            kb += KT;
        }
        if d < QT {
            unsafe { *lsum.add(d) = l_row };
        }
        thread::sync_threads();
        #[unroll]
        for qi in 0..16 {
            if qi < nq {
                let t = t0 + qi;
                let o = rb(acc[qi] / unsafe { *lsum.add(qi) });
                let g = unsafe { ld(qkv, t * ld_qkv as usize + head * 2 * D + D + d) };
                unsafe { st(out, t * h * D + head * D + d, o * rb(sigmoid(g))) };
            }
        }
    }

    // ====================================================================================
    // Tensor-core prefill attention (mma.sync m16n8k16 bf16 -> f32; sm_80+, incl. sm_120/121)
    // ====================================================================================

    /// FlashAttention-2-style causal GQA prefill attention on tensor cores, gate fused.
    ///
    /// One block per (query tile of up to `16 * QT` tokens of one sequence, KV head), with
    /// `QT = blockDim / (32 * group)` sub-tiles of 16 tokens. The block has one warp per (sub-tile,
    /// query head of the KV group), so each K/V tile loaded into shared memory is used by every
    /// query head that shares it and by `QT * 16` tokens: at long context this kernel is bound by
    /// K/V traffic, which a larger QT divides. Warp `w` owns the 16 x 256 output rows of head
    /// `kv * group + w % group` for tokens `16 * (w / group)..+16` of the tile: `S = Q K^T` and `O += P V` run as m16n8k16 MMAs with operands
    /// fed by `ldmatrix` from XOR-swizzled shared tiles; K/V tiles of 32 keys arrive by `cp.async`.
    /// Softmax is online (exp2 with the scale folded in); P is rounded to bf16 for the PV MMA.
    ///
    /// Dynamic shared memory: `(QT * group * 16 + nbuf * 2 * 32) * 256 * 2` bytes (64 KiB for
    /// group 4, QT 1, one K/V buffer pair). grid = (n_tiles, n_kv), block = (32 * group * QT).
    /// A group wider than a block can hold is split: `n_kv` is then `kv_share` times the
    /// stored KV heads, each of its groups reading the keys and values of head `kv / kv_share`.
    #[kernel]
    pub unsafe fn attention_tc(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        qkv: *const u16,
        ld_qkv: u32,
        out: *mut u16,
        tile_start: *const u32,
        tile_len: *const u32,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
        nbuf: u32,
        kv8: u32,
        kv_share: u32,
    ) {
        use cuda_device::DynamicSharedArray;
        use cuda_device::async_copy::{cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_all, cp_async_wait_group};
        use cuda_device::convert::cvt_bf16x2_f32;
        use cuda_device::float::ex2_approx_ftz_f32;
        use cuda_device::wmma::{ldmatrix_x4, ldmatrix_x4_trans, mma_m16n8k16_f32_bf16};

        const D: usize = 256;
        const KT: usize = 32;
        const CH: usize = 32; // 16-byte chunks per 256-wide bf16 row
        const LOG2E: f32 = 1.442_695_f32;

        let smem = DynamicSharedArray::<u16, 128>::get();
        let tile = thread::blockIdx_x() as usize;
        let kv = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let nthreads = thread::blockDim_x() as usize;
        let warp_id = tid / 32;
        let lane = tid % 32;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let grp = h / hkv;
        // `n_kv` counts `kv_share` groups of query heads per stored KV head.
        let (hkv_r, kv_r) = (hkv / kv_share as usize, kv / kv_share as usize);
        let nwarps = nthreads / 32;
        let head = kv * grp + warp_id % grp;
        // First token (within the tile) of this warp's 16-token sub-tile.
        let qoff = (warp_id / grp) * 16;
        let t0 = unsafe { *tile_start.add(tile) } as usize;
        let nq = unsafe { *tile_len.add(tile) } as usize;
        let pos0 = unsafe { *tok_pos.add(t0) } as usize;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t0) as usize * max_blocks as usize) };
        let n_keys = pos0 + nq;

        let qs = smem;
        // K/V tile buffers: one pair, or two so that the next tile loads while this one is used.
        let ks0 = unsafe { smem.add(nwarps * 16 * D) };
        let vs0 = unsafe { ks0.add(KT * D) };
        let ks1 = unsafe { vs0.add(KT * D) };
        let vs1 = unsafe { ks1.add(KT * D) };
        let piped = nbuf == 2;
        // 8-bit cache: the tile is requested as stored into a staging area after the first
        // buffer pair and expanded to bf16 when it is used, so the next tile's fetch overlaps
        // this one's MMAs with one bf16 pair.
        let kv8 = kv8 != 0;
        let stage = ks1 as *mut u8;
        if kv8 {
            unsafe {
                kv_tile8_cp_async(stage, stage.add(KT * KV8_ROW), k_cache as *const u8, v_cache as *const u8, table, bs, hkv_r, kv_r, 0, n_keys, 16, tid, nthreads);
                kv_tile8_cp_async(stage.add(16 * KV8_ROW), stage.add((KT + 16) * KV8_ROW), k_cache as *const u8, v_cache as *const u8, table, bs, hkv_r, kv_r, 16, n_keys, 16, tid, nthreads);
                cp_async_commit_group();
            }
        }

        // Q tile: nwarps * 16 rows (warp w -> rows 16w..16w+16 = its sub-tile's tokens, its head).
        let qrows = nwarps * 16;
        let mut i = tid;
        while i < qrows * CH {
            let row = i / CH;
            let c = i % CH;
            let w = row / 16;
            let r = (w / grp) * 16 + row % 16;
            let dst = unsafe { qs.add(sw256(row, c)) } as *mut u32;
            if r < nq {
                let src = unsafe { q.add(((t0 + r) * h + kv * grp + w % grp) * D + c * 8) } as *const u32;
                unsafe { cp_async_cg_16(dst, src) };
            } else {
                unsafe { cp_async_cg_zfill_16(dst, q as *const u8, 0) };
            }
            i += nthreads;
        }

        // Fragment coordinates.
        let g = lane / 4; // row within the 8-row half
        let tq = lane % 4; // column pair
        let lm = lane / 8; // which of the 4 ldmatrix sub-matrices this lane addresses
        let lr = lane % 8; // row within that sub-matrix
        let row_a = qoff + g; // token index (within the tile) of accumulator regs 0,1
        let row_b = qoff + g + 8; // token index of accumulator regs 2,3

        let mut acc = [[0.0f32; 4]; 32];
        let mut m_a = -1.0e30f32;
        let mut m_b = -1.0e30f32;
        let mut l_a = 0.0f32;
        let mut l_b = 0.0f32;
        let sl2 = scale * LOG2E;

        // K/V tile: KT keys x 256 dims each, zero-filled past the end, as two 16-key halves
        // (each inside one KV page). With two buffers, tile i + 1 is requested before tile i is
        // waited for, so its fetch overlaps the MMAs of tile i: one block scans the whole
        // context serially and is otherwise bound by one fetch latency per tile.
        if piped {
            unsafe {
                kv_tile_cp_async(ks0, vs0, k_cache, v_cache, table, bs, hkv_r, kv_r, 0, n_keys, 16, tid, nthreads);
                kv_tile_cp_async(ks0.add(16 * D), vs0.add(16 * D), k_cache, v_cache, table, bs, hkv_r, kv_r, 16, n_keys, 16, tid, nthreads);
                cp_async_commit_group();
            }
        }
        let mut kb = 0usize;
        let mut cur = 0usize;
        while kb < n_keys {
            let ks = if cur == 0 { ks0 } else { ks1 };
            let vs = if cur == 0 { vs0 } else { vs1 };
            if kv8 {
                unsafe { cp_async_wait_all() };
                thread::sync_threads();
                unsafe { kv_tile8_unpack(stage, ks, vs, KT, tid, nthreads) };
                thread::sync_threads();
                if kb + KT < n_keys {
                    unsafe {
                        kv_tile8_cp_async(stage, stage.add(KT * KV8_ROW), k_cache as *const u8, v_cache as *const u8, table, bs, hkv_r, kv_r, kb + KT, n_keys, 16, tid, nthreads);
                        kv_tile8_cp_async(stage.add(16 * KV8_ROW), stage.add((KT + 16) * KV8_ROW), k_cache as *const u8, v_cache as *const u8, table, bs, hkv_r, kv_r, kb + KT + 16, n_keys, 16, tid, nthreads);
                        cp_async_commit_group();
                    }
                }
            } else if piped {
                let nx = kb + KT;
                if nx < n_keys {
                    let kn = if cur == 0 { ks1 } else { ks0 };
                    let vn = if cur == 0 { vs1 } else { vs0 };
                    unsafe {
                        kv_tile_cp_async(kn, vn, k_cache, v_cache, table, bs, hkv_r, kv_r, nx, n_keys, 16, tid, nthreads);
                        kv_tile_cp_async(kn.add(16 * D), vn.add(16 * D), k_cache, v_cache, table, bs, hkv_r, kv_r, nx + 16, n_keys, 16, tid, nthreads);
                        cp_async_commit_group();
                        cp_async_wait_group(1);
                    }
                } else {
                    unsafe { cp_async_wait_all() };
                }
            } else {
                unsafe {
                    kv_tile_cp_async(ks, vs, k_cache, v_cache, table, bs, hkv_r, kv_r, kb, n_keys, 16, tid, nthreads);
                    kv_tile_cp_async(ks.add(16 * D), vs.add(16 * D), k_cache, v_cache, table, bs, hkv_r, kv_r, kb + 16, n_keys, 16, tid, nthreads);
                    cp_async_commit_group();
                    cp_async_wait_all();
                }
            }
            thread::sync_threads();

            // S = Q K^T for this warp's 16 rows x 32 keys (4 n-tiles of 8 keys).
            let mut sacc = [[0.0f32; 4]; 4];
            #[unroll]
            for kk in 0..16 {
                let arow = warp_id * 16 + lr + 8 * (lm & 1);
                let a = unsafe { ldmatrix_x4(qs.add(sw256(arow, 2 * kk + (lm >> 1))) as *const u32) };
                #[unroll]
                for np in 0..2 {
                    let krow = np * 16 + 8 * (lm >> 1) + lr;
                    let b = unsafe { ldmatrix_x4(ks.add(sw256(krow, 2 * kk + (lm & 1))) as *const u32) };
                    sacc[2 * np] = unsafe { mma_m16n8k16_f32_bf16(sacc[2 * np], a, [b[0], b[1]]) };
                    sacc[2 * np + 1] = unsafe { mma_m16n8k16_f32_bf16(sacc[2 * np + 1], a, [b[2], b[3]]) };
                }
            }

            // Causal mask + online softmax (base 2, scale folded in). Rows: token g and g + 8.
            let mut mx_a = -1.0e30f32;
            let mut mx_b = -1.0e30f32;
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for j in 0..4 {
                    let row = if j < 2 { row_a } else { row_b };
                    let key = kb + nt * 8 + 2 * tq + (j & 1);
                    let valid = row < nq && key <= pos0 + row;
                    let v = if valid { sacc[nt][j] * sl2 } else { -1.0e30 };
                    sacc[nt][j] = v;
                    if j < 2 {
                        mx_a = mx_a.max(v);
                    } else {
                        mx_b = mx_b.max(v);
                    }
                }
            }
            mx_a = mx_a.max(warp::shuffle_xor_f32_sync(FULL, mx_a, 1));
            mx_a = mx_a.max(warp::shuffle_xor_f32_sync(FULL, mx_a, 2));
            mx_b = mx_b.max(warp::shuffle_xor_f32_sync(FULL, mx_b, 1));
            mx_b = mx_b.max(warp::shuffle_xor_f32_sync(FULL, mx_b, 2));
            let mn_a = m_a.max(mx_a);
            let mn_b = m_b.max(mx_b);
            let corr_a = ex2_approx_ftz_f32(m_a - mn_a);
            let corr_b = ex2_approx_ftz_f32(m_b - mn_b);
            m_a = mn_a;
            m_b = mn_b;
            let mut ps_a = 0.0f32;
            let mut ps_b = 0.0f32;
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for j in 0..4 {
                    let v = sacc[nt][j];
                    let mref = if j < 2 { mn_a } else { mn_b };
                    let p = if v <= -1.0e29 { 0.0 } else { ex2_approx_ftz_f32(v - mref) };
                    sacc[nt][j] = p;
                    if j < 2 {
                        ps_a += p;
                    } else {
                        ps_b += p;
                    }
                }
            }
            l_a = l_a * corr_a + ps_a;
            l_b = l_b * corr_b + ps_b;
            #[unroll]
            for nt in 0..32 {
                acc[nt][0] *= corr_a;
                acc[nt][1] *= corr_a;
                acc[nt][2] *= corr_b;
                acc[nt][3] *= corr_b;
            }

            // O += P V: P (16 x 32 keys) as two k-steps of A fragments, V via ldmatrix.trans.
            #[unroll]
            for kk in 0..2 {
                let pa = [
                    cvt_bf16x2_f32(sacc[2 * kk][0], sacc[2 * kk][1]),
                    cvt_bf16x2_f32(sacc[2 * kk][2], sacc[2 * kk][3]),
                    cvt_bf16x2_f32(sacc[2 * kk + 1][0], sacc[2 * kk + 1][1]),
                    cvt_bf16x2_f32(sacc[2 * kk + 1][2], sacc[2 * kk + 1][3]),
                ];
                #[unroll]
                for nd in 0..16 {
                    let vrow = kk * 16 + 8 * (lm & 1) + lr;
                    let b = unsafe { ldmatrix_x4_trans(vs.add(sw256(vrow, 2 * nd + (lm >> 1))) as *const u32) };
                    acc[2 * nd] = unsafe { mma_m16n8k16_f32_bf16(acc[2 * nd], pa, [b[0], b[1]]) };
                    acc[2 * nd + 1] = unsafe { mma_m16n8k16_f32_bf16(acc[2 * nd + 1], pa, [b[2], b[3]]) };
                }
            }
            thread::sync_threads();
            kb += KT;
            if piped {
                cur ^= 1;
            }
        }

        // Finish the row sums across the quad, normalize, round, gate, store pairs.
        l_a += warp::shuffle_xor_f32_sync(FULL, l_a, 1);
        l_a += warp::shuffle_xor_f32_sync(FULL, l_a, 2);
        l_b += warp::shuffle_xor_f32_sync(FULL, l_b, 1);
        l_b += warp::shuffle_xor_f32_sync(FULL, l_b, 2);
        let inv_a = 1.0 / l_a;
        let inv_b = 1.0 / l_b;
        let ldq = ld_qkv as usize;
        #[unroll]
        for half in 0..2 {
            let row = if half == 0 { row_a } else { row_b };
            if row < nq {
                let t = t0 + row;
                let inv = if half == 0 { inv_a } else { inv_b };
                let gbase = t * ldq + head * 2 * D + D;
                let obase = t * h * D + head * D;
                #[unroll]
                for nt in 0..32 {
                    let d = nt * 8 + 2 * tq;
                    let o0 = rb(acc[nt][2 * half] * inv);
                    let o1 = rb(acc[nt][2 * half + 1] * inv);
                    let g0 = rb(sigmoid(unsafe { ld(qkv, gbase + d) }));
                    let g1 = rb(sigmoid(unsafe { ld(qkv, gbase + d + 1) }));
                    let packed = (f32_to_bf16_rne(o0 * g0) as u32) | ((f32_to_bf16_rne(o1 * g1) as u32) << 16);
                    unsafe { *(out.add(obase + d) as *mut u32) = packed };
                }
            }
        }
    }

    /// Split-K decode attention on tensor cores, phase 1 (pairs with `attn_decode_combine`).
    ///
    /// One block per (decode tile, KV head, slice). A tile is a few rows of one sequence at
    /// consecutive positions (the token to verify plus its drafts). A warp holds 8 query slots
    /// (slot `j` = tile row `j / group`, head `j % group`, so `8 / group` rows per warp; `group`
    /// must divide 8), and the block has as many warps as the longest tile needs
    /// (`blockDim / 32`). One pass over the keys then serves the whole tile: each K/V tile is
    /// fetched once per block, not once per row and query head.
    ///
    /// Slice `s` covers the keys `bounds[s]..bounds[s + 1]` (absolute positions, multiples of
    /// the KV page size, the same table for every row), scanned in 16-key tiles. Per tile:
    /// `S^T = K Q^T` (16 keys x 8 slots, 16 MMAs), the online softmax (base 2, scale folded
    /// in) down each slot's column, `P^T` parked in shared memory as bf16, and
    /// `O^T += V^T P` (256 dims x 8 slots, 16 MMAs). A row's partials therefore depend only on
    /// its own position and keys -- not on the batch, the other rows of its tile, or how many
    /// slices the step runs (slices past a row's keys are written as empty). Output: the
    /// unnormalized `(m, l, acc[256])` of each row and head, `m` in natural log units.
    ///
    /// Dynamic shared memory: `(8 * warps + 2 * 16) * 512 + 256 * warps` bytes.
    /// grid = (n_tiles, n_kv, n_splits <= len(bounds) - 1), block = (32 * warps).
    #[kernel]
    pub unsafe fn attn_decode_tq(
        q: *const u16,
        k_cache: *const u16,
        v_cache: *const u16,
        tile_start: *const u32,
        tile_len: *const u32,
        bounds: *const u32,
        tok_pos: *const u32,
        tok_seq: *const u32,
        block_tables: *const u32,
        max_blocks: u32,
        block_size: u32,
        n_heads: u32,
        n_kv: u32,
        scale: f32,
        part_acc: *mut f32,
        part_ml: *mut f32,
        kv8: u32,
        kv_share: u32,
    ) {
        use cuda_device::DynamicSharedArray;
        use cuda_device::async_copy::{cp_async_cg_16, cp_async_cg_zfill_16, cp_async_commit_group, cp_async_wait_all};
        use cuda_device::float::ex2_approx_ftz_f32;
        use cuda_device::wmma::{ldmatrix_x4, ldmatrix_x4_trans, mma_m16n8k16_f32_bf16};

        const D: usize = 256;
        const KT: usize = 16;
        const CH: usize = 32;
        const LOG2E: f32 = 1.442_695_f32;
        const LN2: f32 = 0.693_147_2_f32;

        let smem = DynamicSharedArray::<u16, 128>::get();
        let tile = thread::blockIdx_x() as usize;
        let kv = thread::blockIdx_y() as usize;
        let split = thread::blockIdx_z() as usize;
        let n_splits = thread::gridDim_z() as usize;
        let tid = thread::threadIdx_x() as usize;
        let nthreads = thread::blockDim_x() as usize;
        let warp_id = tid / 32;
        let lane = tid % 32;
        let h = n_heads as usize;
        let hkv = n_kv as usize;
        let grp = h / hkv;
        // `n_kv` counts `kv_share` groups of query heads per stored KV head.
        let (hkv_r, kv_r) = (hkv / kv_share as usize, kv / kv_share as usize);
        let nwarps = nthreads / 32;
        let rpw = 8 / grp;
        let r0 = warp_id * rpw;
        let t0 = unsafe { *tile_start.add(tile) } as usize;
        let nq = unsafe { *tile_len.add(tile) } as usize;
        let pos0 = unsafe { *tok_pos.add(t0) } as usize;
        let bs = block_size as usize;
        let table = unsafe { block_tables.add(*tok_seq.add(t0) as usize * max_blocks as usize) };
        let n_keys = if nq == 0 { 0 } else { pos0 + nq };
        let k0 = (unsafe { *bounds.add(split) } as usize).min(n_keys);
        let k1 = (unsafe { *bounds.add(split + 1) } as usize).min(n_keys);

        // Shared: Q [8 * warps][256] | K [16][256] | V [16][256] | P^T [warps][8][16].
        let qs = smem;
        let ks = unsafe { smem.add(nwarps * 8 * D) };
        let vs = unsafe { ks.add(KT * D) };
        let ps = unsafe { vs.add(KT * D).add(warp_id * 128) };

        // Q: row 8w + j = tile row w * rpw + j / grp, head j % grp.
        let mut i = if k0 < k1 { tid } else { usize::MAX };
        while i < nwarps * 8 * CH {
            let row = i / CH;
            let c = i % CH;
            let r = (row / 8) * rpw + (row % 8) / grp;
            let dst = unsafe { qs.add(sw256(row, c)) } as *mut u32;
            if r < nq {
                let src = unsafe { q.add(((t0 + r) * h + kv * grp + (row % 8) % grp) * D + c * 8) } as *const u32;
                unsafe { cp_async_cg_16(dst, src) };
            } else {
                unsafe { cp_async_cg_zfill_16(dst, q as *const u8, 0) };
            }
            i += nthreads;
        }

        // Fragment coordinates. C regs: [0] (m g, n 2tq) [1] (m g, n 2tq+1) [2] (m g+8, n 2tq)
        // [3] (m g+8, n 2tq+1). For S^T, m = key and n = slot; for O^T, m = dim and n = slot.
        let g = lane / 4;
        let tq = lane % 4;
        let lm = lane / 8;
        let lr = lane % 8;
        let slot0 = 2 * tq;
        let slot1 = 2 * tq + 1;
        let row0 = r0 + slot0 / grp;
        let row1 = r0 + slot1 / grp;

        let mut acc = [[0.0f32; 4]; 16];
        let mut m0 = -1.0e30f32;
        let mut m1 = -1.0e30f32;
        let mut l0 = 0.0f32;
        let mut l1 = 0.0f32;
        let sl2 = scale * LOG2E;

        let mut kb = k0;
        while kb < k1 {
            if kv8 != 0 {
                // 8-bit cache: stage the tile as stored (after the P^T rows), then expand it.
                let stage = unsafe { vs.add(KT * D).add(nwarps * 128) } as *mut u8;
                unsafe {
                    kv_tile8_cp_async(stage, stage.add(KT * KV8_ROW), k_cache as *const u8, v_cache as *const u8, table, bs, hkv_r, kv_r, kb, k1, KT, tid, nthreads);
                    cp_async_commit_group();
                    cp_async_wait_all();
                }
                thread::sync_threads();
                unsafe { kv_tile8_unpack(stage, ks, vs, KT, tid, nthreads) };
            } else {
                unsafe {
                    kv_tile_cp_async(ks, vs, k_cache, v_cache, table, bs, hkv_r, kv_r, kb, k1, KT, tid, nthreads);
                    cp_async_commit_group();
                    cp_async_wait_all();
                }
            }
            thread::sync_threads();

            // S^T = K Q^T: two interleaved accumulation chains (even and odd k-steps).
            let mut se = [0.0f32; 4];
            let mut so = [0.0f32; 4];
            #[unroll]
            for kk in 0..8 {
                let a = unsafe { ldmatrix_x4(ks.add(sw256(lr + 8 * (lm & 1), 4 * kk + (lm >> 1))) as *const u32) };
                let b = unsafe { ldmatrix_x4(qs.add(sw256(warp_id * 8 + lr, 4 * kk + (lm & 1))) as *const u32) };
                se = unsafe { mma_m16n8k16_f32_bf16(se, a, [b[0], b[1]]) };
                let a = unsafe { ldmatrix_x4(ks.add(sw256(lr + 8 * (lm & 1), 4 * kk + 2 + (lm >> 1))) as *const u32) };
                let b = unsafe { ldmatrix_x4(qs.add(sw256(warp_id * 8 + lr, 4 * kk + 2 + (lm & 1))) as *const u32) };
                so = unsafe { mma_m16n8k16_f32_bf16(so, a, [b[0], b[1]]) };
            }

            // Causal mask, scale, and each slot's max over the 16 keys.
            let mut v = [0.0f32; 4];
            #[unroll]
            for j in 0..4 {
                let key = kb + g + 8 * (j >> 1);
                let row = if j & 1 == 0 { row0 } else { row1 };
                let valid = row < nq && key < k1 && key <= pos0 + row;
                v[j] = if valid { (se[j] + so[j]) * sl2 } else { -1.0e30 };
            }
            let mut mx0 = v[0].max(v[2]);
            let mut mx1 = v[1].max(v[3]);
            #[unroll]
            for sh in 0..3 {
                mx0 = mx0.max(warp::shuffle_xor_f32_sync(FULL, mx0, 4 << sh));
                mx1 = mx1.max(warp::shuffle_xor_f32_sync(FULL, mx1, 4 << sh));
            }
            let mn0 = m0.max(mx0);
            let mn1 = m1.max(mx1);
            let corr0 = ex2_approx_ftz_f32(m0 - mn0);
            let corr1 = ex2_approx_ftz_f32(m1 - mn1);
            m0 = mn0;
            m1 = mn1;
            let mut p = [0.0f32; 4];
            #[unroll]
            for j in 0..4 {
                let mref = if j & 1 == 0 { mn0 } else { mn1 };
                p[j] = if v[j] <= -1.0e29 { 0.0 } else { ex2_approx_ftz_f32(v[j] - mref) };
            }
            // P^T [slot][key] to shared memory, rounded to bf16 (the PV operand).
            unsafe {
                *ps.add(slot0 * 16 + g) = f32_to_bf16_rne(p[0]);
                *ps.add(slot1 * 16 + g) = f32_to_bf16_rne(p[1]);
                *ps.add(slot0 * 16 + g + 8) = f32_to_bf16_rne(p[2]);
                *ps.add(slot1 * 16 + g + 8) = f32_to_bf16_rne(p[3]);
            }
            // Slot sums over the 16 keys; the shuffles also order the warp's P stores before
            // its P loads below.
            let mut s0 = p[0] + p[2];
            let mut s1 = p[1] + p[3];
            #[unroll]
            for sh in 0..3 {
                s0 += warp::shuffle_xor_f32_sync(FULL, s0, 4 << sh);
                s1 += warp::shuffle_xor_f32_sync(FULL, s1, 4 << sh);
            }
            l0 = l0 * corr0 + s0;
            l1 = l1 * corr1 + s1;
            #[unroll]
            for nd in 0..16 {
                acc[nd][0] *= corr0;
                acc[nd][1] *= corr1;
                acc[nd][2] *= corr0;
                acc[nd][3] *= corr1;
            }

            // O^T += V^T P: B = P [16 keys][slot g], A = V^T [16 dims][16 keys] via ldmatrix.trans.
            let pb = unsafe { [lds32(ps, g * 16 + 2 * tq), lds32(ps, g * 16 + 8 + 2 * tq)] };
            #[unroll]
            for nd in 0..16 {
                let a = unsafe { ldmatrix_x4_trans(vs.add(sw256(8 * (lm >> 1) + lr, 2 * nd + (lm & 1))) as *const u32) };
                acc[nd] = unsafe { mma_m16n8k16_f32_bf16(acc[nd], a, pb) };
            }
            thread::sync_threads();
            kb += KT;
        }

        // Partials of this lane's two slots: dims 16 nd + g and 16 nd + g + 8.
        #[unroll]
        for half in 0..2 {
            let row = if half == 0 { row0 } else { row1 };
            if row < nq {
                let slot = if half == 0 { slot0 } else { slot1 };
                let out = ((t0 + row) * h + kv * grp + slot % grp) * n_splits + split;
                #[unroll]
                for nd in 0..16 {
                    unsafe {
                        *part_acc.add(out * D + nd * 16 + g) = acc[nd][half];
                        *part_acc.add(out * D + nd * 16 + g + 8) = acc[nd][2 + half];
                    }
                }
                if g == 0 {
                    let m = if half == 0 { m0 } else { m1 };
                    let l = if half == 0 { l0 } else { l1 };
                    unsafe {
                        *part_ml.add(out * 2) = if m <= -1.0e29 { -1.0e30 } else { m * LN2 };
                        *part_ml.add(out * 2 + 1) = l;
                    }
                }
            }
        }
    }

    // ====================================================================================
    // Tensor-core chunked gated delta rule (prefill). Chunk = 64 tokens, Dk = Dv = 128.
    // Operands bf16, accumulation f32, recurrent state f32 (as FLA does).
    //
    //   gdn_tc_prep  (chunk, head)            : Qn, Kn (L2-normed), G = cumsum(g),
    //                                           X = (I + A)^-1, U = X(bV), W = X(b e^G Kn)
    //   gdn_tc_scan  (v-block, head, sequence): per chunk, snapshot h, then
    //                                           Vn = U - W h;  h = e^{G_last} h + Kn^T (e^{G_last-G} Vn)
    //   gdn_tc_out   (chunk, head)            : o = e^G (Qn h) + mask(e^{Gi-Gj} Qn Kn^T) Vn,
    //                                           fused gated RMSNorm
    //
    // Scratch (per chunk-head ib = chunk * Hv + h): Qn, Kn, U, W [64][128] bf16, G [64] f32,
    // hT [128 v][128 k] bf16 (state before the chunk), VnT [128 v][64 c] bf16.
    // All global->shared copies go through `tile_g2s`, which keeps 16 loads in flight.
    // ====================================================================================

    /// grid = (n_chunks, Hv), block = (128), dynamic shared memory = 70672 bytes.
    #[kernel]
    pub unsafe fn gdn_tc_prep(
        conv: *const u16,
        conv_dim: u32,
        proj: *const u16,
        ld_proj: u32,
        b_off: u32,
        a_off: u32,
        a_log: *const f32,
        dt_bias: *const f32,
        chunk_start: *const u32,
        chunk_len: *const u32,
        qn_out: *mut u16,
        kn_out: *mut u16,
        u_out: *mut u16,
        w_out: *mut u16,
        g_out: *mut f32,
        n_k_heads: u32,
        n_v_heads: u32,
    ) {
        const C: usize = 64;
        const D: usize = 128;
        const LK: usize = D + 8;
        const LC: usize = C + 8;
        const LA: usize = C + 1;
        let sm = DynamicSharedArray::<u8, 16>::get_raw();
        let ks = sm.cast::<u16>(); // [64][136] bf16: raw then normalized keys
        let am = unsafe { sm.add(17408) }.cast::<f32>(); // R1 [64][65] f32: A
        let xbv = unsafe { sm.add(17408) }.cast::<u16>(); // R1 (later) [64][72] bf16: X diag(beta)
        let xbk = unsafe { sm.add(26624) }.cast::<u16>(); // R1 (later) [64][72] bf16: X diag(beta e^G)
        let xm = unsafe { sm.add(35840) }.cast::<f32>(); // [64][65] f32: X = (I + A)^-1
        let r3 = unsafe { sm.add(52480) }.cast::<u16>(); // R3 [64][136] bf16: raw q, later raw v
        let gs = unsafe { sm.add(69888) }.cast::<f32>(); // [64] G
        let bsm = unsafe { sm.add(70144) }.cast::<f32>(); // [64] beta
        let fac = unsafe { sm.add(70400) }.cast::<f32>(); // [64] beta e^G
        let tot = unsafe { sm.add(70656) }.cast::<f32>(); // [4] scan carry

        let ci = thread::blockIdx_x() as usize;
        let h = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let g = lane / 4;
        let t = lane % 4;
        let hk = n_k_heads as usize;
        let hv = n_v_heads as usize;
        let kh = h / (hv / hk);
        let cd = conv_dim as usize;
        let key_dim = hk * D;
        let c0 = unsafe { *chunk_start.add(ci) } as usize;
        let len = unsafe { *chunk_len.add(ci) } as usize;
        let ib = ci * hv + h;
        let row0 = unsafe { conv.add(c0 * cd) };

        // 1. Stage raw k (R0) and q (R3) rows; rows past the chunk end are zeroed. Warps 0-1
        //    compute g and beta for the 64 rows and an in-warp inclusive scan of g.
        unsafe {
            tile_cp_async(ks, LK, row0.add(key_dim + kh * D), cd, len, D, tid, 128);
            tile_cp_async(r3, LK, row0.add(kh * D), cd, len, D, tid, 128);
            cuda_device::async_copy::cp_async_commit_group();
        }
        let mut idx = len * 64 + tid;
        while idx < C * 64 {
            let r = idx / 64;
            let cp = idx % 64;
            unsafe {
                *(ks.add(r * LK + 2 * cp) as *mut u32) = 0;
                *(r3.add(r * LK + 2 * cp) as *mut u32) = 0;
            }
            idx += 128;
        }
        if tid < C {
            let (gv, beta) = if tid < len {
                let prow = (c0 + tid) * ld_proj as usize;
                let a = unsafe { ld(proj, prow + a_off as usize + h) };
                let b = unsafe { ld(proj, prow + b_off as usize + h) };
                (-libm::expf(unsafe { *a_log.add(h) }) * softplus(a + unsafe { *dt_bias.add(h) }), rb(sigmoid(b)))
            } else {
                (0.0, 0.0)
            };
            let mut s = gv;
            let mut off = 1u32;
            while off < 32 {
                let o = f32::from_bits(warp::shuffle_up_sync(FULL, s.to_bits(), off));
                if lane as u32 >= off {
                    s += o;
                }
                off *= 2;
            }
            unsafe {
                *gs.add(tid) = s;
                *bsm.add(tid) = beta;
            }
            if lane == 31 {
                unsafe { *tot.add(w) = s };
            }
        }
        unsafe { cuda_device::async_copy::cp_async_wait_all() };
        thread::sync_threads();

        // 2. Finish the scan (rows 32..63 add warp 0's total) and beta e^G; L2-normalize rows
        //    (warp w: rows 16w..16w+16; q also gets 1/sqrt(D)) and write Qn, Kn to global.
        if tid < C {
            let gg = if tid >= 32 { unsafe { *gs.add(tid) + *tot.add(0) } } else { unsafe { *gs.add(tid) } };
            unsafe {
                *gs.add(tid) = gg;
                *fac.add(tid) = *bsm.add(tid) * libm::expf(gg);
            }
        }
        // Four rows at a time so their shuffle reductions overlap.
        let mut rq = 0usize;
        while rq < 4 {
            let mut kv = [[0.0f32; 4]; 4];
            let mut qv = [[0.0f32; 4]; 4];
            let mut sk = [0.0f32; 4];
            let mut sq = [0.0f32; 4];
            #[unroll]
            for u in 0..4 {
                let r = 16 * w + 4 * rq + u;
                let k01 = unsafe { lds32(ks, r * LK + 4 * lane) };
                let k23 = unsafe { lds32(ks, r * LK + 4 * lane + 2) };
                let q01 = unsafe { lds32(r3, r * LK + 4 * lane) };
                let q23 = unsafe { lds32(r3, r * LK + 4 * lane + 2) };
                kv[u] = [
                    bf16_to_f32((k01 & 0xffff) as u16),
                    bf16_to_f32((k01 >> 16) as u16),
                    bf16_to_f32((k23 & 0xffff) as u16),
                    bf16_to_f32((k23 >> 16) as u16),
                ];
                qv[u] = [
                    bf16_to_f32((q01 & 0xffff) as u16),
                    bf16_to_f32((q01 >> 16) as u16),
                    bf16_to_f32((q23 & 0xffff) as u16),
                    bf16_to_f32((q23 >> 16) as u16),
                ];
                sk[u] = kv[u][0] * kv[u][0] + kv[u][1] * kv[u][1] + kv[u][2] * kv[u][2] + kv[u][3] * kv[u][3];
                sq[u] = qv[u][0] * qv[u][0] + qv[u][1] * qv[u][1] + qv[u][2] * qv[u][2] + qv[u][3] * qv[u][3];
            }
            #[unroll]
            for m in 0..5 {
                let off = 16u32 >> m;
                #[unroll]
                for u in 0..4 {
                    sk[u] += warp::shuffle_xor_f32_sync(FULL, sk[u], off);
                    sq[u] += warp::shuffle_xor_f32_sync(FULL, sq[u], off);
                }
            }
            #[unroll]
            for u in 0..4 {
                let r = 16 * w + 4 * rq + u;
                let ik = 1.0 / libm::sqrtf(sk[u] + 1.0e-6);
                let iq = 1.0 / libm::sqrtf(sq[u] + 1.0e-6) / libm::sqrtf(D as f32);
                let kn01 = pk(kv[u][0] * ik, kv[u][1] * ik);
                let kn23 = pk(kv[u][2] * ik, kv[u][3] * ik);
                let o = (ib * C + r) * D + 4 * lane;
                unsafe {
                    *(ks.add(r * LK + 4 * lane) as *mut u32) = kn01;
                    *(ks.add(r * LK + 4 * lane + 2) as *mut u32) = kn23;
                    *(kn_out.add(o) as *mut u32) = kn01;
                    *(kn_out.add(o + 2) as *mut u32) = kn23;
                    *(qn_out.add(o) as *mut u32) = pk(qv[u][0] * iq, qv[u][1] * iq);
                    *(qn_out.add(o + 2) as *mut u32) = pk(qv[u][2] * iq, qv[u][3] * iq);
                }
            }
            rq += 1;
        }
        thread::sync_threads();

        // 3. Raw V streams into R3 (queries are done) while A is computed.
        //    A = strict_lower(beta_i e^{G_i - G_j} Kn Kn^T): warp w owns rows 16w..16w+16.
        unsafe {
            tile_cp_async(r3, LK, row0.add(2 * key_dim + h * D), cd, len, D, tid, 128);
            cuda_device::async_copy::cp_async_commit_group();
        }
        let mut idx = len * 64 + tid;
        while idx < C * 64 {
            unsafe { *(r3.add((idx / 64) * LK + 2 * (idx % 64)) as *mut u32) = 0 };
            idx += 128;
        }
        let r0 = w * 16;
        let mut acc = [[0.0f32; 4]; 8];
        #[unroll]
        for kk in 0..8 {
            let a = unsafe { frag_a(ks, LK, r0, kk * 16, g, t) };
            #[unroll]
            for nt in 0..8 {
                let b = unsafe { frag_b(ks, LK, nt * 8, kk * 16, g, t) };
                acc[nt] = unsafe { mma(acc[nt], a, b) };
            }
        }
        #[unroll]
        for nt in 0..8 {
            #[unroll]
            for e in 0..4 {
                let i = r0 + g + if e >= 2 { 8 } else { 0 };
                let j = nt * 8 + 2 * t + (e & 1);
                let v = if j < i {
                    let f = unsafe { *bsm.add(i) * libm::expf(*gs.add(i) - *gs.add(j)) };
                    f * acc[nt][e]
                } else {
                    0.0
                };
                unsafe { *am.add(i * LA + j) = v };
            }
        }
        thread::sync_threads();

        // 4a. Invert the four 16x16 diagonal blocks of L = I + A: warp w takes block w, lane m
        //     keeps column m of the inverse in registers (it only depends on itself).
        if lane < 16 {
            let base = 16 * w;
            let m = lane;
            let mut x = [0.0f32; 16];
            #[unroll]
            for i in 0..16 {
                let mut s = if i == m { 1.0f32 } else { 0.0 };
                if i > m {
                    s = -unsafe { *am.add((base + i) * LA + base + m) };
                    #[unroll]
                    for j in 0..16 {
                        if j > m && j < i {
                            s -= unsafe { *am.add((base + i) * LA + base + j) } * x[j];
                        }
                    }
                }
                x[i] = s;
            }
            #[unroll]
            for i in 0..16 {
                unsafe { *xm.add((base + i) * LA + base + m) = x[i] };
            }
        }
        thread::sync_threads();

        // 4b. Off-diagonal blocks by distance dd: X_ij = -X_ii sum_{k=j}^{i-1} A_ik X_kj.
        //     Warp w < 4 - dd computes block (w + dd, w). Lane owns row rr = lane / 2 and 8
        //     columns ch..ch+8; the intermediate T is exchanged with shuffles.
        let rr = lane / 2;
        let ch = (lane % 2) * 8;
        let mut dd = 1usize;
        while dd < 4 {
            if w + dd < 4 {
                let bi = w + dd;
                let bj = w;
                let mut tv = [0.0f32; 8];
                let mut kb = bj;
                while kb < bi {
                    #[unroll]
                    for q in 0..16 {
                        let a = unsafe { *am.add((16 * bi + rr) * LA + 16 * kb + q) };
                        #[unroll]
                        for c in 0..8 {
                            tv[c] += a * unsafe { *xm.add((16 * kb + q) * LA + 16 * bj + ch + c) };
                        }
                    }
                    kb += 1;
                }
                let mut s = [0.0f32; 8];
                #[unroll]
                for q in 0..16 {
                    let xq = if q <= rr { unsafe { *xm.add((16 * bi + rr) * LA + 16 * bi + q) } } else { 0.0 };
                    let src = (2 * q + (lane % 2)) as u32;
                    #[unroll]
                    for c in 0..8 {
                        s[c] += xq * warp::shuffle_f32_sync(FULL, tv[c], src);
                    }
                }
                #[unroll]
                for c in 0..8 {
                    unsafe { *xm.add((16 * bi + rr) * LA + 16 * bj + ch + c) = -s[c] };
                }
            }
            thread::sync_threads();
            dd += 1;
        }

        // 4c. X diag(beta) and X diag(beta e^G) -> bf16 (lower triangle incl. unit diagonal),
        //     over the dead A region.
        #[unroll]
        for e in 0..32 {
            let idx = tid + 128 * e;
            let i = idx / C;
            let j = idx % C;
            let x = if j < i {
                unsafe { *xm.add(i * LA + j) }
            } else if j == i {
                1.0
            } else {
                0.0
            };
            unsafe {
                *xbv.add(i * LC + j) = f32_to_bf16_rne(x * *bsm.add(j));
                *xbk.add(i * LC + j) = f32_to_bf16_rne(x * *fac.add(j));
            }
        }
        unsafe { cuda_device::async_copy::cp_async_wait_all() };
        thread::sync_threads();

        // 5. U = (X diag beta) V, W = (X diag beta e^G) Kn: warp w owns rows 16w..16w+16,
        //    16 n-tiles over D. B fragments come straight from the row-major V / Kn tiles via
        //    ldmatrix.trans, two n-tiles per instruction.
        let lrow = (lane & 7) + 8 * ((lane >> 3) & 1);
        let lcol = 8 * (lane >> 4);
        let mut pass = 0usize;
        while pass < 2 {
            let (xb, src, dst) = if pass == 0 { (xbv, r3, u_out) } else { (xbk, ks, w_out) };
            let mut acc = [[0.0f32; 4]; 16];
            #[unroll]
            for kk in 0..4 {
                let a = unsafe { frag_a(xb, LC, r0, kk * 16, g, t) };
                #[unroll]
                for np in 0..8 {
                    let b4 = unsafe {
                        cuda_device::wmma::ldmatrix_x4_trans(src.add((kk * 16 + lrow) * LK + np * 16 + lcol) as *const u32)
                    };
                    acc[2 * np] = unsafe { mma(acc[2 * np], a, [b4[0], b4[1]]) };
                    acc[2 * np + 1] = unsafe { mma(acc[2 * np + 1], a, [b4[2], b4[3]]) };
                }
            }
            #[unroll]
            for nt in 0..16 {
                let col = nt * 8 + 2 * t;
                unsafe {
                    *(dst.add((ib * C + r0 + g) * D + col) as *mut u32) = pk(acc[nt][0], acc[nt][1]);
                    *(dst.add((ib * C + r0 + g + 8) * D + col) as *mut u32) = pk(acc[nt][2], acc[nt][3]);
                }
            }
            pass += 1;
        }
        if tid < C {
            unsafe { *g_out.add(ib * C + tid) = *gs.add(tid) };
        }
    }

    /// grid = (Dv / 32, Hv, n_prefill_seqs), block = (256), dynamic shared memory = 93696.
    /// Warp w (8 warps) keeps state rows k in 16w..16w+16 for the block's 32 value columns as
    /// MMA accumulators. Chunk i+1's W, Kn, U and G stream in with cp.async (double buffer)
    /// while chunk i is processed.
    #[kernel]
    pub unsafe fn gdn_tc_scan(
        kn_in: *const u16,
        u_in: *const u16,
        w_in: *const u16,
        g_in: *const f32,
        seq_chunk0: *const u32,
        seq_nchunk: *const u32,
        seq_slot: *const u32,
        seq_off: u32,
        state_base: *mut f32,
        ht_out: *mut u16,
        vnt_out: *mut u16,
        n_v_heads: u32,
        seq_tap_n: *const u32,
        seq_tap_slot: *const u32,
    ) {
        const C: usize = 64;
        const D: usize = 128;
        const VB: usize = 32;
        const LK: usize = D + 8;
        const LC: usize = C + 8;
        const LU: usize = VB + 8;
        const STAGE: usize = 40192;
        let sm = DynamicSharedArray::<u8, 16>::get_raw();
        let hts = unsafe { sm.add(2 * STAGE) }.cast::<u16>(); // [32 v][136 k] bf16: h^T
        let vns = unsafe { sm.add(2 * STAGE + 8704) }.cast::<u16>(); // [32 v][72 c]: (e^{Gl-G} Vn)^T

        let vb = thread::blockIdx_x() as usize;
        let h = thread::blockIdx_y() as usize;
        let p = thread::blockIdx_z() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let g = lane / 4;
        let t = lane % 4;
        let hv = n_v_heads as usize;
        let slot = unsafe { *seq_slot.add(p + seq_off as usize) } as usize;
        let sp = unsafe { state_base.add((slot * hv + h) * D * D) };
        let c_first = unsafe { *seq_chunk0.add(p) } as usize;
        let n_chunks = unsafe { *seq_nchunk.add(p) } as usize;
        // State tap: after `tap_n` chunks (0 = none) the state is also written to another
        // slot, which gives a checkpoint inside the prompt without a separate step. The
        // tapped state is the f32 register state, exactly what a prefill stopped there holds.
        let tap_n = unsafe { *seq_tap_n.add(p + seq_off as usize) } as usize;
        let tap_slot = unsafe { *seq_tap_slot.add(p + seq_off as usize) } as usize;
        let tp = unsafe { state_base.add((tap_slot * hv + h) * D * D) };
        // (b) work split: rows c in 16 (w % 4).., n-tiles 2 (w / 4)..+2
        let vr0 = 16 * (w % 4);
        let vn0 = 2 * (w / 4);

        if n_chunks > 0 {
            unsafe { gdn_scan_issue(sm, kn_in, u_in, w_in, g_in, c_first * hv + h, vb, tid) };
        }
        // state fragment (nt, e): k = 16w + g (+8 for e >= 2), v = 32vb + 8nt + 2t + (e & 1)
        let mut hs = [[0.0f32; 4]; 4];
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..4 {
                let k = 16 * w + g + if e >= 2 { 8 } else { 0 };
                let v = VB * vb + 8 * nt + 2 * t + (e & 1);
                hs[nt][e] = unsafe { *sp.add(k * D + v) };
            }
        }
        // ldmatrix.x4.trans lanes for A = Kn^T: rows c = kk*16 + (l & 7) + 8 (l >> 4), cols k0 + 8 ((l >> 3) & 1)
        let arow = (lane & 7) + 8 * (lane >> 4);
        let acol = 8 * ((lane >> 3) & 1);

        let mut cc = 0usize;
        while cc < n_chunks {
            let s = cc % 2;
            let sb = unsafe { sm.add(s * STAGE) };
            let wsm = sb.cast::<u16>();
            let kns = unsafe { sb.add(17408) }.cast::<u16>();
            let us = unsafe { sb.add(34816) }.cast::<u16>();
            let gs = unsafe { sb.add(39936) }.cast::<f32>();
            let ib = (c_first + cc) * hv + h;
            if cc + 1 < n_chunks {
                unsafe {
                    gdn_scan_issue(sm.add((1 - s) * STAGE), kn_in, u_in, w_in, g_in, (c_first + cc + 1) * hv + h, vb, tid);
                    cuda_device::async_copy::cp_async_wait_group(1);
                }
            } else {
                unsafe { cuda_device::async_copy::cp_async_wait_all() };
            }
            // (a) h^T (state before this chunk) into shared memory.
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for e in 0..4 {
                    let k = 16 * w + g + if e >= 2 { 8 } else { 0 };
                    let vl = 8 * nt + 2 * t + (e & 1);
                    unsafe { *hts.add(vl * LK + k) = f32_to_bf16_rne(hs[nt][e]) };
                }
            }
            thread::sync_threads();
            // Snapshot h^T to global, coalesced.
            #[unroll]
            for m in 0..8 {
                let idx = tid + 256 * m;
                let row = idx / 64;
                let cp = idx % 64;
                unsafe {
                    *(ht_out.add((ib * D + VB * vb + row) * D + 2 * cp) as *mut u32) = lds32(hts, row * LK + 2 * cp);
                }
            }
            let glast = unsafe { *gs.add(C - 1) };
            // (b) Vn = U - W h: 16 rows x 2 n-tiles per warp, K = 128.
            let mut vn = [[0.0f32; 4]; 2];
            #[unroll]
            for kk in 0..8 {
                let a = unsafe { frag_a(wsm, LK, vr0, kk * 16, g, t) };
                #[unroll]
                for j in 0..2 {
                    let b = unsafe { frag_b(hts, LK, (vn0 + j) * 8, kk * 16, g, t) };
                    vn[j] = unsafe { mma(vn[j], a, b) };
                }
            }
            #[unroll]
            for half in 0..2 {
                let c = vr0 + g + 8 * half;
                let f = libm::expf(glast - unsafe { *gs.add(c) });
                #[unroll]
                for j in 0..2 {
                    let vl = 8 * (vn0 + j) + 2 * t;
                    let u = unsafe { lds32(us, c * LU + vl) };
                    let x0 = bf16_to_f32((u & 0xffff) as u16) - vn[j][2 * half];
                    let x1 = bf16_to_f32((u >> 16) as u16) - vn[j][2 * half + 1];
                    let b0 = f32_to_bf16_rne(x0);
                    let b1 = f32_to_bf16_rne(x1);
                    unsafe {
                        *vnt_out.add((ib * D + VB * vb + vl) * C + c) = b0;
                        *vnt_out.add((ib * D + VB * vb + vl + 1) * C + c) = b1;
                        *vns.add(vl * LC + c) = f32_to_bf16_rne(bf16_to_f32(b0) * f);
                        *vns.add((vl + 1) * LC + c) = f32_to_bf16_rne(bf16_to_f32(b1) * f);
                    }
                }
            }
            thread::sync_threads();
            // (c) h = e^{G_last} h + Kn^T (e^{G_last - G} Vn): warp w rows k 16w..16w+16, N = 32, K = 64.
            let dec = libm::expf(glast);
            #[unroll]
            for nt in 0..4 {
                #[unroll]
                for e in 0..4 {
                    hs[nt][e] *= dec;
                }
            }
            #[unroll]
            for kk in 0..4 {
                let a = unsafe { cuda_device::wmma::ldmatrix_x4_trans(kns.add((kk * 16 + arow) * LK + 16 * w + acol) as *const u32) };
                #[unroll]
                for nt in 0..4 {
                    let b = unsafe { frag_b(vns, LC, nt * 8, kk * 16, g, t) };
                    hs[nt] = unsafe { mma(hs[nt], a, b) };
                }
            }
            thread::sync_threads();
            cc += 1;
            if cc == tap_n {
                #[unroll]
                for nt in 0..4 {
                    #[unroll]
                    for e in 0..4 {
                        let k = 16 * w + g + if e >= 2 { 8 } else { 0 };
                        let v = VB * vb + 8 * nt + 2 * t + (e & 1);
                        unsafe { *tp.add(k * D + v) = hs[nt][e] };
                    }
                }
            }
        }
        #[unroll]
        for nt in 0..4 {
            #[unroll]
            for e in 0..4 {
                let k = 16 * w + g + if e >= 2 { 8 } else { 0 };
                let v = VB * vb + 8 * nt + 2 * t + (e & 1);
                unsafe { *sp.add(k * D + v) = hs[nt][e] };
            }
        }
    }

    /// grid = (n_chunks, Hv), block = (128), dynamic shared memory = 88832.
    /// Warp w owns chunk rows 16w..16w+16 and all 128 value columns, so the gated RMSNorm
    /// over a row is a reduction across the 4 lanes of a quad.
    #[kernel]
    pub unsafe fn gdn_tc_out(
        qn_in: *const u16,
        kn_in: *const u16,
        g_in: *const f32,
        ht_in: *const u16,
        vnt_in: *const u16,
        chunk_start: *const u32,
        chunk_len: *const u32,
        proj: *const u16,
        ld_proj: u32,
        z_off: u32,
        norm_w: *const f32,
        out: *mut u16,
        n_v_heads: u32,
        eps: f32,
        sigmoid_gate: u32,
    ) {
        const C: usize = 64;
        const D: usize = 128;
        const LK: usize = D + 8;
        const LC: usize = C + 8;
        let sm = DynamicSharedArray::<u8, 16>::get_raw();
        let qs = sm.cast::<u16>(); // [64][136]
        let ks = unsafe { sm.add(17408) }.cast::<u16>(); // [64][136]
        let hts = unsafe { sm.add(34816) }.cast::<u16>(); // [128 v][136 k]
        let vns = unsafe { sm.add(69632) }.cast::<u16>(); // [128 v][72 c]
        let gs = unsafe { sm.add(88064) }.cast::<f32>(); // [64]
        let nw = unsafe { sm.add(88320) }.cast::<f32>(); // [128] norm weight

        let ci = thread::blockIdx_x() as usize;
        let h = thread::blockIdx_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let lane = tid % 32;
        let w = tid / 32;
        let g = lane / 4;
        let t = lane % 4;
        let hv = n_v_heads as usize;
        let ib = ci * hv + h;
        let c0 = unsafe { *chunk_start.add(ci) } as usize;
        let len = unsafe { *chunk_len.add(ci) } as usize;

        // Group 0: q, k, h^T, G (the first products). Group 1: Vn^T and the norm weight.
        unsafe {
            tile_cp_async(qs, LK, qn_in.add(ib * C * D), D, C, D, tid, 128);
            tile_cp_async(ks, LK, kn_in.add(ib * C * D), D, C, D, tid, 128);
            tile_cp_async(hts, LK, ht_in.add(ib * D * D), D, D, D, tid, 128);
            if tid < 16 {
                cuda_device::async_copy::cp_async_cg_16(gs.add(4 * tid) as *mut u32, g_in.add(ib * C + 4 * tid) as *const u32);
            }
            cuda_device::async_copy::cp_async_commit_group();
            tile_cp_async(vns, LC, vnt_in.add(ib * D * C), C, D, C, tid, 128);
            if tid >= 16 && tid < 48 {
                cuda_device::async_copy::cp_async_cg_16(nw.add(4 * (tid - 16)) as *mut u32, norm_w.add(4 * (tid - 16)) as *const u32);
            }
            cuda_device::async_copy::cp_async_commit_group();
        }
        // The gate values for this thread's outputs, loaded while the tiles stream in.
        let r0 = 16 * w;
        let i0 = r0 + g;
        let i1 = r0 + g + 8;
        let ldp = ld_proj as usize;
        let mut zz = [0u32; 32];
        #[unroll]
        for nt in 0..16 {
            #[unroll]
            for half in 0..2 {
                let i = if half == 0 { i0 } else { i1 };
                if i < len {
                    zz[2 * nt + half] = unsafe { lds32(proj, (c0 + i) * ldp + z_off as usize + h * D + nt * 8 + 2 * t) };
                }
            }
        }
        unsafe { cuda_device::async_copy::cp_async_wait_group(1) };
        thread::sync_threads();

        let mut o = [[0.0f32; 4]; 16];
        let mut sc = [[0.0f32; 4]; 8];
        #[unroll]
        for kk in 0..8 {
            let a = unsafe { frag_a(qs, LK, r0, kk * 16, g, t) };
            #[unroll]
            for nt in 0..16 {
                let b = unsafe { frag_b(hts, LK, nt * 8, kk * 16, g, t) };
                o[nt] = unsafe { mma(o[nt], a, b) };
            }
            #[unroll]
            for nt in 0..8 {
                let b = unsafe { frag_b(ks, LK, nt * 8, kk * 16, g, t) };
                sc[nt] = unsafe { mma(sc[nt], a, b) };
            }
        }
        let g0 = unsafe { *gs.add(i0) };
        let g1 = unsafe { *gs.add(i1) };
        let e0 = libm::expf(g0);
        let e1 = libm::expf(g1);
        #[unroll]
        for nt in 0..16 {
            o[nt][0] *= e0;
            o[nt][1] *= e0;
            o[nt][2] *= e1;
            o[nt][3] *= e1;
        }
        // P = causal mask(e^{Gi-Gj} S), straight from accumulators into A fragments.
        #[unroll]
        for nt in 0..8 {
            #[unroll]
            for e in 0..4 {
                let (i, gi) = if e >= 2 { (i1, g1) } else { (i0, g0) };
                let j = nt * 8 + 2 * t + (e & 1);
                sc[nt][e] = if j <= i { sc[nt][e] * libm::expf(gi - unsafe { *gs.add(j) }) } else { 0.0 };
            }
        }
        unsafe { cuda_device::async_copy::cp_async_wait_all() };
        thread::sync_threads();
        #[unroll]
        for kk in 0..4 {
            let a = [
                pk(sc[2 * kk][0], sc[2 * kk][1]),
                pk(sc[2 * kk][2], sc[2 * kk][3]),
                pk(sc[2 * kk + 1][0], sc[2 * kk + 1][1]),
                pk(sc[2 * kk + 1][2], sc[2 * kk + 1][3]),
            ];
            #[unroll]
            for nt in 0..16 {
                let b = unsafe { frag_b(vns, LC, nt * 8, kk * 16, g, t) };
                o[nt] = unsafe { mma(o[nt], a, b) };
            }
        }
        // Gated RMSNorm per row (the reference rounds core output to bf16 first).
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        #[unroll]
        for nt in 0..16 {
            #[unroll]
            for e in 0..4 {
                o[nt][e] = rb(o[nt][e]);
            }
            s0 += o[nt][0] * o[nt][0] + o[nt][1] * o[nt][1];
            s1 += o[nt][2] * o[nt][2] + o[nt][3] * o[nt][3];
        }
        s0 += warp::shuffle_xor_f32_sync(FULL, s0, 1);
        s0 += warp::shuffle_xor_f32_sync(FULL, s0, 2);
        s1 += warp::shuffle_xor_f32_sync(FULL, s1, 1);
        s1 += warp::shuffle_xor_f32_sync(FULL, s1, 2);
        let inv0 = 1.0 / libm::sqrtf(s0 / D as f32 + eps);
        let inv1 = 1.0 / libm::sqrtf(s1 / D as f32 + eps);
        #[unroll]
        for nt in 0..16 {
            #[unroll]
            for half in 0..2 {
                let (i, inv) = if half == 0 { (i0, inv0) } else { (i1, inv1) };
                if i < len {
                    let col = nt * 8 + 2 * t;
                    let z = zz[2 * nt + half];
                    let y0 = unsafe { *nw.add(col) } * rb(o[nt][2 * half] * inv) * gate_act(bf16_to_f32((z & 0xffff) as u16), sigmoid_gate);
                    let y1 = unsafe { *nw.add(col + 1) } * rb(o[nt][2 * half + 1] * inv) * gate_act(bf16_to_f32((z >> 16) as u16), sigmoid_gate);
                    unsafe { *(out.add((c0 + i) * hv * D + h * D + col) as *mut u32) = pk(y0, y1) };
                }
            }
        }
    }

    /// Final residual add + RMSNorm for the rows that will be sampled only:
    /// `out[s] = rmsnorm(x[rows[s]] + delta[rows[s]])`. grid = (S), block = (256).
    #[kernel]
    pub unsafe fn final_norm_rows(
        x: *const u16,
        delta: *const u16,
        w: *const u16,
        out: *mut u16,
        rows: *const u32,
        hidden: u32,
        eps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut ROWV: SharedArray<f32, 8192> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let rv = (&raw mut ROWV).cast::<f32>();
        let s = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let src = unsafe { *rows.add(s) } as usize * h;
        let bd = thread::blockDim_x() as usize;
        let mut ss = 0.0f32;
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            let v = rb(unsafe { ld(x, src + i) + ld(delta, src + i) });
            unsafe { *rv.add(i) = v };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            unsafe { st(out, s * h + i, *rv.add(i) * inv * (1.0 + ld(w, i))) };
            i += bd;
        }
    }

    /// Input rows of the multi-token-prediction (MTP) head, `2 * hidden` wide:
    /// `out[t] = [rmsnorm(table[tok], we), rmsnorm(rmsnorm(x[t] + delta[t], wf), wh)]`, where
    /// `tok` is the token that follows row t: `next[t]`, or, when its top bit is set, the
    /// token just sampled at sample index `next[t] & 0x7fff_ffff`. `wf` is the backbone's
    /// final norm; its output is rounded to bf16 as the backbone would store it.
    /// Requires hidden <= 8192. grid = (T), block = (256).
    #[kernel]
    pub unsafe fn mtp_prep(
        x: *const u16,
        delta: *const u16,
        wf: *const u16,
        wh: *const u16,
        we: *const u16,
        table: *const u16,
        slot_of: *const u32,
        next: *const u32,
        sampled: *const u32,
        out: *mut u16,
        hidden: u32,
        eps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut ROWV: SharedArray<f32, 8192> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let rv = (&raw mut ROWV).cast::<f32>();
        let t = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let src = t * h;
        let bd = thread::blockDim_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        // Backbone final norm.
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < h {
            let v = rb(unsafe { ld(x, src + i) + ld(delta, src + i) });
            unsafe { *rv.add(i) = v };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < h {
            let v = rb(unsafe { *rv.add(i) * inv * (1.0 + ld(wf, i)) });
            unsafe { *rv.add(i) = v };
            ss += v * v;
            i += bd;
        }
        // pre_fc_norm_hidden -> second half.
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = tid;
        while i < h {
            unsafe { st(out, t * 2 * h + h + i, *rv.add(i) * inv * (1.0 + ld(wh, i))) };
            i += bd;
        }
        // pre_fc_norm_embedding of the next token -> first half.
        let nx = unsafe { *next.add(t) };
        let id = if nx & 0x8000_0000 != 0 { unsafe { *sampled.add((nx & 0x7fff_ffff) as usize) } } else { nx } as usize;
        // With `slot_of` (token -> row of `table`, u32::MAX = absent) the table holds only some
        // tokens' rows; an absent token reads as zeros.
        let row = if slot_of.is_null() { id } else { (unsafe { *slot_of.add(id) }) as usize };
        let have = slot_of.is_null() || row != u32::MAX as usize;
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < h {
            let v = if have { unsafe { ld(table, row * h + i) } } else { 0.0 };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = tid;
        while i < h {
            unsafe { st(out, t * 2 * h + i, (if have { ld(table, row * h + i) } else { 0.0 }) * inv * (1.0 + ld(we, i))) };
            i += bd;
        }
    }

    /// Input rows of a chained MTP pass (drafts after the first), `2 * hidden` wide:
    /// `out[t] = [rmsnorm(table[tok[sel[t]]], we), rmsnorm(hid[sel[t]], wh)]`, where `hid` and
    /// `tok` are the previous MTP pass's normed output rows and predicted tokens.
    /// Requires hidden <= 8192. grid = (T), block = (256).
    #[kernel]
    pub unsafe fn mtp_chain_prep(
        hid: *const u16,
        tok: *const u32,
        sel: *const u32,
        wh: *const u16,
        we: *const u16,
        table: *const u16,
        slot_of: *const u32,
        out: *mut u16,
        hidden: u32,
        eps: f32,
    ) {
        static mut RED: SharedArray<f32, 32> = SharedArray::UNINIT;
        let red = (&raw mut RED).cast::<f32>();
        let t = thread::blockIdx_x() as usize;
        let h = hidden as usize;
        let bd = thread::blockDim_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let src = unsafe { *sel.add(t) } as usize;
        let id = unsafe { *tok.add(src) } as usize;
        // With `slot_of` (token -> row of `table`, u32::MAX = absent) the table holds only some
        // tokens' rows; an absent token reads as zeros.
        let row = if slot_of.is_null() { id } else { (unsafe { *slot_of.add(id) }) as usize };
        let have = slot_of.is_null() || row != u32::MAX as usize;
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < h {
            let v = unsafe { ld(hid, src * h + i) };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = tid;
        while i < h {
            unsafe { st(out, t * 2 * h + h + i, ld(hid, src * h + i) * inv * (1.0 + ld(wh, i))) };
            i += bd;
        }
        let mut ss = 0.0f32;
        let mut i = tid;
        while i < h {
            let v = if have { unsafe { ld(table, row * h + i) } } else { 0.0 };
            ss += v * v;
            i += bd;
        }
        let total = unsafe { block_sum(ss, red) };
        let inv = 1.0 / libm::sqrtf(total / hidden as f32 + eps);
        let mut i = tid;
        while i < h {
            unsafe { st(out, t * 2 * h + i, (if have { ld(table, row * h + i) } else { 0.0 }) * inv * (1.0 + ld(we, i))) };
            i += bd;
        }
    }

    /// `out[slots[b], :] = table[ids[slots[b]], :]`: (re)fill rows of a gathered copy of an
    /// embedding-shaped table: bf16 (`tq` = 0), or int8 with one bf16 magnitude per block of 32
    /// at `tscale` (the `quant_i8b` layout). grid = (number of slots), block = (256).
    #[kernel]
    pub unsafe fn gather_rows(table: *const u16, tscale: *const u16, tq: u32, ids: *const u32, slots: *const u32, out: *mut u16, hidden: u32) {
        let slot = unsafe { *slots.add(thread::blockIdx_x() as usize) } as usize;
        let id = unsafe { *ids.add(slot) } as usize;
        let h = hidden as usize;
        let mut i = thread::threadIdx_x() as usize;
        while i < h {
            if tq == 0 {
                unsafe { *out.add(slot * h + i) = *table.add(id * h + i) };
            } else {
                let e = id * h + i;
                unsafe { st(out, slot * h + i, *(table as *const i8).add(e) as f32 * bf16_to_f32(*tscale.add(e / 32)) * (1.0 / 127.0)) };
            }
            i += thread::blockDim_x() as usize;
        }
    }

    /// `tok[i] = ids[tok[i]]` for i < n: shortlist indices back to token ids.
    /// grid = (ceil(n / 256)), block = (256).
    #[kernel]
    pub unsafe fn remap_ids(tok: *mut u32, ids: *const u32, n: u32) {
        let i = (thread::blockIdx_x() * thread::blockDim_x() + thread::threadIdx_x()) as usize;
        if i < n as usize {
            unsafe { *tok.add(i) = *ids.add(*tok.add(i) as usize) };
        }
    }

    /// Row-wise argmax, phase 1: grid = (S, P) blocks of 256 threads; block p scans its 1/P of
    /// the row with 16-byte loads and writes its (value, index). Ties go to the lowest index.
    /// Requires n % 8 == 0.
    #[kernel]
    pub unsafe fn argmax_part(logits: *const u16, n: u32, part_v: *mut f32, part_i: *mut u32) {
        static mut BV: SharedArray<f32, 8> = SharedArray::UNINIT;
        static mut BI: SharedArray<u32, 8> = SharedArray::UNINIT;
        let bv = (&raw mut BV).cast::<f32>();
        let bi = (&raw mut BI).cast::<u32>();
        let row = thread::blockIdx_x() as usize;
        let p = thread::blockIdx_y() as usize;
        let np = thread::gridDim_y() as usize;
        let tid = thread::threadIdx_x() as usize;
        let nv = n as usize / 8;
        let per = nv.div_ceil(np);
        let v0 = p * per;
        let v1 = (v0 + per).min(nv);
        let base = unsafe { logits.cast::<U32x4>().add(row * nv) };
        let mut best = -3.0e38f32;
        let mut idx = u32::MAX;
        let mut v = v0 + tid;
        while v < v1 {
            let w = unsafe { *base.add(v) }.to_array();
            #[unroll]
            for e in 0..4 {
                let (a0, a1) = cvt_f32x2_bf16x2(w[e]);
                let i0 = (v * 8 + 2 * e) as u32;
                if a0 > best {
                    best = a0;
                    idx = i0;
                }
                if a1 > best {
                    best = a1;
                    idx = i0 + 1;
                }
            }
            v += 256;
        }
        let mut off = 16u32;
        while off > 0 {
            let ov = warp::shuffle_xor_f32_sync(FULL, best, off);
            let oi = warp::shuffle_xor_sync(FULL, idx, off);
            if ov > best || (ov == best && oi < idx) {
                best = ov;
                idx = oi;
            }
            off /= 2;
        }
        if tid % 32 == 0 {
            unsafe {
                *bv.add(tid / 32) = best;
                *bi.add(tid / 32) = idx;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let mut k = 1usize;
            while k < 8 {
                let v = unsafe { *bv.add(k) };
                let j = unsafe { *bi.add(k) };
                if v > best || (v == best && j < idx) {
                    best = v;
                    idx = j;
                }
                k += 1;
            }
            unsafe {
                *part_v.add(row * np + p) = best;
                *part_i.add(row * np + p) = idx;
            }
        }
    }

    /// Row-wise argmax, phase 2: reduce the P partials of each row. grid = (S), block = (32).
    #[kernel]
    pub unsafe fn argmax_final(part_v: *const f32, part_i: *const u32, np: u32, out: *mut u32) {
        let row = thread::blockIdx_x() as usize;
        let lane = thread::threadIdx_x() as usize;
        let np = np as usize;
        let mut best = -3.0e38f32;
        let mut idx = u32::MAX;
        let mut k = lane;
        while k < np {
            let v = unsafe { *part_v.add(row * np + k) };
            let j = unsafe { *part_i.add(row * np + k) };
            if v > best || (v == best && j < idx) {
                best = v;
                idx = j;
            }
            k += 32;
        }
        let mut off = 16u32;
        while off > 0 {
            let ov = warp::shuffle_xor_f32_sync(FULL, best, off);
            let oi = warp::shuffle_xor_sync(FULL, idx, off);
            if ov > best || (ov == best && oi < idx) {
                best = ov;
                idx = oi;
            }
            off /= 2;
        }
        if lane == 0 {
            unsafe { *out.add(row) = idx };
        }
    }

    /// Row-wise argmax over `[S, n]` f32 logits; grid = (S), block = (1024).
    /// Ties resolve to the lowest index.
    #[kernel]
    pub unsafe fn argmax(logits: *const u16, n: u32, out_base: *mut u32) {
        let x = unsafe { logits.add(thread::blockIdx_x() as usize * n as usize) };
        let out = unsafe { out_base.add(thread::blockIdx_x() as usize) };
        static mut BV: SharedArray<f32, 32> = SharedArray::UNINIT;
        static mut BI: SharedArray<u32, 32> = SharedArray::UNINIT;
        let bv = (&raw mut BV).cast::<f32>();
        let bi = (&raw mut BI).cast::<u32>();
        let tid = thread::threadIdx_x();
        let mut best = -3.0e38f32;
        let mut idx = u32::MAX;
        let mut i = tid;
        while i < n {
            let v = unsafe { ld(x, i as usize) };
            if v > best {
                best = v;
                idx = i;
            }
            i += thread::blockDim_x();
        }
        let mut off = 16u32;
        while off > 0 {
            let ov = warp::shuffle_xor_f32_sync(FULL, best, off);
            let oi = warp::shuffle_xor_sync(FULL, idx, off);
            if ov > best || (ov == best && oi < idx) {
                best = ov;
                idx = oi;
            }
            off /= 2;
        }
        if tid % 32 == 0 {
            unsafe {
                *bv.add((tid / 32) as usize) = best;
                *bi.add((tid / 32) as usize) = idx;
            }
        }
        thread::sync_threads();
        if tid == 0 {
            let mut k = 0usize;
            while k < thread::blockDim_x().div_ceil(32) as usize {
                let v = unsafe { *bv.add(k) };
                let j = unsafe { *bi.add(k) };
                if v > best || (v == best && j < idx) {
                    best = v;
                    idx = j;
                }
                k += 1;
            }
            unsafe { *out = idx };
        }
    }

    /// Hardware probe: an empty kernel, for measuring launch overhead.
    #[kernel]
    pub unsafe fn hw_noop(p: *const u32) {
        let _ = p;
    }

    /// Hardware probe (read bandwidth): block b reads `per` 16-byte vectors starting at vector
    /// `b * per` (threads stride by the block size) and stores each thread's xor in
    /// `out[b * blockDim + tid]`.
    #[kernel]
    pub unsafe fn hw_read(src: *const u32, per: u32, out: *mut u32) {
        let b = thread::blockIdx_x() as usize;
        let tid = thread::threadIdx_x() as usize;
        let nt = thread::blockDim_x() as usize;
        let base = unsafe { src.cast::<U32x4>().add(b * per as usize) };
        let mut acc = 0u32;
        let mut v = tid;
        while v < per as usize {
            let w = unsafe { *base.add(v) }.to_array();
            acc ^= w[0] ^ w[1] ^ w[2] ^ w[3];
            v += nt;
        }
        unsafe { *out.add(b * nt + tid) = acc };
    }

    /// Hardware probe (copy bandwidth): block b copies `per` 16-byte vectors starting at
    /// vector `b * per` from `src` to `dst`.
    #[kernel]
    pub unsafe fn hw_copy(src: *const u32, dst: *mut u32, per: u32) {
        let b = thread::blockIdx_x() as usize;
        let nt = thread::blockDim_x() as usize;
        let sv = unsafe { src.cast::<U32x4>().add(b * per as usize) };
        let dv = unsafe { dst.cast::<U32x4>().add(b * per as usize) };
        let mut v = thread::threadIdx_x() as usize;
        while v < per as usize {
            unsafe { *dv.add(v) = *sv.add(v) };
            v += nt;
        }
    }
}
