/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! RaBitQ — 1-bit-per-dim quantized base store with a random
//! orthogonal rotation up front.
//!
//! Reference: Gao & Long, *"RaBitQ: Quantizing High-Dimensional Vectors
//! with a Theoretical Error Bound for Approximate Nearest Neighbor
//! Search"*, SIGMOD 2024.
//!
//! ## Pipeline
//!
//! 1. Build a deterministic random orthogonal matrix `P ∈ R^{N×N}`
//!    (Gaussian seed → modified Gram-Schmidt). Stored alongside the
//!    sidecar so query rotation uses the same `P`.
//! 2. For each base vector `x`, compute `rotated_x = P @ x`. Store
//!    `sign(rotated_x)` as a bit vector (`N` bits = `ceil(N/8)` bytes)
//!    plus `||x||` as a single `f32` (orthogonal rotation preserves
//!    norm, so `||rotated_x|| == ||x||`).
//! 3. At query time the query is rotated once (`q' = P @ q`, kept in
//!    `f32`) and fed to the Stage-1 kernel that computes a signed sum
//!    of f32 components masked by the base vector's bit code. The
//!    result is an unbiased estimator of `<q', x'>`, which by
//!    `||q-x||² = ||q||² + ||x||² - 2<q,x>` converts to an unbiased
//!    L2-distance estimator with Hoeffding-style error bound
//!    `O(||x|| / sqrt(N))`.
//!
//! ## Status
//!
//! Reference (scalar) implementation. Correctness-focused: rotation
//! generation, encoder, save/load, distance estimator. NEON kernel and
//! integration into `compressed_index.rs` /
//! `ensure_quantized_dataset_rabitq()` land in a follow-up pass.

use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use diskann::model::InmemDataset;
use std::path::Path;
use std::sync::OnceLock;

// Disk-format magic for the RaBitQ sidecar (`.qrbq`). Bumped on
// breaking layout changes so older caches refuse to load instead of
// silently mis-decoding.
pub const RABITQ_MAGIC: u32 = 0x5152_4251; // "QRBQ" little-endian-printable

/// Per-vertex code stride in bytes. Returns the padded length of the
/// sign-bit packed code so `vld1q_u8` loads stay 16-byte aligned and
/// the per-vertex blob can be indexed by a simple `vertex_id * STRIDE`.
/// Does **not** include the per-vertex `f32` norm — that lives in a
/// separate `Vec<f32>` so the bit-dot kernel only loads code bytes.
pub const fn rabitq_code_stride(dim: usize) -> usize {
    let bits = (dim + 7) / 8;
    (bits + 15) & !15
}

/// RaBitQ-encoded base store.
///
/// Two parallel arrays:
/// - `codes`: `num_vertices × STRIDE` bytes, sign-bit-packed (LSB-first
///   per byte: bit `i` of byte `b` encodes the sign of dim `b * 8 + i`).
///   Padded per `rabitq_code_stride` so per-vertex offset arithmetic
///   stays trivial.
/// - `norms`: `num_vertices` f32s, one `||x||` per vertex (preserved
///   from the original f32 base since rotation is orthogonal).
///
/// Plus the row-major `N × N` rotation matrix used for both encoding
/// and per-query rotation at search time.
pub struct RabitQDataset<const N: usize> {
    /// Packed sign codes, `num_vertices * STRIDE` bytes, 16-byte aligned.
    pub codes: AlignedBoxWithSlice<u8>,
    /// Per-vertex `||x||` in f32. Length = `num_vertices`.
    pub norms: Vec<f32>,
    /// **Per-vertex correction** `s_x = ||x'||₁ / ||x'||₂` (= `<unit_x',
    /// sign(x')>` from the paper's Lemma). This is the exact value the
    /// asymptotic `scale = sqrt(2N/π)` approximates — replacing the
    /// global constant with the per-vertex value tightens the
    /// estimator's error bound by 2-4× on the datasets the paper
    /// benchmarks. Length = `num_vertices`.
    ///
    /// Adds one f32 per vertex (4 B); negligible vs the codes slab.
    pub s_values: Vec<f32>,
    /// Row-major `N × N` orthogonal rotation. `P @ x` is `for r in 0..N:
    /// out[r] = dot(P[r*N..(r+1)*N], x)`.
    pub rotation: AlignedBoxWithSlice<f32>,
    /// Number of vertices encoded.
    pub num_vertices: usize,
    /// Per-vertex code stride in bytes. Cached to avoid recomputing.
    pub stride: usize,
    /// Asymptotic scale factor `E[<u, sign(u)>] = sqrt(2N/π)` for unit
    /// Gaussian `u` in N dims. Kept as a fallback for vertices whose
    /// `s_x` came out near zero (ill-defined per-vertex correction).
    pub scale: f32,
}

impl<const N: usize> RabitQDataset<N> {
    /// Per-vertex code stride in bytes — same value as
    /// [`rabitq_code_stride`] but exposed as an associated const for
    /// callers that want it at compile time.
    pub const STRIDE: usize = rabitq_code_stride(N);

    /// Encode an `InmemDataset` end-to-end. `seed` controls the
    /// deterministic rotation matrix so two builds with the same seed
    /// produce identical sidecars — useful for cache + reproducible
    /// benchmarks.
    pub fn build_from(dataset: &InmemDataset<f32, N>, seed: u64) -> Self {
        let num_vertices = dataset.num_points;
        let stride = Self::STRIDE;
        let scale = ((2.0 * N as f64) / std::f64::consts::PI).sqrt() as f32;

        // 1. Build the rotation matrix P (N×N, row-major).
        let rotation = build_orthogonal_rotation::<N>(seed);

        // 2. Encode each base vector: rotate, sign-pack, store norm + s.
        let mut codes =
            AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16).expect("RabitQ codes alloc");
        let mut norms = Vec::with_capacity(num_vertices);
        let mut s_values = Vec::with_capacity(num_vertices);
        let mut rotated = vec![0.0f32; N];

        // Floor on per-vertex `s_x` to avoid divide-by-near-zero in the
        // estimator. For pathological vectors (constant value, all-zero
        // dims) `||x'||_1` can collapse — fall back to the asymptotic
        // `scale` in those cases. The threshold is small enough never
        // to fire on real high-dim data but defensive enough to keep
        // the math well-defined.
        let s_floor = 1e-6 * scale;

        let base = dataset.data.as_slice();
        for vid in 0..num_vertices {
            let x = &base[vid * N..(vid + 1) * N];

            // P @ x. Norm is preserved (orthogonal rotation), so we use
            // the rotated form for both L2 norm and L1 norm (s_x is
            // defined on the rotated space — that's where the sign
            // code lives).
            apply_rotation(rotation.as_slice(), x, &mut rotated);

            let mut l2_sq = 0.0f32;
            let mut l1 = 0.0f32;
            for &v in rotated.iter() {
                l2_sq += v * v;
                l1 += v.abs();
            }
            let l2 = l2_sq.sqrt();
            norms.push(l2);

            // s_x = ||rotated_x||_1 / ||rotated_x||_2 — the paper's
            // per-vertex correction. Equals <unit_rotated_x, sign(x')>.
            // For unit Gaussian vectors this concentrates around
            // sqrt(2N/π); per-vertex variance is what we exploit.
            let s_x = if l2 > 1e-12 {
                let s = l1 / l2;
                if s > s_floor { s } else { scale }
            } else {
                scale
            };
            s_values.push(s_x);

            // sign(rotated) → bit-packed code, LSB-first within byte.
            let code_off = vid * stride;
            let code = &mut codes.as_mut_slice()[code_off..code_off + stride];
            code.fill(0);
            for (d, &v) in rotated.iter().enumerate() {
                if v >= 0.0 {
                    code[d >> 3] |= 1u8 << (d & 7);
                }
            }
        }

        Self {
            codes,
            norms,
            s_values,
            rotation,
            num_vertices,
            stride,
            scale,
        }
    }

    /// Rotate a query into the same f32 space the codes were generated
    /// in. Output is written into `out` (length N, caller-allocated).
    /// This is the per-query setup cost — `O(N²)` FLOPs.
    #[inline]
    pub fn rotate_query(&self, q: &[f32; N], out: &mut [f32; N]) {
        apply_rotation(self.rotation.as_slice(), q, out);
    }

    /// Reference scalar L2-distance estimator (no SIMD). Returns the
    /// estimated `||q - x||²` for vertex `vertex_id`, given the
    /// rotated query `rotated_q = P @ q` and its precomputed
    /// `||q||² = q . q`. Output is approximate; error scales as
    /// `O(||x|| / sqrt(N))` per the paper's Theorem 3.3.
    ///
    /// This exists for correctness regression tests and as the
    /// reference the future NEON kernel will be benchmarked against.
    pub fn estimate_l2_sq(&self, rotated_q: &[f32; N], q_norm_sq: f32, vertex_id: u32) -> f32 {
        let off = (vertex_id as usize) * self.stride;
        let code = &self.codes.as_slice()[off..off + self.stride];

        // <rotated_q, sign(rotated_x)> = sum_d (rotated_q[d] * sign_d)
        // where sign_d ∈ {+1, -1} ← bit d of `code` (1 → +1, 0 → -1).
        // Use NEON kernel when available; scalar fallback otherwise.
        let signed_sum = signed_sum_dispatch::<N>(code, rotated_q);

        // Per-vertex-corrected estimator of <q', x'>:
        //   <q', x'> ≈ signed_sum * (||x|| / s_x)
        // where s_x = ||x'||_1 / ||x'||_2 = <unit_x', sign(x')>.
        // This is the paper's Theorem 3.3 estimator with the per-vertex
        // correction; the asymptotic `scale = sqrt(2N/π)` is only used
        // as a fallback for vertices whose s_x came out near zero
        // (already substituted at encode time).
        //
        // STAGED_RBQ_GLOBAL_SCALE=1 forces the asymptotic-only fallback
        // (legacy v1 behavior) for A/B testing the per-vertex correction.
        // Empirically on SIFT (D=128) the per-vertex s_x distribution is
        // tightly concentrated (CV ≈ 0.02) around the asymptotic value,
        // so per-vertex correction adds variance without bias-reduction
        // and slightly hurts recall — the v1 global-scale path is the
        // current default. Resolved once at startup via OnceLock to keep
        // the env-var check off the hot path (otherwise every estimate
        // becomes a syscall and QPS collapses ~10×).
        static USE_GLOBAL_SCALE: OnceLock<bool> = OnceLock::new();
        let use_global = *USE_GLOBAL_SCALE
            .get_or_init(|| std::env::var("STAGED_RBQ_GLOBAL_SCALE").as_deref() != Ok("0"));
        let x_norm = self.norms[vertex_id as usize];
        let divisor = if use_global {
            self.scale
        } else {
            self.s_values[vertex_id as usize]
        };
        let inner_est = signed_sum * x_norm / divisor;

        // ||q - x||² = ||q||² + ||x||² - 2<q, x>.
        // Clamp to ≥0 since the estimator can occasionally produce
        // very small negatives near identical vectors.
        let est = q_norm_sq + x_norm * x_norm - 2.0 * inner_est;
        est.max(0.0)
    }

    /// Header layout (32 B):
    ///   [u32 magic = RABITQ_MAGIC]
    ///   [u32 version = 2]            ← v2 adds per-vertex `s_values`
    ///   [u32 num_vertices][u32 dim = N]
    ///   [u32 stride][f32 scale]
    ///   [u64 reserved]                ← zero
    /// Body:
    ///   [N*N f32 rotation, row-major]
    ///   [num_vertices f32 norms]
    ///   [num_vertices f32 s_values]   ← v2 only
    ///   [num_vertices * STRIDE u8 codes]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(&RABITQ_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&2u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(N as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.stride as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&self.scale.to_le_bytes());
        // hdr[24..32] reserved (zero).
        w.write_all(&hdr)?;

        // Rotation matrix.
        let rot_bytes = unsafe {
            std::slice::from_raw_parts(
                self.rotation.as_slice().as_ptr() as *const u8,
                N * N * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(rot_bytes)?;

        // Norms.
        let norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.norms.as_ptr() as *const u8,
                self.norms.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(norm_bytes)?;

        // Per-vertex s_values (v2 addition).
        let s_bytes = unsafe {
            std::slice::from_raw_parts(
                self.s_values.as_ptr() as *const u8,
                self.s_values.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(s_bytes)?;

        // Codes.
        w.write_all(self.codes.as_slice())?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        use std::io::Read;
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 32];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != RABITQ_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = QRBQ)",
                    RABITQ_MAGIC
                ),
            )
            .into());
        }
        let num_vertices = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if dim != N {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("dim mismatch: file={dim} expected={N}"),
            )
            .into());
        }
        let stride = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        if stride != Self::STRIDE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("stride mismatch: file={stride} expected={}", Self::STRIDE),
            )
            .into());
        }
        let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if version != 2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported RaBitQ sidecar version {version} (only v2 with per-vertex s correction is supported; delete the `.qrbq` cache to force rebuild)",
                ),
            ).into());
        }
        let scale = f32::from_le_bytes(hdr[20..24].try_into().unwrap());

        let mut rotation = AlignedBoxWithSlice::<f32>::new(N * N, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        let rot_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                rotation.as_mut_slice().as_mut_ptr() as *mut u8,
                N * N * std::mem::size_of::<f32>(),
            )
        };
        r.read_exact(rot_bytes)?;

        let mut norms = vec![0.0f32; num_vertices];
        let norm_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                norms.as_mut_ptr() as *mut u8,
                norms.len() * std::mem::size_of::<f32>(),
            )
        };
        r.read_exact(norm_bytes)?;

        let mut s_values = vec![0.0f32; num_vertices];
        let s_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                s_values.as_mut_ptr() as *mut u8,
                s_values.len() * std::mem::size_of::<f32>(),
            )
        };
        r.read_exact(s_bytes)?;

        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        r.read_exact(codes.as_mut_slice())?;

        Ok(Self {
            codes,
            norms,
            s_values,
            rotation,
            num_vertices,
            stride,
            scale,
        })
    }
}

// ── Signed-sum kernel: <rotated_q, sign(rotated_x)> ───────────────────
//
// This is the inner loop of the RaBitQ estimator. For each dimension
// `d`, the corresponding bit of `code` is 1 if `rotated_x[d] >= 0`,
// else 0. The dot product against `rotated_q` is therefore a signed
// sum: lanes whose bit is 1 add `+rotated_q[d]`, lanes whose bit is 0
// add `-rotated_q[d]`. Naïvely this is `N` scalar fmas. The NEON
// kernel processes 4 lanes per nibble and 8 dims per code byte, using
// `vbslq_f32` to select between `+q` and `-q` based on the bit mask.

#[inline]
fn signed_sum_dispatch<const N: usize>(code: &[u8], rotated_q: &[f32; N]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // Whole-byte path covers the contiguous chunk; remainder
        // (D not a multiple of 8) falls through to scalar. The two
        // dataset dimensions we ship (SIFT 128, GIST 960) are both
        // multiples of 8, so the tail loop is dead code there.
        let bytes = N / 8;
        let neon_part = unsafe { signed_sum_neon::<N>(code.as_ptr(), rotated_q.as_ptr(), bytes) };
        let scalar_tail = signed_sum_scalar_tail::<N>(code, rotated_q, bytes * 8);
        return neon_part + scalar_tail;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // AVX-512: 16-bit mask per iter (= 2 code bytes = 16 dims).
        // Per-pair tail catches the trailing 1 byte / 8 dims case
        // (SIFT 128, GIST 960 both align, but D=100 trips this).
        let pairs = N / 16;
        let avx_part = unsafe { signed_sum_avx512::<N>(code.as_ptr(), rotated_q.as_ptr(), pairs) };
        let scalar_tail = signed_sum_scalar_tail::<N>(code, rotated_q, pairs * 16);
        return avx_part + scalar_tail;
    }
    #[allow(unreachable_code)]
    signed_sum_scalar::<N>(code, rotated_q)
}

/// Scalar reference path. Used by tests and on non-NEON targets.
#[inline]
fn signed_sum_scalar<const N: usize>(code: &[u8], rotated_q: &[f32; N]) -> f32 {
    let mut acc = 0.0f32;
    for d in 0..N {
        let bit = (code[d >> 3] >> (d & 7)) & 1;
        let sign = if bit != 0 { 1.0 } else { -1.0 };
        acc += rotated_q[d] * sign;
    }
    acc
}

/// Scalar leftover for the (theoretical) `N % 8 != 0` (NEON) or
/// `N % 16 != 0` (AVX-512) case. Inlined out of the SIMD paths so
/// the compiler can constant-fold to a no-op when N is known to
/// be a clean multiple at the call site. `#[allow(dead_code)]`:
/// it's unused on the scalar-only fallback build (no SIMD arm).
#[allow(dead_code)]
#[inline]
fn signed_sum_scalar_tail<const N: usize>(code: &[u8], rotated_q: &[f32; N], start: usize) -> f32 {
    let mut acc = 0.0f32;
    for d in start..N {
        let bit = (code[d >> 3] >> (d & 7)) & 1;
        let sign = if bit != 0 { 1.0 } else { -1.0 };
        acc += rotated_q[d] * sign;
    }
    acc
}

/// NEON kernel for the signed dot product. Processes 8 dims per code
/// byte = one (low nibble, high nibble) pair of 4-lane operations:
///
/// 1. Load 8 f32 from `rotated_q[d..d+8]` as two `float32x4_t` and
///    pre-negate one copy (`vnegq_f32`).
/// 2. For each nibble (4 bits), broadcast it to a `uint32x4_t`, AND
///    with per-lane masks `[1, 2, 4, 8]`, and `vceqq_u32` against the
///    same masks — yields all-1s in lanes where the bit is set, else
///    all-0s.
/// 3. `vbslq_f32(mask, q, -q)` selects `+q` when bit=1 (positive sign
///    in the code) and `-q` when bit=0 (negative sign).
/// 4. Accumulate two `float32x4_t` accumulators (one per nibble) to
///    hide the dependency chain; horizontal-add at the end.
///
/// Stride: 8 dims per loop iteration. For SIFT D=128: 16 iterations.
/// For GIST D=960: 120 iterations.
///
/// # Safety
///
/// Caller must ensure:
/// - `code` points to at least `bytes` valid bytes
/// - `rotated_q` points to at least `bytes * 8` valid `f32`s
/// - `bytes * 8 <= N` (caller bounds the dimension)
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn signed_sum_neon<const N: usize>(
    code: *const u8,
    rotated_q: *const f32,
    bytes: usize,
) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        // Per-lane bit masks: lane i tests bit i of the nibble.
        let lane_masks_arr = [1u32, 2, 4, 8];
        let lane_masks = vld1q_u32(lane_masks_arr.as_ptr());

        // Two accumulators (one per nibble) break the loop-carried
        // dependency so the M2 OoO engine can pipeline both vaddq_f32s
        // every iteration.
        let mut acc_lo = vdupq_n_f32(0.0);
        let mut acc_hi = vdupq_n_f32(0.0);

        for byte_idx in 0..bytes {
            let byte = *code.add(byte_idx) as u32;
            let d = byte_idx * 8;

            // Load 8 f32 lanes of rotated_q + precompute their negations.
            let q_lo = vld1q_f32(rotated_q.add(d));
            let q_hi = vld1q_f32(rotated_q.add(d + 4));
            let neg_lo = vnegq_f32(q_lo);
            let neg_hi = vnegq_f32(q_hi);

            // Low nibble → lanes 0..=3.
            let low = vdupq_n_u32(byte & 0xf);
            let low_masked = vandq_u32(low, lane_masks);
            let low_set = vceqq_u32(low_masked, lane_masks);
            // bsl: result = (mask & b) | (!mask & c) — equivalent to
            // "if mask bit is 1 pick lane from `q`, else pick from `-q`".
            let signed_lo = vbslq_f32(low_set, q_lo, neg_lo);
            acc_lo = vaddq_f32(acc_lo, signed_lo);

            // High nibble → lanes 4..=7.
            let high = vdupq_n_u32((byte >> 4) & 0xf);
            let high_masked = vandq_u32(high, lane_masks);
            let high_set = vceqq_u32(high_masked, lane_masks);
            let signed_hi = vbslq_f32(high_set, q_hi, neg_hi);
            acc_hi = vaddq_f32(acc_hi, signed_hi);
        }

        // Horizontal sum: combine both accumulators and reduce.
        let acc = vaddq_f32(acc_lo, acc_hi);
        vaddvq_f32(acc)
    }
}

/// AVX-512 kernel for the signed dot product. 16 dims per outer
/// iteration via a 16-bit mask built from two consecutive code
/// bytes. `_mm512_mask_blend_ps` selects `+q` (mask bit 1) or
/// `-q` (mask bit 0) lane-wise — same logical contract as the
/// NEON `vbslq_f32` but at twice the width.
///
/// 2-way unroll across consecutive byte-pairs hides the load
/// latency for `rotated_q`. Net throughput on Sapphire Rapids
/// (~4 FMA-IPC at f32 / 16-wide) lands roughly on par with the
/// NEON M2 path despite NEON's narrower SIMD width — AVX-512's
/// mask-blend is a single uop where NEON needs the `vdupq_n_u32 +
/// vandq + vceqq + vbslq` chain per nibble.
///
/// # Safety
///
/// Caller must ensure:
/// - `code` points to at least `pairs * 2` valid bytes
/// - `rotated_q` points to at least `pairs * 16` valid `f32`s
/// - `pairs * 16 <= N`
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
unsafe fn signed_sum_avx512<const N: usize>(
    code: *const u8,
    rotated_q: *const f32,
    pairs: usize,
) -> f32 {
    use std::arch::x86_64::*;

    // Two accumulators break the loop-carried dep — mirrors NEON
    // `acc_lo` / `acc_hi`. With 16-wide we no longer split on
    // nibble; we split on consecutive byte-pairs (= 32 dims per
    // outer iter when both lanes contribute).
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();

    let pair_chunks = pairs / 2;
    let mut p = 0usize;
    while p < pair_chunks {
        // Pair 0: bytes [2p..2p+2], dims [16p..16p+16].
        let b0 = (*code.add(2 * p) as u16) | ((*code.add(2 * p + 1) as u16) << 8);
        let q0 = _mm512_loadu_ps(rotated_q.add(16 * p));
        // `_mm512_xor_ps` with sign-bit mask flips floats in-place
        // (cheaper than `_mm512_sub_ps(zero, q)`).
        let neg0 = _mm512_xor_ps(q0, _mm512_set1_ps(-0.0_f32));
        let signed0 = _mm512_mask_blend_ps(b0 as __mmask16, neg0, q0);
        acc0 = _mm512_add_ps(acc0, signed0);

        // Pair 1: bytes [2p+2..2p+4], dims [16p+16..16p+32].
        let b1 = (*code.add(2 * p + 2) as u16) | ((*code.add(2 * p + 3) as u16) << 8);
        let q1 = _mm512_loadu_ps(rotated_q.add(16 * p + 16));
        let neg1 = _mm512_xor_ps(q1, _mm512_set1_ps(-0.0_f32));
        let signed1 = _mm512_mask_blend_ps(b1 as __mmask16, neg1, q1);
        acc1 = _mm512_add_ps(acc1, signed1);

        p += 2;
    }
    // Odd-pair tail.
    let mut t = pair_chunks * 2;
    while t < pairs {
        let b = (*code.add(2 * t) as u16) | ((*code.add(2 * t + 1) as u16) << 8);
        let q = _mm512_loadu_ps(rotated_q.add(16 * t));
        let neg = _mm512_xor_ps(q, _mm512_set1_ps(-0.0_f32));
        let signed = _mm512_mask_blend_ps(b as __mmask16, neg, q);
        acc0 = _mm512_add_ps(acc0, signed);
        t += 1;
    }

    _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1))
}

// ── Public re-exports for sibling codecs (rabitq_b4_dataset) ──────────
//
// The B=4 variant in `rabitq_b4_dataset.rs` reuses the same rotation
// machinery (matrix construction + matvec) but its own struct. Expose
// thin pub wrappers here so the B=4 file doesn't need access to
// crate-private helpers.

/// Public wrapper for [`build_orthogonal_rotation`] used by sibling
/// codecs (B=4 dataset).
pub fn build_orthogonal_rotation_pub<const N: usize>(seed: u64) -> AlignedBoxWithSlice<f32> {
    build_orthogonal_rotation::<N>(seed)
}

/// Public wrapper for [`apply_rotation`].
#[inline]
pub fn apply_rotation_pub(m: &[f32], x: &[f32], y: &mut [f32]) {
    apply_rotation(m, x, y);
}

/// Public re-export of the deterministic PRNG so the B=4 test module
/// can produce the same Gaussian samples without duplicating the
/// xorshift code. Inner state is hidden — callers only use
/// `new`/`next_u64`/`gaussian_pub`.
pub struct XorShiftPub {
    inner: XorShift64,
}
impl XorShiftPub {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: XorShift64(seed),
        }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
}

/// Public wrapper for [`gaussian`].
pub fn gaussian_pub(rng: &mut XorShiftPub) -> f32 {
    gaussian(&mut rng.inner)
}

// ── Rotation matrix construction (deterministic, seeded) ──────────────

/// Build a row-major `N × N` orthogonal matrix from `seed` via
/// modified Gram-Schmidt on a Gaussian random matrix. Deterministic
/// for a given seed — same seed yields the same matrix bit-for-bit.
///
/// Modified Gram-Schmidt is `O(N³)` but stable enough for the
/// dimensions we care about (N ≤ 1024). For very large N the QR
/// decomposition via Householder would be preferable.
fn build_orthogonal_rotation<const N: usize>(seed: u64) -> AlignedBoxWithSlice<f32> {
    let mut mat = AlignedBoxWithSlice::<f32>::new(N * N, 16).expect("rotation alloc");
    let mut rng = XorShift64(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));

    // Fill with standard normal samples (row-major).
    for i in 0..(N * N) {
        mat.as_mut_slice()[i] = gaussian(&mut rng);
    }

    // Modified Gram-Schmidt on rows. After this, row vectors are
    // pairwise orthonormal, so the matrix is orthogonal (P @ P^T = I).
    for i in 0..N {
        // Normalize row i.
        let (i_start, after_i) = mat.as_mut_slice().split_at_mut((i + 1) * N);
        let row_i = &mut i_start[i * N..(i + 1) * N];
        let norm: f32 = row_i.iter().map(|v| v * v).sum::<f32>().sqrt();
        // Guard against the (vanishingly rare) zero-norm seed sample.
        let inv = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
        for v in row_i.iter_mut() {
            *v *= inv;
        }

        // Project out row i from each subsequent row j > i.
        for jrow in after_i.chunks_exact_mut(N) {
            let mut dot = 0.0f32;
            for d in 0..N {
                dot += jrow[d] * row_i[d];
            }
            for d in 0..N {
                jrow[d] -= dot * row_i[d];
            }
        }
    }

    mat
}

/// `y = M @ x`, row-major M (N×N), x and y length N. Reference scalar
/// implementation. Dispatches to NEON when the f32 inner-loop length
/// is a multiple of 16 (= 4 lanes × 4 ILP); falls back to scalar
/// otherwise (still triggers auto-vectorization on tight inputs).
#[inline]
fn apply_rotation(m: &[f32], x: &[f32], y: &mut [f32]) {
    let n = y.len();
    debug_assert_eq!(m.len(), n * n);
    debug_assert_eq!(x.len(), n);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if n % 16 == 0 {
            unsafe { apply_rotation_neon(m, x, y, n) };
            return;
        }
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        if n % 64 == 0 {
            unsafe { apply_rotation_avx512(m, x, y, n) };
            return;
        }
        // n % 16 == 0 but not % 64: use 16-wide single accumulator.
        if n % 16 == 0 {
            unsafe { apply_rotation_avx512_16(m, x, y, n) };
            return;
        }
    }
    apply_rotation_scalar(m, x, y, n);
}

#[inline]
fn apply_rotation_scalar(m: &[f32], x: &[f32], y: &mut [f32], n: usize) {
    for r in 0..n {
        let row = &m[r * n..(r + 1) * n];
        let mut acc = 0.0f32;
        for d in 0..n {
            acc += row[d] * x[d];
        }
        y[r] = acc;
    }
}

/// NEON `apply_rotation` — 4-lane × 4-way-ILP dot product per row.
/// Requires `n % 16 == 0`. For D=960 (GIST) this is 60 iterations
/// × 4 FMAs/iter = 240 FMAs/dot, ×960 rows = ~230k FMAs. With Apple
/// M2's ~4 FMA-IPC at f32 this lands ~15-25µs vs the scalar 436µs.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn apply_rotation_neon(m: &[f32], x: &[f32], y: &mut [f32], n: usize) {
    unsafe {
        use std::arch::aarch64::*;
        let chunks = n / 16;
        let x_ptr = x.as_ptr();
        for r in 0..n {
            let row_ptr = m.as_ptr().add(r * n);
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            let mut a2 = vdupq_n_f32(0.0);
            let mut a3 = vdupq_n_f32(0.0);
            for c in 0..chunks {
                let off = c * 16;
                let r0 = vld1q_f32(row_ptr.add(off));
                let r1 = vld1q_f32(row_ptr.add(off + 4));
                let r2 = vld1q_f32(row_ptr.add(off + 8));
                let r3 = vld1q_f32(row_ptr.add(off + 12));
                let x0 = vld1q_f32(x_ptr.add(off));
                let x1 = vld1q_f32(x_ptr.add(off + 4));
                let x2 = vld1q_f32(x_ptr.add(off + 8));
                let x3 = vld1q_f32(x_ptr.add(off + 12));
                a0 = vfmaq_f32(a0, r0, x0);
                a1 = vfmaq_f32(a1, r1, x1);
                a2 = vfmaq_f32(a2, r2, x2);
                a3 = vfmaq_f32(a3, r3, x3);
            }
            let s01 = vaddq_f32(a0, a1);
            let s23 = vaddq_f32(a2, a3);
            let s = vaddq_f32(s01, s23);
            *y.get_unchecked_mut(r) = vaddvq_f32(s);
        }
    }
}

/// AVX-512 `apply_rotation` — 16-lane × 4-way-ILP dot product per
/// row. Requires `n % 64 == 0`. For SIFT D=128: 2 inner iters / row.
/// For GIST D=960: 15 inner iters / row.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
unsafe fn apply_rotation_avx512(m: &[f32], x: &[f32], y: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 64;
    let x_ptr = x.as_ptr();
    for r in 0..n {
        let row_ptr = m.as_ptr().add(r * n);
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        for c in 0..chunks {
            let off = c * 64;
            let r0 = _mm512_loadu_ps(row_ptr.add(off));
            let r1 = _mm512_loadu_ps(row_ptr.add(off + 16));
            let r2 = _mm512_loadu_ps(row_ptr.add(off + 32));
            let r3 = _mm512_loadu_ps(row_ptr.add(off + 48));
            let x0 = _mm512_loadu_ps(x_ptr.add(off));
            let x1 = _mm512_loadu_ps(x_ptr.add(off + 16));
            let x2 = _mm512_loadu_ps(x_ptr.add(off + 32));
            let x3 = _mm512_loadu_ps(x_ptr.add(off + 48));
            a0 = _mm512_fmadd_ps(r0, x0, a0);
            a1 = _mm512_fmadd_ps(r1, x1, a1);
            a2 = _mm512_fmadd_ps(r2, x2, a2);
            a3 = _mm512_fmadd_ps(r3, x3, a3);
        }
        let s01 = _mm512_add_ps(a0, a1);
        let s23 = _mm512_add_ps(a2, a3);
        *y.get_unchecked_mut(r) = _mm512_reduce_add_ps(_mm512_add_ps(s01, s23));
    }
}

/// Fallback for `n % 16 == 0` but `n % 64 != 0` — single
/// accumulator, no 4-way unroll. Catches dims like 80, 112, 144.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
unsafe fn apply_rotation_avx512_16(m: &[f32], x: &[f32], y: &mut [f32], n: usize) {
    use std::arch::x86_64::*;
    let chunks = n / 16;
    let x_ptr = x.as_ptr();
    for r in 0..n {
        let row_ptr = m.as_ptr().add(r * n);
        let mut acc = _mm512_setzero_ps();
        for c in 0..chunks {
            let off = c * 16;
            let rv = _mm512_loadu_ps(row_ptr.add(off));
            let xv = _mm512_loadu_ps(x_ptr.add(off));
            acc = _mm512_fmadd_ps(rv, xv, acc);
        }
        *y.get_unchecked_mut(r) = _mm512_reduce_add_ps(acc);
    }
}

// ── Deterministic PRNG + Gaussian sampler ─────────────────────────────

struct XorShift64(u64);

impl XorShift64 {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.0 = s;
        s
    }

    /// Uniform f32 in [1e-7, 1).  Lower-bounded so `ln(u)` in
    /// Box-Muller never returns `-inf` on a 0 sample.
    #[inline]
    fn next_unit(&mut self) -> f32 {
        let bits = self.next_u64();
        let u = (bits >> 40) as f32 / ((1u32 << 24) as f32); // 24-bit precision
        u.max(1e-7).min(1.0 - 1e-7)
    }
}

/// Standard normal sample via Box-Muller. Returns one of the two
/// independent samples per call (the other is discarded — fine for
/// our use case since the rotation generation is one-shot at build
/// time and Box-Muller's discarded sample matters only for sampler
/// throughput, not correctness).
#[inline]
fn gaussian(rng: &mut XorShift64) -> f32 {
    let u1 = rng.next_unit();
    let u2 = rng.next_unit();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_matches_paper_examples() {
        // SIFT D=128 → 16 B code, no pad needed
        assert_eq!(rabitq_code_stride(128), 16);
        // GIST D=960 → ceil(960/8)=120 B → padded 128
        assert_eq!(rabitq_code_stride(960), 128);
        // Edge: D=1, 1 bit → padded 16
        assert_eq!(rabitq_code_stride(1), 16);
    }

    #[test]
    fn rotation_is_orthogonal() {
        const N: usize = 32;
        let rot = build_orthogonal_rotation::<N>(42);
        // Verify P @ P^T = I by checking row dot products.
        for i in 0..N {
            for j in i..N {
                let row_i = &rot.as_slice()[i * N..(i + 1) * N];
                let row_j = &rot.as_slice()[j * N..(j + 1) * N];
                let dot: f32 = row_i.iter().zip(row_j).map(|(a, b)| a * b).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "row {i}.row {j} dot = {dot}, expected {expected}",
                );
            }
        }
    }

    #[test]
    fn rotation_preserves_norm() {
        const N: usize = 32;
        let rot = build_orthogonal_rotation::<N>(7);
        let x: [f32; N] = std::array::from_fn(|i| ((i as f32) * 0.31 - 5.0).sin());
        let x_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let mut y = [0.0f32; N];
        apply_rotation(rot.as_slice(), &x, &mut y);
        let y_norm: f32 = y.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(
            (x_norm - y_norm).abs() < 1e-3,
            "norm not preserved: ||x||={x_norm} ||Px||={y_norm}",
        );
    }

    #[test]
    fn build_and_estimate_roundtrip() {
        // Small (N=64) sanity test: encode 100 random vectors, then
        // estimate self-distance (which should be ≈ 0) and pairwise
        // distance (which should be roughly within Hoeffding bound of
        // the true L2² distance).
        const N: usize = 64;
        let num = 100;

        // Deterministic data generation.
        let mut rng = XorShift64(99);
        let mut flat = vec![0.0f32; num * N];
        for v in flat.iter_mut() {
            *v = gaussian(&mut rng);
        }

        let mut ds = InmemDataset::<f32, N>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let rbq = RabitQDataset::<N>::build_from(&ds, 12345);

        // Self-distance: pick vertex 0, rotate it, estimate against itself.
        let v0: [f32; N] = std::array::from_fn(|i| flat[i]);
        let v0_norm_sq: f32 = v0.iter().map(|x| x * x).sum();
        let mut rv0 = [0.0f32; N];
        rbq.rotate_query(&v0, &mut rv0);
        let est_self = rbq.estimate_l2_sq(&rv0, v0_norm_sq, 0);
        // Per-vertex `s` correction tightens self-distance estimate
        // substantially vs the asymptotic-only version. At N=64 with
        // mean ||x||≈8 the residual should easily fit under ||v0||².
        assert!(
            est_self < v0_norm_sq,
            "self-distance estimate {est_self} > ||v0||² ({})",
            v0_norm_sq,
        );

        // Pairwise: vertex 0 vs vertex 1, estimate vs ground truth.
        let v1: [f32; N] = std::array::from_fn(|i| flat[N + i]);
        let true_l2_sq: f32 = v0.iter().zip(v1.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let est_01 = rbq.estimate_l2_sq(&rv0, v0_norm_sq, 1);
        // With per-vertex `s` we expect tighter error than the basic
        // (asymptotic-scale-only) version; allow up to 50% relative
        // error on a single-sample pair-distance estimate at N=64.
        let rel_err = ((est_01 - true_l2_sq) / true_l2_sq.max(1e-3)).abs();
        assert!(
            rel_err < 0.5,
            "pair-distance estimate {est_01} vs true {true_l2_sq} → rel_err {rel_err}",
        );
    }

    #[test]
    fn neon_signed_sum_matches_scalar() {
        // Exercise the dispatch path against the scalar reference on
        // representative production sizes (SIFT D=128, GIST D=960).
        // Both kernels operate on the same bit code and rotated query;
        // they should agree to within f32 round-off.
        const SIFT: usize = 128;
        const GIST: usize = 960;

        // Helper: drive both kernels with deterministic data.
        fn check<const D: usize>(seed: u64) {
            let mut rng = XorShift64(seed);
            let code_bytes = (D + 7) / 8;
            let stride = rabitq_code_stride(D);
            let mut code = vec![0u8; stride];
            for b in 0..code_bytes {
                code[b] = (rng.next_u64() & 0xff) as u8;
            }
            let mut q = [0f32; D];
            for v in q.iter_mut() {
                *v = gaussian(&mut rng);
            }
            let s = signed_sum_dispatch::<D>(&code, &q);
            let r = signed_sum_scalar::<D>(&code, &q);
            // Allow a relative tolerance proportional to sqrt(D) since
            // f32 sum accumulation drifts with the number of terms.
            let tol = (D as f32).sqrt() * 1e-5 * r.abs().max(1.0);
            assert!(
                (s - r).abs() < tol,
                "D={D}: neon={s} scalar={r} diff={} tol={tol}",
                (s - r).abs(),
            );
        }

        check::<SIFT>(42);
        check::<SIFT>(7);
        check::<GIST>(42);
        check::<GIST>(7);
    }

    #[test]
    fn save_load_roundtrip() {
        const N: usize = 32;
        let num = 16;
        let mut rng = XorShift64(1);
        let mut flat = vec![0.0f32; num * N];
        for v in flat.iter_mut() {
            *v = gaussian(&mut rng);
        }
        let mut ds = InmemDataset::<f32, N>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let a = RabitQDataset::<N>::build_from(&ds, 777);

        let path = std::env::temp_dir().join(format!("rabitq_test_{}.qrbq", std::process::id()));
        a.save(&path).expect("save");
        let b = RabitQDataset::<N>::load(&path).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(a.num_vertices, b.num_vertices);
        assert_eq!(a.stride, b.stride);
        assert!((a.scale - b.scale).abs() < 1e-6);
        assert_eq!(a.codes.as_slice(), b.codes.as_slice());
        assert_eq!(a.norms, b.norms);
        assert_eq!(a.s_values, b.s_values);
        for i in 0..(N * N) {
            assert!((a.rotation.as_slice()[i] - b.rotation.as_slice()[i]).abs() < 1e-6);
        }
    }
}
