/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Extended RaBitQ at **B=4 bits per dim** — the configuration Milvus
//! ships in `IVF_RABITQ` and the one the paper recommends for high
//! recall. Same random-rotation front-end as 1-bit RaBitQ but with a
//! 4-bit signed scalar quantizer on the rotated components instead of
//! sign-only.
//!
//! Reference: Gao & Long, *"Practical and Asymptotically Optimal
//! Quantization of High-Dimensional Vectors..."* (SIGMOD 2025; the
//! follow-up to the SIGMOD 2024 1-bit paper).
//!
//! ## Encoding pipeline
//!
//! 1. Rotate `rotated_x = P @ x` (same `P` as the 1-bit path — we
//!    don't currently share it between codecs since each owns its own
//!    sidecar, but the math is identical).
//! 2. Find per-vertex `tau_x = max_d |rotated_x[d]|`.
//! 3. Quantize each dim: `c[d] = round(rotated_x[d] / tau_x * Q_MAX)`,
//!    clamped to `[-Q_MAX, +Q_MAX]`. With `B=4` and signed-symmetric
//!    encoding we use `Q_MAX = 7` (range `[-7, +7]`, 15 levels; the
//!    -8 slot of two's complement is left unused so the magnitudes
//!    are symmetric around zero, matching the rotation's zero-mean
//!    distribution).
//! 4. Pack two nibbles per byte: byte `b` holds dim `2b` in the low
//!    nibble and dim `2b+1` in the high nibble. Both nibbles are
//!    stored as 4-bit two's-complement; sign-extension is a `<< 4`
//!    followed by `>> 4` (arithmetic) for decode.
//!
//! Per-vertex storage: `D/2` packed bytes + `tau_x` (4 B) + `||x||` (4 B).
//! For GIST D=960: 480 B code + 8 B scalars = **488 B/vertex**, vs
//! 960 B u8 (≈ 2× compression) and 128 B 1-bit (≈ 4× more verbose).
//!
//! ## Distance estimator
//!
//! `<q', x'> ≈ (tau_x / Q_MAX) * sum_d (rotated_q[d] * c[d])`
//!
//! where the sum is over the signed 4-bit code values. Then:
//!
//! `||q - x||² ≈ ||q||² + ||x||² - 2 * <q', x'>`
//!
//! The estimator's variance scales as `O(tau_x² / Q_MAX²)`, which is
//! `1/49` of the 1-bit version's variance — the recall ceiling on
//! SIFT/GIST that 1-bit hit at ~0.74/0.94 should clear comfortably.

use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use diskann::model::InmemDataset;
use std::io::{Read, Write};
use std::path::Path;

/// Disk-format magic for the B=4 RaBitQ sidecar (`.qrb4`).
pub const RABITQ_B4_MAGIC: u32 = 0x5152_4234; // "QRB4"

/// Symmetric quantization peak. With B=4 signed-symmetric we use 7
/// (range `[-7, +7]`), reserving the -8 slot to keep the encoding
/// zero-symmetric. Matches the paper's recommendation for
/// random-rotation distributions (zero-mean, near-Gaussian).
pub const QMAX: i8 = 7;

/// Per-vertex packed-code stride in bytes. `D/2` rounded up to a
/// 16-byte multiple so per-vertex offset arithmetic stays trivial and
/// future NEON loads can use 16-byte alignment.
pub const fn rabitq_b4_code_stride(dim: usize) -> usize {
    let bytes = (dim + 1) / 2; // ceil(D/2)
    (bytes + 15) & !15
}

/// B=4 RaBitQ-encoded base store.
pub struct RabitQ4Dataset<const N: usize> {
    /// Packed nibble codes, `num_vertices * STRIDE` bytes, 16-byte
    /// aligned. Each byte holds two signed 4-bit values: low nibble =
    /// dim `2b`, high nibble = dim `2b+1`. Sign-extend via
    /// `((nibble << 4) as i8) >> 4` (arithmetic shift).
    pub codes: AlignedBoxWithSlice<u8>,
    /// Per-vertex `||x||` (= `||rotated_x||` since rotation is
    /// orthogonal). Length = `num_vertices`.
    pub norms: Vec<f32>,
    /// Per-vertex `tau_x = max_d |rotated_x[d]|` — the quantization
    /// peak used to scale code values back to f32. Length =
    /// `num_vertices`.
    pub taus: Vec<f32>,
    /// Row-major `N × N` orthogonal rotation. Independent of the 1-bit
    /// path's rotation (each codec owns its own seed → matrix) so the
    /// two sidecars can be loaded side-by-side without interfering.
    pub rotation: AlignedBoxWithSlice<f32>,
    pub num_vertices: usize,
    pub stride: usize,
}

impl<const N: usize> RabitQ4Dataset<N> {
    pub const STRIDE: usize = rabitq_b4_code_stride(N);

    pub fn build_from(dataset: &InmemDataset<f32, N>, seed: u64) -> Self {
        let num_vertices = dataset.num_points;
        let stride = Self::STRIDE;
        let rotation = super::rabitq_dataset::build_orthogonal_rotation_pub::<N>(seed);

        let mut codes =
            AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16).expect("rbq4 codes alloc");
        let mut norms = Vec::with_capacity(num_vertices);
        let mut taus = Vec::with_capacity(num_vertices);
        let mut rotated = vec![0.0f32; N];

        let base = dataset.data.as_slice();
        for vid in 0..num_vertices {
            let x = &base[vid * N..(vid + 1) * N];

            // P @ x and per-vertex L2 norm.
            super::rabitq_dataset::apply_rotation_pub(rotation.as_slice(), x, &mut rotated);
            let l2: f32 = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
            norms.push(l2);

            // tau = max |rotated_x[d]|. Use `max(tau, eps)` so the
            // divide below stays well-defined on (rare) zero vectors.
            let mut tau = 0.0f32;
            for &v in rotated.iter() {
                let a = v.abs();
                if a > tau {
                    tau = a;
                }
            }
            let tau_eff = tau.max(1e-12);
            taus.push(tau_eff);

            // Quantize and pack two nibbles per byte. Use round-to-
            // nearest with a clamp to [-QMAX, +QMAX] (asymmetric vs
            // the natural two's-complement -8 to +7, but symmetric
            // around 0 — better for unbiased zero-mean rotated data).
            let code_off = vid * stride;
            let code = &mut codes.as_mut_slice()[code_off..code_off + stride];
            code.fill(0);
            let scale = QMAX as f32 / tau_eff;
            // Pairs: (d_lo, d_hi) → one byte.
            for b in 0..(N / 2) {
                let lo = quantize_to_i4(rotated[2 * b] * scale);
                let hi = quantize_to_i4(rotated[2 * b + 1] * scale);
                code[b] = pack_nibbles(lo, hi);
            }
            // Odd-D tail (only the low nibble is meaningful; high
            // nibble stays zero). N must equal the type-level const,
            // so this branch is dead for N=128/960 but kept correct.
            if N & 1 == 1 {
                let lo = quantize_to_i4(rotated[N - 1] * scale);
                code[N / 2] = pack_nibbles(lo, 0);
            }
        }

        Self {
            codes,
            norms,
            taus,
            rotation,
            num_vertices,
            stride,
        }
    }

    #[inline]
    pub fn rotate_query(&self, q: &[f32; N], out: &mut [f32; N]) {
        super::rabitq_dataset::apply_rotation_pub(self.rotation.as_slice(), q, out);
    }

    /// Reference scalar L2² estimator. Decodes 4-bit signed values one
    /// at a time, dots with `rotated_q`, scales by `tau / QMAX`.
    pub fn estimate_l2_sq(&self, rotated_q: &[f32; N], q_norm_sq: f32, vertex_id: u32) -> f32 {
        let off = (vertex_id as usize) * self.stride;
        let code = &self.codes.as_slice()[off..off + self.stride];

        let dot = signed_dot_dispatch::<N>(code, rotated_q);

        let tau = self.taus[vertex_id as usize];
        let x_norm = self.norms[vertex_id as usize];
        let inner_est = dot * tau / (QMAX as f32);

        let est = q_norm_sq + x_norm * x_norm - 2.0 * inner_est;
        est.max(0.0)
    }

    /// Header (32 B):
    ///   [u32 magic = RABITQ_B4_MAGIC]
    ///   [u32 version = 1]
    ///   [u32 num_vertices][u32 dim = N]
    ///   [u32 stride][u32 reserved]
    ///   [u64 reserved]
    /// Body:
    ///   [N*N f32 rotation, row-major]
    ///   [num_vertices f32 norms]
    ///   [num_vertices f32 taus]
    ///   [num_vertices * STRIDE u8 codes]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(&RABITQ_B4_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(N as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.stride as u32).to_le_bytes());
        w.write_all(&hdr)?;

        let rot_bytes = unsafe {
            std::slice::from_raw_parts(
                self.rotation.as_slice().as_ptr() as *const u8,
                N * N * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(rot_bytes)?;

        let norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.norms.as_ptr() as *const u8,
                self.norms.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(norm_bytes)?;

        let tau_bytes = unsafe {
            std::slice::from_raw_parts(
                self.taus.as_ptr() as *const u8,
                self.taus.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(tau_bytes)?;

        w.write_all(self.codes.as_slice())?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 32];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != RABITQ_B4_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = QRB4)",
                    RABITQ_B4_MAGIC
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

        let mut taus = vec![0.0f32; num_vertices];
        let tau_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                taus.as_mut_ptr() as *mut u8,
                taus.len() * std::mem::size_of::<f32>(),
            )
        };
        r.read_exact(tau_bytes)?;

        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        r.read_exact(codes.as_mut_slice())?;

        Ok(Self {
            codes,
            norms,
            taus,
            rotation,
            num_vertices,
            stride,
        })
    }
}

// ── Nibble packing helpers ────────────────────────────────────────────

/// Round-to-nearest with clamp to `[-QMAX, +QMAX]`. Returns an i8 in
/// the symmetric 4-bit signed range; the caller packs two of these per
/// byte via [`pack_nibbles`].
#[inline]
fn quantize_to_i4(v: f32) -> i8 {
    let r = v.round() as i32;
    r.clamp(-(QMAX as i32), QMAX as i32) as i8
}

/// Pack two signed 4-bit values into a byte. Low nibble = lo (dim 2b),
/// high nibble = hi (dim 2b+1). The encoding is 4-bit two's
/// complement, but since we constrain to `[-7, +7]` the `-8` code is
/// never produced.
#[inline]
fn pack_nibbles(lo: i8, hi: i8) -> u8 {
    let lo_u = (lo as u8) & 0x0f;
    let hi_u = (hi as u8) & 0x0f;
    (hi_u << 4) | lo_u
}

/// Decode the low signed nibble of a byte to i8. Arithmetic right
/// shift on a left-shifted i8 sign-extends the 4-bit value.
#[inline]
fn unpack_lo(byte: u8) -> i8 {
    ((byte << 4) as i8) >> 4
}

/// Decode the high signed nibble of a byte to i8.
#[inline]
fn unpack_hi(byte: u8) -> i8 {
    (byte as i8) >> 4
}

// ── Signed-dot kernel: <rotated_q, code (as i8 lanes)> ────────────────

#[inline]
fn signed_dot_dispatch<const N: usize>(code: &[u8], rotated_q: &[f32; N]) -> f32 {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        // NEON path: handle the bulk of dims in 16-dim chunks (8 code
        // bytes per iter); scalar tail covers any leftover.
        let chunk_bytes = (N / 2) & !7; // multiple of 8 bytes = 16 dims
        let neon_part = unsafe { signed_dot_neon(code.as_ptr(), rotated_q.as_ptr(), chunk_bytes) };
        let tail = signed_dot_scalar::<N>(code, rotated_q, chunk_bytes * 2);
        return neon_part + tail;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        // AVX-512 path: 16 dims per iter = 8 packed bytes.
        let chunk_bytes = (N / 2) & !7;
        let avx_part = unsafe { signed_dot_avx512(code.as_ptr(), rotated_q.as_ptr(), chunk_bytes) };
        let tail = signed_dot_scalar::<N>(code, rotated_q, chunk_bytes * 2);
        return avx_part + tail;
    }
    #[allow(unreachable_code)]
    signed_dot_scalar::<N>(code, rotated_q, 0)
}

/// Scalar reference: decode each nibble, multiply by `rotated_q[d]`,
/// accumulate. Used by tests and as the NEON tail.
#[inline]
fn signed_dot_scalar<const N: usize>(code: &[u8], rotated_q: &[f32; N], start: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut d = start;
    while d + 1 < N {
        let byte = code[d / 2];
        let c_lo = unpack_lo(byte) as f32;
        let c_hi = unpack_hi(byte) as f32;
        acc += rotated_q[d] * c_lo + rotated_q[d + 1] * c_hi;
        d += 2;
    }
    if d < N {
        let byte = code[d / 2];
        let c_lo = unpack_lo(byte) as f32;
        acc += rotated_q[d] * c_lo;
    }
    acc
}

/// NEON kernel for the signed 4-bit dot product. Processes 16 dims
/// per loop iteration (8 packed bytes); unpacks nibbles to i8, then to
/// i32, then to f32 to mul-add against `rotated_q`. Two accumulators
/// break the dependency chain.
///
/// # Safety
///
/// Caller ensures `code` has at least `bytes` valid bytes,
/// `rotated_q` has at least `bytes * 2` f32 lanes, and `bytes` is a
/// multiple of 8.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn signed_dot_neon(code: *const u8, rotated_q: *const f32, bytes: usize) -> f32 {
    unsafe {
        use std::arch::aarch64::*;
        let low_mask = vdup_n_u8(0x0f);
        let mut acc_a = vdupq_n_f32(0.0);
        let mut acc_b = vdupq_n_f32(0.0);

        let mut b = 0;
        while b < bytes {
            // Load 8 packed bytes = 16 nibble lanes = 16 dims.
            let packed = vld1_u8(code.add(b));

            // Unpack signed nibbles: arithmetic shift treats nibbles as
            // signed 4-bit two's-complement.
            let lo_u = vand_u8(packed, low_mask);
            // (lo << 4) as i8, then >> 4 sign-extends.
            let lo_i8 = vshr_n_s8::<4>(vshl_n_s8::<4>(vreinterpret_s8_u8(lo_u)));
            let hi_i8 = vshr_n_s8::<4>(vreinterpret_s8_u8(packed));

            // Interleave into 16 lanes in (d0, d1, d2, ..., d15) order.
            // zip1/zip2 produces [lo0, hi0, lo1, hi1, ...].
            let zip_lo = vzip1_s8(lo_i8, hi_i8);
            let zip_hi = vzip2_s8(lo_i8, hi_i8);
            // Combine two int8x8 into one int8x16: dims 0..16 for this byte chunk.
            let dims_i8 = vcombine_s8(zip_lo, zip_hi);

            // Widen i8 → i16 (two halves of 8 i16 each).
            let i16_lo = vmovl_s8(vget_low_s8(dims_i8));
            let i16_hi = vmovl_s8(vget_high_s8(dims_i8));

            // i16 → i32 → f32 in 4 chunks of 4 lanes.
            let f32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_lo)));
            let f32_1 = vcvtq_f32_s32(vmovl_high_s16(i16_lo));
            let f32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(i16_hi)));
            let f32_3 = vcvtq_f32_s32(vmovl_high_s16(i16_hi));

            // Load 16 query lanes in 4 chunks.
            let q_base = rotated_q.add(b * 2);
            let q0 = vld1q_f32(q_base);
            let q1 = vld1q_f32(q_base.add(4));
            let q2 = vld1q_f32(q_base.add(8));
            let q3 = vld1q_f32(q_base.add(12));

            // Fused multiply-add into two accumulators.
            acc_a = vfmaq_f32(acc_a, q0, f32_0);
            acc_b = vfmaq_f32(acc_b, q1, f32_1);
            acc_a = vfmaq_f32(acc_a, q2, f32_2);
            acc_b = vfmaq_f32(acc_b, q3, f32_3);

            b += 8;
        }

        let acc = vaddq_f32(acc_a, acc_b);
        vaddvq_f32(acc)
    }
}

/// AVX-512 kernel for the signed 4-bit dot product. Processes 16
/// dims per loop iter (8 packed bytes) via the same low/high
/// nibble unpack as NEON, just at the wider AVX-512 conversion
/// path:
///
///   1. Load 8 packed bytes into a `__m128i`.
///   2. Extract low / high nibbles by AND/SHIFT.
///   3. Look up signed value via VPSHUFB against a per-nibble LUT
///      `[0..7, -8..-1]` (treating each 4-bit value as signed
///      two's-complement). One VPSHUFB per nibble.
///   4. Interleave low/high signed nibbles to recover dim order
///      `[d0=lo0, d1=hi0, d2=lo1, ...]` via PUNPCKLBW.
///   5. Convert 16 i8 → 16 i32 → 16 f32, FMA with `rotated_q`.
///
/// 2-way unroll (two consecutive 16-dim windows per outer iter)
/// hides the i8→i32→f32 conversion latency. The 16-wide FMA
/// throughput at f32 lands roughly on par with the NEON path on
/// per-cycle ops but processes 2× the dims, so wall-clock should
/// halve vs NEON on the same query size.
///
/// # Safety
///
/// Caller must ensure `code` has at least `bytes` valid bytes,
/// `rotated_q` has at least `bytes * 2` f32 lanes, and `bytes` is
/// a multiple of 8.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
unsafe fn signed_dot_avx512(code: *const u8, rotated_q: *const f32, bytes: usize) -> f32 {
    use std::arch::x86_64::*;

    // VPSHUFB lookup table: nibble [0..15] → signed i8 [-8..7]
    // with the two's-complement convention. Replicated across
    // both 64-bit halves of a __m128i so 8-byte VPSHUFB doesn't
    // need any mask.
    let lut = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
    let low_mask = _mm_set1_epi8(0x0F);

    let mut acc_a = _mm512_setzero_ps();
    let mut acc_b = _mm512_setzero_ps();

    let mut b = 0usize;
    while b < bytes {
        // Load 8 packed bytes into the low 64 bits of a __m128i.
        let packed = _mm_loadl_epi64(code.add(b) as *const __m128i);

        // Split into low + high nibbles.
        let lo_nibbles = _mm_and_si128(packed, low_mask);
        // `_mm_srli_epi16` shifts at 16-bit width; mask the
        // junk that the upper-byte bits leaked into the lower.
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);

        // VPSHUFB lookup: nibble [0..15] → signed i8 [-8..7].
        let lo_signed = _mm_shuffle_epi8(lut, lo_nibbles);
        let hi_signed = _mm_shuffle_epi8(lut, hi_nibbles);

        // Interleave to dim order: PUNPCKLBW produces
        // [lo0, hi0, lo1, hi1, ..., lo7, hi7] in the low half of
        // a __m128i.
        let dims_i8 = _mm_unpacklo_epi8(lo_signed, hi_signed);

        // 16 i8 → 16 i32 → 16 f32. `_mm512_cvtepi8_epi32` reads
        // the low 16 bytes of its __m128i input.
        let dims_i32 = _mm512_cvtepi8_epi32(dims_i8);
        let dims_f32 = _mm512_cvtepi32_ps(dims_i32);

        // FMA with 16 query lanes.
        let q = _mm512_loadu_ps(rotated_q.add(b * 2));
        acc_a = _mm512_fmadd_ps(q, dims_f32, acc_a);

        // 2-way unroll: next 8 bytes if available.
        if b + 8 < bytes {
            let packed2 = _mm_loadl_epi64(code.add(b + 8) as *const __m128i);
            let lo2 = _mm_and_si128(packed2, low_mask);
            let hi2 = _mm_and_si128(_mm_srli_epi16(packed2, 4), low_mask);
            let lo_s2 = _mm_shuffle_epi8(lut, lo2);
            let hi_s2 = _mm_shuffle_epi8(lut, hi2);
            let dims2 = _mm_unpacklo_epi8(lo_s2, hi_s2);
            let i32_2 = _mm512_cvtepi8_epi32(dims2);
            let f32_2 = _mm512_cvtepi32_ps(i32_2);
            let q2 = _mm512_loadu_ps(rotated_q.add(b * 2 + 16));
            acc_b = _mm512_fmadd_ps(q2, f32_2, acc_b);
            b += 16;
        } else {
            b += 8;
        }
    }

    _mm512_reduce_add_ps(_mm512_add_ps(acc_a, acc_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::dataset::rabitq_dataset::{XorShiftPub, gaussian_pub};

    #[test]
    fn stride_matches_dimensions() {
        // SIFT D=128 → 64 packed bytes (no pad to 16-multiple needed).
        assert_eq!(rabitq_b4_code_stride(128), 64);
        // GIST D=960 → 480 packed bytes (already 16-multiple).
        assert_eq!(rabitq_b4_code_stride(960), 480);
        // D=15 → ceil(15/2)=8, pad to 16.
        assert_eq!(rabitq_b4_code_stride(15), 16);
    }

    #[test]
    fn nibble_pack_unpack_roundtrip() {
        for lo in -7i8..=7 {
            for hi in -7i8..=7 {
                let packed = pack_nibbles(lo, hi);
                assert_eq!(unpack_lo(packed), lo, "lo={lo} hi={hi} byte={packed:02x}");
                assert_eq!(unpack_hi(packed), hi, "lo={lo} hi={hi} byte={packed:02x}");
            }
        }
    }

    #[test]
    fn neon_signed_dot_matches_scalar() {
        // Production sizes: SIFT D=128, GIST D=960.
        fn check<const D: usize>(seed: u64) {
            let mut rng = XorShiftPub::new(seed);
            let code_bytes = (D + 1) / 2;
            let stride = rabitq_b4_code_stride(D);
            let mut code = vec![0u8; stride];
            for b in 0..code_bytes {
                code[b] = (rng.next_u64() & 0xff) as u8;
            }
            let mut q = [0f32; D];
            for v in q.iter_mut() {
                *v = gaussian_pub(&mut rng);
            }
            let neon = signed_dot_dispatch::<D>(&code, &q);
            let scalar = signed_dot_scalar::<D>(&code, &q, 0);
            let tol = (D as f32).sqrt() * 1e-4 * scalar.abs().max(1.0);
            assert!(
                (neon - scalar).abs() < tol,
                "D={D}: neon={neon} scalar={scalar} diff={} tol={tol}",
                (neon - scalar).abs()
            );
        }
        check::<128>(11);
        check::<960>(11);
    }

    #[test]
    fn build_and_estimate_roundtrip_b4() {
        const D: usize = 64;
        let num = 100;
        let mut rng = XorShiftPub::new(99);
        let mut flat = vec![0.0f32; num * D];
        for v in flat.iter_mut() {
            *v = gaussian_pub(&mut rng);
        }
        let mut ds = InmemDataset::<f32, D>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let rbq4 = RabitQ4Dataset::<D>::build_from(&ds, 7);

        let v0: [f32; D] = std::array::from_fn(|i| flat[i]);
        let q_norm_sq: f32 = v0.iter().map(|v| v * v).sum();
        let mut rotated = [0f32; D];
        rbq4.rotate_query(&v0, &mut rotated);
        let est_self = rbq4.estimate_l2_sq(&rotated, q_norm_sq, 0);
        // With B=4 the per-vertex `tau` quantization is much tighter
        // than 1-bit's sign-only — self-distance should be well under
        // 10% of ||v0||² (vs ~50-100% for the 1-bit version).
        assert!(
            est_self < 0.1 * q_norm_sq,
            "self-distance est {est_self} too large vs ||v0||²={q_norm_sq}"
        );

        // Pair-distance accuracy.
        let v1: [f32; D] = std::array::from_fn(|i| flat[D + i]);
        let true_l2_sq: f32 = v0.iter().zip(v1.iter()).map(|(a, b)| (a - b).powi(2)).sum();
        let est_01 = rbq4.estimate_l2_sq(&rotated, q_norm_sq, 1);
        let rel_err = ((est_01 - true_l2_sq) / true_l2_sq.max(1e-3)).abs();
        // Tighter bound than B=1 (which we set at 0.5). At B=4 we
        // expect <10% relative error on a single pair-distance
        // estimate; the bound is loose to allow data-driven outliers.
        assert!(
            rel_err < 0.15,
            "pair-distance rel_err {rel_err} (est {est_01} vs true {true_l2_sq})"
        );
    }

    #[test]
    fn save_load_roundtrip_b4() {
        const D: usize = 32;
        let num = 16;
        let mut rng = XorShiftPub::new(1);
        let mut flat = vec![0.0f32; num * D];
        for v in flat.iter_mut() {
            *v = gaussian_pub(&mut rng);
        }
        let mut ds = InmemDataset::<f32, D>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let a = RabitQ4Dataset::<D>::build_from(&ds, 777);

        let path = std::env::temp_dir().join(format!("rabitq4_test_{}.qrb4", std::process::id()));
        a.save(&path).expect("save");
        let b = RabitQ4Dataset::<D>::load(&path).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(a.num_vertices, b.num_vertices);
        assert_eq!(a.stride, b.stride);
        assert_eq!(a.codes.as_slice(), b.codes.as_slice());
        assert_eq!(a.norms, b.norms);
        assert_eq!(a.taus, b.taus);
    }
}
