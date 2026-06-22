/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! JL Hadamard — dense, structured-random binary signature for the
//! prefilter tier.
//!
//! Each output bit is the sign of a sum-of-all-D-coordinates with
//! pseudorandom `±1` weights, computed via three rounds of
//! `random-sign-flip + FWHT (HDHDHDx)`. Same on-disk + in-RAM layout
//! as [`super::jl_sparse_dataset::JLSparseDataset`] (bit-packed sign
//! codes, LSB-first per byte) so the search-time hot path reuses
//! `JLHammingDistance` unchanged.
//!
//! ## Why HDHD over JL Sparse
//!
//! JL Sparse with NZ=9 gives each bit ~9 random projections of D=960
//! → bit-level signal is noisy. The HDHD construction (Ailon-Chazelle
//! 2006) makes each output bit a sign of a Gaussian-like sum over
//! **all D dims**, jumping per-bit information content from ~3
//! effective bits to ~7+. Same byte budget (D_PAD bits = 128 B/vert
//! on GIST) → tighter Hamming-vs-true-L2 correlation → more
//! discriminative cutoff at iso-recall.
//!
//! ## FLOP budget
//!
//! Per encode (vertex or query) at D=960, D_PAD=1024:
//!   * 3 × FWHT-1024 = 3 × 1024 × 10 ≈ 30,720 add/sub ops
//!   * 3 × D_PAD sign-flip multiplies ≈ 3,072
//!   * 1 × D_PAD comparisons + bit-packing ≈ 1,024
//!   * Total ≈ 34k FLOPs, ~15-25µs scalar.
//!
//! JL Sparse: ~6k FLOPs / ~7µs setup. Hadamard pays 2-3× the setup
//! for dense-per-bit signal — still <1% of total wall-time at L=192.

use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use diskann::model::InmemDataset;
use rayon::prelude::*;
use std::io::{Read, Write};
use std::path::Path;

/// Disk-format magic for the JL Hadamard sidecar (`.jlh`). Bumped on
/// any layout change so stale caches refuse to load instead of
/// silently mis-decoding.
pub const JL_HADAMARD_MAGIC: u32 = 0x4A4C_4843; // "JLHC"

/// Round `n` up to the next power of two.
pub const fn next_pow2(n: usize) -> usize {
    if n.is_power_of_two() {
        return n;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Per-vertex code stride in bytes — derived from `D_PAD` plus a
/// floor at 32 bytes so the `JLHammingDistance` NEON kernel
/// (32-byte chunk) never reads past the buffer end. Padding bytes
/// XOR to zero between matched queries / codes (both stay all-zero
/// in the tail), so the padding doesn't bias Hamming distance.
pub const fn jl_hadamard_stride(d_pad: usize) -> usize {
    let raw = (d_pad + 7) / 8;
    let floored = if raw < 32 { 32 } else { raw };
    (floored + 15) & !15
}

/// JL Hadamard signature dataset.
///
/// Generic over the source f32 dim `N` and the padded FWHT length
/// `D_PAD` (must be a power of two ≥ `N`). For datasets whose `N` is
/// already a power of two, `D_PAD == N`.
pub struct JlHadamardDataset<const N: usize, const D_PAD: usize> {
    /// Packed sign bits, `num_vertices × STRIDE` bytes, 16-byte
    /// aligned. Bit `i` of byte `b` encodes signature bit `b × 8 + i`.
    pub codes: AlignedBoxWithSlice<u8>,
    /// First `±1` sign-flip diagonal, length `D_PAD`. Stored as `i8`
    /// for cache compactness (3 × D_PAD bytes for all three signs <
    /// 1 cache line at D_PAD=1024).
    pub signs1: Vec<i8>,
    pub signs2: Vec<i8>,
    pub signs3: Vec<i8>,
    pub num_vertices: usize,
    /// Cached per-vertex stride in bytes.
    pub stride: usize,
    /// Seed used to derive `signs{1,2,3}`. Saved in the sidecar.
    pub seed: u64,
}

impl<const N: usize, const D_PAD: usize> JlHadamardDataset<N, D_PAD> {
    pub const STRIDE: usize = jl_hadamard_stride(D_PAD);

    /// Build the signature dataset from an `InmemDataset`. Parallelised
    /// over vertices via rayon. Deterministic for a given seed.
    pub fn build_from(dataset: &InmemDataset<f32, N>, seed: u64) -> Self {
        assert!(
            D_PAD.is_power_of_two() && D_PAD >= N,
            "D_PAD must be a power of two ≥ N (got D_PAD={D_PAD}, N={N})"
        );

        let num_vertices = dataset.num_points;
        let stride = Self::STRIDE;

        // Derive three sign-flip diagonals from the seed. xorshift64
        // for determinism; offsets pull non-overlapping streams.
        let mut rng = XorShift64(seed.wrapping_add(0x8B9C_4D7E_25F1_A302));
        let mut signs1 = vec![0i8; D_PAD];
        let mut signs2 = vec![0i8; D_PAD];
        let mut signs3 = vec![0i8; D_PAD];
        for s in [&mut signs1, &mut signs2, &mut signs3] {
            for v in s.iter_mut() {
                *v = if (rng.next_u64() & 1) == 0 { 1 } else { -1 };
            }
        }

        // Allocate the code slab and encode each vertex in parallel.
        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .expect("JlHadamardDataset codes alloc");

        let base = dataset.data.as_slice();
        let codes_slice = codes.as_mut_slice();

        codes_slice
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(vid, code)| {
                let x = &base[vid * N..(vid + 1) * N];
                let mut buf = vec![0.0f32; D_PAD];
                encode_vector::<N, D_PAD>(x, &signs1, &signs2, &signs3, &mut buf, code);
            });

        Self {
            codes,
            signs1,
            signs2,
            signs3,
            num_vertices,
            stride,
            seed,
        }
    }

    /// Encode a query into a fixed-size signature.
    pub fn encode_query(&self, q: &[f32; N]) -> Box<[u8]> {
        let mut out = vec![0u8; self.stride].into_boxed_slice();
        let mut buf = vec![0.0f32; D_PAD];
        encode_vector::<N, D_PAD>(
            q,
            &self.signs1,
            &self.signs2,
            &self.signs3,
            &mut buf,
            &mut out,
        );
        out
    }

    /// Hamming distance between a query signature and the signature
    /// of vertex `vid`. Reuses the JL Sparse kernel since the byte
    /// layout is identical.
    #[inline]
    pub fn hamming(&self, q_sig: &[u8], vid: u32) -> u32 {
        let off = (vid as usize) * self.stride;
        let v_sig = &self.codes.as_slice()[off..off + self.stride];
        super::jl_sparse_dataset::hamming_dispatch(q_sig, v_sig)
    }

    /// Sidecar header (32 B):
    ///   [u32 magic = JL_HADAMARD_MAGIC]
    ///   [u32 version = 1]
    ///   [u32 num_vertices][u32 dim = N]
    ///   [u32 d_pad = D_PAD][u32 stride]
    ///   [u64 seed]
    /// Body:
    ///   [D_PAD i8 signs1][D_PAD i8 signs2][D_PAD i8 signs3]
    ///   [num_vertices × STRIDE u8 codes]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(&JL_HADAMARD_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(N as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(D_PAD as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&(self.stride as u32).to_le_bytes());
        hdr[24..32].copy_from_slice(&self.seed.to_le_bytes());
        w.write_all(&hdr)?;

        let s_bytes = |s: &[i8]| -> &[u8] {
            unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
        };
        w.write_all(s_bytes(&self.signs1))?;
        w.write_all(s_bytes(&self.signs2))?;
        w.write_all(s_bytes(&self.signs3))?;
        w.write_all(self.codes.as_slice())?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 32];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != JL_HADAMARD_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = JLHC)",
                    JL_HADAMARD_MAGIC
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
        let d_pad = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        if d_pad != D_PAD {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("d_pad mismatch: file={d_pad} expected={D_PAD}"),
            )
            .into());
        }
        let stride = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        if stride != Self::STRIDE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("stride mismatch: file={stride} expected={}", Self::STRIDE),
            )
            .into());
        }
        let seed = u64::from_le_bytes(hdr[24..32].try_into().unwrap());

        let mut signs1 = vec![0i8; D_PAD];
        let mut signs2 = vec![0i8; D_PAD];
        let mut signs3 = vec![0i8; D_PAD];
        let read_signs = |r: &mut std::io::BufReader<std::fs::File>,
                          s: &mut [i8]|
         -> ANNResult<()> {
            let buf = unsafe { std::slice::from_raw_parts_mut(s.as_mut_ptr() as *mut u8, s.len()) };
            r.read_exact(buf)?;
            Ok(())
        };
        read_signs(&mut r, &mut signs1)?;
        read_signs(&mut r, &mut signs2)?;
        read_signs(&mut r, &mut signs3)?;

        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        r.read_exact(codes.as_mut_slice())?;

        Ok(Self {
            codes,
            signs1,
            signs2,
            signs3,
            num_vertices,
            stride,
            seed,
        })
    }
}

// ── FWHT + encoder ────────────────────────────────────────────────────

/// Encode one vector (vertex or query) — zero-pad to `D_PAD`, run
/// `H D₃ H D₂ H D₁` in place on the buffer, then sign-pack the first
/// `D_PAD` bits into the output code slot.
#[inline]
fn encode_vector<const N: usize, const D_PAD: usize>(
    x: &[f32],
    signs1: &[i8],
    signs2: &[i8],
    signs3: &[i8],
    buf: &mut [f32],
    code: &mut [u8],
) {
    debug_assert_eq!(buf.len(), D_PAD);

    // Zero-extend x into the FWHT scratch buffer.
    buf[..N].copy_from_slice(x);
    for v in &mut buf[N..] {
        *v = 0.0;
    }

    // Round 1: D₁ then H
    apply_signs(buf, signs1);
    fwht(buf);
    // Round 2: D₂ then H
    apply_signs(buf, signs2);
    fwht(buf);
    // Round 3: D₃ then H
    apply_signs(buf, signs3);
    fwht(buf);

    // Sign-pack into the code byte slot. Padded code tail (beyond
    // D_PAD bits) stays zero — same for query and base, so it
    // contributes zero to Hamming distance.
    for byte in code.iter_mut() {
        *byte = 0;
    }
    for bit_idx in 0..D_PAD {
        if buf[bit_idx] > 0.0 {
            code[bit_idx >> 3] |= 1u8 << (bit_idx & 7);
        }
    }
}

/// Apply a `±1` diagonal in place. `signs` is i8 with values in
/// `{+1, -1}`; we cast to f32 and multiply. Trivially auto-vectorisable.
#[inline]
fn apply_signs(x: &mut [f32], signs: &[i8]) {
    debug_assert_eq!(x.len(), signs.len());
    for i in 0..x.len() {
        x[i] *= signs[i] as f32;
    }
}

/// In-place Walsh-Hadamard transform on `x` (length must be a power
/// of two). Scalar; relies on auto-vectorisation for the tight inner
/// loop. NEON specialisation is a follow-up if this becomes the
/// hotspot.
#[inline]
fn fwht(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut half = 1;
    while half < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + half) {
                let a = x[j];
                let b = x[j + half];
                x[j] = a + b;
                x[j + half] = a - b;
            }
            i += 2 * half;
        }
        half <<= 1;
    }
}

// ── Deterministic PRNG ────────────────────────────────────────────────

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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_self_inverse_up_to_scale() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let orig = x;
        fwht(&mut x);
        fwht(&mut x);
        let n = 4.0;
        for i in 0..4 {
            assert!((x[i] - orig[i] * n).abs() < 1e-5);
        }
    }

    #[test]
    fn hadamard_encode_round_trip_signs() {
        // Two identical queries produce identical signatures.
        let q = [0.5f32; 8];
        let mut buf1 = [0.0f32; 8];
        let mut buf2 = [0.0f32; 8];
        let mut code1 = [0u8; 32];
        let mut code2 = [0u8; 32];
        let signs1 = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
        let signs2 = vec![1i8, 1, -1, -1, 1, 1, -1, -1];
        let signs3 = vec![1i8, 1, 1, 1, -1, -1, -1, -1];
        encode_vector::<8, 8>(&q, &signs1, &signs2, &signs3, &mut buf1, &mut code1);
        encode_vector::<8, 8>(&q, &signs1, &signs2, &signs3, &mut buf2, &mut code2);
        assert_eq!(code1, code2);
    }
}
