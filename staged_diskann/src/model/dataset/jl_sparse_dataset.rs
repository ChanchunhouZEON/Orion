/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Johnson-Lindenstrauss sparse-projection binary signature for
//! ultra-cheap pre-filter distance compute in the L2 search cascade.
//!
//! Mirrors ParlayANN's `Euclidean_JL_Sparse_Point<1024>`
//! (`algorithms/utils/euclidian_point.h`). Each vertex is encoded as a
//! `BITS`-bit signature where bit `i` is:
//!
//! ```text
//! sign( sum( x[idx[i,0..3]] ) - sum( x[idx[i,3..6]] ) )
//! ```
//!
//! `idx` is a fixed (seed-derived) table of `BITS × NZ` random
//! dimension indices into the f32 vector. Same `idx` is used for
//! every base vector and at query time so the Hamming distance
//! between two signatures correlates with the L2 distance between
//! their originals (Johnson-Lindenstrauss lemma — sparse variant
//! with sign quantization).
//!
//! ## Storage
//!
//! - `codes: AlignedBoxWithSlice<u8>` — `num_vertices × STRIDE` bytes,
//!   16-byte aligned for `vld1q_u8`. `STRIDE` = ceil(BITS/8) rounded
//!   up to a 16-byte multiple.
//! - `indices: Vec<u32>` — `BITS × NZ` random dim indices, shared
//!   across all vertices. Reproduced from `seed` so two builds with
//!   the same seed produce bit-identical signatures.
//!
//! For `BITS = 1024` (the default and what PA uses): 128 bytes/vertex.
//! GIST 1M base shrinks from 960 MB (u8) → 128 MB (JL B=1024).
//!
//! ## Distance kernel
//!
//! `hamming(p, q) = popcount(p XOR q)` — on ARM via `vcntq_u8` +
//! pairwise-add reduction to u32; on x86 via `popcnt` (scalar
//! fallback). For STRIDE=128: ~8 NEON loop iterations, ~15 cycles
//! per distance vs ~360 cycles for our u8 NEON L2 kernel at D=960 —
//! **~24× speedup per cmp**.
//!
//! Hamming distance is NOT a faithful L2 estimate — it's a noisy
//! correlate with bounded relative error. So the JL signature is
//! used **only as the cheapest prefilter** in the search cascade,
//! never as the final ranker.

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

use diskann::common::ANNResult;
use diskann::common::AlignedBoxWithSlice;
use diskann::model::InmemDataset;
use rayon::prelude::*;
use std::io::{Read, Write};
use std::path::Path;

/// Disk-format magic for the JL Sparse sidecar (`.jls`). Bumped each
/// time `NZ` changes so any cached sidecar from a previous projection
/// auto-invalidates on load (the stored sign bits depend on the same
/// `NZ` the query encoder uses, so loading codes built with one NZ
/// while the query encoder uses another would silently produce
/// garbage Hamming distances). Current value tracks NZ=12.
pub const JL_SPARSE_MAGIC: u32 = 0x4A4C_5346; // "JLSF" — bumped when L2 dropped the per-vertex norms sidecar (norms now only on JLSparseDatasetMips)

/// **Default** number of dimensions sampled per output bit. Carried
/// over from the GIST L2 sweep — see the table below. The MIPS path
/// instantiates [`JLSparseDataset`] with `NZ=11` instead (set on the
/// type via the third const-generic parameter), since the asymmetric
/// per-bit distribution on raw-MIPS data benefits from a slightly
/// wider sampling window.
///
/// Number of dimensions sampled per output bit. **NZ=9 — odd**.
///
/// GIST D=960 sweep (L2-Q, R=100):
///
/// |  NZ | parity |  pos/neg | L=48 R / QPS    | L=192 R / QPS    |
/// |-----|--------|----------|-----------------|------------------|
/// |  5  | odd    | 2/3      | 0.8962 / 43.1k  | 0.9752 / 18.1k   |
/// |  7  | odd    | 3/4      | 0.9123 / 47.3k  | 0.9795 / 17.9k   |
/// |  8  | even   | 4/4      | 0.9125 / 40.2k  | 0.9815 / 15.5k   |
/// |  9  | odd    | 4/5      | 0.9166 / 44.3k  | 0.9827 / 17.4k   |
/// | 11  | odd    | 5/6      | 0.9164 / 43.0k  | 0.9810 / 16.9k   |
/// | 12  | even   | 6/6      | 0.9116 / 39.7k  | 0.9816 / 15.8k   |
///
/// All odd-NZ probes consistently beat the even-NZ baselines —
/// confirmed parity effect. NZ=9 picks the sweet spot: higher recall
/// than NZ=12 at every L AND +10-17% QPS.
///
/// Mechanism (hypothesis): with odd NZ, the `NZ/2` integer split
/// gives asymmetric +/- counts (4 pos + 5 neg here). Each output bit
/// has a negative drift, which introduces correlation between
/// signature bits. Correlated bits → lower entropy → Hamming
/// distances cluster more tightly around their typical value →
/// running-mean × slack threshold becomes more discriminative →
/// tighter cutoff at iso-recall.
pub const NZ: usize = 9;
pub const NZ_MIPS: usize = 11;

// NZ_POS — the positive-direction index count — is now derived from
// NZ as `NZ / 2` on the impl block (see `Self::NZ_POS`). For odd NZ
// this gives a mild (NZ/2, NZ/2+1) split that drives the prefilter's
// threshold-mean discriminator. The L2 sweep below picked NZ=9 / split
// 4-5; symmetric exchanges (5-4) within noise; pushing past ~55/45
// collapses recall as the signature is dominated by one direction.

/// Per-vertex code stride in bytes. Returns the padded length of the
/// sign-bit packed code so `vld1q_u8` loads stay 16-byte aligned.
pub const fn jl_sparse_stride(bits: usize) -> usize {
    let raw = (bits + 7) / 8;
    (raw + 15) & !15
}

/// JL Sparse signature dataset.
pub struct JLSparseDataset<const N: usize, const BITS: usize = 1024, const NZ: usize = 9> {
    /// Packed sign bits, `num_vertices × STRIDE` bytes, 16-byte
    /// aligned. Bit `i` of byte `b` encodes signature bit `b × 8 + i`.
    pub codes: AlignedBoxWithSlice<u8>,
    /// Random dim indices, `BITS × NZ` u32s. Bit `i` uses
    /// `indices[i * NZ + 0..NZ/2]` (positive) and `indices[i * NZ +
    /// NZ/2..NZ]` (negative).
    pub indices: Vec<u32>,
    /// Number of vertices encoded.
    pub num_vertices: usize,
    /// Cached per-vertex stride in bytes.
    pub stride: usize,
    /// Seed used to derive `indices`. Saved in the sidecar so two
    /// readers of the same `.jls` file agree on the projection.
    pub seed: u64,
}

impl<const N: usize, const BITS: usize, const NZ: usize> JLSparseDataset<N, BITS, NZ> {
    pub const STRIDE: usize = jl_sparse_stride(BITS);
    /// First `NZ_POS` indices contribute positively, the remaining
    /// `NZ - NZ_POS` contribute negatively. Derived as `NZ / 2` so
    /// odd NZ gives a mild (NZ/2, NZ/2+1) split — the asymmetry that
    /// drives the threshold-mean discriminator in the prefilter.
    pub const NZ_POS: usize = NZ / 2;

    /// Build the signature dataset from an `InmemDataset`. Parallelised
    /// over vertices via rayon. Deterministic for a given seed.
    pub fn build_from(dataset: &InmemDataset<f32, N>, seed: u64) -> Self {
        let num_vertices = dataset.num_points;
        let stride = Self::STRIDE;

        // 1. Derive the index table from the seed via **balanced
        //    coverage**: each dim appears exactly `⌈BITS·NZ/D⌉` or
        //    `⌊BITS·NZ/D⌋` times across all `BITS × NZ` slots, then
        //    the whole multiset is Fisher-Yates shuffled. Compared to
        //    naive uniform-with-replacement sampling this:
        //      * removes per-dim coverage variance (was σ≈3.1 hits/dim
        //        on GIST; now ±1);
        //      * eliminates per-bit duplicate dims for any
        //        `NZ ≤ ⌊BITS·NZ/D⌋` — the post-shuffle slot allocation
        //        still permits the same dim landing twice in one bit's
        //        NZ window, but only with the multiset constraint;
        //      * still produces a deterministic projection from `seed`
        //        (same xorshift drives both fill and shuffle).
        let total_slots = BITS * NZ;
        let base_hits = total_slots / N;
        let extras = total_slots - base_hits * N;
        let mut indices = Vec::with_capacity(total_slots);
        for d in 0..N {
            let hits = base_hits + if d < extras { 1 } else { 0 };
            for _ in 0..hits {
                indices.push(d as u32);
            }
        }
        debug_assert_eq!(indices.len(), total_slots);

        // Fisher-Yates shuffle in place (same xorshift stream).
        let mut rng = XorShift64(seed.wrapping_add(0xD1B5_4A32_D192_ED03));
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        // 2. Allocate the code slab and encode each vertex in parallel.
        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .expect("JLSparseDataset codes alloc");

        let base = dataset.data.as_slice();
        let codes_slice = codes.as_mut_slice();

        codes_slice
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(vid, code)| {
                let x = &base[vid * N..(vid + 1) * N];
                encode_vertex_scalar::<N, BITS, NZ>(x, &indices, code);
            });

        Self {
            codes,
            indices,
            num_vertices,
            stride,
            seed,
        }
    }

    /// Encode a query into a fixed-size signature. Allocates a small
    /// stride-sized buffer on the heap; the caller is expected to do
    /// this once per query in the search setup phase.
    pub fn encode_query(&self, q: &[f32; N]) -> Box<[u8]> {
        let mut out = vec![0u8; self.stride].into_boxed_slice();
        encode_vertex_scalar::<N, BITS, NZ>(q, &self.indices, &mut out);
        out
    }

    /// Hamming distance between a query signature and the signature
    /// of vertex `vid`. Uses the NEON popcount kernel when available;
    /// scalar otherwise.
    #[inline]
    pub fn hamming(&self, q_sig: &[u8], vid: u32) -> u32 {
        let off = (vid as usize) * self.stride;
        let v_sig = &self.codes.as_slice()[off..off + self.stride];
        hamming_dispatch(q_sig, v_sig)
    }

    /// Sidecar header (32 B):
    ///   [u32 magic = JL_SPARSE_MAGIC]
    ///   [u32 version = 1]
    ///   [u32 num_vertices][u32 bits]
    ///   [u32 stride][u32 dim = N]
    ///   [u32 nz][u32 _pad]
    ///   [u64 seed]
    /// Body:
    ///   [BITS × NZ u32 indices]
    ///   [num_vertices × STRIDE u8 codes]
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 40];
        hdr[0..4].copy_from_slice(&JL_SPARSE_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&2u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(BITS as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.stride as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&(N as u32).to_le_bytes());
        hdr[24..28].copy_from_slice(&(NZ as u32).to_le_bytes());
        // hdr[28..32]: _pad
        hdr[32..40].copy_from_slice(&self.seed.to_le_bytes());
        w.write_all(&hdr)?;

        // Indices.
        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                self.indices.as_ptr() as *const u8,
                self.indices.len() * std::mem::size_of::<u32>(),
            )
        };
        w.write_all(idx_bytes)?;

        // Codes.
        w.write_all(self.codes.as_slice())?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 40];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != JL_SPARSE_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = JLSF)",
                    JL_SPARSE_MAGIC
                ),
            )
            .into());
        }
        let num_vertices = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let bits = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if bits != BITS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bits mismatch: file={bits} expected={BITS}"),
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
        let dim = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        if dim != N {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("dim mismatch: file={dim} expected={N}"),
            )
            .into());
        }
        let file_nz = u32::from_le_bytes(hdr[24..28].try_into().unwrap()) as usize;
        if file_nz != NZ {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("nz mismatch: file={file_nz} expected={NZ}"),
            )
            .into());
        }
        let seed = u64::from_le_bytes(hdr[32..40].try_into().unwrap());

        // Indices.
        let mut indices = vec![0u32; BITS * NZ];
        let idx_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                indices.as_mut_ptr() as *mut u8,
                indices.len() * std::mem::size_of::<u32>(),
            )
        };
        r.read_exact(idx_bytes)?;

        // Codes.
        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        r.read_exact(codes.as_mut_slice())?;

        Ok(Self {
            codes,
            indices,
            num_vertices,
            stride,
            seed,
        })
    }
}

// ── Encoder (scalar; parallelised over vertices in build_from) ────────

/// Encode one vector into a signature byte buffer of length
/// `STRIDE`. `code` is zero-filled before encoding.
#[inline]
fn encode_vertex_scalar<const N: usize, const BITS: usize, const NZ: usize>(
    x: &[f32],
    indices: &[u32],
    code: &mut [u8],
) {
    let nz_pos = NZ / 2;
    code.fill(0);
    for bit_idx in 0..BITS {
        let off = bit_idx * NZ;
        // Positive contribution: first NZ_POS indices.
        // Negative contribution: remaining (NZ - NZ_POS) indices.
        let mut v = 0.0f32;
        for j in 0..nz_pos {
            v += x[indices[off + j] as usize];
        }
        for j in nz_pos..NZ {
            v -= x[indices[off + j] as usize];
        }
        // Sign bit. `v > 0.0` matches PA's `(vv > 0)` convention.
        if v > 0.0 {
            code[bit_idx >> 3] |= 1u8 << (bit_idx & 7);
        }
    }
}

// ── MIPS variant ──────────────────────────────────────────────────────

/// Disk-format magic for the MIPS JL Sparse sidecar (`.jls_mips`).
/// Distinct from [`JL_SPARSE_MAGIC`] so a wrong-mode cache load fails
/// at the magic-check step.
pub const JL_SPARSE_MIPS_MAGIC: u32 = 0x4A4C_4D38; // "JLM8"

/// MIPS twin of [`JLSparseDataset`]. Mirrors PA's
/// `Mips_JL_Sparse_Point_Normalized` (`jl_point.h:286`): same sparse
/// sign-projection codes, but with a per-vertex L2 norm sidecar that
/// the MIPS prefilter weights into its distance estimate as
/// `popcount × ‖v‖` (PA `jl_point.h:319`).
///
/// `NZ` defaults to 8 — different from the L2 default (9) per the
/// empirical sweep below. The encoding shape is identical to
/// [`JLSparseDataset`], so the two share
/// [`encode_vertex_scalar`].
pub struct JLSparseDatasetMips<const N: usize, const BITS: usize = 1024, const NZ: usize = 9> {
    /// Same sign-bit code layout as [`JLSparseDataset::codes`].
    pub codes: AlignedBoxWithSlice<u8>,
    /// Same `BITS × NZ` index table as [`JLSparseDataset::indices`].
    pub indices: Vec<u32>,
    /// Per-vertex L2 norm `‖x‖` (f32). Cost 4 B/vertex (negligible
    /// vs the 128 B/vertex code slab at BITS=1024). PA reads this
    /// as `radius` and multiplies it into the per-candidate
    /// distance estimate (`jl_point.h:319`).
    pub norms: AlignedBoxWithSlice<f32>,
    pub num_vertices: usize,
    pub stride: usize,
    pub seed: u64,
}

impl<const N: usize, const BITS: usize, const NZ: usize> JLSparseDatasetMips<N, BITS, NZ> {
    pub const STRIDE: usize = jl_sparse_stride(BITS);
    pub const NZ_POS: usize = NZ / 2;

    pub fn build_from(dataset: &diskann::model::InmemDataset<f32, N>, seed: u64) -> Self {
        let num_vertices = dataset.num_points;
        let stride = Self::STRIDE;

        // Same balanced-coverage index layout as the L2 variant —
        // only `NZ` differs.
        let total_slots = BITS * NZ;
        let base_hits = total_slots / N;
        let extras = total_slots - base_hits * N;
        let mut indices = Vec::with_capacity(total_slots);
        for d in 0..N {
            let hits = base_hits + if d < extras { 1 } else { 0 };
            for _ in 0..hits {
                indices.push(d as u32);
            }
        }
        debug_assert_eq!(indices.len(), total_slots);

        let mut rng = XorShift64(seed.wrapping_add(0xD1B5_4A32_D192_ED03));
        for i in (1..indices.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            indices.swap(i, j);
        }

        // Allocate codes + norms; encode in a single parallel sweep.
        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .expect("JLSparseDatasetMips codes alloc");
        let mut norms = AlignedBoxWithSlice::<f32>::new(num_vertices, 32)
            .expect("JLSparseDatasetMips norms alloc");

        let base = dataset.data.as_slice();
        let codes_slice = codes.as_mut_slice();
        let norms_slice = norms.as_mut_slice();

        codes_slice
            .par_chunks_mut(stride)
            .zip(norms_slice.par_iter_mut())
            .enumerate()
            .for_each(|(vid, (code, norm_slot))| {
                let x = &base[vid * N..(vid + 1) * N];
                encode_vertex_scalar::<N, BITS, NZ>(x, &indices, code);
                let mut s = 0.0f64;
                for &v in x {
                    s += (v as f64) * (v as f64);
                }
                *norm_slot = s.sqrt() as f32;
            });

        Self {
            codes,
            indices,
            norms,
            num_vertices,
            stride,
            seed,
        }
    }

    pub fn encode_query(&self, q: &[f32; N]) -> Box<[u8]> {
        let mut out = vec![0u8; self.stride].into_boxed_slice();
        encode_vertex_scalar::<N, BITS, NZ>(q, &self.indices, &mut out);
        out
    }

    #[inline]
    pub fn hamming(&self, q_sig: &[u8], vid: u32) -> u32 {
        let off = (vid as usize) * self.stride;
        let v_sig = &self.codes.as_slice()[off..off + self.stride];
        hamming_dispatch(q_sig, v_sig)
    }

    /// MIPS-weighted JL distance estimate: `popcount × ‖v‖`. The
    /// per-query `‖q‖` constant is dropped from the formula (PA does
    /// the same — `jl_point.h:319` comments out `* qr`) because it
    /// does not affect ranking. `smaller == closer` (signature
    /// agreement combined with small norm contribution wins).
    #[inline]
    pub fn mips_distance(&self, q_sig: &[u8], vid: u32) -> f32 {
        let h = self.hamming(q_sig, vid) as f32;
        h * self.norms.as_slice()[vid as usize]
    }

    /// Sidecar header (40 B), same shape as the L2 variant's but
    /// with [`JL_SPARSE_MIPS_MAGIC`] in the magic slot.
    /// Body: `BITS × NZ` u32 indices, then `num_vertices × STRIDE`
    /// u8 codes, then `num_vertices` × f32 norms.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut hdr = [0u8; 40];
        hdr[0..4].copy_from_slice(&JL_SPARSE_MIPS_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(BITS as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(self.stride as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&(N as u32).to_le_bytes());
        hdr[24..28].copy_from_slice(&(NZ as u32).to_le_bytes());
        hdr[32..40].copy_from_slice(&self.seed.to_le_bytes());
        w.write_all(&hdr)?;

        let idx_bytes = unsafe {
            std::slice::from_raw_parts(
                self.indices.as_ptr() as *const u8,
                self.indices.len() * std::mem::size_of::<u32>(),
            )
        };
        w.write_all(idx_bytes)?;
        w.write_all(self.codes.as_slice())?;
        let norm_bytes = unsafe {
            std::slice::from_raw_parts(
                self.norms.as_slice().as_ptr() as *const u8,
                self.norms.len() * std::mem::size_of::<f32>(),
            )
        };
        w.write_all(norm_bytes)?;
        w.flush()?;
        Ok(())
    }

    pub fn load<P: AsRef<Path>>(path: P) -> ANNResult<Self> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut hdr = [0u8; 40];
        r.read_exact(&mut hdr)?;
        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != JL_SPARSE_MIPS_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bad magic 0x{magic:08x} (expected 0x{:08x} = JLM8)",
                    JL_SPARSE_MIPS_MAGIC
                ),
            )
            .into());
        }
        let num_vertices = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let bits = u32::from_le_bytes(hdr[12..16].try_into().unwrap()) as usize;
        if bits != BITS {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bits mismatch: file={bits} expected={BITS}"),
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
        let dim = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        if dim != N {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("dim mismatch: file={dim} expected={N}"),
            )
            .into());
        }
        let file_nz = u32::from_le_bytes(hdr[24..28].try_into().unwrap()) as usize;
        if file_nz != NZ {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("nz mismatch: file={file_nz} expected={NZ}"),
            )
            .into());
        }
        let seed = u64::from_le_bytes(hdr[32..40].try_into().unwrap());

        let mut indices = vec![0u32; BITS * NZ];
        let idx_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                indices.as_mut_ptr() as *mut u8,
                indices.len() * std::mem::size_of::<u32>(),
            )
        };
        r.read_exact(idx_bytes)?;

        let mut codes = AlignedBoxWithSlice::<u8>::new(num_vertices * stride, 16)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        r.read_exact(codes.as_mut_slice())?;

        let mut norms = AlignedBoxWithSlice::<f32>::new(num_vertices, 32)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{e:?}")))?;
        let norm_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                norms.as_mut_slice().as_mut_ptr() as *mut u8,
                norms.len() * std::mem::size_of::<f32>(),
            )
        };
        r.read_exact(norm_bytes)?;

        Ok(Self {
            codes,
            indices,
            norms,
            num_vertices,
            stride,
            seed,
        })
    }
}

// ── Hamming distance kernel ───────────────────────────────────────────

/// Dispatch: NEON on AArch64, scalar otherwise. The two paths are
/// kept bit-identical and exercised against each other in the unit
/// test below.
#[inline]
pub fn hamming_dispatch(p: &[u8], q: &[u8]) -> u32 {
    debug_assert_eq!(p.len(), q.len(), "Hamming inputs must match length");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        return unsafe { hamming_neon(p.as_ptr(), q.as_ptr(), p.len()) };
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512vpopcntdq"
    ))]
    {
        return unsafe { hamming_avx512_vpopcnt(p.as_ptr(), q.as_ptr(), p.len()) };
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        not(target_feature = "avx512vpopcntdq")
    ))]
    {
        return unsafe { hamming_avx512_harley_seal(p.as_ptr(), q.as_ptr(), p.len()) };
    }
    #[allow(unreachable_code)]
    hamming_scalar(p, q)
}

#[inline]
pub fn hamming_scalar(p: &[u8], q: &[u8]) -> u32 {
    let mut sum: u32 = 0;
    for i in 0..p.len() {
        sum += (p[i] ^ q[i]).count_ones();
    }
    sum
}

/// NEON popcount kernel. Processes 16 bytes per iteration via
/// `veorq_u8` + `vcntq_u8` + pairwise-add widen to u32. Accumulator
/// is u32 with horizontal sum at the end.
///
/// # Safety
///
/// Caller must ensure both `p` and `q` point to at least `len` valid
/// bytes and `len` is a 16-byte multiple.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn hamming_neon(p: *const u8, q: *const u8, len: usize) -> u32 {
    unsafe {
        debug_assert_eq!(len % 16, 0);
        let mut acc = vdupq_n_u32(0);
        let n_regs = len / 16;
        for i in 0..n_regs {
            let vp = vld1q_u8(p.add(i * 16));
            let vq = vld1q_u8(q.add(i * 16));
            // popcount(p XOR q): per-byte popcount of XOR result.
            let vc = vcntq_u8(veorq_u8(vp, vq));
            // Widen u8 lanes → u32 lanes via two pairwise-add stages.
            // After vpaddlq_u8: 8 lanes of u16. After vpaddlq_u16: 4 lanes
            // of u32. Each u32 lane holds the sum of 4 input u8 bytes.
            let h16 = vpaddlq_u8(vc);
            let h32 = vpaddlq_u16(h16);
            acc = vaddq_u32(acc, h32);
        }
        vaddvq_u32(acc)
    }
}

/// AVX-512 fast path using `_mm512_popcnt_epi64` (requires
/// `avx512vpopcntdq`, available on Ice Lake-Server, Tiger Lake,
/// Sapphire Rapids, Zen 4+). One VPOPCNTQ per 64-byte block vs
/// the multi-stage Harley-Seal fallback below.
///
/// # Safety
/// `p` and `q` must be valid for `len` bytes; `len` must be a
/// 16-byte multiple to match the NEON contract.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vpopcntdq"
))]
#[inline]
unsafe fn hamming_avx512_vpopcnt(p: *const u8, q: *const u8, len: usize) -> u32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(len % 16, 0);
    let mut acc = _mm512_setzero_si512();
    let n_blocks = len / 64;
    for i in 0..n_blocks {
        let vp = _mm512_loadu_si512(p.add(i * 64) as *const __m512i);
        let vq = _mm512_loadu_si512(q.add(i * 64) as *const __m512i);
        let vx = _mm512_xor_si512(vp, vq);
        // 8× i64 lanes each holding popcount of a 64-bit chunk.
        let pc = _mm512_popcnt_epi64(vx);
        acc = _mm512_add_epi64(acc, pc);
    }
    let mut total = _mm512_reduce_add_epi64(acc) as u32;

    // Tail: 16-byte chunks (NEON's natural stride). Process via
    // SSE2 + scalar popcount since AVX-512 wants 64-byte aligned
    // ops at full width.
    let mut i = n_blocks * 64;
    while i + 16 <= len {
        let lo =
            (p.add(i) as *const u64).read_unaligned() ^ (q.add(i) as *const u64).read_unaligned();
        let hi = (p.add(i + 8) as *const u64).read_unaligned()
            ^ (q.add(i + 8) as *const u64).read_unaligned();
        total += lo.count_ones() + hi.count_ones();
        i += 16;
    }
    total
}

/// AVX-512 fallback when `avx512vpopcntdq` isn't available
/// (Skylake-SP, Cascade Lake). Uses the Wojciech Mula bit-slicing
/// popcount: bit-parallel half-adders that emulate per-byte
/// popcount in ~6 vector ops over 64 bytes.
///
/// Slower than the dedicated VPOPCNTQ instruction (~3× more
/// uops) but still 4-5× faster than Harley-Seal on AVX2.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    not(target_feature = "avx512vpopcntdq")
))]
#[inline]
unsafe fn hamming_avx512_harley_seal(p: *const u8, q: *const u8, len: usize) -> u32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(len % 16, 0);

    // Per-nibble popcount LUT broadcast to all 64 lanes of a __m512i.
    // _mm512_shuffle_epi8 selects per-byte; we shuffle low nibbles
    // and shifted-high nibbles separately and add.
    let lut = _mm512_set_epi8(
        4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1,
        1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0, 4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1,
        2, 1, 1, 0,
    );
    let low_mask = _mm512_set1_epi8(0x0F);

    let mut acc = _mm512_setzero_si512();
    let n_blocks = len / 64;
    for i in 0..n_blocks {
        let vp = _mm512_loadu_si512(p.add(i * 64) as *const __m512i);
        let vq = _mm512_loadu_si512(q.add(i * 64) as *const __m512i);
        let vx = _mm512_xor_si512(vp, vq);
        // Low nibble lookup.
        let lo = _mm512_and_si512(vx, low_mask);
        // High nibble = (x >> 4) & 0x0F.
        let hi = _mm512_and_si512(_mm512_srli_epi16(vx, 4), low_mask);
        let pl = _mm512_shuffle_epi8(lut, lo);
        let ph = _mm512_shuffle_epi8(lut, hi);
        let per_byte = _mm512_add_epi8(pl, ph);
        // Sum byte lanes into 8× i64 accumulators via VPSADBW
        // (reduces 8 bytes → one i64 per dqword).
        let widened = _mm512_sad_epu8(per_byte, _mm512_setzero_si512());
        acc = _mm512_add_epi64(acc, widened);
    }
    let mut total = _mm512_reduce_add_epi64(acc) as u32;

    // Tail (same shape as the vpopcnt path).
    let mut i = n_blocks * 64;
    while i + 16 <= len {
        let lo =
            (p.add(i) as *const u64).read_unaligned() ^ (q.add(i) as *const u64).read_unaligned();
        let hi = (p.add(i + 8) as *const u64).read_unaligned()
            ^ (q.add(i + 8) as *const u64).read_unaligned();
        total += lo.count_ones() + hi.count_ones();
        i += 16;
    }
    total
}

// ── Deterministic PRNG (shared shape with rabitq_dataset) ─────────────

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
    fn stride_matches_paper_examples() {
        // BITS=1024 (PA default): 128 bytes, no padding needed.
        assert_eq!(jl_sparse_stride(1024), 128);
        // BITS=512: 64 bytes.
        assert_eq!(jl_sparse_stride(512), 64);
        // BITS=100: ceil(100/8)=13 bytes → padded to 16.
        assert_eq!(jl_sparse_stride(100), 16);
    }

    #[test]
    fn neon_hamming_matches_scalar() {
        // STRIDE=128 (BITS=1024 production size).
        let mut p = vec![0u8; 128];
        let mut q = vec![0u8; 128];
        // Seed with a deterministic xorshift so test is reproducible.
        let mut rng = XorShift64(42);
        for i in 0..128 {
            p[i] = (rng.next_u64() & 0xff) as u8;
            q[i] = (rng.next_u64() & 0xff) as u8;
        }
        let neon = hamming_dispatch(&p, &q);
        let scalar = hamming_scalar(&p, &q);
        assert_eq!(neon, scalar, "NEON popcount must equal scalar");

        // Also exercise STRIDE=64 (BITS=512) to catch any STRIDE
        // assumptions inside the NEON loop.
        let neon_64 = hamming_dispatch(&p[..64], &q[..64]);
        let scalar_64 = hamming_scalar(&p[..64], &q[..64]);
        assert_eq!(neon_64, scalar_64);
    }

    #[test]
    fn hamming_self_distance_zero() {
        // A signature compared to itself must have Hamming distance 0.
        let mut rng = XorShift64(7);
        let p: Vec<u8> = (0..128).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        assert_eq!(hamming_dispatch(&p, &p), 0);
    }

    #[test]
    fn build_and_query_correlation() {
        // Sanity: queries near a vertex should have small Hamming
        // distance to that vertex's signature; queries far should
        // have large Hamming. Test on a small synthetic dataset.
        const D: usize = 64;
        const BITS: usize = 256;
        let num = 20;

        let mut rng = XorShift64(1);
        let mut flat = vec![0f32; num * D];
        for v in flat.iter_mut() {
            *v = gaussian(&mut rng);
        }
        let mut ds = InmemDataset::<f32, D>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let jl = JLSparseDataset::<D, BITS>::build_from(&ds, 99);

        // Query equal to vertex 0 → Hamming distance to vertex 0 must
        // be 0 (since the encoder is deterministic).
        let q0: [f32; D] = std::array::from_fn(|i| flat[i]);
        let q0_sig = jl.encode_query(&q0);
        assert_eq!(jl.hamming(&q0_sig, 0), 0);

        // Query equal to vertex 0 should have Hamming distance >0 to
        // (essentially every) other vertex. Spot-check vertex 1.
        let h01 = jl.hamming(&q0_sig, 1);
        assert!(h01 > 0, "Hamming dist q0 → vertex 1 should be non-zero");
        assert!(
            h01 < BITS as u32,
            "Hamming dist must be < BITS (signatures shouldn't be bit-flipped of each other)"
        );
    }

    #[test]
    fn save_load_roundtrip() {
        const D: usize = 32;
        const BITS: usize = 128;
        let num = 16;
        let mut rng = XorShift64(11);
        let mut flat = vec![0f32; num * D];
        for v in flat.iter_mut() {
            *v = gaussian(&mut rng);
        }
        let mut ds = InmemDataset::<f32, D>::new(num, 1.0).unwrap();
        ds.data.memcpy(&flat).unwrap();
        let a = JLSparseDataset::<D, BITS>::build_from(&ds, 777);

        let path = std::env::temp_dir().join(format!("jl_test_{}.jls", std::process::id()));
        a.save(&path).expect("save");
        let b = JLSparseDataset::<D, BITS>::load(&path).expect("load");
        std::fs::remove_file(&path).ok();

        assert_eq!(a.num_vertices, b.num_vertices);
        assert_eq!(a.stride, b.stride);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.indices, b.indices);
        assert_eq!(a.codes.as_slice(), b.codes.as_slice());
    }

    // Minimal Box-Muller for test data generation. Inlined here so the
    // test file doesn't depend on `rabitq_dataset`'s public xor/gauss
    // helpers (keeps the module self-contained).
    fn gaussian(rng: &mut XorShift64) -> f32 {
        let u1 = ((rng.next_u64() >> 40) as f32 / ((1u32 << 24) as f32))
            .max(1e-7)
            .min(1.0 - 1e-7);
        let u2 = ((rng.next_u64() >> 40) as f32 / ((1u32 << 24) as f32))
            .max(1e-7)
            .min(1.0 - 1e-7);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}
