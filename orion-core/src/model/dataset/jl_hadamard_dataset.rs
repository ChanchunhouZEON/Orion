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
//! effective bits to ~7+. Same byte budget (d_pad bits = 128 B/vert
//! on GIST) → tighter Hamming-vs-true-L2 correlation → more
//! discriminative cutoff at iso-recall.
//!
//! ## FLOP budget
//!
//! Per encode (vertex or query) at D=960, d_pad=1024:
//!   * 3 × FWHT-1024 = 3 × 1024 × 10 ≈ 30,720 add/sub ops
//!   * 3 × d_pad sign-flip multiplies ≈ 3,072
//!   * 1 × d_pad comparisons + bit-packing ≈ 1,024
//!   * Total ≈ 34k FLOPs, ~15-25µs scalar.
//!
//! JL Sparse: ~6k FLOPs / ~7µs setup. Hadamard pays 2-3× the setup
//! for dense-per-bit signal — still <1% of total wall-time at L=192.

use super::quantized_dataset::{QuantizedDataset, QuantizedDatasetShape, ResidentMemory};
use super::rows::F32Rows;
use super::validation::{self, CheckedRows};
use crate::Result;
use orion_ann_diskann::common::AlignedBoxWithSlice;
use rayon::prelude::*;

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

/// Per-vertex code stride in bytes — derived from `d_pad` plus a
/// floor at one `JLHammingDistance` block so neither NEON nor AVX-512
/// reads past the buffer end. Padding bytes
/// XOR to zero between matched queries / codes (both stay all-zero
/// in the tail), so the padding doesn't bias Hamming distance.
pub const fn jl_hadamard_stride(d_pad: usize) -> usize {
    let bytes = jl_hadamard_stride_disk(d_pad);
    let block = <orion_ann_vector::JLHammingDistance as orion_ann_vector::DistanceKernelBackend>::CHUNK_BYTES;
    bytes.div_ceil(block) * block
}

pub const fn jl_hadamard_stride_disk(d_pad: usize) -> usize {
    let raw = (d_pad + 7) / 8;
    let floored = if raw < 32 { 32 } else { raw };
    (floored + 15) & !15
}

/// JL Hadamard signature dataset.
///
/// Both the logical input dimension and FWHT output length are runtime values.
/// Default recipes keep 1024 output bits through dimension 1024, then use the
/// next power of two. `build_with_padding` selects an explicit output width.
pub struct JlHadamardDataset {
    pub(crate) dimension: usize,
    pub(crate) d_pad: usize,
    /// Packed sign bits, `num_vertices × STRIDE` bytes, 16-byte
    /// aligned. Bit `i` of byte `b` encodes signature bit `b × 8 + i`.
    pub(crate) codes: AlignedBoxWithSlice<u8>,
    /// First `±1` sign-flip diagonal, length `d_pad`. Stored as `i8`
    /// for cache compactness (3 × d_pad bytes for all three signs).
    pub(crate) signs1: Vec<i8>,
    pub(crate) signs2: Vec<i8>,
    pub(crate) signs3: Vec<i8>,
    pub(crate) num_vertices: usize,
    /// Cached per-vertex stride in bytes.
    pub(crate) stride: usize,
    /// Seed used to derive `signs{1,2,3}`. Saved in the sidecar.
    pub(crate) seed: u64,
}

impl JlHadamardDataset {
    /// Build the signature dataset from an `InmemDataset`. Parallelised
    /// over vertices via rayon. Deterministic for a given seed.
    /// Convenience constructor for already validated data.
    ///
    /// # Panics
    /// Panics on invalid input or allocation failure. Use [`Self::try_build_from`]
    /// to propagate a structured error at application boundaries.
    pub fn build_from(dataset: &impl F32Rows, seed: u64) -> Self {
        Self::try_build_from(dataset, seed).expect("invalid sidecar input")
    }

    /// Construct a sidecar after validating shape, arithmetic and finite input.
    ///
    /// # Errors
    /// Returns an error for invalid rows, unsupported dimensions, invalid fitted
    /// parameters, or allocation failure. Logical rows exclude SIMD padding.
    pub fn try_build_from(dataset: &impl F32Rows, seed: u64) -> Result<Self> {
        let d_pad = dataset
            .dimension()
            .checked_next_power_of_two()
            .ok_or_else(|| crate::Error::invalid_argument("Hadamard dimension overflow"))?
            .max(1024);
        Self::try_build_with_padding(dataset, seed, d_pad)
    }
    /// Convenience constructor for already validated data.
    ///
    /// # Panics
    /// Panics on invalid input or allocation failure. Use [`Self::try_build_with_padding`]
    /// to propagate a structured error at application boundaries.
    pub fn build_with_padding(dataset: &impl F32Rows, seed: u64, d_pad: usize) -> Self {
        Self::try_build_with_padding(dataset, seed, d_pad).expect("invalid sidecar input")
    }

    /// Construct a sidecar after validating shape, arithmetic and finite input.
    ///
    /// # Errors
    /// Returns an error for invalid rows, unsupported dimensions, invalid fitted
    /// parameters, or allocation failure. Logical rows exclude SIMD padding.
    pub fn try_build_with_padding(dataset: &impl F32Rows, seed: u64, d_pad: usize) -> Result<Self> {
        let checked = CheckedRows::new(dataset, u32::MAX as usize)?;
        let dataset = &checked;
        let dim = dataset.dimension();
        assert!(dim > 0);
        if !d_pad.is_power_of_two() || d_pad < dim || d_pad > u32::MAX as usize {
            return Err(crate::Error::invalid_argument(
                "Hadamard padding must be a representable power of two >= dimension",
            ));
        }

        let num_vertices = dataset.len();
        let stride = jl_hadamard_stride(d_pad);

        // Derive three sign-flip diagonals from the seed. xorshift64
        // for determinism; offsets pull non-overlapping streams.
        let mut rng = XorShift64(seed.wrapping_add(0x8B9C_4D7E_25F1_A302));
        let mut signs1 = vec![0i8; d_pad];
        let mut signs2 = vec![0i8; d_pad];
        let mut signs3 = vec![0i8; d_pad];
        for s in [&mut signs1, &mut signs2, &mut signs3] {
            for v in s.iter_mut() {
                *v = if (rng.next_u64() & 1) == 0 { 1 } else { -1 };
            }
        }

        // Allocate the code slab and encode each vertex in parallel.
        let mut codes =
            AlignedBoxWithSlice::<u8>::new(validation::product(num_vertices, stride)?, 16)?;

        let codes_slice = codes.as_mut_slice();

        codes_slice
            .par_chunks_mut(stride)
            .enumerate()
            .for_each(|(vid, code)| {
                let x = dataset.row(vid);
                let mut buf = vec![0.0f32; d_pad];
                encode_vector(x, &signs1, &signs2, &signs3, &mut buf, code);
            });

        Ok(Self {
            dimension: dim,
            d_pad,
            codes,
            signs1,
            signs2,
            signs3,
            num_vertices,
            stride,
            seed,
        })
    }

    /// Encode a query into a fixed-size signature.
    pub fn encode_query(&self, q: &[f32]) -> Box<[u8]> {
        let d_pad = self.d_pad;
        assert_eq!(q.len(), self.dimension);
        let mut out = vec![0u8; self.stride].into_boxed_slice();
        let mut buf = vec![0.0f32; d_pad];
        encode_vector(
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
    ///   [u32 d_pad = d_pad][u32 stride]
    ///   [u64 seed]
    /// Body:
    ///   [d_pad i8 signs1][d_pad i8 signs2][d_pad i8 signs3]
    ///   [num_vertices × STRIDE u8 codes]
    pub fn encode(&self, w: &mut dyn std::io::Write) -> Result<()> {
        let d_pad = self.d_pad;
        let dim = self.dimension;
        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(&JL_HADAMARD_MAGIC.to_le_bytes());
        hdr[4..8].copy_from_slice(&1u32.to_le_bytes());
        hdr[8..12].copy_from_slice(&(self.num_vertices as u32).to_le_bytes());
        hdr[12..16].copy_from_slice(&(dim as u32).to_le_bytes());
        hdr[16..20].copy_from_slice(&(d_pad as u32).to_le_bytes());
        hdr[20..24].copy_from_slice(&(jl_hadamard_stride_disk(d_pad) as u32).to_le_bytes());
        hdr[24..32].copy_from_slice(&self.seed.to_le_bytes());
        w.write_all(&hdr)?;

        for signs in [&self.signs1, &self.signs2, &self.signs3] {
            w.write_all(bytemuck::cast_slice(signs))?;
        }
        super::rows::write_rows(
            w,
            self.codes.as_slice(),
            self.stride,
            jl_hadamard_stride_disk(d_pad),
        )?;
        Ok(())
    }

    pub fn decode(r: &mut dyn std::io::Read, byte_len: u64) -> Result<Self> {
        let mut hdr = [0u8; 32];
        r.read_exact(&mut hdr)?;
        if u32::from_le_bytes(hdr[4..8].try_into().unwrap()) != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "unsupported sidecar version",
            )
            .into());
        }
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
        if dim == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "zero sidecar dimension",
            )
            .into());
        }
        let d_pad = u32::from_le_bytes(hdr[16..20].try_into().unwrap()) as usize;
        if !d_pad.is_power_of_two() || d_pad < dim {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("d_pad mismatch: file={d_pad} expected={d_pad}"),
            )
            .into());
        }
        let stride = u32::from_le_bytes(hdr[20..24].try_into().unwrap()) as usize;
        if stride != jl_hadamard_stride_disk(d_pad) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "stride mismatch: file={stride} expected={}",
                    jl_hadamard_stride(d_pad)
                ),
            )
            .into());
        }
        let seed = u64::from_le_bytes(hdr[24..32].try_into().unwrap());

        validation::shape(num_vertices, dim, u32::MAX as usize)?;
        validation::file_length(byte_len, 32, &[(d_pad, 3), (num_vertices, stride)])?;

        let mut signs1 = vec![0i8; d_pad];
        let mut signs2 = vec![0i8; d_pad];
        let mut signs3 = vec![0i8; d_pad];
        let read_signs = |r: &mut dyn std::io::Read, s: &mut [i8]| -> Result<()> {
            r.read_exact(bytemuck::cast_slice_mut(s))?;
            Ok(())
        };
        read_signs(r, &mut signs1)?;
        read_signs(r, &mut signs2)?;
        read_signs(r, &mut signs3)?;

        if signs1
            .iter()
            .chain(&signs2)
            .chain(&signs3)
            .any(|&s| s != -1 && s != 1)
        {
            return Err(crate::Error::invalid_data(
                "Hadamard diagonals must contain only -1 or +1",
            ));
        }
        let disk_stride = stride;
        let stride = jl_hadamard_stride(d_pad);
        let codes = super::rows::read_rows::<u8>(r, num_vertices, stride, disk_stride)?;

        Ok(Self {
            dimension: dim,
            d_pad,
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

/// Encode one vector (vertex or query) — zero-pad to `d_pad`, run
/// `H D₃ H D₂ H D₁` in place on the buffer, then sign-pack the first
/// `d_pad` bits into the output code slot.
#[inline]
fn encode_vector(
    x: &[f32],
    signs1: &[i8],
    signs2: &[i8],
    signs3: &[i8],
    buf: &mut [f32],
    code: &mut [u8],
) {
    let d_pad = buf.len();

    // Zero-extend x into the FWHT scratch buffer.
    buf[..x.len()].copy_from_slice(x);
    for v in &mut buf[x.len()..] {
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
    // d_pad bits) stays zero — same for query and base, so it
    // contributes zero to Hamming distance.
    for byte in code.iter_mut() {
        *byte = 0;
    }
    for bit_idx in 0..d_pad {
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

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for JlHadamardDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JlHadamardDataset")
            .field("dimension", &self.dimension)
            .field("num_vertices", &self.num_vertices)
            .field("stride", &self.stride)
            .finish_non_exhaustive()
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
        encode_vector(&q, &signs1, &signs2, &signs3, &mut buf1, &mut code1);
        encode_vector(&q, &signs1, &signs2, &signs3, &mut buf2, &mut code2);
        assert_eq!(code1, code2);
    }
}

impl JlHadamardDataset {
    /// Read-only access to the sidecar dimension.
    #[inline]
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Read-only access to the sidecar d pad.
    #[inline]
    pub fn d_pad(&self) -> usize {
        self.d_pad
    }
    /// Read-only access to the sidecar codes.
    #[inline]
    pub fn codes(&self) -> &[u8] {
        self.codes.as_slice()
    }
    /// Read-only access to the sidecar signs1.
    #[inline]
    pub fn signs1(&self) -> &[i8] {
        &self.signs1
    }
    /// Read-only access to the sidecar signs2.
    #[inline]
    pub fn signs2(&self) -> &[i8] {
        &self.signs2
    }
    /// Read-only access to the sidecar signs3.
    #[inline]
    pub fn signs3(&self) -> &[i8] {
        &self.signs3
    }
    /// Read-only access to the sidecar num vertices.
    #[inline]
    pub fn len(&self) -> usize {
        self.num_vertices
    }
    /// Whether this sidecar contains no rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_vertices == 0
    }
    /// Read-only access to the sidecar seed.
    #[inline]
    pub fn seed(&self) -> u64 {
        self.seed
    }
}

impl super::quantized_dataset::sealed::Sealed for JlHadamardDataset {}
impl QuantizedDataset for JlHadamardDataset {
    type BuildOptions = super::quantized_dataset::HadamardOptions;
    fn default_build_options() -> Self::BuildOptions {
        super::quantized_dataset::HadamardOptions {
            seed: JL_HADAMARD_MAGIC as u64,
            padded_dimension: None,
        }
    }
    fn build(base: &impl F32Rows, options: &Self::BuildOptions) -> Result<Self> {
        Self::estimate_resident_memory(
            QuantizedDatasetShape {
                rows: base.len(),
                dimension: base.dimension(),
            },
            options,
        )?;
        Self::try_build_with_padding(
            base,
            options.seed,
            options.padded_dimension(base.dimension())?,
        )
    }
    fn estimate_resident_memory(
        input_shape: QuantizedDatasetShape,
        options: &Self::BuildOptions,
    ) -> Result<ResidentMemory> {
        input_shape.validate()?;
        let padded = options.padded_dimension(input_shape.dimension)?;
        ResidentMemory::new(
            validation::product(input_shape.rows, jl_hadamard_stride(padded))?,
            0,
            validation::product(padded, 3)?,
        )
    }
    fn resident_memory(&self) -> ResidentMemory {
        ResidentMemory::new(
            self.codes.len(),
            0,
            self.signs1.capacity() + self.signs2.capacity() + self.signs3.capacity(),
        )
        .expect("validated resident allocation sizes")
    }
}
