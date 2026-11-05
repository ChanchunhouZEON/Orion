/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Lazy runtime-dimensional sidecars shared by Orion and its typed adapter.
use super::CompressedIndex;
use crate::model::dataset::jl_hadamard_dataset::JlHadamardDataset;
use crate::model::dataset::jl_sparse_dataset::{JLSparseDataset, JLSparseDatasetMips};
use crate::model::dataset::l2_kt_dataset::L2KTDataset;
use crate::model::dataset::rabitq_b4_dataset::RabitQ4Dataset;
use crate::model::dataset::rabitq_dataset::RabitQDataset;
use crate::model::dataset::rows::F32Rows;
use crate::model::{L2U8, L2U16, MipsI8, MipsI16, QuantSpec, ScalarQuantizedDataset};

use crate::{Error, Result};
use std::{
    io::{Read, Write},
    sync::{Mutex, OnceLock},
};
mod storage;
pub use storage::{DecodeSidecar, EncodeSidecar, SidecarCache};

#[derive(Default)]
pub struct DynamicSidecars {
    pub(super) cache: Option<Box<dyn SidecarCache>>,
    /// U8-quantized base for the L2 prefilter (ParlayANN-style). Built
    /// lazily on first `ensure_quantized_dataset`. Sidecar `.qds`.
    q_dataset: SidecarCell<ScalarQuantizedDataset<L2U8>>,

    /// U16-quantized base — precise PQ-admission tier of the L2
    /// cascade. 256× finer per-dim resolution than u8 at 2× storage,
    /// pairs with the u8 sidecar via the PA `quantize_bits 16` recipe
    /// (u8 = cheap filter, u16 = PQ admission distance). Built lazily
    /// on first `ensure_quantized_dataset_l2_u16`. Sidecar `.qds6`.
    q_dataset_l2_u16: SidecarCell<ScalarQuantizedDataset<L2U16>>,

    /// I8-quantized base for the MIPS search path on unit-normalized
    /// data. Built lazily on first `ensure_quantized_dataset_mips`.
    /// Sidecar `.qdm8`.
    q_dataset_mips: SidecarCell<ScalarQuantizedDataset<MipsI8>>,

    /// I16 twin of the above — 2× per-element precision for the
    /// high-recall band where i8's distance-collision ceiling caps
    /// recall on hard angular workloads. Built lazily on first
    /// `ensure_quantized_dataset_mips_i16`. Sidecar `.qdm16`.
    q_dataset_mips_i16: SidecarCell<ScalarQuantizedDataset<MipsI16>>,

    /// RaBitQ 1-bit-per-dim quantized base for the high-dim
    /// bandwidth-bound L2 path (Gao & Long, SIGMOD 2024). Built lazily
    /// on first `ensure_quantized_dataset_rabitq`. Sidecar `.qrbq`.
    /// Targets GIST-class datasets where the entire quantized base
    /// fits into L1 cache.
    q_dataset_rabitq: SidecarCell<RabitQDataset>,

    /// Extended RaBitQ at B=4 bits/dim (Gao & Long, SIGMOD 2025).
    /// Same random-rotation front-end but a 4-bit signed scalar
    /// quantizer on the rotated components. Sidecar `.qrb4`. Lifts
    /// the recall ceiling at the cost of 4× more storage per vertex
    /// vs B=1 (still ~2× compression vs u8 on GIST D=960).
    q_dataset_rabitq_b4: SidecarCell<RabitQ4Dataset>,

    /// Johnson-Lindenstrauss sparse-projection binary signature
    /// (1024 bits per vertex) — used as the cheapest prefilter tier
    /// in the L2 search cascade, mirroring ParlayANN's
    /// `Euclidean_JL_Sparse_Point<1024>`. Sidecar `.jls`. Hamming
    /// distance via NEON `vcntq_u8` is ~24× cheaper per cmp than the
    /// u8 NEON L2 kernel on GIST D=960, but is only a rough L2
    /// correlate — strictly a prefilter, never the final ranker.
    q_dataset_jl: SidecarCell<JLSparseDataset<1024>>,

    /// MIPS variant of the JL sparse signature — distinct type
    /// ([`JLSparseDatasetMips`]) so the L2 path doesn't carry the
    /// per-vertex `‖v‖` slab. Default NZ=9 per the current MIPS
    /// sweep. Sidecar `.jls_mips`.
    q_dataset_jl_mips: SidecarCell<JLSparseDatasetMips<1024, 9>>,

    /// JL Hadamard 1024-bit signature dataset (HDHDHD-encoded). Lazy-
    /// built on first `ensure_quantized_dataset_jl_hadamard`. Sidecar
    /// `.jlh`.
    q_dataset_jl_hadamard: SidecarCell<JlHadamardDataset>,

    /// **L2 kernel-trick** sidecar — i8 base + per-vertex `‖x_i8‖²`
    /// (i32). Pairs with `IpI8Distance` (sdot) to reconstruct
    /// `‖q-x‖² = ‖q‖² + ‖x‖² - 2·⟨q,x⟩` per hop, giving MIPS-like
    /// kernel speed but L2-rank-exact (modulo the same per-dim
    /// quantization noise as the u8 sidecar). Sidecar `.qdsl2kt`.
    q_dataset_l2_kt: SidecarCell<L2KTDataset>,
}

// Quantized sidecars belong only to the f32 storage path. Native-byte indexes
// borrow their original dataset directly and cannot accidentally build a copy.
impl<D: F32Rows> CompressedIndex<D, DynamicSidecars> {
    /// Lazily obtain the u8 quantized dataset (L2 prefilter path).
    /// Tries the sidecar `.qds` first (memcpy load); on miss, builds
    /// from the f32 dataset and writes back.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset(&self) -> &ScalarQuantizedDataset<L2U8> {
        self.try_ensure_quantized_dataset()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset(&self) -> Result<&ScalarQuantizedDataset<L2U8>> {
        self.sidecars
            .q_dataset
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the **u16** quantized dataset (precise
    /// PQ-admission tier of the L2 cascade). Sidecar `.qds6`. Goes
    /// through the same `build_quant` plumbing as the u8 path; the
    /// only differences are storage type + quantization scale (see
    /// the `L2U16` spec).
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_l2_u16`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_l2_u16(&self) -> &ScalarQuantizedDataset<L2U16> {
        self.try_ensure_quantized_dataset_l2_u16()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_l2_u16(&self) -> Result<&ScalarQuantizedDataset<L2U16>> {
        self.sidecars
            .q_dataset_l2_u16
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the i8 MIPS quantized dataset. Sidecar `.qdm8`.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_mips`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_mips(&self) -> &ScalarQuantizedDataset<MipsI8> {
        self.try_ensure_quantized_dataset_mips()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_mips(&self) -> Result<&ScalarQuantizedDataset<MipsI8>> {
        self.sidecars
            .q_dataset_mips
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the i16 MIPS quantized dataset. Sidecar `.qdm16`.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_mips_i16`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_mips_i16(&self) -> &ScalarQuantizedDataset<MipsI16> {
        self.try_ensure_quantized_dataset_mips_i16()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_mips_i16(
        &self,
    ) -> Result<&ScalarQuantizedDataset<MipsI16>> {
        self.sidecars
            .q_dataset_mips_i16
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the RaBitQ 1-bit quantized dataset. Sidecar
    /// `.qrbq`. On first access: try the sidecar (memcpy load); on
    /// miss build from the f32 dataset (rotation gen + encode) and
    /// write back. Build uses a fixed seed so every cache slot is
    /// reproducible across runs.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_rabitq`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_rabitq(&self) -> &RabitQDataset {
        self.try_ensure_quantized_dataset_rabitq()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_rabitq(&self) -> Result<&RabitQDataset> {
        self.sidecars
            .q_dataset_rabitq
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the **B=4** RaBitQ quantized dataset. Sidecar
    /// `.qrb4`. Distinct from the B=1 sidecar (`.qrbq`) so both can
    /// coexist on disk and be A/B'd at search time.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_rabitq_b4`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_rabitq_b4(&self) -> &RabitQ4Dataset {
        self.try_ensure_quantized_dataset_rabitq_b4()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_rabitq_b4(&self) -> Result<&RabitQ4Dataset> {
        self.sidecars
            .q_dataset_rabitq_b4
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the JL Sparse 1024-bit signature dataset.
    /// Sidecar `.jls`. Built fresh on first access — encoding is
    /// `O(N · BITS · NZ)` random reads on the f32 base (~10s for
    /// GIST 1M D=960 on M2). Used as the cheapest tier of the L2
    /// search cascade; never the final ranker.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_jl`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_jl(&self) -> &JLSparseDataset<1024> {
        self.try_ensure_quantized_dataset_jl()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_jl(&self) -> Result<&JLSparseDataset<1024>> {
        self.sidecars
            .q_dataset_jl
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// MIPS twin of [`Self::ensure_quantized_dataset_jl`] — builds (or
    /// loads from the `.jls_mips` sidecar) the NZ=9
    /// [`JLSparseDatasetMips`] designed for raw-MIPS data, per PA's
    /// `Mips_JL_Sparse_Point_Normalized` recipe. The per-vertex
    /// `‖x‖` norms used by the MIPS prefilter live inside the
    /// returned dataset alongside the sign-bit codes.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_jl_mips`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_jl_mips(&self) -> &JLSparseDatasetMips<1024, 9> {
        self.try_ensure_quantized_dataset_jl_mips()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_jl_mips(&self) -> Result<&JLSparseDatasetMips<1024, 9>> {
        self.sidecars
            .q_dataset_jl_mips
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the **JL Hadamard** 1024-bit signature dataset
    /// (HDHDHD-encoded). Sidecar `.jlh`. Same byte layout as the JL
    /// Sparse signature so the search-time hot path reuses the
    /// `JLHammingDistance` kernel; per-bit information content is
    /// significantly higher (each bit sees all D dims via Hadamard
    /// mixing, vs JL Sparse's NZ=9 dims).
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_jl_hadamard`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_jl_hadamard(&self) -> &JlHadamardDataset {
        self.try_ensure_quantized_dataset_jl_hadamard()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_jl_hadamard(&self) -> Result<&JlHadamardDataset> {
        self.sidecars
            .q_dataset_jl_hadamard
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }

    /// Lazily obtain the **L2 kernel-trick** sidecar (i8 base +
    /// per-vert `‖x_i8‖²`). Sidecar `.qdsl2kt`. Tries the on-disk
    /// cache first (memcpy load); on miss, builds via parallel
    /// `L2KTDataset::build_from` and writes back.
    /// # Panics
    /// Panics when sidecar construction fails. Use [`Self::try_ensure_quantized_dataset_l2_kt`] for
    /// structured errors in services and language bindings.
    pub fn ensure_quantized_dataset_l2_kt(&self) -> &L2KTDataset {
        self.try_ensure_quantized_dataset_l2_kt()
            .expect("sidecar construction failed")
    }

    /// Obtain the sidecar, validating cache identity and propagating build errors.
    ///
    /// # Errors
    /// Returns invalid-input, allocation or initialization errors. An invalid
    /// cache is rebuilt; inability to persist an optional cache only logs a warning.
    pub fn try_ensure_quantized_dataset_l2_kt(&self) -> Result<&L2KTDataset> {
        self.sidecars
            .q_dataset_l2_kt
            .ensure(&self.dataset, self.sidecars.cache.as_deref())
    }
}

/// Initialization is serialized only on a cache miss. Searches borrow the
/// resulting immutable sidecar and never acquire this mutex. Failed builds are
/// not cached, so transient allocation/I/O failures can be retried.
struct SidecarCell<T> {
    value: OnceLock<T>,
    initialization: Mutex<()>,
}
impl<T> Default for SidecarCell<T> {
    fn default() -> Self {
        Self {
            value: OnceLock::new(),
            initialization: Mutex::new(()),
        }
    }
}
impl<T> SidecarCell<T> {
    fn get(&self) -> Option<&T> {
        self.value.get()
    }
    fn get_or_try_init(&self, build: impl FnOnce() -> Result<T>) -> Result<&T> {
        if let Some(value) = self.value.get() {
            return Ok(value);
        }
        let _guard = self
            .initialization
            .lock()
            .map_err(|_| Error::invalid_state("sidecar initialization lock poisoned"))?;
        if let Some(value) = self.value.get() {
            return Ok(value);
        }
        let value = build()?;
        let _ = self.value.set(value);
        Ok(self
            .value
            .get()
            .expect("sidecar initialized under exclusive initialization lock"))
    }
}

trait SidecarCodec: crate::model::dataset::QuantizedDataset {
    const EXTENSION: &'static str;
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self>;
    fn encode(&self, writer: &mut dyn Write) -> Result<()>;
    fn shape(&self) -> (usize, usize);
}

impl<Q: QuantSpec> SidecarCodec for ScalarQuantizedDataset<Q> {
    const EXTENSION: &'static str = Q::FILE_EXT;
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.n)
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl std::fmt::Debug for DynamicSidecars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicSidecars")
            .field("q_dataset", &self.q_dataset.get().is_some())
            .field("q_dataset_l2_u16", &self.q_dataset_l2_u16.get().is_some())
            .field("q_dataset_mips", &self.q_dataset_mips.get().is_some())
            .field(
                "q_dataset_mips_i16",
                &self.q_dataset_mips_i16.get().is_some(),
            )
            .field("q_dataset_rabitq", &self.q_dataset_rabitq.get().is_some())
            .field(
                "q_dataset_rabitq_b4",
                &self.q_dataset_rabitq_b4.get().is_some(),
            )
            .field("q_dataset_jl", &self.q_dataset_jl.get().is_some())
            .field("q_dataset_jl_mips", &self.q_dataset_jl_mips.get().is_some())
            .field(
                "q_dataset_jl_hadamard",
                &self.q_dataset_jl_hadamard.get().is_some(),
            )
            .field("q_dataset_l2_kt", &self.q_dataset_l2_kt.get().is_some())
            .finish_non_exhaustive()
    }
}

impl SidecarCodec for RabitQDataset {
    const EXTENSION: &'static str = "qrbq";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

impl SidecarCodec for RabitQ4Dataset {
    const EXTENSION: &'static str = "qrb4";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

impl SidecarCodec for JLSparseDataset<1024> {
    const EXTENSION: &'static str = "jls";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

impl SidecarCodec for JLSparseDatasetMips<1024, 9> {
    const EXTENSION: &'static str = "jls_mips";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

impl SidecarCodec for JlHadamardDataset {
    const EXTENSION: &'static str = "jlh";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

impl SidecarCodec for L2KTDataset {
    const EXTENSION: &'static str = "qdsl2kt";
    fn decode(reader: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode(reader, bytes)
    }
    fn encode(&self, writer: &mut dyn Write) -> Result<()> {
        self.encode(writer)
    }
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.num_vertices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn failed_initialization_can_retry_without_poisoning() {
        let cell = SidecarCell::<usize>::default();
        assert!(
            cell.get_or_try_init(|| Err(Error::invalid_argument("probe")))
                .is_err()
        );
        assert!(cell.get().is_none());
        assert_eq!(*cell.get_or_try_init(|| Ok(42)).unwrap(), 42);
        assert_eq!(
            *cell
                .get_or_try_init(|| panic!("already initialized"))
                .unwrap(),
            42
        );
    }

    #[test]
    fn concurrent_initialization_publishes_one_immutable_value() {
        let cell = SidecarCell::default();
        let builds = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for _ in 0..16 {
                        let value = cell
                            .get_or_try_init(|| {
                                builds.fetch_add(1, Ordering::Relaxed);
                                Ok(vec![1u32, 2, 3])
                            })
                            .unwrap();
                        assert_eq!(value.as_slice(), &[1, 2, 3]);
                    }
                });
            }
        });
        assert_eq!(builds.load(Ordering::Relaxed), 1);
    }
}
