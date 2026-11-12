/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! memory: N * M u8 codes + codebook {M+1 offsets, D f32 mean, 256*D f32 centers}
//! file:   ["ORIONPQ\0": 8 B][N/D/M: u64 each][M+1 u64 offsets]
//!         [D f32 mean][256*D f32 centers][N*M u8 codes]
//! ```
//!
//! M is the number of subspaces; each code selects one of 256 centroids.
//! Codes are packed row-major without SIMD row padding. The query builds an
//! M-by-256 ADC lookup table; base vectors are not materialized as f32 copies.
//! File fields are little-endian; `codec` owns encoding and validation.
//! Codebook pin guards are dropped before their backing allocations.

use super::super::{
    PreparedQuery, QuantizedDataset, QuantizedDatasetCompute, QuantizedDatasetShape,
    ResidentMemory, sealed,
};
use super::{FixedChunkPQTable, PqBuildOptions, PqCodeStorage, storage::Codes};
use crate::{Error, Result, model::dataset::rows::Rows};
use orion_ann_platform::residency::PinGuard;
use orion_ann_vector::{DistanceStream, PqAdcDistance};

/// Score reconstructed subvectors using squared L2 or negative inner product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PqMetric {
    L2,
    InnerProduct,
}

/// PQ codes and a small, immutable codebook. Codebook pages are best-effort
/// pinned by default when owned; mapped codebooks stay pageable. Codes can be
/// explicitly pinned with `pin_codes`. Lock denial is observable via `codebook_pin_error`.
/// Pinning keeps RAM resident, not CPU-cache lines resident.
pub struct PqDataset {
    // Drop the lock before its backing allocations. The codebook cannot be
    // mutated/replaced through any public API; its Box allocations never move.
    pub(super) codebook_pin: PinGuard<'static>,
    pub(super) pin_error: Option<String>,
    pub(super) table: FixedChunkPQTable,
    pub(super) codes: Codes,
    pub(super) rows: usize,
}
impl std::fmt::Debug for PqDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PqDataset")
            .field("rows", &self.rows)
            .field("codebook", &self.table)
            .field("codes", &self.code_storage())
            .field("codebook_pinned_bytes", &self.codebook_pin.pinned_bytes())
            .field("pin_error", &self.pin_error)
            .finish_non_exhaustive()
    }
}
impl PqDataset {
    /// Fit the codebook, then encode directly into owned or mapped code storage.
    /// Training retains its bounded sample; only encoding is block-streamed.
    /// # Errors
    /// Rejects invalid recipes/input and reports allocation or file I/O failures.
    pub fn build_with_storage(
        base: &impl Rows,
        options: &PqBuildOptions,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        Self::estimate_resident_memory(
            QuantizedDatasetShape {
                rows: base.len(),
                dimension: base.dimension(),
            },
            options,
        )?;
        let table = FixedChunkPQTable::train_rows(base, *options)?.with_storage(storage)?;
        let codes = orion_ann_platform::region::ArrayView::build_rows::<Error>(
            base.len(),
            table.get_num_chunks(),
            64,
            storage,
            |start, out| table.encode_into(base, start, out),
        )?;
        Self::from_parts(table, Codes(codes), base.len())
    }

    pub(super) fn from_parts(table: FixedChunkPQTable, codes: Codes, rows: usize) -> Result<Self> {
        if rows == 0 || rows.checked_mul(table.get_num_chunks()) != Some(codes.as_slice().len()) {
            return Err(Error::invalid_argument(
                "PQ code count does not match dataset shape",
            ));
        }
        let mut pin = PinGuard::default();
        let regions = if table.centers.placement() == orion_ann_platform::region::Placement::Mapped
        {
            Vec::new()
        } else {
            table.regions()
        };
        // SAFETY: only immutable payload addresses are borrowed. Moving `table` moves
        // handles, not payloads. They remain private and immutable until drop,
        // where `codebook_pin` is dropped first. No 'static slice is exposed.
        let regions: Vec<&'static [u8]> = regions
            .into_iter()
            .map(|r| unsafe { std::slice::from_raw_parts(r.as_ptr(), r.len()) })
            .collect();
        let pin_error = pin
            .try_pin(&regions, 64 * 1024 * 1024)
            .err()
            .map(|e| e.to_string());
        if let Some(error) = &pin_error {
            log::warn!("PQ codebook remains pageable: {error}");
        }
        Ok(Self {
            codebook_pin: pin,
            pin_error,
            table,
            codes,
            rows,
        })
    }
    pub fn len(&self) -> usize {
        self.rows
    }
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }
    pub fn dimension(&self) -> usize {
        self.table.dim()
    }
    /// Mapped payload span; not instantaneous resident memory.
    pub fn mapped_bytes(&self) -> usize {
        let table =
            if self.table.centers.placement() == orion_ann_platform::region::Placement::Mapped {
                self.table.bytes()
            } else {
                0
            };
        table
            + if self.code_storage() == PqCodeStorage::OnDemand {
                self.codes().len()
            } else {
                0
            }
    }
    pub fn codebook(&self) -> &FixedChunkPQTable {
        &self.table
    }
    pub fn codes(&self) -> &[u8] {
        self.codes.as_slice()
    }
    pub fn code_storage(&self) -> PqCodeStorage {
        self.codes.policy()
    }
    pub fn codebook_pin_error(&self) -> Option<&str> {
        self.pin_error.as_deref()
    }
    pub fn codebook_pinned_bytes(&self) -> usize {
        self.codebook_pin.pinned_bytes()
    }
    /// Change code placement. Codebook, IDs and scores are unchanged.
    /// # Errors
    /// Returns temporary-file/allocation/I/O errors. Existing storage survives failure.
    pub fn set_code_storage(&mut self, policy: PqCodeStorage) -> Result<()> {
        if self.code_storage() != policy {
            let replacement = Codes::from_reader(
                &mut self.codes.as_slice(),
                self.codes.as_slice().len(),
                policy,
            )?;
            self.codes = replacement;
        }
        Ok(())
    }
    /// Lock codes in RAM until the returned guard is dropped.
    /// # Errors
    /// Reports budget overflow or OS refusal. It never silently claims residency.
    pub fn pin_codes(&self, budget_bytes: usize) -> Result<PinGuard<'_>> {
        let mut guard = PinGuard::default();
        guard
            .try_pin(&[self.codes()], budget_bytes)
            .map_err(|e| Error::invalid_state(e.to_string()))?;
        Ok(guard)
    }
    /// Prepare an ADC query without quantizing query coordinates.
    /// # Errors
    /// Rejects dimension mismatch, nonfinite input or overflow in the lookup table.
    pub fn query(&self, query: &[f32], metric: PqMetric) -> Result<PqQuery<'_>> {
        if query.len() != self.dimension() {
            return Err(Error::dimension("PQ query", self.dimension(), query.len()));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid_argument("PQ query must be finite"));
        }
        let lookup = self
            .table
            .lookup(query, metric == PqMetric::InnerProduct, false);
        if lookup.iter().any(|x| !x.is_finite()) {
            return Err(Error::invalid_argument("PQ distance table overflow"));
        }
        Ok(PqQuery {
            dataset: self,
            lookup,
        })
    }
}
impl sealed::Sealed for PqDataset {}
impl QuantizedDatasetCompute for PqDataset {
    fn prepare_query(&self, query: &[f32]) -> Result<impl PreparedQuery + '_> {
        self.query(query, PqMetric::L2)
    }
}
impl QuantizedDataset for PqDataset {
    type BuildOptions = PqBuildOptions;
    fn default_build_options() -> PqBuildOptions {
        Default::default()
    }
    fn build(base: &impl Rows, options: &PqBuildOptions) -> Result<Self> {
        Self::build_with_storage(
            base,
            options,
            &orion_ann_platform::region::RegionOptions::default(),
        )
    }
    fn estimate_resident_memory(
        shape: QuantizedDatasetShape,
        options: &PqBuildOptions,
    ) -> Result<ResidentMemory> {
        shape.validate()?;
        let m = options.chunks(shape.dimension)?;
        let codes = shape
            .rows
            .checked_mul(m)
            .ok_or_else(|| Error::invalid_argument("PQ payload overflow"))?;
        let centers = shape
            .dimension
            .checked_mul(257 * 4)
            .and_then(|v| v.checked_add((m + 1) * size_of::<usize>()))
            .ok_or_else(|| Error::invalid_argument("PQ codebook overflow"))?;
        ResidentMemory::new(codes, 0, centers)
    }
    fn resident_memory(&self) -> ResidentMemory {
        // Mapped payload is potential resident storage, not a current RSS reading.
        ResidentMemory::new(self.codes().len(), 0, self.table.bytes())
            .expect("allocated payload fits address space")
    }
}
impl super::super::residency::ResidentRegions for PqDataset {
    fn resident_regions(&self) -> Vec<&[u8]> {
        let mut r = self.table.regions();
        r.push(self.codes());
        r
    }
}

/// Query-bound lookup table. Scoring reuses DistanceStream and preserves ID order.
pub struct PqQuery<'a> {
    dataset: &'a PqDataset,
    lookup: Vec<f32>,
}
impl std::fmt::Debug for PqQuery<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PqQuery")
            .field("chunks", &self.dataset.table.get_num_chunks())
            .finish_non_exhaustive()
    }
}
impl sealed::Sealed for PqQuery<'_> {}
impl PreparedQuery for PqQuery<'_> {
    fn distance(&self, id: u32) -> Result<f32> {
        let mut distance = 0.0;
        self.stream(&[id], 1, |_, d| distance = d)?;
        Ok(distance)
    }
}
impl PqQuery<'_> {
    /// Score IDs in their input order, including duplicates.
    /// # Errors
    /// Rejects all invalid IDs before emitting any distances.
    pub fn stream(&self, ids: &[u32], lookahead: usize, sink: impl FnMut(u32, f32)) -> Result<()> {
        if ids.iter().any(|&id| id as usize >= self.dataset.rows) {
            return Err(Error::invalid_argument("PQ row ID out of bounds"));
        }
        // SAFETY: every ID validated; m code bytes and m aligned tables per row.
        unsafe {
            self.stream_unchecked(ids, lookahead, sink);
        }
        Ok(())
    }
    pub(crate) unsafe fn stream_unchecked(
        &self,
        ids: &[u32],
        lookahead: usize,
        sink: impl FnMut(u32, f32),
    ) {
        let m = self.dataset.table.get_num_chunks();
        unsafe {
            DistanceStream::<PqAdcDistance>::new(
                self.dataset.codes().as_ptr(),
                m,
                m,
                self.lookup.as_ptr().cast(),
                ids,
                lookahead,
            )
            .run(sink);
        }
    }
}
