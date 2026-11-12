/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Read-only access to state whose consistency is established during construction.
use super::CompressedIndex;
use crate::model::FixedChunkPQTable;

impl<D, S> CompressedIndex<D, S> {
    /// Borrow the base vectors without copying their allocation.
    ///
    /// Replacing or editing vectors would invalidate graph topology, fitted
    /// sidecars and calibration. Rebuild an index to change its base; use
    /// [`Self::into_dataset`] to recover the original allocation first.
    ///
    /// ```compile_fail
    /// fn replace(index: &mut orion_ann_core::Orion, base: orion_ann_core::PaddedDataset<f32>) {
    ///     index.dataset = base;
    /// }
    /// ```
    /// ```compile_fail
    /// fn edit(index: &mut orion_ann_core::Orion) {
    ///     index.dataset().row_mut(0).unwrap()[0] = 0.0;
    /// }
    /// ```
    #[inline]
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// Consume the index and return its base allocation without copying.
    /// Graph storage, scratch and fitted sidecars are dropped, so they cannot
    /// accidentally be reused with subsequently modified vectors.
    #[inline]
    pub fn into_dataset(self) -> D {
        self.dataset
    }

    /// Search entry point, validated against the base and graph at construction.
    ///
    /// ```compile_fail
    /// fn change_entry(index: &mut orion_ann_core::Orion) {
    ///     index.entry = u32::MAX;
    /// }
    /// ```
    #[inline]
    pub fn entry_point(&self) -> u32 {
        self.entry
    }

    /// Number of indexed vectors, equal to the graph node count.
    ///
    /// ```compile_fail
    /// fn change_count(index: &mut orion_ann_core::Orion) {
    ///     index.num_nodes = 1;
    /// }
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.num_nodes
    }

    /// Whether the index contains no vectors. Current constructors reject empty
    /// input; this method accompanies [`Self::len`] for collection-style code.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.num_nodes == 0
    }

    /// Borrow the optional legacy PQ table supplied during construction.
    /// The table and codes cannot be replaced independently on a live index.
    #[inline]
    pub fn pq_table(&self) -> Option<&FixedChunkPQTable> {
        self.pq.as_deref()
    }

    /// Borrow optional legacy PQ codes without exposing a mutable code buffer.
    ///
    /// ```compile_fail
    /// fn replace_codes(index: &mut orion_ann_core::Orion) {
    ///     index.pq_codes = Some(vec![]);
    /// }
    /// ```
    #[inline]
    pub fn pq_codes(&self) -> Option<&[u8]> {
        self.pq_codes.as_deref()
    }

    /// Number of chunks in the optional legacy PQ table.
    #[inline]
    pub fn num_pq_chunks(&self) -> Option<usize> {
        self.num_pq_chunks
    }
}

impl<D: crate::model::dataset::base::IndexData, S> CompressedIndex<D, S> {
    /// Attach fitted product-quantization data to an owned index.
    /// # Errors
    /// Rejects incompatible dimensions or packed code lengths. This is an
    /// experimental PQ search facility, separate from scalar-quantized cascades.
    pub fn with_product_quantization(
        mut self,
        table: std::sync::Arc<FixedChunkPQTable>,
        codes: Vec<u8>,
    ) -> crate::Result<Self> {
        let chunks = table.get_num_chunks();
        if table.dim() != self.dataset.dimension()
            || chunks == 0
            || self.num_nodes.checked_mul(chunks) != Some(codes.len())
        {
            return Err(crate::Error::invalid_argument(
                "PQ table/codes do not match the base shape",
            ));
        }
        self.num_pq_chunks = Some(chunks);
        self.pq = Some(table);
        self.pq_codes = Some(codes);
        Ok(self)
    }
}
