/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! memory: row 0 [D values of T | zero padding] ... row N-1 [D values | padding]
//!         row start = base + row_id * stride_bytes
//!         stride_elements = stride_bytes / size_of::<T>()
//! ```
//!
//! `T: Pod` describes storage, independently of distance-kernel support. The
//! layout must declare the same element width as `T`; logical rows exclude
//! padding. Owned and private-mapped allocations obey the same layout and
//! lifetime rules. This module has no file codec: base loaders decode packed
//! input rows into this layout, and persistence is owned by the runtime.

//! Runtime-dimensional storage shared by Orion and DiskANN adapters.

use bytemuck::Pod;
use orion_ann_vector::{DistanceKernelBackend, DistanceStream, PaddedLayout};

use crate::common::AlignedBoxWithSlice;

/// Owns aligned, zero-padded rows independently of the scalar's available distance kernels.
pub struct PaddedDataset<T: Pod> {
    data: AlignedBoxWithSlice<T>,
    layout: PaddedLayout,
    rows: usize,
}

impl<T: Pod> PaddedDataset<T> {
    /// Allocates zero-initialized resident rows using a checked scalar layout.
    ///
    /// # Errors
    ///
    /// Rejects empty shapes, element-width mismatches and failed or overflowing allocations.
    pub fn new(rows: usize, layout: PaddedLayout) -> Result<Self, String> {
        Self::new_with_storage(rows, layout, &Default::default())
    }

    /// Allocates zero-padded rows under the requested placement and allocation account.
    ///
    /// # Errors
    ///
    /// Rejects shape/type mismatches and propagates allocation or mapping failures.
    pub fn new_with_storage(
        rows: usize,
        layout: PaddedLayout,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self, String> {
        if rows == 0 || layout.element_bytes() != std::mem::size_of::<T>() {
            return Err("invalid padded dataset shape or element type".into());
        }
        let bytes = layout.allocation_bytes(rows)?;
        let data = AlignedBoxWithSlice::new_with_storage(
            bytes / std::mem::size_of::<T>(),
            layout.alignment(),
            storage,
        )
        .map_err(|e| e.to_string())?;
        Ok(Self { data, layout, rows })
    }

    /// Copies directly into the final allocation; also serves ndarray callers
    /// that have already validated a contiguous, row-major source slice.
    pub fn from_packed(values: &[T], layout: PaddedLayout) -> Result<Self, String> {
        if !values.len().is_multiple_of(layout.logical_dim()) {
            return Err("packed input is not a rectangular matrix".into());
        }
        let mut result = Self::new(values.len() / layout.logical_dim(), layout)?;
        if let Some(packed) = result.packed_mut() {
            packed.copy_from_slice(values);
            return Ok(result);
        }
        for (id, row) in values.chunks_exact(layout.logical_dim()).enumerate() {
            result.set_row(id, row)?;
        }
        Ok(result)
    }

    /// Transfer an initialized row-padded allocation without a second base copy.
    pub fn from_allocation(
        data: AlignedBoxWithSlice<T>,
        rows: usize,
        layout: PaddedLayout,
    ) -> Result<Self, String> {
        if rows == 0
            || layout.element_bytes() != std::mem::size_of::<T>()
            || data.len() * std::mem::size_of::<T>() != layout.allocation_bytes(rows)?
            || !(data.as_ptr() as usize).is_multiple_of(layout.alignment())
        {
            return Err("invalid padded allocation shape or alignment".into());
        }
        if layout.stride_elements() != layout.logical_dim() {
            for row in data.chunks_exact(layout.stride_elements()) {
                if bytemuck::cast_slice::<T, u8>(&row[layout.logical_dim()..])
                    .iter()
                    .any(|&b| b != 0)
                {
                    return Err("nonzero SIMD padding".into());
                }
            }
        }
        Ok(Self { data, rows, layout })
    }

    /// Transfers backing ownership without copying rows or releasing their budget charge.
    pub fn into_allocation(self) -> AlignedBoxWithSlice<T> {
        self.data
    }

    /// Reports heap or mapped backing, independently of actual page residency.
    pub fn placement(&self) -> orion_ann_platform::region::Placement {
        self.data.placement()
    }

    /// Flushes completed rows and requests file-page reclamation without losing ownership.
    /// Resident rows are unchanged. Page boundaries may include adjacent rows;
    /// callers must finish writes before requesting reclamation.
    ///
    /// # Errors
    /// Rejects invalid row ranges and propagates storage/advice failures.
    pub fn flush_and_reclaim_rows(&mut self, rows: std::ops::Range<usize>) -> std::io::Result<()> {
        if rows.start > rows.end || rows.end > self.rows {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "row reclaim range out of bounds",
            ));
        }
        let stride = self.layout.stride_bytes();
        self.data
            .flush_and_reclaim(rows.start * stride..rows.end * stride)
    }

    /// Returns the checked relationship between logical width and physical row stride.
    pub fn layout(&self) -> PaddedLayout {
        self.layout
    }

    /// Returns logical row count, not padded scalar count.
    pub fn len(&self) -> usize {
        self.rows
    }

    /// Reports whether there are no logical rows.
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Borrows the complete allocation, including zero-filled row padding.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Contiguous fast path only when there are no padding bytes to protect.
    pub fn packed_mut(&mut self) -> Option<&mut [T]> {
        (self.layout.logical_dim() == self.layout.stride_elements())
            .then(|| self.data.as_mut_slice())
    }

    /// Borrows only logical coordinates, returning None for an out-of-range row ID.
    pub fn row(&self, id: usize) -> Option<&[T]> {
        if id >= self.rows {
            return None;
        }
        let start = id * self.layout.stride_elements();
        Some(&self.data[start..start + self.layout.logical_dim()])
    }

    /// Restrict mutation to logical coordinates so safe callers cannot poison
    /// padding. Both the base loader and query scratch use this contract.
    pub fn row_mut(&mut self, id: usize) -> Option<&mut [T]> {
        if id >= self.rows {
            return None;
        }
        let start = id * self.layout.stride_elements();
        Some(&mut self.data[start..start + self.layout.logical_dim()])
    }

    /// Copies one logical row without exposing or overwriting SIMD padding.
    ///
    /// # Errors
    ///
    /// Rejects wrong row width or an out-of-range row ID before copying.
    pub fn set_row(&mut self, id: usize, values: &[T]) -> Result<(), String> {
        if values.len() != self.layout.logical_dim() {
            return Err("query/vector dimension mismatch".into());
        }
        self.row_mut(id)
            .ok_or("row ID out of bounds")?
            .copy_from_slice(values);
        Ok(())
    }

    /// Borrow a stream over padded rows. Validation occurs once, before the
    /// hot loop; the loop itself is the existing SIMD/prefetch implementation.
    /// Scratch is another dataset (usually one row) to guarantee alignment.
    pub fn stream<'a, K: DistanceKernelBackend<Storage = T>>(
        &'a self,
        query: &'a Self,
        query_id: usize,
        ids: &'a [u32],
        lookahead_lines: usize,
    ) -> Result<DistanceStream<'a, K>, String> {
        if self.layout != query.layout || query_id >= query.rows {
            return Err("query layout or row mismatch".into());
        }
        if self.layout.logical_dim() > K::MAX_COORDINATES {
            return Err("dimension exceeds the kernel's integer accumulator range".into());
        }
        let block = K::stream_block_bytes(self.layout.compute_bytes());
        if block == 0 || !self.layout.compute_bytes().is_multiple_of(block) {
            return Err("layout does not contain complete kernel blocks".into());
        }
        if ids.iter().any(|&id| id as usize >= self.rows) {
            return Err("candidate ID out of bounds".into());
        }
        Ok(unsafe { self.stream_unchecked::<K>(query, query_id, ids, lookahead_lines) })
    }

    /// Stream candidates whose IDs and query layout are already validated.
    /// Graph search uses this after preparing the query, avoiding another ID
    /// scan and layout check for every visited vertex's adjacency list.
    ///
    /// # Safety
    /// Every ID must address a row in `self`; `query_id` must address a row in
    /// `query`. Both layouts must match, contain complete `K` blocks and stay
    /// within `K::MAX_COORDINATES`. Use [`Self::stream`] for unchecked inputs.
    #[inline(always)]
    pub unsafe fn stream_unchecked<'a, K: DistanceKernelBackend<Storage = T>>(
        &'a self,
        query: &'a Self,
        query_id: usize,
        ids: &'a [u32],
        lookahead_lines: usize,
    ) -> DistanceStream<'a, K> {
        // Each row has initialized padding, including the very last row.
        // The borrows keep both allocations and the ID slice alive.
        unsafe {
            DistanceStream::new(
                self.data.as_ptr().cast(),
                self.layout.stride_bytes(),
                self.layout.compute_bytes(),
                query
                    .data
                    .as_ptr()
                    .add(query_id * self.layout.stride_elements())
                    .cast(),
                ids,
                lookahead_lines,
            )
        }
    }

    /// Single-row full distance using the same kernel and reduction order as
    /// streaming. No per-call allocation or coordinate-wise dynamic dispatch.
    pub fn distance<K: DistanceKernelBackend<Storage = T>>(
        &self,
        id: u32,
        query: &Self,
        query_id: usize,
    ) -> Result<f32, String> {
        let ids = [id];
        let stream = self.stream::<K>(query, query_id, &ids, 1)?;
        let mut result = 0.0;
        stream.run(|_, value| result = value);
        Ok(result)
    }
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<T: Pod> std::fmt::Debug for PaddedDataset<T> {
    /// Formats `rows`, `layout`, `allocated_elements` for diagnostic inspection.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PaddedDataset")
            .field("rows", &self.rows)
            .field("layout", &self.layout)
            .field("allocated_elements", &self.data.len())
            .finish_non_exhaustive()
    }
}
