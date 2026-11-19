/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime row layout for full-block SIMD computation.
//!
//! Padding is per row, not a readable tail after the entire allocation.
//! Logical coordinates remain separate from storage so normalization and
//! quantizer fitting never accidentally include padding in their statistics.

use std::mem::{align_of, size_of};

use crate::{BaseScalar, DistanceKernelBackend};

/// Checked per-row layout separating logical coordinates from SIMD compute and allocation spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedLayout {
    logical_dim: usize,
    element_bytes: usize,
    compute_bytes: usize,
    alignment: usize,
}

impl PaddedLayout {
    /// Round rows to a whole number of stream blocks. The block width is
    /// independent of the machine's cache-line width. All current kernels
    /// supported by this layout use power-of-two blocks of at least 16 bytes.
    /// Built-in x86 kernels retain 32-byte short rows even in AVX-512 builds;
    /// their primitive `step` contract remains unchanged.
    pub fn for_kernel<K: DistanceKernelBackend>(logical_dim: usize) -> Result<Self, String> {
        Self::for_storage::<K::Storage>(logical_dim, K::MAX_COORDINATES, K::stream_block_bytes)
    }

    /// Computes the physical layout of a base row from its scalar representation.
    ///
    /// This chooses padding only, not a search metric, calibration policy or CPU
    /// backend. Encoded sidecars should continue to use [`Self::for_kernel`]
    /// because their computation layout follows the quantized representation.
    ///
    /// # Errors
    ///
    /// Rejects zero or unaddressable dimensions and dimensions beyond the
    /// scalar layout's supported range. No backing memory is allocated.
    ///
    /// ```
    /// use orion_ann_vector::PaddedLayout;
    /// let layout = PaddedLayout::for_base_scalar::<u16>(25)?;
    /// assert_eq!(layout.element_bytes(), 2);
    /// assert!(layout.stride_elements() >= 25);
    /// # Ok::<(), String>(())
    /// ```
    pub fn for_base_scalar<T: BaseScalar>(logical_dim: usize) -> Result<Self, String> {
        Self::for_storage::<T>(logical_dim, T::MAX_DIMENSION, T::block_bytes)
    }

    /// Applies shared size/range checks before rounding storage to compute blocks.
    fn for_storage<T>(
        logical_dim: usize,
        max_coordinates: usize,
        block_bytes: impl FnOnce(usize) -> usize,
    ) -> Result<Self, String> {
        if logical_dim > max_coordinates {
            return Err("dimension exceeds supported coordinate range".into());
        }
        let row_bytes = logical_dim
            .checked_mul(size_of::<T>())
            .ok_or("row size overflow")?;
        Self::new::<T>(logical_dim, block_bytes(row_bytes))
    }

    /// Checks a scalar/block layout and rounds each row to whole compute blocks.
    ///
    /// # Errors
    ///
    /// Rejects zero dimensions, invalid block/element sizes and unaddressable or
    /// overflowing row sizes. This constructor does not check a kernel's accumulator limit.
    pub fn new<T>(logical_dim: usize, block_bytes: usize) -> Result<Self, String> {
        let element_bytes = size_of::<T>();
        if logical_dim == 0
            || element_bytes == 0
            || block_bytes < 16
            || !block_bytes.is_power_of_two()
            || !block_bytes.is_multiple_of(element_bytes)
        {
            return Err("invalid dimension, element size, or SIMD block size".into());
        }
        let bytes = logical_dim
            .checked_mul(element_bytes)
            .ok_or("row size overflow")?;
        let compute_bytes = bytes
            .checked_add(block_bytes - 1)
            .ok_or("padded row size overflow")?
            & !(block_bytes - 1);
        if compute_bytes > isize::MAX as usize {
            return Err("padded row exceeds addressable allocation size".into());
        }
        Ok(Self {
            logical_dim,
            element_bytes,
            compute_bytes,
            alignment: block_bytes.max(align_of::<T>()),
        })
    }

    /// Returns meaningful coordinates per row, excluding the zero-padded tail.
    pub fn logical_dim(self) -> usize {
        self.logical_dim
    }

    /// Returns bytes per stored scalar, independent of the logical dimension.
    pub fn element_bytes(self) -> usize {
        self.element_bytes
    }

    /// Returns padded bytes consumed when a kernel evaluates one row.
    pub fn compute_bytes(self) -> usize {
        self.compute_bytes
    }

    /// Returns the byte distance between consecutive stored rows.
    pub fn stride_bytes(self) -> usize {
        self.compute_bytes
    }

    /// Returns the padded row stride in scalar elements rather than bytes.
    pub fn stride_elements(self) -> usize {
        self.compute_bytes / self.element_bytes
    }

    /// Returns the required base allocation alignment for this layout.
    pub fn alignment(self) -> usize {
        self.alignment
    }

    /// Includes row padding, with checked arithmetic for preflight callers.
    pub fn allocation_bytes(self, rows: usize) -> Result<usize, String> {
        self.stride_bytes()
            .checked_mul(rows)
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or_else(|| "padded allocation size overflow".into())
    }

    /// Counts row-tail padding across the dataset using checked allocation bounds.
    pub fn padding_bytes(self, rows: usize) -> Result<usize, String> {
        self.allocation_bytes(rows)?;
        Ok((self.stride_bytes() - self.logical_dim * self.element_bytes) * rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IpF32Distance, IpI16Distance, IpI8Distance, L2U8Distance};

    #[test]
    fn glove25_layout_depends_on_storage_bytes_not_cache_lines() {
        for dim in [25, 32] {
            assert_eq!(
                PaddedLayout::for_kernel::<IpI8Distance>(dim)
                    .unwrap()
                    .stride_bytes(),
                32
            );
            assert_eq!(
                PaddedLayout::for_kernel::<IpI16Distance>(dim)
                    .unwrap()
                    .stride_bytes(),
                64
            );
            assert_eq!(
                PaddedLayout::for_kernel::<IpF32Distance>(dim)
                    .unwrap()
                    .stride_bytes(),
                128
            );
        }
    }

    #[test]
    fn longer_rows_keep_the_existing_compile_time_padding() {
        fn check<K: DistanceKernelBackend>() {
            for dim in [33, 63, 64, 65, 96, 100, 128, 960, 1536] {
                let expected = PaddedLayout::new::<K::Storage>(dim, K::CHUNK_BYTES).unwrap();
                assert_eq!(PaddedLayout::for_kernel::<K>(dim).unwrap(), expected);
            }
        }
        check::<IpI8Distance>();
        check::<IpI16Distance>();
        check::<IpF32Distance>();
        check::<L2U8Distance>();
    }
}
