//! Runtime row layout for full-block SIMD computation.
//!
//! Padding is per row, not a readable tail after the entire allocation.
//! Logical coordinates remain separate from storage so normalization and
//! quantizer fitting never accidentally include padding in their statistics.

use std::mem::{align_of, size_of};

use crate::DistanceFn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaddedLayout {
    logical_dim: usize,
    element_bytes: usize,
    compute_bytes: usize,
    alignment: usize,
}

impl PaddedLayout {
    /// Round rows to a whole number of kernel blocks. The block width is
    /// independent of the machine's cache-line width. All current kernels
    /// use power-of-two blocks of at least 16 bytes.
    pub fn for_kernel<K: DistanceFn>(logical_dim: usize) -> Result<Self, String> {
        if logical_dim > K::MAX_COORDINATES {
            return Err("dimension exceeds kernel accumulator range".into());
        }
        Self::new::<K::Storage>(logical_dim, K::CHUNK_BYTES)
    }

    pub fn new<T>(logical_dim: usize, block_bytes: usize) -> Result<Self, String> {
        let element_bytes = size_of::<T>();
        if logical_dim == 0
            || element_bytes == 0
            || block_bytes < 16
            || !block_bytes.is_power_of_two()
            || block_bytes % element_bytes != 0
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

    pub fn logical_dim(self) -> usize {
        self.logical_dim
    }
    pub fn element_bytes(self) -> usize {
        self.element_bytes
    }
    pub fn compute_bytes(self) -> usize {
        self.compute_bytes
    }
    pub fn stride_bytes(self) -> usize {
        self.compute_bytes
    }
    pub fn stride_elements(self) -> usize {
        self.compute_bytes / self.element_bytes
    }
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

    pub fn padding_bytes(self, rows: usize) -> Result<usize, String> {
        self.allocation_bytes(rows)?;
        Ok((self.stride_bytes() - self.logical_dim * self.element_bytes) * rows)
    }
}
