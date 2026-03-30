/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Trait for on-demand vector access, enabling both in-memory slices and
/// mmap-backed storage to be used interchangeably in search algorithms.
pub trait VectorStorage<T: Copy, const N: usize> {
    fn get_vector(&self, id: u32) -> [T; N];
    fn num_vectors(&self) -> usize;
}

/// Blanket impl for contiguous slices — ensures backward compatibility.
impl<T: Copy, const N: usize> VectorStorage<T, N> for [[T; N]] {
    #[inline(always)]
    fn get_vector(&self, id: u32) -> [T; N] {
        self[id as usize]
    }

    #[inline(always)]
    fn num_vectors(&self) -> usize {
        self.len()
    }
}

/// Impl for Vec — delegates to the slice impl via Deref.
/// This avoids needing explicit `.as_slice()` at every call site.
impl<T: Copy, const N: usize> VectorStorage<T, N> for Vec<[T; N]> {
    #[inline(always)]
    fn get_vector(&self, id: u32) -> [T; N] {
        self[id as usize]
    }

    #[inline(always)]
    fn num_vectors(&self) -> usize {
        self.len()
    }
}
