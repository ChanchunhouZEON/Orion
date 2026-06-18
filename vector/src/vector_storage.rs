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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_impl_round_trip() {
        let data: &[[f32; 4]] = &[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        assert_eq!(<[[f32; 4]] as VectorStorage<f32, 4>>::num_vectors(data), 2);
        assert_eq!(
            <[[f32; 4]] as VectorStorage<f32, 4>>::get_vector(data, 0),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            <[[f32; 4]] as VectorStorage<f32, 4>>::get_vector(data, 1),
            [5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn vec_impl_round_trip() {
        let data: Vec<[i32; 3]> = vec![[10, 20, 30], [40, 50, 60], [70, 80, 90]];
        assert_eq!(<Vec<[i32; 3]> as VectorStorage<i32, 3>>::num_vectors(&data), 3);
        assert_eq!(
            <Vec<[i32; 3]> as VectorStorage<i32, 3>>::get_vector(&data, 2),
            [70, 80, 90]
        );
    }
}
