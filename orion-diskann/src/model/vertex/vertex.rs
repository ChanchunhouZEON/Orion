/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_vector::{FullPrecisionDistance, Metric};
use std::array::TryFromSliceError;

/// Borrowed fixed-width coordinates with their internal graph ID.
#[derive(Debug)]
pub struct Vertex<'a, T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    val: &'a [T; N],
    id: u32,
}

impl<'a, T, const N: usize> Vertex<'a, T, N>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Borrows a fixed-width vector and associates its compact internal ID without copying coordinates.
    pub fn new(val: &'a [T; N], id: u32) -> Self {
        Self { val, id }
    }

    /// Computes the configured legacy distance between two borrowed fixed-width vectors.
    #[inline(always)]
    pub fn compare(&self, other: &Vertex<'a, T, N>, metric: Metric) -> f32 {
        <[T; N]>::distance_compare(self.val, other.val, metric)
    }

    /// L2 distance with early abandon. Returns distance if < upper_bound,
    /// or a negative value if partial distance exceeds the bound.
    #[inline(always)]
    pub fn compare_with_bound(&self, other: &Vertex<'a, T, N>, upper_bound: f32) -> f32 {
        <[T; N]>::distance_compare_with_bound(self.val, other.val, upper_bound)
    }

    /// ADSampling L2 distance: assumes `self` (query) and `other` (stored
    /// vector) have been pre-rotated by a random orthogonal matrix. Abandons
    /// when the scaled partial sum exceeds `upper_bound × (1 + ε/√d')`.
    #[inline(always)]
    #[cfg(feature = "benchmark")]
    pub fn compare_adsampling(
        &self,
        other: &Vertex<'a, T, N>,
        upper_bound: f32,
        epsilon: f32,
    ) -> f32 {
        <[T; N]>::distance_compare_adsampling(self.val, other.val, upper_bound, epsilon)
    }

    /// Borrows the vector coordinates retained by this vertex view.
    #[inline]
    pub fn vector(&self) -> &[T; N] {
        self.val
    }

    /// Returns the internal graph ID carried by this view.
    #[inline]
    pub fn vertex_id(&self) -> u32 {
        self.id
    }
}

impl<'a, T, const N: usize> TryFrom<(&'a [T], u32)> for Vertex<'a, T, N>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    type Error = TryFromSliceError;

    /// Borrows an exactly N-element vector with its ID; rejects a mismatched slice length.
    fn try_from((mem_slice, id): (&'a [T], u32)) -> Result<Self, Self::Error> {
        let array: &[T; N] = mem_slice.try_into()?;
        Ok(Vertex::new(array, id))
    }
}
