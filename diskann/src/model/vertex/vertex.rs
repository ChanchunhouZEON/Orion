/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::array::TryFromSliceError;
use vector::{FullPrecisionDistance, Metric};

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
    pub fn new(val: &'a [T; N], id: u32) -> Self {
        Self { val, id }
    }

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

    #[inline]
    pub fn vector(&self) -> &[T; N] {
        self.val
    }

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

    fn try_from((mem_slice, id): (&'a [T], u32)) -> Result<Self, Self::Error> {
        let array: &[T; N] = mem_slice.try_into()?;
        Ok(Vertex::new(array, id))
    }
}
