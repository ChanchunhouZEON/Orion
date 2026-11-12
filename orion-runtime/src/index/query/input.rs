/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Borrowed query shape. Coordinates and conversions are owned by core query
//! preparation; base dtype does not constrain accepted query dtype.
use crate::{Error, Result};
use orion_ann_core::search::query::{QueryElement, QueryInput};

/// Read-only packed rows; synchronous operations retain no caller borrows.
#[derive(Clone, Copy)]
pub struct QueryBatch<'a> {
    pub(in crate::index) values: QueryInput<'a>,
    pub(super) dimension: usize,
}
impl<'a> QueryBatch<'a> {
    /// Borrow any supported scalar without copying. Coordinates are validated
    /// before execution; construction checks a positive, exactly dividing width.
    pub fn from_slice<T: QueryElement>(values: &'a [T], dimension: usize) -> Result<Self> {
        Self::from_input(T::view(values), dimension)
    }
    /// Wrap typed stream/FFI transport without converting the entire batch.
    pub fn from_input(values: QueryInput<'a>, dimension: usize) -> Result<Self> {
        if dimension == 0 || !values.len().is_multiple_of(dimension) {
            return Err(Error::packed_shape("queries", dimension, values.len()).at("read_queries"));
        }
        Ok(Self { values, dimension })
    }
    pub fn from_f32(values: &'a [f32], dimension: usize) -> Result<Self> {
        Self::from_slice(values, dimension)
    }
    pub fn from_u8(values: &'a [u8], dimension: usize) -> Result<Self> {
        Self::from_slice(values, dimension)
    }
    pub fn from_i8(values: &'a [i8], dimension: usize) -> Result<Self> {
        Self::from_slice(values, dimension)
    }
    pub fn from_u16(values: &'a [u16], dimension: usize) -> Result<Self> {
        Self::from_slice(values, dimension)
    }
    pub fn from_u32(values: &'a [u32], dimension: usize) -> Result<Self> {
        Self::from_slice(values, dimension)
    }
    pub fn dimension(self) -> usize {
        self.dimension
    }
    pub fn len(self) -> usize {
        self.values.len() / self.dimension
    }
    pub fn is_empty(self) -> bool {
        self.values.is_empty()
    }
}
impl std::fmt::Debug for QueryBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryBatch")
            .field("rows", &self.len())
            .field("dimension", &self.dimension)
            .field("storage", &self.values.element_type())
            .finish()
    }
}
