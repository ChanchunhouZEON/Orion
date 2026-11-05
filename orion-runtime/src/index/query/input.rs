/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Borrowed query shape and representation; no index state or allocation.
use crate::{Error, Result};

/// A read-only, packed row-major view. Constructing a batch validates its shape;
/// search validates coordinates against the selected index and metric. The
/// borrowed storage is needed only until the synchronous operation returns.
#[derive(Clone, Copy)]
pub struct QueryBatch<'a> {
    pub(in crate::index) values: QueryValues<'a>,
    pub(super) dimension: usize,
}

#[derive(Clone, Copy)]
pub(in crate::index) enum QueryValues<'a> {
    F32(&'a [f32]),
    U8(&'a [u8]),
}

impl<'a> QueryBatch<'a> {
    /// Borrow f32 coordinates without copying or taking ownership.
    ///
    /// # Errors
    /// Dimension must be nonzero and divide the number of coordinates exactly.
    pub fn from_f32(values: &'a [f32], dimension: usize) -> Result<Self> {
        Self::new(QueryValues::F32(values), dimension)
    }
    /// Borrow byte coordinates. Native-u8 search consumes these directly;
    /// an f32 index converts them to its query representation internally.
    ///
    /// # Errors
    /// Dimension must be nonzero and divide the number of coordinates exactly.
    pub fn from_u8(values: &'a [u8], dimension: usize) -> Result<Self> {
        Self::new(QueryValues::U8(values), dimension)
    }
    fn new(values: QueryValues<'a>, dimension: usize) -> Result<Self> {
        let elements = match values {
            QueryValues::F32(v) => v.len(),
            QueryValues::U8(v) => v.len(),
        };
        if dimension == 0 || elements % dimension != 0 {
            return Err(Error::packed_shape("queries", dimension, elements).at("read_queries"));
        }
        Ok(Self { values, dimension })
    }
    /// Logical coordinates per row, excluding internal SIMD padding.
    pub fn dimension(self) -> usize {
        self.dimension
    }
    /// Number of query rows.
    pub fn len(self) -> usize {
        (match self.values {
            QueryValues::F32(v) => v.len(),
            QueryValues::U8(v) => v.len(),
        }) / self.dimension
    }
    /// Empty batches are allowed but still require valid search controls and calibration.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for QueryBatch<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryBatch")
            .field("rows", &self.len())
            .field("dimension", &self.dimension)
            .field(
                "storage",
                &match self.values {
                    QueryValues::F32(_) => "f32",
                    QueryValues::U8(_) => "u8",
                },
            )
            .finish_non_exhaustive()
    }
}
