/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! base: PaddedDataset<T> -> N rows, each [D logical T values | zero padding]
//!       InmemDataset<T, N> -> legacy fixed-width rows
//! ```
//!
//! This module provides index-facing shape, coordinate validation and query
//! preparation over storage owned by the dataset. It introduces no second
//! vector allocation or file format; base-file decoding belongs to runtime I/O.

//! Minimal dataset metadata needed by the graph owner and cache format.
//! Graph traversal consumes sessions, never this trait's virtual dispatch.

use orion_ann_diskann::model::{InmemDataset, PaddedDataset};
use orion_ann_vector::FullPrecisionDistance;

pub trait IndexData {
    fn placement(&self) -> orion_ann_platform::region::Placement {
        orion_ann_platform::region::Placement::Resident
    }
    fn point_count(&self) -> usize;
    fn dimension(&self) -> usize;
    fn element_type(&self) -> &'static str;
}

impl<T, const N: usize> IndexData for InmemDataset<T, N>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    fn point_count(&self) -> usize {
        self.num_active_pts
    }
    fn dimension(&self) -> usize {
        N
    }
    fn element_type(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

impl<T: bytemuck::Pod> IndexData for PaddedDataset<T> {
    fn placement(&self) -> orion_ann_platform::region::Placement {
        self.placement()
    }
    fn point_count(&self) -> usize {
        self.len()
    }
    fn dimension(&self) -> usize {
        self.layout().logical_dim()
    }
    fn element_type(&self) -> &'static str {
        std::any::type_name::<T>()
    }
}

mod element;
pub use element::DynamicElement;
