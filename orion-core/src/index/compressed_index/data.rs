/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Minimal dataset metadata needed by the graph owner and cache format.
//! Graph traversal consumes sessions, never this trait's virtual dispatch.

use orion_ann_diskann::model::{InmemDataset, PaddedDataset};
use orion_ann_vector::FullPrecisionDistance;

pub trait IndexData {
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
