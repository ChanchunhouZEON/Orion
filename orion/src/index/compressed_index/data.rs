//! Minimal dataset metadata needed by the graph owner and cache format.
//! Graph traversal consumes sessions, never this trait's virtual dispatch.

use diskann::model::{InmemDataset, PaddedDataset};
use vector::FullPrecisionDistance;

pub trait IndexData {
    fn point_count(&self) -> usize;
    fn dimension(&self) -> usize;
    fn element_type(&self) -> &'static str;
    /// V5 metadata did not record dimension/type. Only the existing typed
    /// loader may rely on its caller's legacy provenance checks.
    const ACCEPT_LEGACY_CACHE: bool;
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
    const ACCEPT_LEGACY_CACHE: bool = true;
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
    const ACCEPT_LEGACY_CACHE: bool = false;
}
