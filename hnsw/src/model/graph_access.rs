use crate::common::HNSWResult;

/// Trait for graph access in HNSW search, enabling both in-memory and mmap-backed graphs.
pub trait HNSWGraphAccess {
    fn get_neighbors(&self, node_id: u32, layer: usize) -> HNSWResult<Vec<u32>>;
    fn num_nodes(&self) -> usize;
    fn max_level(&self) -> usize;
    fn entry_point(&self) -> u32;
}
