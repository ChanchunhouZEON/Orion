use crate::common::NSGResult;

/// Trait for graph access in NSG search, enabling both in-memory and mmap-backed graphs.
pub trait NSGGraphAccess {
    fn get_neighbors(&self, node_id: u32) -> NSGResult<Vec<u32>>;
    fn num_nodes(&self) -> usize;
    fn navigation_node(&self) -> u32;
}
