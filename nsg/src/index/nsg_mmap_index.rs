use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::algorithm::search::greedy_search;
use crate::common::{NSGError, NSGResult};
use crate::model::{NSGConfig, NSGGraphAccess, Neighbor};

use super::NSGIndex;

/// NSG mmap file format:
///
/// ```text
/// [Common header: 32 bytes] (platform::graph_mmap format)
/// [NSG header: 8 bytes]
///   navigation_node: u32
///   r (max_degree): u32
/// [Adjacency data: num_nodes * (1 + max_degree) * 4 bytes]
/// [Vector data: num_nodes * dimension * 4 bytes]
/// ```
const NSG_ALGO_HEADER_SIZE: usize = 8;

/// Read-only mmap-backed NSG search index.
pub struct NSGMmapIndex<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    mmap: Mmap,
    pub config: NSGConfig,
    pub metric: Metric,
    pub navigation_node: u32,
    pub num_nodes: u32,
    max_degree: usize,
    /// Offset where adjacency data starts
    adj_offset: usize,
    /// Offset where vector data starts
    vec_offset: usize,
}

impl<const N: usize> NSGMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Save an in-memory NSGIndex to an mmap-compatible file.
    pub fn save_mmap<T>(index: &NSGIndex<T, N>, path: &str) -> NSGResult<()>
    where
        T: Default + Copy + Sync + Send + Into<f32>,
        [T; N]: FullPrecisionDistance<T, N>,
    {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let n = index.data.len() as u32;

        // Common header (32 bytes)
        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(&0x414E4E53u32.to_le_bytes()); // magic
        header[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        header[8..12].copy_from_slice(&n.to_le_bytes());
        header[12..16].copy_from_slice(&(index.config.r as u32).to_le_bytes());
        header[16..20].copy_from_slice(&(N as u32).to_le_bytes());
        header[20] = index.metric as u8;
        header[21] = 2; // AlgorithmId::Nsg
        file.write_all(&header)?;

        // NSG-specific header
        file.write_all(&index.graph.navigation_node.to_le_bytes())?;
        file.write_all(&(index.config.r as u32).to_le_bytes())?;

        // Adjacency data (fixed stride)
        let max_degree = index.config.r;
        for i in 0..n {
            let neighbors = index.graph.get_neighbors(i)?;
            let count = neighbors.len().min(max_degree) as u32;
            file.write_all(&count.to_le_bytes())?;
            for &nid in neighbors.iter().take(max_degree) {
                file.write_all(&nid.to_le_bytes())?;
            }
            // Pad remaining slots
            for _ in neighbors.len()..max_degree {
                file.write_all(&0u32.to_le_bytes())?;
            }
        }

        // Vector data
        for i in 0..n as usize {
            for j in 0..N {
                let val: f32 = index.data[i][j].into();
                file.write_all(&val.to_le_bytes())?;
            }
        }

        file.flush()?;
        Ok(())
    }

    /// Load a read-only mmap NSG index from file.
    pub fn load_mmap(path: &str) -> NSGResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < 32 + NSG_ALGO_HEADER_SIZE {
            return Err(NSGError::InvalidConfig(
                "File too small for header".to_string(),
            ));
        }

        // Parse common header
        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != 0x414E4E53 {
            return Err(NSGError::InvalidConfig(format!(
                "Invalid magic: 0x{magic:08X}"
            )));
        }

        let num_nodes = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let max_degree = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let dim = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        if dim != N {
            return Err(NSGError::InvalidConfig(format!(
                "Dimension mismatch: file has {dim}, expected {N}"
            )));
        }
        let metric_byte = mmap[20];
        let metric = match metric_byte {
            0 => Metric::L2,
            1 => Metric::Cosine,
            _ => {
                return Err(NSGError::InvalidConfig(format!(
                    "Unknown metric: {metric_byte}"
                )))
            }
        };

        // Parse NSG header
        let mut off = 32usize;
        let navigation_node = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap());
        off += 4;
        let r = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;

        let adj_offset = off;
        let adj_stride = (1 + max_degree) * 4;
        let vec_offset = adj_offset + (num_nodes as usize) * adj_stride;

        let config = NSGConfig::new(r, 100, 200, 50, 1);

        Ok(Self {
            mmap,
            config,
            metric,
            navigation_node,
            num_nodes,
            max_degree,
            adj_offset,
            vec_offset,
        })
    }

    /// Get neighbors of a node (zero-copy).
    pub fn get_neighbors_slice(&self, node_id: u32) -> &[u32] {
        let stride = (1 + self.max_degree) * 4;
        let base = self.adj_offset + (node_id as usize) * stride;
        let count = u32::from_le_bytes(self.mmap[base..base + 4].try_into().unwrap()) as usize;
        let count = count.min(self.max_degree);
        let ptr = &self.mmap[base + 4..base + 4 + count * 4];
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u32, count) }
    }

    /// Get vector data for a node (zero-copy).
    pub fn vector_f32(&self, node_id: u32) -> &[f32] {
        let base = self.vec_offset + (node_id as usize) * N * 4;
        let ptr = &self.mmap[base..base + N * 4];
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, N) }
    }

    /// Search for K nearest neighbors using the mmap-backed index.
    pub fn search(&self, query: &[f32; N], k: usize) -> NSGResult<Vec<Neighbor>> {
        self.search_with_l(query, k, self.config.l)
    }

    /// Warm the OS page cache by BFS from the navigation node.
    ///
    /// Reads neighbors and vectors in the BFS neighborhood, triggering page faults
    /// that bring the mmap pages into the OS cache.
    pub fn warm_cache(&self, max_hops: usize) {
        use crate::algorithm::warm_cache::bfs_neighborhood;

        if let Ok(nodes) = bfs_neighborhood(self, self.navigation_node, max_hops) {
            for &node in &nodes {
                let _ = self.get_neighbors_slice(node);
                let _ = self.vector_f32(node);
            }
        }
    }

    /// Search with a custom L value.
    ///
    /// Uses the graph trait to read neighbors directly from mmap (no full graph reconstruction).
    /// Reads vectors on-demand from mmap via `VectorStorage` — zero heap allocation.
    pub fn search_with_l(&self, query: &[f32; N], k: usize, l: usize) -> NSGResult<Vec<Neighbor>> {
        if self.num_nodes == 0 {
            return Ok(Vec::new());
        }

        let l = l.max(k);

        // Use `self` as both graph (NSGGraphAccess) and data (VectorStorage)
        // — reads neighbors and vectors on-demand from mmap, zero heap allocation
        let mut results = greedy_search(query, self.navigation_node, l, self, self, self.metric)?;

        results.truncate(k);
        Ok(results)
    }
}

/// Implement NSGGraphAccess for mmap index so search functions work directly.
impl<const N: usize> NSGGraphAccess for NSGMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    fn get_neighbors(&self, node_id: u32) -> NSGResult<Vec<u32>> {
        Ok(self.get_neighbors_slice(node_id).to_vec())
    }
    fn num_nodes(&self) -> usize {
        self.num_nodes as usize
    }
    fn navigation_node(&self) -> u32 {
        self.navigation_node
    }
}

/// Implement VectorStorage so mmap index can serve as both graph and data source.
/// Reads vectors on-demand from mmap — no heap allocation for the full vector set.
impl<const N: usize> VectorStorage<f32, N> for NSGMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[inline]
    fn get_vector(&self, id: u32) -> [f32; N] {
        let slice = self.vector_f32(id);
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(slice);
        arr
    }
    fn num_vectors(&self) -> usize {
        self.num_nodes as usize
    }
}
