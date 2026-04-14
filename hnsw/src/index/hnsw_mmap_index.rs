use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::Write;
use vector::{FullPrecisionDistance, Metric, VectorStorage};

use crate::algorithm::search::{search_layer, search_upper_layers};
use crate::common::{HNSWError, HNSWResult};
use crate::model::{HNSWConfig, HNSWGraphAccess, Neighbor};

use super::HNSWIndex;

/// HNSW mmap file format:
///
/// ```text
/// [Common header: 32 bytes] (platform::graph_mmap format)
/// [HNSW header]
///   entry_point: u32
///   max_level: u32
///   m: u32
///   m_max0: u32
///   ef_construction: u32
///   ef_search: u32
///   node_levels: [u32; num_nodes]
/// [Layer 0 adjacency: num_nodes * (1 + m_max0) * 4 bytes]
/// [Layer 1..=max_level adjacency: num_nodes * (1 + m) * 4 bytes each]
/// [Vector data: num_nodes * dimension * 4 bytes]
/// ```
/// Read-only mmap-backed HNSW search index.
pub struct HNSWMmapIndex<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    mmap: Mmap,
    pub config: HNSWConfig,
    pub metric: Metric,
    pub entry_point: u32,
    pub max_level: usize,
    pub num_nodes: u32,
    m_max0: usize,
    m: usize,
    /// Per-node level assignment
    #[allow(dead_code)]
    node_levels: Vec<usize>,
    /// Offsets into the mmap for each layer's adjacency data
    layer_offsets: Vec<usize>,
    /// Offset where vector data starts
    vec_offset: usize,
}

impl<const N: usize> HNSWMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Save an in-memory HNSWIndex to an mmap-compatible file.
    pub fn save_mmap<T>(index: &HNSWIndex<T, N>, path: &str) -> HNSWResult<()>
    where
        T: Default + Copy + Sync + Send + Into<f32>,
        [T; N]: FullPrecisionDistance<T, N>,
    {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let n = index.graph.num_nodes() as u32;

        // Common header (32 bytes)
        let mut header = [0u8; 32];
        header[0..4].copy_from_slice(&0x414E4E53u32.to_le_bytes()); // magic
        header[4..8].copy_from_slice(&1u32.to_le_bytes()); // version
        header[8..12].copy_from_slice(&n.to_le_bytes());
        // max_degree: max(m_max0, m) for the common header
        let max_degree = index.config.m_max0.max(index.config.m) as u32;
        header[12..16].copy_from_slice(&max_degree.to_le_bytes());
        header[16..20].copy_from_slice(&(N as u32).to_le_bytes());
        header[20] = index.metric as u8;
        header[21] = 1; // AlgorithmId::Hnsw
        file.write_all(&header)?;

        // HNSW-specific header
        file.write_all(&index.graph.entry_point.to_le_bytes())?;
        file.write_all(&(index.graph.max_level as u32).to_le_bytes())?;
        file.write_all(&(index.config.m as u32).to_le_bytes())?;
        file.write_all(&(index.config.m_max0 as u32).to_le_bytes())?;
        file.write_all(&(index.config.ef_construction as u32).to_le_bytes())?;
        file.write_all(&(index.config.ef_search as u32).to_le_bytes())?;

        // Node levels
        for i in 0..n {
            file.write_all(&(index.graph.node_level(i) as u32).to_le_bytes())?;
        }

        // Layer adjacency data with fixed stride per layer
        for layer in 0..=index.graph.max_level {
            let max_conn = if layer == 0 {
                index.config.m_max0
            } else {
                index.config.m
            };
            for i in 0..n {
                let neighbors = index.graph.get_neighbors(i, layer)?;
                let count = neighbors.len().min(max_conn) as u32;
                file.write_all(&count.to_le_bytes())?;
                for &nid in neighbors.iter().take(max_conn) {
                    file.write_all(&nid.to_le_bytes())?;
                }
                // Pad remaining slots
                for _ in neighbors.len()..max_conn {
                    file.write_all(&0u32.to_le_bytes())?;
                }
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

    /// Load a read-only mmap HNSW index from file.
    pub fn load_mmap(path: &str) -> HNSWResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < 32 {
            return Err(HNSWError::InvalidConfig(
                "File too small for header".to_string(),
            ));
        }

        // Parse common header
        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != 0x414E4E53 {
            return Err(HNSWError::InvalidConfig(format!(
                "Invalid magic: 0x{magic:08X}"
            )));
        }

        let num_nodes = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let dim = u32::from_le_bytes(mmap[16..20].try_into().unwrap()) as usize;
        if dim != N {
            return Err(HNSWError::InvalidConfig(format!(
                "Dimension mismatch: file has {dim}, expected {N}"
            )));
        }
        let metric_byte = mmap[20];
        let metric = match metric_byte {
            0 => Metric::L2,
            1 => Metric::Cosine,
            _ => {
                return Err(HNSWError::InvalidConfig(format!(
                    "Unknown metric: {metric_byte}"
                )))
            }
        };

        // Parse HNSW header
        let mut off = 32usize;
        let entry_point = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap());
        off += 4;
        let max_level = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let m = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let m_max0 = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let ef_construction = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let ef_search = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
        off += 4;

        // Node levels
        let mut node_levels = Vec::with_capacity(num_nodes as usize);
        for _ in 0..num_nodes {
            let level = u32::from_le_bytes(mmap[off..off + 4].try_into().unwrap()) as usize;
            node_levels.push(level);
            off += 4;
        }

        // Compute layer offsets
        let mut layer_offsets = Vec::with_capacity(max_level + 1);
        for layer in 0..=max_level {
            layer_offsets.push(off);
            let max_conn = if layer == 0 { m_max0 } else { m };
            let stride = (1 + max_conn) * 4;
            off += (num_nodes as usize) * stride;
        }

        let vec_offset = off;

        let mut config = HNSWConfig::new(m, ef_construction, ef_search, 1);
        config.m_max0 = m_max0;

        Ok(Self {
            mmap,
            config,
            metric,
            entry_point,
            max_level,
            num_nodes,
            m_max0,
            m,
            node_levels,
            layer_offsets,
            vec_offset,
        })
    }

    /// Get neighbors of a node at a given layer (zero-copy).
    pub fn get_neighbors_slice(&self, node_id: u32, layer: usize) -> &[u32] {
        if layer > self.max_level {
            return &[];
        }
        let max_conn = if layer == 0 { self.m_max0 } else { self.m };
        let stride = (1 + max_conn) * 4;
        let base = self.layer_offsets[layer] + (node_id as usize) * stride;
        let count = u32::from_le_bytes(self.mmap[base..base + 4].try_into().unwrap()) as usize;
        let count = count.min(max_conn);
        let ptr = &self.mmap[base + 4..base + 4 + count * 4];
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u32, count) }
    }

    /// Get vector data for a node (zero-copy).
    pub fn vector_f32(&self, node_id: u32) -> &[f32] {
        let base = self.vec_offset + (node_id as usize) * N * 4;
        let ptr = &self.mmap[base..base + N * 4];
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, N) }
    }

    /// Convert a zero-copy f32 slice to a fixed-size array reference.
    fn vector_array(&self, node_id: u32) -> [f32; N] {
        let slice = self.vector_f32(node_id);
        let mut arr = [0.0f32; N];
        arr.copy_from_slice(slice);
        arr
    }

    /// Search for K nearest neighbors using the mmap-backed index.
    pub fn search(&self, query: &[f32; N], k: usize) -> HNSWResult<Vec<Neighbor>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Warm the OS page cache by BFS from entry point.
    ///
    /// Reads neighbors and vectors in the BFS neighborhood, triggering page faults
    /// that bring the mmap pages into the OS cache. Upper layers (1..max_level) are
    /// always fully cached since they're small; only layer 0 uses selective prefetch.
    pub fn warm_cache(&self, max_hops: usize) {
        use crate::algorithm::warm_cache::bfs_neighborhood;

        // Warm all upper layers fully (they're small)
        for layer in (1..=self.max_level).rev() {
            if let Ok(nodes) = bfs_neighborhood(self, self.entry_point, layer, usize::MAX) {
                for &node in &nodes {
                    // Touch neighbor data to fault pages in
                    let _ = self.get_neighbors_slice(node, layer);
                }
            }
        }

        // Warm layer 0 selectively (max_hops BFS)
        if let Ok(nodes) = bfs_neighborhood(self, self.entry_point, 0, max_hops) {
            for &node in &nodes {
                let _ = self.get_neighbors_slice(node, 0);
                let _ = self.vector_f32(node);
            }
        }
    }

    /// Search with a custom ef value.
    ///
    /// Uses the graph trait to read neighbors directly from mmap (no full graph reconstruction).
    /// Reads vectors on-demand from mmap via `VectorStorage` — zero heap allocation.
    pub fn search_with_ef(
        &self,
        query: &[f32; N],
        k: usize,
        ef: usize,
    ) -> HNSWResult<Vec<Neighbor>> {
        if self.num_nodes == 0 {
            return Ok(Vec::new());
        }

        let ef = ef.max(k);

        // Use `self` as both graph (HNSWGraphAccess) and data (VectorStorage)
        // — reads neighbors and vectors on-demand from mmap, zero heap allocation
        let mut entry_point = self.entry_point;
        if self.max_level > 0 {
            entry_point = search_upper_layers(
                query,
                entry_point,
                self.max_level,
                1,
                self,
                self,
                self.metric,
            )?;
        }

        // Search layer 0
        let mut results = search_layer(query, entry_point, ef, 0, self, self, self.metric)?;

        results.truncate(k);
        Ok(results)
    }
}

/// Implement HNSWGraphAccess for mmap index so search functions work directly.
impl<const N: usize> HNSWGraphAccess for HNSWMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    fn get_neighbors(&self, node_id: u32, layer: usize) -> HNSWResult<Vec<u32>> {
        Ok(self.get_neighbors_slice(node_id, layer).to_vec())
    }
    fn num_nodes(&self) -> usize {
        self.num_nodes as usize
    }
    fn max_level(&self) -> usize {
        self.max_level
    }
    fn entry_point(&self) -> u32 {
        self.entry_point
    }
}

/// Implement VectorStorage so mmap index can serve as both graph and data source.
/// Reads vectors on-demand from mmap — no heap allocation for the full vector set.
impl<const N: usize> VectorStorage<f32, N> for HNSWMmapIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    #[inline]
    fn get_vector(&self, id: u32) -> [f32; N] {
        self.vector_array(id)
    }
    fn num_vectors(&self) -> usize {
        self.num_nodes as usize
    }
}
