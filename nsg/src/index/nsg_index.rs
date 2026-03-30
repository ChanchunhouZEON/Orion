use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use vector::{FullPrecisionDistance, Metric};

use crate::algorithm::edge_selection::select_edges;
use crate::algorithm::knn_graph::build_knn_graph;
use crate::algorithm::search::greedy_search;
use crate::common::{NSGError, NSGResult};
use crate::model::{NSGConfig, NSGGraph, Neighbor};
use crate::utils::calculate_medoid;

/// In-memory NSG index.
pub struct NSGIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Configuration.
    pub config: NSGConfig,
    /// Distance metric.
    pub metric: Metric,
    /// Stored vectors.
    pub data: Vec<[T; N]>,
    /// NSG graph.
    pub graph: NSGGraph,
}

impl<T, const N: usize> NSGIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create a new empty NSG index.
    pub fn new(config: NSGConfig, metric: Metric, capacity: usize) -> Self {
        Self {
            graph: NSGGraph::new(capacity, config.r),
            config,
            metric,
            data: Vec::with_capacity(capacity),
        }
    }

    /// Build NSG from a set of vectors.
    ///
    /// Steps:
    /// 1. Build initial k-NN graph (brute-force).
    /// 2. Select navigation node (closest to centroid).
    /// 3. For each node: greedy search from navigation node to get candidates,
    ///    then apply NSG edge selection.
    pub fn build(&mut self, vectors: Vec<[T; N]>) -> NSGResult<()> {
        let n = vectors.len();
        if n == 0 {
            return Err(NSGError::IndexError(
                "Cannot build index with 0 vectors".to_string(),
            ));
        }

        self.data = vectors;
        self.graph = NSGGraph::new(n, self.config.r);

        // Step 1: Build initial k-NN graph.
        println!("Building initial k-NN graph (k={})...", self.config.k);
        let knn = build_knn_graph(&self.data, self.config.k, self.metric);

        // Initialize the NSG graph with k-NN edges.
        for (i, neighbors) in knn.iter().enumerate() {
            let neighbor_ids: Vec<u32> = neighbors.iter().map(|n| n.id).collect();
            self.graph.set_neighbors(i as u32, neighbor_ids)?;
        }

        // Step 2: Select navigation node (medoid).
        self.graph.navigation_node = calculate_medoid(&self.data, self.metric);
        println!("Navigation node: {}", self.graph.navigation_node);

        // Step 3: For each node, refine edges using NSG edge selection.
        println!(
            "Refining edges with NSG edge selection (R={}, L={}, C={})...",
            self.config.r, self.config.l, self.config.c
        );

        for i in 0..n {
            // Greedy search from navigation node to find candidates.
            let candidates = greedy_search(
                &self.data[i],
                self.graph.navigation_node,
                self.config.c,
                &self.graph,
                &self.data,
                self.metric,
            )?;

            // Filter out self.
            let filtered: Vec<Neighbor> = candidates
                .into_iter()
                .filter(|c| c.id != i as u32)
                .collect();

            // Select edges using NSG criterion.
            let selected = select_edges(&filtered, self.config.r, &self.data, self.metric);
            self.graph.set_neighbors(i as u32, selected)?;

            if (i + 1) % 10000 == 0 {
                println!("  NSG edge selection: {}/{} points", i + 1, n);
            }
        }

        println!(
            "NSG index built. Navigation node: {}",
            self.graph.navigation_node
        );
        Ok(())
    }

    /// Search for K nearest neighbors.
    pub fn search(&self, query: &[T; N], k: usize) -> NSGResult<Vec<Neighbor>> {
        self.search_with_l(query, k, self.config.l)
    }

    /// Search with a custom L value.
    pub fn search_with_l(&self, query: &[T; N], k: usize, l: usize) -> NSGResult<Vec<Neighbor>> {
        if self.data.is_empty() {
            return Ok(Vec::new());
        }

        let l = l.max(k);
        let mut results = greedy_search(
            query,
            self.graph.navigation_node,
            l,
            &self.graph,
            &self.data,
            self.metric,
        )?;

        results.truncate(k);
        Ok(results)
    }

    /// Save the index to a file.
    pub fn save(&self, filename: &str) -> NSGResult<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        let n = self.data.len() as u32;
        writer.write_u32::<LittleEndian>(n)?;
        writer.write_u32::<LittleEndian>(N as u32)?;
        writer.write_u32::<LittleEndian>(self.graph.navigation_node)?;
        writer.write_u32::<LittleEndian>(self.config.r as u32)?;
        writer.write_u8(self.metric as u8)?;

        // Write adjacency lists.
        for i in 0..n {
            let neighbors = self.graph.get_neighbors(i)?;
            writer.write_u32::<LittleEndian>(neighbors.len() as u32)?;
            for &nid in &neighbors {
                writer.write_u32::<LittleEndian>(nid)?;
            }
        }

        // Write vector data.
        for i in 0..n as usize {
            for j in 0..N {
                let val: f32 = self.data[i][j].into();
                writer.write_f32::<LittleEndian>(val)?;
            }
        }

        writer.flush()?;
        println!("NSG index saved to {}", filename);
        Ok(())
    }

    /// Load the index from a file.
    pub fn load(filename: &str) -> NSGResult<NSGIndex<f32, N>>
    where
        [f32; N]: FullPrecisionDistance<f32, N>,
    {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let n = reader.read_u32::<LittleEndian>()? as usize;
        let dim = reader.read_u32::<LittleEndian>()? as usize;
        let navigation_node = reader.read_u32::<LittleEndian>()?;
        let r = reader.read_u32::<LittleEndian>()? as usize;
        let metric_byte = reader.read_u8()?;

        if dim != N {
            return Err(NSGError::InvalidConfig(format!(
                "Dimension mismatch: file has {}, expected {}",
                dim, N
            )));
        }

        let metric = match metric_byte {
            0 => Metric::L2,
            1 => Metric::Cosine,
            _ => {
                return Err(NSGError::InvalidConfig(format!(
                    "Unknown metric: {}",
                    metric_byte
                )))
            }
        };

        let config = NSGConfig::new(r, 100, 200, 50, 1);
        let mut graph = NSGGraph::new(n, r);
        graph.navigation_node = navigation_node;

        // Read adjacency lists.
        for i in 0..n {
            let num_neighbors = reader.read_u32::<LittleEndian>()? as usize;
            let mut neighbors = Vec::with_capacity(num_neighbors);
            for _ in 0..num_neighbors {
                neighbors.push(reader.read_u32::<LittleEndian>()?);
            }
            graph.set_neighbors(i as u32, neighbors)?;
        }

        // Read vector data.
        let mut data: Vec<[f32; N]> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut vec = [0.0f32; N];
            for v in vec.iter_mut() {
                *v = reader.read_f32::<LittleEndian>()?;
            }
            data.push(vec);
        }

        println!("NSG index loaded from {}. {} points", filename, n);

        Ok(NSGIndex {
            config,
            metric,
            data,
            graph,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_data(n: usize) -> Vec<[f32; 8]> {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut v = [0.0f32; 8];
            for j in 0..8 {
                v[j] = ((i * 7 + j * 13) % 100) as f32 / 100.0;
            }
            data.push(v);
        }
        data
    }

    #[test]
    fn test_build_and_search_e2e() {
        let data = random_data(50);
        let config = NSGConfig::new(8, 30, 40, 10, 1);
        let mut index = NSGIndex::<f32, 8>::new(config, Metric::L2, 50);
        index.build(data.clone()).unwrap();

        let results = index.search(&data[0], 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_search_returns_k_results() {
        let data = random_data(50);
        let config = NSGConfig::new(8, 30, 40, 10, 1);
        let mut index = NSGIndex::<f32, 8>::new(config, Metric::L2, 50);
        index.build(data).unwrap();

        let query = [0.5; 8];
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_results_ordered() {
        let data = random_data(50);
        let config = NSGConfig::new(8, 30, 40, 10, 1);
        let mut index = NSGIndex::<f32, 8>::new(config, Metric::L2, 50);
        index.build(data).unwrap();

        let query = [0.3; 8];
        let results = index.search(&query, 10).unwrap();
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let data = random_data(30);
        let config = NSGConfig::new(6, 20, 30, 8, 1);
        let mut index = NSGIndex::<f32, 8>::new(config, Metric::L2, 30);
        index.build(data).unwrap();

        let query = [0.5; 8];
        let original_results = index.search(&query, 5).unwrap();

        let tmp = std::env::temp_dir().join("nsg_test_save_load.bin");
        let path = tmp.to_str().unwrap();
        index.save(path).unwrap();

        let loaded = NSGIndex::<f32, 8>::load(path).unwrap();
        let loaded_results = loaded.search(&query, 5).unwrap();

        assert_eq!(original_results.len(), loaded_results.len());
        for (a, b) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.distance - b.distance).abs() < 1e-6);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_build_empty_returns_error() {
        let config = NSGConfig::new(4, 20, 30, 8, 1);
        let mut index = NSGIndex::<f32, 8>::new(config, Metric::L2, 0);
        let result = index.build(Vec::new());
        assert!(result.is_err());
    }
}
