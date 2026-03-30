use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use vector::{FullPrecisionDistance, Metric};

use crate::algorithm::insert::insert_node;
use crate::algorithm::search::{search_layer, search_upper_layers};
use crate::common::{HNSWError, HNSWResult};
use crate::model::{HNSWConfig, HNSWGraph, Neighbor};
use crate::utils::random_level;

/// In-memory HNSW index.
pub struct HNSWIndex<T, const N: usize>
where
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Configuration
    pub config: HNSWConfig,
    /// Distance metric
    pub metric: Metric,
    /// Stored vectors (flat array of [T; N])
    pub data: Vec<[T; N]>,
    /// Multi-layer graph
    pub graph: HNSWGraph,
}

impl<T, const N: usize> HNSWIndex<T, N>
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    /// Create a new empty HNSW index.
    pub fn new(config: HNSWConfig, metric: Metric, capacity: usize) -> Self {
        Self {
            graph: HNSWGraph::new(capacity, config.m, config.m_max0),
            config,
            metric,
            data: Vec::with_capacity(capacity),
        }
    }

    /// Build index from a set of vectors.
    pub fn build(&mut self, vectors: Vec<[T; N]>) -> HNSWResult<()> {
        let n = vectors.len();
        if n == 0 {
            return Err(HNSWError::IndexError(
                "Cannot build index with 0 vectors".to_string(),
            ));
        }

        self.data = vectors;
        self.graph = HNSWGraph::new(n, self.config.m, self.config.m_max0);

        println!(
            "Building HNSW index with {} points, M={}, ef_construction={}",
            n, self.config.m, self.config.ef_construction
        );

        for i in 0..n {
            let level = random_level(self.config.ml);
            insert_node(
                i as u32,
                level,
                &mut self.graph,
                &self.data,
                self.metric,
                self.config.ef_construction,
                true,
            )?;

            if (i + 1) % 10000 == 0 {
                println!("  Inserted {}/{} points", i + 1, n);
            }
        }

        println!(
            "HNSW index built. Entry point: {}, max level: {}",
            self.graph.entry_point, self.graph.max_level
        );

        Ok(())
    }

    /// Search for K nearest neighbors.
    ///
    /// Returns (ids, distances) sorted by distance.
    pub fn search(&self, query: &[T; N], k: usize) -> HNSWResult<Vec<Neighbor>> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search with a custom ef value.
    pub fn search_with_ef(&self, query: &[T; N], k: usize, ef: usize) -> HNSWResult<Vec<Neighbor>> {
        if self.graph.num_nodes() == 0 {
            return Ok(Vec::new());
        }

        let ef = ef.max(k);

        // Greedy search through upper layers.
        let mut entry_point = self.graph.entry_point;
        if self.graph.max_level > 0 {
            entry_point = search_upper_layers(
                query,
                entry_point,
                self.graph.max_level,
                1,
                &self.graph,
                &self.data,
                self.metric,
            )?;
        }

        // Search layer 0 with ef.
        let mut results = search_layer(
            query,
            entry_point,
            ef,
            0,
            &self.graph,
            &self.data,
            self.metric,
        )?;

        // Return top-K results.
        results.truncate(k);
        Ok(results)
    }

    /// Save the index to a file.
    pub fn save(&self, filename: &str) -> HNSWResult<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        // Header: num_nodes, dim, max_level, entry_point, m, m_max0, ef_construction, ef_search
        let n = self.graph.num_nodes() as u32;
        writer.write_u32::<LittleEndian>(n)?;
        writer.write_u32::<LittleEndian>(N as u32)?;
        writer.write_u32::<LittleEndian>(self.graph.max_level as u32)?;
        writer.write_u32::<LittleEndian>(self.graph.entry_point)?;
        writer.write_u32::<LittleEndian>(self.config.m as u32)?;
        writer.write_u32::<LittleEndian>(self.config.m_max0 as u32)?;
        writer.write_u32::<LittleEndian>(self.config.ef_construction as u32)?;
        writer.write_u32::<LittleEndian>(self.config.ef_search as u32)?;
        writer.write_u8(self.metric as u8)?;

        // Write node levels.
        for i in 0..n {
            writer.write_u32::<LittleEndian>(self.graph.node_level(i) as u32)?;
        }

        // Write adjacency lists per layer.
        for layer in 0..=self.graph.max_level {
            for i in 0..n {
                let neighbors = self.graph.get_neighbors(i, layer)?;
                writer.write_u32::<LittleEndian>(neighbors.len() as u32)?;
                for &nid in &neighbors {
                    writer.write_u32::<LittleEndian>(nid)?;
                }
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
        println!("HNSW index saved to {}", filename);
        Ok(())
    }

    /// Load the index from a file.
    pub fn load(filename: &str) -> HNSWResult<HNSWIndex<f32, N>>
    where
        [f32; N]: FullPrecisionDistance<f32, N>,
    {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let n = reader.read_u32::<LittleEndian>()? as usize;
        let dim = reader.read_u32::<LittleEndian>()? as usize;
        let max_level = reader.read_u32::<LittleEndian>()? as usize;
        let entry_point = reader.read_u32::<LittleEndian>()?;
        let m = reader.read_u32::<LittleEndian>()? as usize;
        let m_max0 = reader.read_u32::<LittleEndian>()? as usize;
        let ef_construction = reader.read_u32::<LittleEndian>()? as usize;
        let ef_search = reader.read_u32::<LittleEndian>()? as usize;
        let metric_byte = reader.read_u8()?;

        if dim != N {
            return Err(HNSWError::InvalidConfig(format!(
                "Dimension mismatch: file has {}, expected {}",
                dim, N
            )));
        }

        let metric = match metric_byte {
            0 => Metric::L2,
            1 => Metric::Cosine,
            _ => {
                return Err(HNSWError::InvalidConfig(format!(
                    "Unknown metric: {}",
                    metric_byte
                )))
            }
        };

        let mut config = HNSWConfig::new(m, ef_construction, ef_search, 1);
        config.m_max0 = m_max0;

        let mut graph = HNSWGraph::new(n, m, m_max0);
        graph.ensure_layers(max_level, n);
        graph.entry_point = entry_point;
        graph.max_level = max_level;

        // Read node levels.
        for i in 0..n {
            let level = reader.read_u32::<LittleEndian>()? as usize;
            graph.set_node_level(i as u32, level);
        }

        // Read adjacency lists.
        for layer in 0..=max_level {
            for i in 0..n {
                let num_neighbors = reader.read_u32::<LittleEndian>()? as usize;
                let mut neighbors = Vec::with_capacity(num_neighbors);
                for _ in 0..num_neighbors {
                    neighbors.push(reader.read_u32::<LittleEndian>()?);
                }
                graph.set_neighbors(i as u32, layer, neighbors)?;
            }
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

        graph.set_num_nodes(n);

        println!(
            "HNSW index loaded from {}. {} points, max_level={}",
            filename, n, max_level
        );

        Ok(HNSWIndex {
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
        let data = random_data(100);
        let config = HNSWConfig::new(8, 50, 50, 1);
        let mut index = HNSWIndex::<f32, 8>::new(config, Metric::L2, 100);
        index.build(data.clone()).unwrap();

        // Search for the first vector — it should find itself as nearest
        let results = index.search(&data[0], 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance < 1e-6);
    }

    #[test]
    fn test_search_returns_k_results() {
        let data = random_data(50);
        let config = HNSWConfig::new(8, 50, 50, 1);
        let mut index = HNSWIndex::<f32, 8>::new(config, Metric::L2, 50);
        index.build(data).unwrap();

        let query = [0.5; 8];
        let results = index.search(&query, 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_results_ordered_by_distance() {
        let data = random_data(100);
        let config = HNSWConfig::new(8, 50, 50, 1);
        let mut index = HNSWIndex::<f32, 8>::new(config, Metric::L2, 100);
        index.build(data).unwrap();

        let query = [0.3; 8];
        let results = index.search(&query, 20).unwrap();
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let data = random_data(30);
        let config = HNSWConfig::new(4, 20, 20, 1);
        let mut index = HNSWIndex::<f32, 8>::new(config, Metric::L2, 30);
        index.build(data.clone()).unwrap();

        let query = [0.5; 8];
        let original_results = index.search(&query, 5).unwrap();

        let tmp = std::env::temp_dir().join("hnsw_test_save_load.bin");
        let path = tmp.to_str().unwrap();
        index.save(path).unwrap();

        let loaded = HNSWIndex::<f32, 8>::load(path).unwrap();
        let loaded_results = loaded.search(&query, 5).unwrap();

        // Results should match
        assert_eq!(original_results.len(), loaded_results.len());
        for (a, b) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(a.id, b.id);
            assert!((a.distance - b.distance).abs() < 1e-6);
        }

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_build_empty_returns_error() {
        let config = HNSWConfig::new(4, 20, 20, 1);
        let mut index = HNSWIndex::<f32, 8>::new(config, Metric::L2, 0);
        let result = index.build(Vec::new());
        assert!(result.is_err());
    }
}
