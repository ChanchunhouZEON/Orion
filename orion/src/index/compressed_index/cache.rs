//! Graph persistence shared by typed and runtime-dimensional indexes.
use super::{CompressedIndex, IndexData};
use crate::model::PhasedGraph;
use diskann::common::{ANNError, ANNResult};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::Path,
};

impl<D: IndexData, S: Default> CompressedIndex<D, S> {
    /// Return per-node (degree, local_count) from the PhasedGraph.
    pub fn graph_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.num_nodes;
        (0..n)
            .map(|i| (self.graph.degree(i), self.graph.local_count(i)))
            .collect()
    }

    // --- IO ---

    /// Check shape without scanning a potentially billion-node graph.
    /// Graph edge validity remains the graph producer's responsibility.
    pub fn validate_dataset_graph(&self) -> ANNResult<()> {
        let rows = self.dataset.point_count();
        if rows == 0
            || rows > u32::MAX as usize
            || self.dataset.dimension() == 0
            || rows != self.graph.num_nodes()
            || rows != self.num_nodes
            || self.entry as usize >= rows
        {
            return Err(ANNError::log_index_error(format!(
                "incompatible dataset/graph: rows={rows}, dimension={}, graph_nodes={}, entry={}",
                self.dataset.dimension(),
                self.graph.num_nodes(),
                self.entry
            )));
        }
        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> ANNResult<()> {
        self.validate_dataset_graph()?;
        let dir = path.as_ref().parent().unwrap_or(Path::new("."));
        fs::create_dir_all(dir)?;

        let graph_path = path.as_ref().with_extension("pgraph");
        // Publish completed files only; metadata is the last file published.
        // The CLI additionally requires a matching manifest before cache reuse.
        let graph_tmp = graph_path.with_extension(format!("pgraph.{}.tmp", std::process::id()));
        self.graph.save(&graph_tmp)?;
        File::open(&graph_tmp)?.sync_all()?;
        let meta_tmp = path
            .as_ref()
            .with_extension(format!("bin.{}.tmp", std::process::id()));

        let meta = OrionMeta {
            version: 6,
            entry: self.entry,
        };
        let mut writer = BufWriter::new(File::create(&meta_tmp)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut writer, config)?;
        let shape = DatasetShape {
            dimension: self.dataset.dimension() as u64,
            points: self.dataset.point_count() as u64,
            element_type: self.dataset.element_type().to_owned(),
        };
        bincode::serde::encode_into_std_write(&shape, &mut writer, config)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(graph_tmp, graph_path)?;
        fs::rename(meta_tmp, path.as_ref())?;
        Ok(())
    }

    /// Construct from cached PhasedGraph + metadata on disk.
    /// `path` is the metadata `.bin` file; PhasedGraph is read from `path.with_extension("pgraph")`.
    /// Takes ownership of the caller-provided dataset without copying its payload.
    pub fn load_from_cache<P: AsRef<Path>>(path: P, dataset: D) -> ANNResult<Self> {
        Self::load_cache_impl(path, dataset, false)
    }

    /// Load a legacy cache after the caller has checked its manifest against
    /// the input identity, logical dimension, point count and storage type.
    pub fn load_from_cache_with_provenance<P: AsRef<Path>>(path: P, dataset: D) -> ANNResult<Self> {
        Self::load_cache_impl(path, dataset, true)
    }

    fn load_cache_impl<P: AsRef<Path>>(
        path: P,
        dataset: D,
        verified_legacy: bool,
    ) -> ANNResult<Self> {
        let graph_path = path.as_ref().with_extension("pgraph");
        if !graph_path.exists() {
            return Err(ANNError::log_index_error(format!(
                "PhasedGraph file not found: {:?}",
                graph_path
            )));
        }

        let file = File::open(&path)?;
        let mut reader = std::io::BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: OrionMeta = bincode::serde::decode_from_std_read(&mut reader, config)?;
        // Reject incompatible data before allocating/loading the graph payload.
        match meta.version {
            6 => {
                let shape: DatasetShape =
                    bincode::serde::decode_from_std_read(&mut reader, config.with_limit::<4096>())?;
                if shape.dimension != dataset.dimension() as u64
                    || shape.points != dataset.point_count() as u64
                    || shape.element_type != dataset.element_type()
                {
                    return Err(ANNError::log_index_error(
                        "cached graph dataset dimension, count or element type does not match"
                            .into(),
                    ));
                }
            }
            5 if D::ACCEPT_LEGACY_CACHE || verified_legacy => {}
            _ => {
                return Err(ANNError::log_index_error(format!(
                    "unsupported or unverifiable graph metadata version {}",
                    meta.version
                )));
            }
        }
        let graph = PhasedGraph::load(&graph_path)?;
        let mut idx = Self::from_phased_graph(dataset, graph, meta.entry, None, None);
        idx.validate_dataset_graph()?;
        // Preserve the cache path so `ensure_quantized_dataset` can locate
        // the `.qds` sidecar next to it (load fast path).
        idx.cache_base_path = path.as_ref().to_path_buf();
        Ok(idx)
    }
}

/// Serializable metadata for Orion (separate from PhasedGraph binary).
#[derive(serde::Serialize, serde::Deserialize)]
struct OrionMeta {
    version: u32,
    entry: u32,
}

/// Logical shape only: cache reuse does not depend on SIMD padding width.
#[derive(serde::Serialize, serde::Deserialize)]
struct DatasetShape {
    dimension: u64,
    points: u64,
    element_type: String,
}
