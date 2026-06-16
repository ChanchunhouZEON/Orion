/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::FixedChunkPQTable;
use crate::model::PhasedGraph;
use crate::model::dataset::jl_hadamard_dataset::{JL_HADAMARD_MAGIC, JlHadamardDataset};
use crate::model::dataset::jl_sparse_dataset::{
    JL_SPARSE_MAGIC, JLSparseDataset, JLSparseDatasetMips,
};
use crate::model::dataset::l2_kt_dataset::L2KTDataset;
use crate::model::dataset::rabitq_b4_dataset::{RABITQ_B4_MAGIC, RabitQ4Dataset};
use crate::model::dataset::rabitq_dataset::{RABITQ_MAGIC, RabitQDataset};
use crate::model::scratch::InMemScratchPool;
use crate::model::{L2U8, L2U16, MipsI8, MipsI16, QuantSpec, QuantizedDataset};
use diskann::model::InmemDataset;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock}; // Arc still needed for pq field
use std::time::Instant;
use vector::FullPrecisionDistance;

/// Staged DiskANN with PhasedGraph-based two-phase search.
///
/// Uses a `PhasedGraph` where each node's neighbors are partitioned:
/// - `local_neighbors`: candidate-set-confirmed (navigation + reranking)
/// - `remote_neighbors`: long-range shortcuts (navigation only)
/// - `extra_candidates`: non-graph candidate-set points (reranking only)
pub struct StagedDiskANN<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Vector data storage taken from the DiskANN `InmemIndex`.
    pub dataset: InmemDataset<f32, N>,
    /// PhasedGraph with local/remote/extra layout + bidir bits.
    pub graph: PhasedGraph,
    pub entry: u32,
    pub num_nodes: usize,
    /// Base path for on-disk caches of this index. The metadata blob
    /// is written here directly; every sidecar (PhasedGraph `.pgraph`,
    /// quantized datasets `.qds` / `.qdm8` / `.qdm16` / `.qrbq`) is
    /// derived by `path.with_extension(...)`. Empty `PathBuf` means
    /// "in-memory only, do not persist."
    #[allow(dead_code)]
    pub cache_base_path: PathBuf,

    #[allow(dead_code)]
    pub is_save: bool,

    // PQ relevant
    pub pq: Option<Arc<FixedChunkPQTable>>,
    pub pq_codes: Option<Vec<u8>>,
    pub num_pq_chunks: Option<usize>,

    /// Pool of pre-allocated scratch spaces for in-memory search.
    /// Lazily initialized on first call to `search()`.
    pub(crate) inmem_scratch_pool: OnceLock<InMemScratchPool>,

    /// U8-quantized base for the L2 prefilter (ParlayANN-style). Built
    /// lazily on first `ensure_quantized_dataset`. Sidecar `.qds`.
    pub(crate) q_dataset: OnceLock<QuantizedDataset<L2U8, N>>,

    /// U16-quantized base — precise PQ-admission tier of the L2
    /// cascade. 256× finer per-dim resolution than u8 at 2× storage,
    /// pairs with the u8 sidecar via the PA `quantize_bits 16` recipe
    /// (u8 = cheap filter, u16 = PQ admission distance). Built lazily
    /// on first `ensure_quantized_dataset_l2_u16`. Sidecar `.qds6`.
    pub(crate) q_dataset_l2_u16: OnceLock<QuantizedDataset<L2U16, N>>,

    /// I8-quantized base for the MIPS search path on unit-normalized
    /// data. Built lazily on first `ensure_quantized_dataset_mips`.
    /// Sidecar `.qdm8`.
    pub(crate) q_dataset_mips: OnceLock<QuantizedDataset<MipsI8, N>>,

    /// I16 twin of the above — 2× per-element precision for the
    /// high-recall band where i8's distance-collision ceiling caps
    /// recall on hard angular workloads. Built lazily on first
    /// `ensure_quantized_dataset_mips_i16`. Sidecar `.qdm16`.
    pub(crate) q_dataset_mips_i16: OnceLock<QuantizedDataset<MipsI16, N>>,

    /// RaBitQ 1-bit-per-dim quantized base for the high-dim
    /// bandwidth-bound L2 path (Gao & Long, SIGMOD 2024). Built lazily
    /// on first `ensure_quantized_dataset_rabitq`. Sidecar `.qrbq`.
    /// Targets GIST-class datasets where the entire quantized base
    /// fits into L1 cache.
    pub(crate) q_dataset_rabitq: OnceLock<RabitQDataset<N>>,

    /// Extended RaBitQ at B=4 bits/dim (Gao & Long, SIGMOD 2025).
    /// Same random-rotation front-end but a 4-bit signed scalar
    /// quantizer on the rotated components. Sidecar `.qrb4`. Lifts
    /// the recall ceiling at the cost of 4× more storage per vertex
    /// vs B=1 (still ~2× compression vs u8 on GIST D=960).
    pub(crate) q_dataset_rabitq_b4: OnceLock<RabitQ4Dataset<N>>,

    /// Johnson-Lindenstrauss sparse-projection binary signature
    /// (1024 bits per vertex) — used as the cheapest prefilter tier
    /// in the L2 search cascade, mirroring ParlayANN's
    /// `Euclidean_JL_Sparse_Point<1024>`. Sidecar `.jls`. Hamming
    /// distance via NEON `vcntq_u8` is ~24× cheaper per cmp than the
    /// u8 NEON L2 kernel on GIST D=960, but is only a rough L2
    /// correlate — strictly a prefilter, never the final ranker.
    pub(crate) q_dataset_jl: OnceLock<JLSparseDataset<N, 1024>>,


    /// MIPS variant of the JL sparse signature — distinct type
    /// ([`JLSparseDatasetMips`]) so the L2 path doesn't carry the
    /// per-vertex `‖v‖` slab. Default NZ=9 per the current MIPS
    /// sweep. Sidecar `.jls_mips`.
    pub(crate) q_dataset_jl_mips: OnceLock<JLSparseDatasetMips<N, 1024, 9>>,


    /// JL Hadamard 1024-bit signature dataset (HDHDHD-encoded). Lazy-
    /// built on first `ensure_quantized_dataset_jl_hadamard`. Sidecar
    /// `.jlh`.
    pub(crate) q_dataset_jl_hadamard: OnceLock<JlHadamardDataset<N, 1024>>,

    /// **L2 kernel-trick** sidecar — i8 base + per-vertex `‖x_i8‖²`
    /// (i32). Pairs with `IpI8Distance` (sdot) to reconstruct
    /// `‖q-x‖² = ‖q‖² + ‖x‖² - 2·⟨q,x⟩` per hop, giving MIPS-like
    /// kernel speed but L2-rank-exact (modulo the same per-dim
    /// quantization noise as the u8 sidecar). Sidecar `.qdsl2kt`.
    pub(crate) q_dataset_l2_kt: OnceLock<L2KTDataset<N>>,
}

impl<const N: usize> StagedDiskANN<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build a StagedDiskANN from pre-computed partitions.
    pub fn new(
        dataset: InmemDataset<f32, N>,
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        entry: u32,
        max_degree: u32,
        max_extra: usize,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
        cache_base_path: Option<PathBuf>,
        is_save: bool,
    ) -> Self {
        let num_nodes = partitions.len();

        let num_pq_chunks = pq.as_ref().map(|p| p.get_num_chunks());

        let cache_base_path = cache_base_path.unwrap_or_else(|| {
            let dir = PathBuf::from("staged_diskann_graphs");
            fs::create_dir_all(&dir).unwrap();
            dir.join(format!("staged_diskann_n{}.bin", num_nodes))
        });

        let t_graph = Instant::now();
        let graph = PhasedGraph::build_from_partitions(partitions, max_degree, max_extra);
        log::info!(
            "🏗️: PhasedGraph built in {:.3}s  mem={}",
            t_graph.elapsed().as_secs_f32(),
            diskann::utils::mem_usage(),
        );
        graph.print_stats();

        let result = StagedDiskANN {
            dataset,
            graph,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            cache_base_path,
            is_save,
            inmem_scratch_pool: OnceLock::new(),
            q_dataset: OnceLock::new(),
            q_dataset_l2_u16: OnceLock::new(),
            q_dataset_mips: OnceLock::new(),
            q_dataset_mips_i16: OnceLock::new(),
            q_dataset_rabitq: OnceLock::new(),
            q_dataset_rabitq_b4: OnceLock::new(),
            q_dataset_jl: OnceLock::new(),
            q_dataset_jl_mips: OnceLock::new(),
            q_dataset_jl_hadamard: OnceLock::new(),
            q_dataset_l2_kt: OnceLock::new(),
        };

        if is_save {
            if let Err(e) = result.save(&result.cache_base_path.clone()) {
                log::warn!("Failed to save graph: {}", e);
            }
        }

        result
    }

    /// Build from a pre-built PhasedGraph (e.g. cloned from another instance).
    pub fn from_phased_graph(
        dataset: InmemDataset<f32, N>,
        graph: PhasedGraph,
        entry: u32,
        pq: Option<Arc<FixedChunkPQTable>>,
        pq_codes: Option<Vec<u8>>,
    ) -> Self {
        let num_nodes = graph.num_nodes();
        let num_pq_chunks = pq.as_ref().map(|p| p.get_num_chunks());
        StagedDiskANN {
            dataset,
            graph,
            entry,
            num_nodes,
            pq,
            pq_codes,
            num_pq_chunks,
            cache_base_path: PathBuf::from(""),
            is_save: false,
            inmem_scratch_pool: OnceLock::new(),
            q_dataset: OnceLock::new(),
            q_dataset_l2_u16: OnceLock::new(),
            q_dataset_mips: OnceLock::new(),
            q_dataset_mips_i16: OnceLock::new(),
            q_dataset_rabitq: OnceLock::new(),
            q_dataset_rabitq_b4: OnceLock::new(),
            q_dataset_jl: OnceLock::new(),
            q_dataset_jl_mips: OnceLock::new(),
            q_dataset_jl_hadamard: OnceLock::new(),
            q_dataset_l2_kt: OnceLock::new(),
        }
    }

    /// Lazily obtain the u8 quantized dataset (L2 prefilter path).
    /// Tries the sidecar `.qds` first (memcpy load); on miss, builds
    /// from the f32 dataset and writes back.
    pub fn ensure_quantized_dataset(&self) -> &QuantizedDataset<L2U8, N> {
        self.q_dataset
            .get_or_init(|| build_quant::<L2U8, N>(&self.dataset, &self.cache_base_path))
    }

    /// Lazily obtain the **u16** quantized dataset (precise
    /// PQ-admission tier of the L2 cascade). Sidecar `.qds6`. Goes
    /// through the same `build_quant` plumbing as the u8 path; the
    /// only differences are storage type + quantization scale (see
    /// the `L2U16` spec).
    pub fn ensure_quantized_dataset_l2_u16(&self) -> &QuantizedDataset<L2U16, N> {
        self.q_dataset_l2_u16
            .get_or_init(|| build_quant::<L2U16, N>(&self.dataset, &self.cache_base_path))
    }

    /// Lazily obtain the i8 MIPS quantized dataset. Sidecar `.qdm8`.
    pub fn ensure_quantized_dataset_mips(&self) -> &QuantizedDataset<MipsI8, N> {
        self.q_dataset_mips.get_or_init(|| {
            build_quant::<MipsI8, N>(&self.dataset, &self.cache_base_path)
        })
    }

    /// Lazily obtain the i16 MIPS quantized dataset. Sidecar `.qdm16`.
    pub fn ensure_quantized_dataset_mips_i16(&self) -> &QuantizedDataset<MipsI16, N> {
        self.q_dataset_mips_i16.get_or_init(|| {
            build_quant::<MipsI16, N>(&self.dataset, &self.cache_base_path)
        })
    }

    /// Lazily obtain the RaBitQ 1-bit quantized dataset. Sidecar
    /// `.qrbq`. On first access: try the sidecar (memcpy load); on
    /// miss build from the f32 dataset (rotation gen + encode) and
    /// write back. Build uses a fixed seed so every cache slot is
    /// reproducible across runs.
    pub fn ensure_quantized_dataset_rabitq(&self) -> &RabitQDataset<N> {
        self.q_dataset_rabitq.get_or_init(|| {
            build_rabitq::<N>(&self.dataset, &self.cache_base_path)
        })
    }

    /// Lazily obtain the **B=4** RaBitQ quantized dataset. Sidecar
    /// `.qrb4`. Distinct from the B=1 sidecar (`.qrbq`) so both can
    /// coexist on disk and be A/B'd at search time.
    pub fn ensure_quantized_dataset_rabitq_b4(&self) -> &RabitQ4Dataset<N> {
        self.q_dataset_rabitq_b4.get_or_init(|| {
            build_rabitq_b4::<N>(&self.dataset, &self.cache_base_path)
        })
    }

    /// Lazily obtain the JL Sparse 1024-bit signature dataset.
    /// Sidecar `.jls`. Built fresh on first access — encoding is
    /// `O(N · BITS · NZ)` random reads on the f32 base (~10s for
    /// GIST 1M D=960 on M2). Used as the cheapest tier of the L2
    /// search cascade; never the final ranker.
    pub fn ensure_quantized_dataset_jl(&self) -> &JLSparseDataset<N, 1024> {
        self.q_dataset_jl
            .get_or_init(|| build_jl_sparse::<N>(&self.dataset, &self.cache_base_path))
    }

    /// MIPS twin of [`ensure_quantized_dataset_jl`] — builds (or
    /// loads from the `.jls_mips` sidecar) the NZ=9
    /// [`JLSparseDatasetMips`] designed for raw-MIPS data, per PA's
    /// `Mips_JL_Sparse_Point_Normalized` recipe. The per-vertex
    /// `‖x‖` norms used by the MIPS prefilter live inside the
    /// returned dataset alongside the sign-bit codes.
    pub fn ensure_quantized_dataset_jl_mips(&self) -> &JLSparseDatasetMips<N, 1024, 9> {
        self.q_dataset_jl_mips
            .get_or_init(|| build_jl_sparse_mips::<N>(&self.dataset, &self.cache_base_path))
    }

    /// Lazily obtain the **JL Hadamard** 1024-bit signature dataset
    /// (HDHDHD-encoded). Sidecar `.jlh`. Same byte layout as the JL
    /// Sparse signature so the search-time hot path reuses the
    /// `JLHammingDistance` kernel; per-bit information content is
    /// significantly higher (each bit sees all D dims via Hadamard
    /// mixing, vs JL Sparse's NZ=9 dims).
    pub fn ensure_quantized_dataset_jl_hadamard(&self) -> &JlHadamardDataset<N, 1024> {
        self.q_dataset_jl_hadamard
            .get_or_init(|| build_jl_hadamard::<N>(&self.dataset, &self.cache_base_path))
    }

    /// Lazily obtain the **L2 kernel-trick** sidecar (i8 base +
    /// per-vert `‖x_i8‖²`). Sidecar `.qdsl2kt`. Tries the on-disk
    /// cache first (memcpy load); on miss, builds via parallel
    /// `L2KTDataset::build_from` and writes back.
    pub fn ensure_quantized_dataset_l2_kt(&self) -> &L2KTDataset<N> {
        self.q_dataset_l2_kt
            .get_or_init(|| build_l2_kt::<N>(&self.dataset, &self.cache_base_path))
    }

    /// Return per-node (degree, local_count) from the PhasedGraph.
    pub fn graph_degree_stats(&self) -> Vec<(usize, usize)> {
        let n = self.num_nodes;
        (0..n)
            .map(|i| (self.graph.degree(i), self.graph.local_count(i)))
            .collect()
    }

    // --- IO ---

    pub fn save<P: AsRef<Path>>(&self, path: P) -> anyhow::Result<()> {
        let dir = path.as_ref().parent().unwrap_or(Path::new("."));
        fs::create_dir_all(dir)?;

        let graph_path = path.as_ref().with_extension("pgraph");
        self.graph.save(&graph_path)?;

        let meta = StagedDiskANNMeta {
            version: 5,
            entry: self.entry,
        };
        let mut writer = BufWriter::new(File::create(path)?);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        bincode::serde::encode_into_std_write(&meta, &mut writer, config)?;
        writer.flush()?;
        Ok(())
    }

    /// Construct from cached PhasedGraph + metadata on disk.
    /// `path` is the metadata `.bin` file; PhasedGraph is read from `path.with_extension("pgraph")`.
    /// The returned instance has an empty `dataset` — caller must install it afterwards.
    pub fn load_from_cache<P: AsRef<Path>>(
        path: P,
        dataset: InmemDataset<f32, N>,
    ) -> anyhow::Result<Self> {
        let graph_path = path.as_ref().with_extension("pgraph");
        if !graph_path.exists() {
            anyhow::bail!("PhasedGraph file not found: {:?}", graph_path);
        }
        let graph = PhasedGraph::load(&graph_path)?;

        let file = File::open(&path)?;
        let mut reader = std::io::BufReader::new(file);
        let config = bincode::config::standard()
            .with_fixed_int_encoding()
            .with_little_endian();
        let meta: StagedDiskANNMeta = bincode::serde::decode_from_std_read(&mut reader, config)?;
        let mut idx = Self::from_phased_graph(dataset, graph, meta.entry, None, None);
        // Preserve the cache path so `ensure_quantized_dataset` can locate
        // the `.qds` sidecar next to it (load fast path).
        idx.cache_base_path = path.as_ref().to_path_buf();
        Ok(idx)
    }
}

/// Serializable metadata for StagedDiskANN (separate from PhasedGraph binary).
#[derive(serde::Serialize, serde::Deserialize)]
struct StagedDiskANNMeta {
    version: u32,
    entry: u32,
}

/// Shared "load sidecar or build from f32, then save" helper for
/// every `QuantSpec`. Per-spec behaviour (file extension, log label,
/// quantization scale, NEON kernel) goes through the trait.
fn build_quant<Q, const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> QuantizedDataset<Q, N>
where
    Q: QuantSpec,
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension(Q::FILE_EXT);
    if path.exists() {
        match QuantizedDataset::<Q, N>::load(&path) {
            Ok(q) => {
                log::info!("QuantizedDataset<{}> loaded from {:?}", Q::LABEL, path);
                return q;
            }
            Err(e) => log::warn!(
                "QuantizedDataset<{}> load failed ({e}), rebuilding",
                Q::LABEL
            ),
        }
    }
    let t = Instant::now();
    let q = QuantizedDataset::<Q, N>::from_f32_dataset(dataset);
    log::info!(
        "QuantizedDataset<{}> built in {:.2}s",
        Q::LABEL,
        t.elapsed().as_secs_f32()
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("QuantizedDataset<{}> save failed: {e}", Q::LABEL);
        } else {
            log::info!("QuantizedDataset<{}> saved to {:?}", Q::LABEL, path);
        }
    }
    q
}

/// RaBitQ-specific "load sidecar or build, then save" helper. Mirrors
/// [`build_quant`] but uses [`RabitQDataset`] (which carries its own
/// magic + rotation matrix and doesn't fit the [`QuantSpec`] trait).
/// Sidecar extension is `.qrbq`. Build seed is fixed at the magic
/// constant for reproducibility — same cache slot, same rotation,
/// same codes across runs.
fn build_rabitq<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> RabitQDataset<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("qrbq");
    if path.exists() {
        match RabitQDataset::<N>::load(&path) {
            Ok(q) => {
                log::info!("RabitQDataset loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("RabitQDataset load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = RabitQDataset::<N>::build_from(dataset, RABITQ_MAGIC as u64);
    log::info!(
        "RabitQDataset built in {:.2}s (N={}, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("RabitQDataset save failed: {e}");
        } else {
            log::info!("RabitQDataset saved to {:?}", path);
        }
    }
    q
}

/// Twin of [`build_rabitq`] for the B=4 extended variant. Sidecar
/// extension `.qrb4`. Seed pinned to `RABITQ_B4_MAGIC` (distinct from
/// the B=1 seed so the two codecs get different rotation matrices,
/// keeping their estimators independent).
fn build_rabitq_b4<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> RabitQ4Dataset<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("qrb4");
    if path.exists() {
        match RabitQ4Dataset::<N>::load(&path) {
            Ok(q) => {
                log::info!("RabitQ4Dataset loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("RabitQ4Dataset load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = RabitQ4Dataset::<N>::build_from(dataset, RABITQ_B4_MAGIC as u64);
    log::info!(
        "RabitQ4Dataset built in {:.2}s (N={}, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("RabitQ4Dataset save failed: {e}");
        } else {
            log::info!("RabitQ4Dataset saved to {:?}", path);
        }
    }
    q
}

/// Twin of [`build_rabitq`] for the L2 JL Sparse 1024-bit signature.
/// Sidecar extension `.jls`. Seed pinned to `JL_SPARSE_MAGIC` for
/// reproducibility — same cache slot, same random index table, same
/// signatures across runs.
fn build_jl_sparse<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> JLSparseDataset<N, 1024>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("jls");
    if path.exists() {
        match JLSparseDataset::<N, 1024>::load(&path) {
            Ok(q) => {
                log::info!("JLSparseDataset (L2, NZ=9) loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("JLSparseDataset (L2) load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = JLSparseDataset::<N, 1024>::build_from(dataset, JL_SPARSE_MAGIC as u64);
    log::info!(
        "JLSparseDataset (L2, NZ=9) built in {:.2}s (N={}, BITS=1024, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("JLSparseDataset save failed: {e}");
        } else {
            log::info!("JLSparseDataset (L2) saved to {:?}", path);
        }
    }
    q
}

/// MIPS twin of [`build_jl_sparse`]. Builds a
/// [`JLSparseDatasetMips`] (with per-vertex `‖v‖` sidecar) at
/// `NZ=9`. Sidecar extension `.jls_mips` so the L2 and MIPS caches
/// don't alias — and a wrong-mode load would fail at the magic
/// check anyway (`JL_SPARSE_MIPS_MAGIC`).
fn build_jl_sparse_mips<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> JLSparseDatasetMips<N, 1024, 9>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("jls_mips");
    if path.exists() {
        match JLSparseDatasetMips::<N, 1024, 9>::load(&path) {
            Ok(q) => {
                log::info!("JLSparseDatasetMips (NZ=9) loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("JLSparseDatasetMips load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = JLSparseDatasetMips::<N, 1024, 9>::build_from(dataset, JL_SPARSE_MAGIC as u64);
    log::info!(
        "JLSparseDatasetMips (NZ=9) built in {:.2}s (N={}, BITS=1024, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("JLSparseDatasetMips save failed: {e}");
        } else {
            log::info!("JLSparseDatasetMips saved to {:?}", path);
        }
    }
    q
}

fn build_jl_hadamard<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> JlHadamardDataset<N, 1024>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("jlh");
    if path.exists() {
        match JlHadamardDataset::<N, 1024>::load(&path) {
            Ok(q) => {
                log::info!("JlHadamardDataset loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("JlHadamardDataset load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = JlHadamardDataset::<N, 1024>::build_from(dataset, JL_HADAMARD_MAGIC as u64);
    log::info!(
        "JlHadamardDataset built in {:.2}s (N={}, D_PAD=1024, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("JlHadamardDataset save failed: {e}");
        } else {
            log::info!("JlHadamardDataset saved to {:?}", path);
        }
    }
    q
}

/// Build-or-load the **L2 kernel-trick** sidecar (i8 base +
/// per-vertex `‖x_i8‖²`) — the storage shape the search path uses
/// to swap the direct L2 kernel for an `sdot` IP kernel via the
/// identity `‖q-x‖² = ‖q‖² + ‖x‖² - 2·⟨q,x⟩`.
///
/// Pipeline:
/// 1. Probe the on-disk cache at `<cache_base>.qdsl2kt`. On hit, the
///    sidecar is `memcpy`-loaded into two aligned slabs (i8 base +
///    i32 norms) with magic / dim / stride verification — same
///    fast-path shape as every other quantized sidecar.
/// 2. On miss (or load-error), build fresh from the f32 base via
///    `L2KTDataset::build_from`. That routine derives the affine
///    quantization params, quantizes every vertex into i8, and
///    accumulates `‖x_i8‖²` in the **same parallel pass** so the
///    second sweep is free. Cost is roughly the L2-u8 build cost
///    plus one extra integer-multiply-accumulate per element.
/// 3. Write back to disk (best-effort — a save error just warns).
///
/// Cache-key parity: shares the same `cache_base_path` stem as
/// every other sidecar (`.qds`, `.qds6`, `.qdm8`, `.jls`, ...), so
/// rebuilding the graph also implicitly invalidates this slab via
/// the rest of the cache stem changing.
fn build_l2_kt<const N: usize>(
    dataset: &InmemDataset<f32, N>,
    cache_base: &Path,
) -> L2KTDataset<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    let path = cache_base.with_extension("qdsl2kt");
    if path.exists() {
        match L2KTDataset::<N>::load(&path) {
            Ok(q) => {
                log::info!("L2KTDataset loaded from {:?}", path);
                return q;
            }
            Err(e) => log::warn!("L2KTDataset load failed ({e}), rebuilding"),
        }
    }
    let t = Instant::now();
    let q = L2KTDataset::<N>::build_from(dataset);
    log::info!(
        "L2KTDataset built in {:.2}s (N={}, num={})",
        t.elapsed().as_secs_f32(),
        N,
        q.num_vertices,
    );
    if !cache_base.as_os_str().is_empty() {
        if let Err(e) = q.save(&path) {
            log::warn!("L2KTDataset save failed: {e}");
        } else {
            log::info!("L2KTDataset saved to {:?}", path);
        }
    }
    q
}
