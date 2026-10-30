//! An owned index session for embedded applications. No CLI parsing or process-wide pool.
use crate::{
    IndexOptions,
    config::{GraphSource, GraphSourcePolicy, ResolvedRunConfig, VectorStorageKind},
    data::memory as resources,
    data::{
        BaseElement, LoadedBase, VectorFormat, inspect_vector_file, validate_and_normalize_vectors,
    },
    graph::cache::CachePlan,
    search::{F32CascadeBackend, NativeU8L2Backend, SearchBackend},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// Array data is copied directly into the final aligned base allocation. Python
/// callers create this while holding the GIL, then release it for graph work.
pub enum BaseInput {
    File(PathBuf),
    F32 {
        data: LoadedBase<f32>,
        dimension: usize,
    },
    U8 {
        data: LoadedBase<u8>,
        dimension: usize,
    },
}

fn copy_array<T: BaseElement>(values: &[T], dimension: usize) -> Result<LoadedBase<T>, String> {
    if dimension == 0 || values.is_empty() || values.len() % dimension != 0 {
        return Err("base must be a nonempty rectangular matrix".into());
    }
    let count = values.len() / dimension;
    if count >= u32::MAX as usize {
        return Err("base point count exceeds u32 IDs".into());
    }
    let layout = vector::PaddedLayout::for_kernel::<T::Kernel>(dimension)?;
    let base = diskann::model::PaddedDataset::from_packed(values, layout)?;
    Ok(LoadedBase {
        base: Some(base.into_allocation()),
        num_points: count,
        layout,
    })
}

impl BaseInput {
    pub fn from_f32(values: &[f32], dimension: usize) -> Result<Self, String> {
        Ok(Self::F32 {
            data: copy_array(values, dimension)?,
            dimension,
        })
    }

    pub fn from_u8(values: &[u8], dimension: usize) -> Result<Self, String> {
        Ok(Self::U8 {
            data: copy_array(values, dimension)?,
            dimension,
        })
    }

    fn describe(
        &self,
        options: &IndexOptions,
    ) -> Result<(PathBuf, VectorFormat, usize, usize, BaseIdentity), String> {
        match self {
            Self::File(path) => {
                let path = path
                    .canonicalize()
                    .map_err(|e| format!("base {}: {e}", path.display()))?;
                let format = match options.base_format {
                    Some(f) => f,
                    None => VectorFormat::infer(&path)?,
                };
                let header = inspect_vector_file(&path, format)?;
                let identity = BaseIdentity::File {
                    path: path.clone(),
                    stamp: FileStamp::read(&path)?,
                };
                Ok((path, format, header.dimension, header.count, identity))
            }
            Self::F32 { data, dimension } => {
                // Hash the owned, unnormalized allocation after the Python GIL
                // has been released. Hashes stay stable across host byte order.
                let base = data.base.as_ref().ok_or("missing array base")?;
                let mut hash = Sha256::new();
                if cfg!(target_endian = "little") && data.layout.stride_elements() == *dimension {
                    hash.update(bytemuck::cast_slice(base.as_slice()));
                } else {
                    for row in base
                        .chunks_exact(data.layout.stride_elements())
                        .take(data.num_points)
                    {
                        if cfg!(target_endian = "little") {
                            hash.update(bytemuck::cast_slice(&row[..*dimension]));
                        } else {
                            for value in &row[..*dimension] {
                                hash.update(value.to_le_bytes());
                            }
                        }
                    }
                }
                self.describe_array(
                    options,
                    data.num_points,
                    *dimension,
                    &format!("{:x}", hash.finalize()),
                    VectorFormat::Fbin,
                    VectorStorageKind::F32,
                )
            }
            Self::U8 { data, dimension } => {
                let base = data.base.as_ref().ok_or("missing array base")?;
                let mut hash = Sha256::new();
                for row in base
                    .chunks_exact(data.layout.stride_elements())
                    .take(data.num_points)
                {
                    hash.update(&row[..*dimension]);
                }
                self.describe_array(
                    options,
                    data.num_points,
                    *dimension,
                    &format!("{:x}", hash.finalize()),
                    VectorFormat::U8bin,
                    VectorStorageKind::U8,
                )
            }
        }
    }

    fn describe_array(
        &self,
        options: &IndexOptions,
        count: usize,
        dimension: usize,
        hash: &str,
        format: VectorFormat,
        storage: VectorStorageKind,
    ) -> Result<(PathBuf, VectorFormat, usize, usize, BaseIdentity), String> {
        if options.max_points.is_some_and(|n| n < count) {
            return Err("slice array input before construction instead of using max_points".into());
        }
        if options.storage.is_some_and(|v| v != storage)
            || options.base_format.is_some_and(|v| v != format)
        {
            return Err(
                "array dtype must match storage; base_format applies only to file inputs".into(),
            );
        }
        // Arrays do not masquerade as readable files. The optional content ID
        // supplies cache identity; declared shape supplies memory accounting.
        Ok((
            PathBuf::from("<array>"),
            format,
            dimension,
            count,
            BaseIdentity::Array {
                sha256: hash.into(),
                count,
                dimension,
                storage,
            },
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct FileStamp {
    bytes: u64,
    modified_ns: u64,
}
impl FileStamp {
    fn read(path: &Path) -> Result<Self, String> {
        let meta = fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let modified_ns = meta
            .modified()
            .map_err(|e| e.to_string())?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_nanos()
            .try_into()
            .map_err(|_| "file modification time exceeds u64 nanoseconds")?;
        Ok(Self {
            bytes: meta.len(),
            modified_ns,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BaseIdentity {
    File {
        path: PathBuf,
        stamp: FileStamp,
    },
    Array {
        sha256: String,
        count: usize,
        dimension: usize,
        storage: VectorStorageKind,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibration {
    pub k: usize,
    pub search_list_size: usize,
    pub samples: usize,
    pub threshold: f32,
    pub early_exit_limit: usize,
}
impl Calibration {
    fn params(&self) -> orion::CalibratedParams {
        orion::CalibratedParams {
            threshold: self.threshold,
            early_exit_limit: self.early_exit_limit,
        }
    }
}

/// IDs preserve query order. Missing neighbors are padded with u32::MAX;
/// counts distinguishes padding from valid IDs without inventing distances.
pub struct SearchResult {
    pub ids: Vec<u32>,
    pub counts: Vec<u32>,
    pub k: usize,
}

trait Engine: Send + Sync {
    fn calibrate(
        &self,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
    ) -> Result<orion::CalibratedParams, String>;
    fn search(
        &self,
        queries: &[f32],
        config: &ResolvedRunConfig,
        beam: usize,
        params: orion::CalibratedParams,
    ) -> Result<Vec<Vec<u32>>, String>;
    fn save(&self, path: &Path) -> Result<(), String>;
}

struct DynamicEngine<B: SearchBackend> {
    index: orion::Orion<B::Element>,
}
impl<B: SearchBackend> Engine for DynamicEngine<B> {
    fn calibrate(
        &self,
        q: &[f32],
        c: &ResolvedRunConfig,
        beam: usize,
    ) -> Result<orion::CalibratedParams, String> {
        self.index
            .calibrate(
                q,
                beam,
                c.window_size,
                orion::CalibrationConfig {
                    k: c.sweep.k,
                    metric: c.metric.calibration_metric(),
                },
            )
            .map_err(|e| e.to_string())
    }
    fn search(
        &self,
        q: &[f32],
        c: &ResolvedRunConfig,
        beam: usize,
        params: orion::CalibratedParams,
    ) -> Result<Vec<Vec<u32>>, String> {
        B::search_batch(&self.index, q, c, beam, params).map_err(|e| e.to_string())
    }
    fn save(&self, path: &Path) -> Result<(), String> {
        self.index.save(path).map_err(|e| e.to_string())
    }
}

fn prepare<B: SearchBackend + 'static>(
    c: &ResolvedRunConfig,
    mut data: LoadedBase<B::Element>,
) -> Result<Box<dyn Engine>, String> {
    let index = B::load_index(c, &mut data)?;
    Ok(Box::new(DynamicEngine::<B> { index }))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Snapshot {
    version: u32,
    options: IndexOptions,
    base: BaseIdentity,
    calibrations: BTreeMap<usize, Calibration>,
    namespace: String,
    execution_signature: String,
    graph_stamp: FileStamp,
    metadata_stamp: FileStamp,
}

/// Each session owns a dedicated Rayon pool. Frontends serialize operations on
/// this object; worker parallelism remains internal to a single batch.
pub struct Index {
    engine: Box<dyn Engine>,
    pool: rayon::ThreadPool,
    config: ResolvedRunConfig,
    options: IndexOptions,
    base: BaseIdentity,
    calibrations: BTreeMap<usize, Calibration>,
    cache_manifest: Vec<u8>,
}

impl Index {
    pub fn new(input: BaseInput, options: IndexOptions) -> Result<Self, String> {
        Self::load(input, options, None)
    }

    /// Open a snapshot strictly: validate input identity and graph files before
    /// loading. Missing or stale files never trigger an implicit rebuild.
    pub fn open(
        path: &Path,
        input: BaseInput,
        threads: Option<usize>,
        memory_budget_gib: Option<f64>,
    ) -> Result<Self, String> {
        let snapshot: Snapshot = serde_json::from_slice(
            &fs::read(path.join("index.json"))
                .map_err(|e| format!("cache {}: {e}", path.display()))?,
        )
        .map_err(|e| e.to_string())?;
        if snapshot.version != 1 {
            return Err("unsupported Index cache version".into());
        }
        let mut options = snapshot.options.clone();
        options.cache_dir = path.canonicalize().map_err(|e| e.to_string())?;
        if let Some(n) = threads {
            options.threads = n;
        }
        if memory_budget_gib.is_some() {
            options.memory_budget_gib = memory_budget_gib;
        }
        Self::load(input, options, Some(snapshot))
    }

    fn load(
        input: BaseInput,
        options: IndexOptions,
        snapshot: Option<Snapshot>,
    ) -> Result<Self, String> {
        let (path, format, dimension, count, identity) = input.describe(&options)?;
        let content_id = match &identity {
            BaseIdentity::Array { sha256, .. } => Some(sha256.clone()),
            _ => None,
        };
        let config = options.resolve(path, format, dimension, count, content_id)?;
        let points = count.min(config.max_points);
        let mut calibrations = BTreeMap::new();
        if let Some(saved) = snapshot {
            if saved.base != identity
                || saved.namespace != config.cache_namespace
                || saved.execution_signature != execution_signature(&config)?
            {
                return Err("cache base identity or graph configuration mismatch".into());
            }
            let cache = CachePlan::resolve(&config, points)?;
            if !cache.is_hit {
                return Err("Index.open requires a complete existing graph cache".into());
            }
            if FileStamp::read(&cache.metadata_path)? != saved.metadata_stamp
                || FileStamp::read(&cache.metadata_path.with_extension("pgraph"))?
                    != saved.graph_stamp
            {
                return Err("cached graph changed since calibration was saved".into());
            }
            for (&k, calibration) in &saved.calibrations {
                validate_search(points, k, calibration.search_list_size)?;
                if k != calibration.k
                    || calibration.samples == 0
                    || !calibration.threshold.is_finite()
                    || !(0.0..=1.0).contains(&calibration.threshold)
                    || calibration.early_exit_limit == 0
                {
                    return Err("invalid saved calibration".into());
                }
            }
            calibrations = saved.calibrations;
        }
        resources::check_memory_budget(&config, points)?;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .map_err(|e| e.to_string())?;
        let engine = match input {
            BaseInput::File(_) if config.search_plan().storage().kind == VectorStorageKind::F32 => {
                let data = LoadedBase::<f32>::load(&config)?;
                pool.install(|| prepare::<F32CascadeBackend>(&config, data))?
            }
            BaseInput::File(_) => {
                let data = LoadedBase::<u8>::load(&config)?;
                pool.install(|| prepare::<NativeU8L2Backend>(&config, data))?
            }
            BaseInput::F32 { mut data, .. } => {
                data.num_points = points;
                for row in data
                    .base
                    .as_mut()
                    .ok_or("missing base")?
                    .chunks_exact_mut(data.layout.stride_elements())
                {
                    f32::normalize(&mut row[..dimension], dimension, config.metric)?;
                }
                pool.install(|| prepare::<F32CascadeBackend>(&config, data))?
            }
            BaseInput::U8 { mut data, .. } => {
                data.num_points = points;
                pool.install(|| prepare::<NativeU8L2Backend>(&config, data))?
            }
        };
        // Retain the provenance of the graph actually loaded. Saving later
        // must not fingerprint a staged export that may since have changed.
        let cache = CachePlan::resolve(&config, points)?;
        let cache_manifest = fs::read(cache.metadata_path.with_extension("cache.json"))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            engine,
            cache_manifest,
            pool,
            config,
            options,
            base: identity,
            calibrations,
        })
    }

    pub fn dimension(&self) -> usize {
        self.config.dimension
    }
    pub fn storage(&self) -> VectorStorageKind {
        self.config.search_plan().storage().kind
    }
    pub fn len(&self) -> usize {
        self.config.num_points.unwrap().min(self.config.max_points)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn queries(&self, queries: &mut [f32]) -> Result<(), String> {
        if queries.len() % self.dimension() != 0 {
            return Err("query dimension does not match index".into());
        }
        if self.storage() == VectorStorageKind::U8
            && queries
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > 255. || v.fract() != 0.)
        {
            return Err("native u8 queries must contain exact byte values".into());
        }
        validate_and_normalize_vectors(queries, self.dimension(), self.config.metric)
    }

    pub fn calibrate(
        &mut self,
        mut queries: Vec<f32>,
        k: usize,
        search_list_size: Option<usize>,
    ) -> Result<Calibration, String> {
        let beam = search_list_size.unwrap_or_else(|| orion::calibration_search_list_size!(48, k));
        validate_search(self.len(), k, beam)?;
        if queries.is_empty() {
            return Err("calibration requires at least one query".into());
        }
        self.queries(&mut queries)?;
        self.config.sweep.k = k;
        let params = self
            .pool
            .install(|| self.engine.calibrate(&queries, &self.config, beam))?;
        let calibration = Calibration {
            k,
            search_list_size: beam,
            samples: queries.len() / self.dimension(),
            threshold: params.threshold,
            early_exit_limit: params.early_exit_limit,
        };
        self.calibrations.insert(k, calibration.clone());
        Ok(calibration)
    }

    pub fn search(
        &mut self,
        mut queries: Vec<f32>,
        k: usize,
        beam: usize,
        adaptive: bool,
    ) -> Result<SearchResult, String> {
        validate_search(self.len(), k, beam)?;
        self.queries(&mut queries)?;
        let params = if adaptive {
            self.calibrations
                .get(&k)
                .ok_or_else(|| format!("calibrate this index for k={k} before adaptive search"))?
                .params()
        } else {
            // Admission rates are nonnegative, so a zero threshold disables
            // convergence; early exit cannot fire before convergence either.
            orion::CalibratedParams {
                threshold: 0.0,
                early_exit_limit: usize::MAX,
            }
        };
        self.config.sweep.k = k;
        let rows = self
            .pool
            .install(|| self.engine.search(&queries, &self.config, beam, params))?;
        let capacity = rows.len().checked_mul(k).ok_or("result size overflow")?;
        let mut ids = vec![u32::MAX; capacity];
        let mut counts = Vec::with_capacity(rows.len());
        for (out, row) in ids.chunks_exact_mut(k).zip(rows) {
            if row.len() > k || row.iter().any(|&id| id as usize >= self.len()) {
                return Err("engine returned invalid result IDs".into());
            }
            out[..row.len()].copy_from_slice(&row);
            counts.push(row.len() as u32);
        }
        Ok(SearchResult { ids, counts, k })
    }

    /// Write a self-describing snapshot into a new directory. Base vectors stay
    /// external. A staging directory is published only after all writes succeed.
    pub fn save(&self, destination: &Path) -> Result<(), String> {
        if destination.try_exists().map_err(|e| e.to_string())? {
            return Err("save destination already exists; choose a new directory".into());
        }
        if let BaseIdentity::File { path, stamp } = &self.base {
            if &FileStamp::read(path)? != stamp {
                return Err("base file changed since index was loaded".into());
            }
        }
        let parent = destination
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let staging = parent.join(format!(
            ".orion-save-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&staging).map_err(|e| e.to_string())?;
        let result = (|| {
            let mut config = self.config.clone();
            config.cache_dir = staging.clone();
            config.staged_file_explicit = false;
            let plan = CachePlan::resolve(&config, self.len())?;
            self.engine.save(&plan.metadata_path)?;
            fs::write(
                plan.metadata_path.with_extension("cache.json"),
                &self.cache_manifest,
            )
            .map_err(|e| e.to_string())?;
            let mut options = self.options.clone();
            // Persist the resolved graph source. Reopening is never automatic.
            options.graph.source = match config.graph_source {
                GraphSource::Rust => GraphSourcePolicy::Rust,
                GraphSource::Parlayann => GraphSourcePolicy::Parlayann,
            };
            options.graph.staged_file = None;
            options.storage = Some(self.storage());
            options.base_format = Some(config.base_format);
            let snapshot = Snapshot {
                version: 1,
                options,
                base: self.base.clone(),
                calibrations: self.calibrations.clone(),
                namespace: config.cache_namespace.clone(),
                execution_signature: execution_signature(&config)?,
                graph_stamp: FileStamp::read(&plan.metadata_path.with_extension("pgraph"))?,
                metadata_stamp: FileStamp::read(&plan.metadata_path)?,
            };
            let file = fs::File::create(staging.join("index.json")).map_err(|e| e.to_string())?;
            serde_json::to_writer_pretty(&file, &snapshot).map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            fs::rename(&staging, destination).map_err(|e| e.to_string())
        })();
        if result.is_err() {
            let _ = fs::remove_dir_all(&staging);
        }
        result
    }

    pub fn info(&self) -> serde_json::Value {
        serde_json::json!({"num_points": self.len(), "dimension": self.dimension(),
            "storage": self.storage(), "metric": self.config.metric,
            "graph_source": self.config.graph_source, "graph_selection_reason": self.config.graph_selection_reason,
            "cascade": self.config.search_plan().cascade(), "threads": self.options.threads,
            "cache_namespace": self.config.cache_namespace, "calibrations": self.calibrations})
    }
}

fn validate_search(points: usize, k: usize, beam: usize) -> Result<(), String> {
    if k == 0 || k > points || beam < k || beam > u32::MAX as usize / 4 {
        return Err("require 1 <= k <= num_points and k <= search_list_size <= u32::MAX/4".into());
    }
    Ok(())
}

fn execution_signature(config: &ResolvedRunConfig) -> Result<String, String> {
    let bytes = serde_json::to_vec(&(
        config.cache_namespace.as_str(),
        config.metric,
        config.search_plan(),
        config.window_size,
    ))
    .map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
