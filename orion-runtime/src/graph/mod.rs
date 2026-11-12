/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve graph provenance, transfer the base allocation, then load/build/import.
use crate::Result;
pub mod cache;
pub mod parlayann;
pub(crate) mod persistence;
mod selection;
pub(crate) mod staged;
#[cfg(feature = "benchmark")]
mod trusted;

/// Fingerprint of the vendored builder and its fixed construction recipe.
pub use orion_ann_core::build::PARLAY_RECIPE;

use crate::{
    config::{GraphSource, ResolvedRunConfig},
    data::{BaseElement, LoadedBase},
    graph::cache::CachePlan,
};
use orion_ann_core::Orion;
use orion_ann_diskann::model::PaddedDataset;
use std::time::Instant;

/// Graph storage and the persisted generation from which a session is assembled.
pub(crate) struct LoadedGraph<T: BaseElement> {
    pub index: Orion<T>,
    pub generation: cache::Generation,
    pub graph_bytes: u64,
    pub metadata_bytes: u64,
    pub store: std::path::PathBuf,
    pub working_key: std::path::PathBuf,
}

/// Consumes the base allocation. Query and evaluation inputs are owned separately.
/// Check the export path/header before transferring ownership of the allocation.
/// The graph payload itself is validated later, during streaming import.
#[cfg(any(feature = "benchmark", all(test, feature = "cli")))]
pub fn load_index(config: &ResolvedRunConfig, data: &mut LoadedBase) -> Result<Orion> {
    let cache = CachePlan::resolve(config, data.num_points)?;
    #[cfg(feature = "benchmark")]
    if config.benchmark_trust_cache && cache.is_hit() {
        return trusted::load(data.take_index_dataset()?, cache, config);
    }
    attach_working(prepare_f32(config, data, cache)?, config)
}

#[cfg(any(feature = "benchmark", all(test, feature = "cli")))]
pub fn load_u8_index(config: &ResolvedRunConfig, data: &mut LoadedBase<u8>) -> Result<Orion<u8>> {
    let cache = CachePlan::resolve(config, data.num_points)?;
    #[cfg(feature = "benchmark")]
    if config.benchmark_trust_cache && cache.is_hit() {
        return trusted::load(data.take_index_dataset()?, cache, config);
    }
    attach_working(prepare_u8(config, data, cache)?, config)
}

pub(crate) fn prepare_f32(
    config: &ResolvedRunConfig,
    data: &mut LoadedBase,
    cache: CachePlan,
) -> Result<LoadedGraph<f32>> {
    load_index_impl(config, data, cache, |config, dataset| {
        let bytes = crate::preflight::graph_bytes(config, dataset.len())?;
        let storage = crate::placement::graph_options(config, &dataset, bytes)?;
        Orion::<f32>::build_with_storage(dataset, config.build_config()?, &storage)
    })
}

pub(crate) fn prepare_u8(
    config: &ResolvedRunConfig,
    data: &mut LoadedBase<u8>,
    cache: CachePlan,
) -> Result<LoadedGraph<u8>> {
    load_index_impl(config, data, cache, |config, dataset| {
        let bytes = crate::preflight::graph_bytes(config, dataset.len())?;
        let storage = crate::placement::graph_options(config, &dataset, bytes)?;
        Orion::<u8>::build_with_storage(dataset, config.build_config()?, &storage)
    })
}

#[cfg(any(feature = "benchmark", all(test, feature = "cli")))]
fn attach_working<T: BaseElement>(
    mut graph: LoadedGraph<T>,
    _config: &ResolvedRunConfig,
) -> Result<Orion<T>> {
    #[cfg(feature = "benchmark")]
    if _config.benchmark_trust_cache {
        crate::sidecars::trusted::attach(&mut graph.index, &graph.working_key)?;
        return Ok(graph.index);
    }
    // Raw driver inputs may lack an authenticated content ID. Fingerprint the actual rows.
    crate::sidecars::track(
        &mut graph.index,
        graph.store,
        Some(graph.working_key),
        Default::default(),
        None,
    );
    Ok(graph.index)
}

fn load_index_impl<T: BaseElement>(
    config: &ResolvedRunConfig,
    data: &mut LoadedBase<T>,
    mut cache: CachePlan,
    build: impl FnOnce(&ResolvedRunConfig, PaddedDataset<T>) -> Result<Orion<T>>,
) -> Result<LoadedGraph<T>> {
    if config.dimension != data.layout.logical_dim() {
        return Err("index dimension does not match input".into());
    }

    let num_points = data.num_points;
    let store = cache.objects()?;
    let import_source = if !cache.is_hit() && config.graph_source == GraphSource::Parlayann {
        staged::parlay_input(config, num_points)?
    } else {
        None
    };

    let dataset = data.take_index_dataset()?;
    let (index, generation, graph_bytes, metadata_bytes) = if let Some(generation) =
        cache.selected.take()
    {
        let record = &generation.manifest;
        let path = crate::storage::object(&store, &record.graph_sha256, "pgraph")?;
        log::info!("Loading cached PhasedGraph from {}", path.display());
        let metadata =
            crate::storage::read_record(&store, &record.metadata_sha256, "bin", 4096, "metadata")?;
        let storage = crate::placement::graph_options(
            config,
            &dataset,
            std::fs::metadata(&path)?.len().saturating_sub(16),
        )?;
        let verified = persistence::load_verified_with_storage(
            &path,
            dataset,
            &metadata,
            &record.graph_sha256,
            &storage,
        )?;
        (
            verified.value,
            generation,
            verified.bytes,
            metadata.len() as u64,
        )
    } else {
        let start = Instant::now();
        let (index, source) = match import_source {
            Some(path) => {
                let (index, identity) = staged::import_staged_graph(config, dataset, &path)?;
                (index, Some(identity))
            }
            None => (build(config, dataset)?, None),
        };

        // Both constructors defer saving. Publish once and propagate IO failures
        // before recording provenance, so an incomplete cache is never advertised.
        let mut header = Vec::new();
        let (graph_bytes, graph) = crate::storage::put(&store, "pgraph", |w| {
            index.encode_native_graph(&mut header, w)
        })?;
        let (metadata_bytes, metadata) = crate::storage::put(&store, "bin", |w| {
            std::io::Write::write_all(w, &header)?;
            Ok(())
        })?;
        let generation = cache.write_manifest(config, num_points, source, &graph, &metadata)?;

        log::info!(
            "Graph preparation+save took {:.1}s (n={num_points})",
            start.elapsed().as_secs_f64()
        );
        (index, generation, graph_bytes, metadata_bytes)
    };
    Ok(LoadedGraph {
        index,
        generation,
        graph_bytes,
        metadata_bytes,
        store,
        working_key: cache.metadata_path,
    })
}

#[cfg(all(test, feature = "cli"))]
mod large_input_tests {
    use super::*;

    use crate::data::LoadedDataset;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn native_byte_import_and_reload_keep_one_buffer_and_no_sidecars() {
        let directory = std::env::temp_dir().join(format!(
            "orion-native-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let base = directory.join("base.bvecs");
        let query = directory.join("query.bvecs");
        let gt = directory.join("gt.ivecs");
        let staged = directory.join("graph.staged");
        let mut payload = Vec::new();
        for i in 0..4u8 {
            payload.extend(128u32.to_le_bytes());
            payload.extend([i; 128]);
        }
        std::fs::write(&base, &payload).unwrap();
        std::fs::write(&query, &payload[..132]).unwrap();
        std::fs::write(
            &gt,
            [1u32, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut file = std::fs::File::create(&staged).unwrap();
        for word in [0x53544147u32, 3, 4, 2, 1, 0] {
            file.write_all(&word.to_le_bytes()).unwrap();
        }
        for i in 0..4u32 {
            for word in [1, 1, 1, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4] {
                file.write_all(&word.to_le_bytes()).unwrap();
            }
        }
        drop(file);
        let mut config = crate::IndexOptions {
            graph: crate::GraphConfig {
                staged_file: Some(staged.clone()),
                degree: 2,
                ..Default::default()
            },
            cache_dir: directory.clone(),
            ..Default::default()
        }
        .resolve(base.clone(), crate::VectorFormat::Bvecs, 128, 4, None)
        .unwrap();
        config.query = Some(query);
        config.groundtruth = Some(gt);
        config.sweep.k = 1;
        config.sweep.search_list_sizes = vec![4];
        // F32-only diagnostic executables must reject this config before loading a base.
        assert!(LoadedDataset::<f32>::load(&config).is_err());
        for reload in [false, true] {
            let mut data = LoadedDataset::<u8>::load(&config).unwrap();
            let pointer = data.base.as_ref().unwrap().as_ptr();
            let index = load_u8_index(&config, &mut data).unwrap();
            assert!(data.base.is_none());
            assert_eq!(index.dataset().as_slice().as_ptr(), pointer);
            assert_eq!(index.dataset().as_slice().len(), 4 * 128);
            assert_eq!(&index.dataset().as_slice()[384..512], &[3; 128]);
            let result = index
                .search(
                    orion_ann_core::search::query::QueryInput::U8(&[0u8; 128]),
                    orion_ann_core::DynamicSearchConfig {
                        k: 1,
                        search_list_size: 4,
                        window_size: 2,
                        epsilon: 1.,
                        early_exit_limit: 100,
                    },
                    orion_ann_core::DynamicCascade {
                        prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                        admission: &index
                            .exact_stage(orion_ann_core::ScoreMetric::SquaredL2)
                            .unwrap(),
                        rerank: &orion_ann_core::search::cascade::stage::rerank::NoRerank,
                    },
                    None,
                )
                .map(|r| r.ids)
                .unwrap();
            assert_eq!(result, vec![0]);
            if !reload {
                std::fs::remove_file(&staged).unwrap();
            }
        }
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            entry
                .unwrap()
                .path()
                .extension()
                .is_none_or(|ext| ext != "qds")
        }));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn byte_input_streamed_graph_and_cache_transfer_the_same_allocation() {
        let dir = std::env::temp_dir().join(format!(
            "orion-large-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.bvecs");
        let query = dir.join("query.bvecs");
        let gt = dir.join("gt.ivecs");
        let staged = dir.join("graph.staged");
        let mut bytes = Vec::new();
        for i in 0..4u8 {
            bytes.extend(128u32.to_le_bytes());
            bytes.extend([i; 128]);
        }
        std::fs::write(&base, &bytes).unwrap();
        std::fs::write(&query, &bytes[..132]).unwrap();
        std::fs::write(
            &gt,
            [1u32, 0]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut f = std::fs::File::create(&staged).unwrap();
        for x in [0x53544147u32, 3, 4, 2, 1, 0] {
            f.write_all(&x.to_le_bytes()).unwrap();
        }
        for i in 0..4u32 {
            for x in [1, 1, 1, (i + 1) % 4, (i + 2) % 4, (i + 3) % 4] {
                f.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        drop(f);
        let mut config = crate::IndexOptions {
            storage: Some(crate::VectorStorageKind::F32),
            graph: crate::GraphConfig {
                staged_file: Some(staged.clone()),
                degree: 2,
                ..Default::default()
            },
            cache_dir: dir.join("cache"),
            ..Default::default()
        }
        .resolve(base.clone(), crate::VectorFormat::Bvecs, 128, 4, None)
        .unwrap();
        config.query = Some(query);
        config.groundtruth = Some(gt);
        config.sweep.k = 1;
        config.sweep.search_list_sizes = vec![4];
        let mut data = LoadedDataset::load(&config).unwrap();
        let ptr = data.base.as_ref().unwrap().as_ptr();
        let old = crate::graph::parlayann::load_from_staged_file(
            &staged,
            &data.base.as_ref().unwrap()[..512],
            128,
        )
        .unwrap();
        let index = load_index(&config, &mut data).unwrap();
        assert!(data.base.is_none());
        assert_eq!(ptr, index.dataset().as_slice().as_ptr());
        assert_eq!(index.entry_point(), old.entry_point);
        // Even callers that bypass CLI resolution cannot run native admission
        // against an f32 index or silently allocate a quantized replacement.
        {
            let rejected = crate::search::cascade::build_admission(
                &index,
                crate::search::cascade::AdmissionChoice::NativeL2U8,
            );
            assert!(
                matches!(rejected, Err(error) if error.to_string().contains("requires u8 base storage"))
            );
        }
        for (i, (l, r, e)) in old.partitions.iter().enumerate() {
            assert_eq!(index.graph().local_neighbors(i), l);
            assert_eq!(index.graph().remote_neighbors(i), r);
            assert_eq!(index.graph().extra_candidates(i), e);
        }
        let saved = index.graph().buffer_bytes().to_vec();
        drop(index);
        let mut again = LoadedDataset::load(&config).unwrap();
        let ptr = again.base.as_ref().unwrap().as_ptr();
        let selected = CachePlan::resolve(&config, again.num_points).unwrap();
        let key = selected.metadata_path.with_extension("cache.json");
        let original = std::fs::read(&key).unwrap();
        let provenance: serde_json::Value = serde_json::from_slice(&original).unwrap();
        assert_eq!(
            provenance["imported_from"]["sha256"],
            crate::integrity::file_sha256(&staged).unwrap()
        );
        // Simulate replacement between admission and decode. Only new sessions observe it.
        std::fs::write(&key, b"changed generation").unwrap();
        assert!(CachePlan::resolve(&config, again.num_points).is_err());
        let loaded = prepare_f32(&config, &mut again, selected).unwrap();
        assert_eq!(loaded.generation.encoded, original);
        let cached = loaded.index;
        assert_eq!(cached.dataset().as_slice().as_ptr(), ptr);
        assert_eq!(cached.graph().buffer_bytes(), saved);
        assert_eq!(&cached.dataset().as_slice()[384..512], &[3.; 128]);
        drop(cached);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
