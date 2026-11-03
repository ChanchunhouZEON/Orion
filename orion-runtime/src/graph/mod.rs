/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve graph provenance, transfer the base allocation, then load/build/import.
use crate::{Error, Result};
pub mod cache;
pub mod parlayann;

/// Fingerprint of the vendored builder and its fixed construction recipe.
pub use orion_ann::build::PARLAY_RECIPE;

use crate::graph::parlayann as parlayann_bridge;
use crate::{
    config::{GraphSource, ResolvedRunConfig},
    data::{BaseElement, LoadedBase},
    graph::cache::CachePlan,
};
use orion_ann::Orion;
use orion_ann_diskann::model::PaddedDataset;
use std::{path::Path, time::Instant};

/// Consumes the base allocation. Query and evaluation inputs are owned separately.
/// Check the export path/header before transferring ownership of the allocation.
/// The graph payload itself is validated later, during streaming import.
pub fn load_index(config: &ResolvedRunConfig, data: &mut LoadedBase) -> Result<Orion> {
    load_index_impl(config, data, |config, dataset| {
        Orion::<f32>::build(dataset, config.build_config())
    })
}

pub fn load_u8_index(config: &ResolvedRunConfig, data: &mut LoadedBase<u8>) -> Result<Orion<u8>> {
    load_index_impl(config, data, |config, dataset| {
        Orion::<u8>::build(dataset, config.build_config())
    })
}

fn load_index_impl<T: BaseElement>(
    config: &ResolvedRunConfig,
    data: &mut LoadedBase<T>,
    build: impl FnOnce(&ResolvedRunConfig, PaddedDataset<T>) -> Result<Orion<T>>,
) -> Result<Orion<T>> {
    if config.dimension != data.layout.logical_dim() {
        return Err("index dimension does not match input".into());
    }

    let num_points = data.num_points;
    let cache = CachePlan::resolve(config, num_points)?;
    let import_source = if !cache.is_hit && config.graph_source == GraphSource::Parlayann {
        CachePlan::parlay_input(config, num_points)?
    } else {
        None
    };

    let dataset = data.take_index_dataset()?;
    let index = if cache.is_hit {
        log::info!(
            "Loading cached PhasedGraph from {:?}",
            cache.metadata_path.with_extension("pgraph")
        );
        Orion::<T>::load_from_cache_with_provenance(&cache.metadata_path, dataset)
            .map_err(Error::from)?
    } else {
        let start = Instant::now();
        let mut index = match config.graph_source {
            GraphSource::Rust => build(config, dataset)?,
            GraphSource::Parlayann => {
                if let Some(staged_path) = import_source.as_deref() {
                    import_staged_graph(config, dataset, staged_path)?
                } else {
                    build(config, dataset)?
                }
            }
        };

        // Both constructors defer saving. Publish once and propagate IO failures
        // before recording provenance, so an incomplete cache is never advertised.
        index.cache_base_path = cache.metadata_path.clone();
        index.save(&cache.metadata_path).map_err(Error::from)?;
        index.is_save = true;
        log::info!(
            "Graph preparation+save took {:.1}s (n={num_points})",
            start.elapsed().as_secs_f64()
        );
        index
    };

    cache.write_manifest(config, num_points, import_source.as_deref())?;
    Ok(index)
}

fn import_staged_graph<T: BaseElement>(
    config: &ResolvedRunConfig,
    dataset: PaddedDataset<T>,
    staged_path: &Path,
) -> Result<Orion<T>> {
    log::info!(
        "Streaming ParlayANN .staged export from {}",
        staged_path.display()
    );
    let num_points = dataset.len();
    let graph = orion_ann::PhasedGraph::load_staged(staged_path, num_points, config.graph_degree)
        .map_err(|e| {
        let message = format!("ParlayANN import failed: {}", e.message());
        Error::from(e).with_message(message)
    })?;

    // Keep the historical sampled-medoid choice; storage refactoring must not
    // silently change the search entry point to the export's header hint.
    let entry = parlayann_bridge::sampled_medoid_rows(num_points, |id| dataset.row(id).unwrap());
    Ok(Orion::from_phased_graph(dataset, graph, entry, None, None))
}

#[cfg(all(test, feature = "cli"))]
mod large_input_tests {
    use super::*;

    use crate::data::LoadedDataset;
    use clap::Parser;
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
        let config = crate::config::cli::Args::try_parse_from([
            "orion",
            "sift10m",
            "--base",
            base.to_str().unwrap(),
            "--query",
            query.to_str().unwrap(),
            "--groundtruth",
            gt.to_str().unwrap(),
            "--staged-file",
            staged.to_str().unwrap(),
            "--cache-dir",
            directory.to_str().unwrap(),
            "--graph-degree",
            "2",
            "--k",
            "1",
            "--search-list-sizes",
            "4",
        ])
        .unwrap()
        .resolve()
        .unwrap();
        // F32-only diagnostic executables must reject this config before loading a base.
        assert!(LoadedDataset::<f32>::load(&config).is_err());
        for reload in [false, true] {
            let mut data = LoadedDataset::<u8>::load(&config).unwrap();
            let pointer = data.base.as_ref().unwrap().as_ptr();
            let index = load_u8_index(&config, &mut data).unwrap();
            assert!(data.base.is_none());
            assert_eq!(index.dataset.as_slice().as_ptr(), pointer);
            assert_eq!(index.dataset.as_slice().len(), 4 * 128);
            assert_eq!(&index.dataset.as_slice()[384..512], &[3; 128]);
            let result = index
                .search_l2(
                    &[0u8; 128],
                    orion_ann::DynamicSearchConfig {
                        k: 1,
                        search_list_size: 4,
                        window_size: 2,
                        epsilon: 1.,
                        early_exit_limit: 100,
                    },
                )
                .unwrap();
            assert_eq!(result, vec![0]);
            if !reload {
                std::fs::remove_file(&staged).unwrap();
            }
        }
        assert!(std::fs::read_dir(&directory).unwrap().all(|entry| {
            !entry
                .unwrap()
                .path()
                .extension()
                .is_some_and(|ext| ext == "qds")
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
        let config = crate::config::cli::Args::try_parse_from([
            "orion",
            "--base",
            base.to_str().unwrap(),
            "--query",
            query.to_str().unwrap(),
            "--groundtruth",
            gt.to_str().unwrap(),
            "--metric",
            "l2",
            "--graph-degree",
            "2",
            "--k",
            "1",
            "--search-list-sizes",
            "4",
            "--graph-source",
            "parlayann",
            "--staged-file",
            staged.to_str().unwrap(),
            "--cache-dir",
            dir.join("cache").to_str().unwrap(),
        ])
        .unwrap()
        .resolve()
        .unwrap();
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
        assert_eq!(ptr, index.dataset.as_slice().as_ptr());
        assert_eq!(index.entry, old.entry_point);
        // Even callers that bypass CLI resolution cannot run native admission
        // against an f32 index or silently allocate a quantized replacement.
        {
            let rejected = crate::search::cascade::build_admission(
                &index,
                crate::search::cascade::AdmissionChoice::NativeL2U8,
            );
            assert!(
                matches!(rejected, Err(error) if error.to_string().contains("requires a native u8 index"))
            );
        }
        for (i, (l, r, e)) in old.partitions.iter().enumerate() {
            assert_eq!(index.graph.local_neighbors(i), l);
            assert_eq!(index.graph.remote_neighbors(i), r);
            assert_eq!(index.graph.extra_candidates(i), e);
        }
        let saved = index.graph.buffer_bytes().to_vec();
        drop(index);
        let mut again = LoadedDataset::load(&config).unwrap();
        let ptr = again.base.as_ref().unwrap().as_ptr();
        let cached = load_index(&config, &mut again).unwrap();
        assert_eq!(cached.dataset.as_slice().as_ptr(), ptr);
        assert_eq!(cached.graph.buffer_bytes(), saved);
        assert_eq!(&cached.dataset.as_slice()[384..512], &[3.; 128]);
        drop(cached);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
