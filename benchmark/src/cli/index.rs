//! Resolve graph provenance, transfer the base allocation, then load/build/import.
use super::{
    cache::CachePlan,
    config::{GraphSource, ResolvedRunConfig},
    data::LoadedDataset,
};
use crate::parlayann_bridge;
use diskann::model::InmemDataset;
use orion::{build_diskann_index, Orion};
use std::{path::Path, time::Instant};

/// Leaves query vectors and ground truth in `data`, but consumes its base allocation.
/// Check the export path/header before transferring ownership of the allocation.
/// The graph payload itself is validated later, during streaming import.
pub fn load_index<const N: usize>(
    config: &ResolvedRunConfig,
    data: &mut LoadedDataset,
) -> Result<Orion<N>, String>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    if config.dimension != N {
        return Err("index dimension does not match input".into());
    }

    let num_points = data.num_points;
    let cache = CachePlan::resolve(config, num_points)?;
    let import_source = if !cache.is_hit && config.graph_source == GraphSource::Parlayann {
        Some(CachePlan::resolve_staged_export(config, num_points)?)
    } else {
        None
    };

    let dataset = data.take_index_dataset::<N>()?;
    let index = if cache.is_hit {
        log::info!(
            "Loading cached PhasedGraph from {:?}",
            cache.metadata_path.with_extension("pgraph")
        );
        Orion::<N>::load_from_cache(&cache.metadata_path, dataset).map_err(|e| e.to_string())?
    } else {
        let start = Instant::now();
        let mut index = match config.graph_source {
            GraphSource::Rust => build_in_process(config, dataset)?,
            GraphSource::Parlayann => {
                let staged_path = import_source
                    .as_deref()
                    .ok_or("missing staged_file on cache miss")?;

                import_staged_graph(config, dataset, staged_path)?
            }
        };

        // Both constructors defer saving. Publish once and propagate IO failures
        // before recording provenance, so an incomplete cache is never advertised.
        index.cache_base_path = cache.metadata_path.clone();
        index
            .save(&cache.metadata_path)
            .map_err(|e| e.to_string())?;
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

fn build_in_process<const N: usize>(
    config: &ResolvedRunConfig,
    dataset: InmemDataset<f32, N>,
) -> Result<Orion<N>, String>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    log::info!(
        "Building Orion ({N}-dim, R={}, L={}, α={}, ex={})",
        config.graph_degree,
        config.build_l,
        config.alpha,
        config.max_extra
    );
    let num_points = dataset.num_points;
    let build = build_diskann_index(
        &dataset.data[..num_points * N],
        num_points,
        N,
        config.alpha,
        config.graph_degree,
        config.build_l as u32,
        false,
        None,
        None,
        true,
        config.max_extra,
    )
    .map_err(|e| e.to_string())?;

    // Release the builder's working index before allocating the final graph slab.
    let entry = build.entry_point;
    drop(build.index);

    // PQ sidecars are not supplied here; load_index saves the completed graph.
    Ok(Orion::new(
        dataset,
        &build.partitions,
        entry,
        config.graph_degree,
        config.max_extra,
        None,
        None,
        None,
        false,
    ))
}

fn import_staged_graph<const N: usize>(
    config: &ResolvedRunConfig,
    dataset: InmemDataset<f32, N>,
    staged_path: &Path,
) -> Result<Orion<N>, String>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    log::info!(
        "Streaming ParlayANN .staged export from {}",
        staged_path.display()
    );
    let num_points = dataset.num_points;
    let graph = orion::PhasedGraph::load_staged(staged_path, num_points, config.graph_degree)
        .map_err(|e| format!("ParlayANN import failed: {e}"))?;

    // Keep the historical sampled-medoid choice; storage refactoring must not
    // silently change the search entry point to the export's header hint.
    let entry = parlayann_bridge::sampled_medoid(&dataset.data[..num_points * N], N);
    Ok(Orion::from_phased_graph(dataset, graph, entry, None, None))
}

#[cfg(test)]
mod large_input_tests {
    use super::*;
    use clap::Parser;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

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
        let config = super::super::config::Args::try_parse_from([
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
        let old = crate::parlayann_bridge::load_from_staged_file(
            &staged,
            &data.base.as_ref().unwrap()[..512],
            128,
        )
        .unwrap();
        let index = load_index::<128>(&config, &mut data).unwrap();
        assert!(data.base.is_none());
        assert_eq!(ptr, index.dataset.data.as_ptr());
        assert_eq!(index.entry, old.entry_point);
        for (i, (l, r, e)) in old.partitions.iter().enumerate() {
            assert_eq!(index.graph.local_neighbors(i), l);
            assert_eq!(index.graph.remote_neighbors(i), r);
            assert_eq!(index.graph.extra_candidates(i), e);
        }
        let saved = index.graph.buffer_bytes().to_vec();
        drop(index);
        let mut again = LoadedDataset::load(&config).unwrap();
        let ptr = again.base.as_ref().unwrap().as_ptr();
        let cached = load_index::<128>(&config, &mut again).unwrap();
        assert_eq!(cached.dataset.data.as_ptr(), ptr);
        assert_eq!(cached.graph.buffer_bytes(), saved);
        assert_eq!(&cached.dataset.data[384..512], &[3.; 128]);
        drop(cached);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
