//! Shared index loading for sweep and diagnostic executables.
use super::{
    cache::CachePlan,
    config::{GraphSource, ResolvedRunConfig},
    data::LoadedDataset,
};
use crate::parlayann_bridge;
use orion::{build_diskann_index, Orion};
use std::time::Instant;

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
    let n = data.num_points;
    let use_pa_graph = config.graph_source == GraphSource::Parlayann;
    let cache = CachePlan::select(config, n)?;
    let import_source = if !cache.hit && use_pa_graph {
        Some(CachePlan::source_for_import(config, n)?)
    } else {
        None
    };
    let cache_path = &cache.path;
    let pgraph_path = cache_path.with_extension("pgraph");

    let allocation = data
        .base
        .take()
        .ok_or("base allocation already transferred")?;
    let ds = diskann::model::InmemDataset::<f32, N> {
        data: allocation,
        num_points: n,
        num_active_pts: n,
        capacity: n * N,
    };
    let idx = if cache.hit {
        log::info!("Loading cached PhasedGraph from {:?}", pgraph_path);
        Orion::<N>::load_from_cache(&cache_path, ds).map_err(|e| e.to_string())?
    } else if !use_pa_graph {
        log::info!(
            "Building Orion ({}-dim, R={}, L={}, α={}, ex={}) — will save to {:?}",
            N,
            config.graph_degree,
            config.build_l,
            config.alpha,
            config.max_extra,
            pgraph_path
        );
        let t0 = Instant::now();
        let result = build_diskann_index(
            &ds.data[..n * N],
            n,
            N,
            config.alpha as f32,
            config.graph_degree as u32,
            config.build_l as u32,
            false,
            None,
            None,
            true,
            config.max_extra,
        )
        .map_err(|e| e.to_string())?;
        let entry = result.entry_point;
        drop(result.index);
        let mut idx = Orion::<N>::new(
            ds,
            &result.partitions,
            entry,
            config.graph_degree,
            config.max_extra,
            None,
            None,
            Some(cache_path.clone()),
            false,
        );
        idx.save(cache_path).map_err(|e| e.to_string())?;
        idx.is_save = true;
        log::info!("Build+save took {:.1}s", t0.elapsed().as_secs_f64());
        idx
    } else {
        // PA-graph cache is missing and build_diskann_index isn't our
        // path — instead import the `.staged` export PA wrote. Path
        // given by `ORION_STAGED_FILE`; we save to the same cache
        // slot on the way out so the second invocation hits the fast
        // load-from-cache arm above.
        let staged_file_path = import_source
            .as_ref()
            .ok_or("missing staged_file on cache miss")?;
        log::info!(
            "Importing ParlayANN .staged export from {} — will save \
             PhasedGraph cache to {:?}",
            staged_file_path.display(),
            pgraph_path
        );
        let t0 = Instant::now();
        let graph = orion::PhasedGraph::load_staged(staged_file_path, n, config.graph_degree)
            .map_err(|e| format!("ParlayANN import failed: {e}"))?;
        let entry = parlayann_bridge::sampled_medoid(&ds.data[..n * N], N);
        let mut idx = Orion::<N>::from_phased_graph(ds, graph, entry, None, None);
        idx.cache_base_path = cache_path.clone();
        idx.save(cache_path).map_err(|e| e.to_string())?;
        idx.is_save = true;
        log::info!(
            "Streaming import+save took {:.1}s (n={n})",
            t0.elapsed().as_secs_f64()
        );
        idx
    };
    cache.record(config, n, import_source.as_deref())?;

    Ok(idx)
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
