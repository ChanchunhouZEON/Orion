/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(feature = "benchmark")]
use orion_ann_core::{
    BeamSearchConfig, FixedChunkPQTable, IoLimiter, async_beam_search,
    storage::{
        mmap_storage::{ClusterPageData, MmapStorage},
        sector_layout::SectorLayout,
    },
};
use std::sync::Arc;

#[test]
fn async_mmap_search_matches_exact_topk_before_and_after_convergence() {
    let path = std::env::temp_dir().join(format!("orion-async-search-{}.bin", std::process::id()));
    let rows: Vec<[f32; 32]> = (0..12).map(|i| [i as f32; 32]).collect();
    // An exact, one-chunk codebook removes quantization error from this test,
    // allowing traversal/cache behavior to be checked against exhaustive L2.
    let pq = Arc::new(FixedChunkPQTable::new(
        32,
        1,
        (0..256).flat_map(|i| [i as f32; 32]).collect(),
        vec![0.; 32],
        vec![0, 32],
    ));
    let codes: Vec<_> = (0..12).collect();
    let layout = SectorLayout::new(12, 32, 12, 12, 12);
    let builder = MmapStorage::new(layout.clone());
    // The entry discovers all candidates; leaf expansions admit no new IDs,
    // so the convergence detector must switch to the compressed phase.
    let edges: Vec<Vec<u32>> = (0..12)
        .map(|i| if i == 0 { (1..12).collect() } else { vec![] })
        .collect();
    let page = builder.build_cluster_page(&ClusterPageData {
        cluster_id: 0,
        records: edges
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, e)| (i as u32, e))
            .collect(),
    });
    let nodes: Vec<_> = rows
        .iter()
        .zip(&edges)
        .map(|(v, e)| (e.clone(), v.to_vec(), page.clone()))
        .collect();
    builder.write_to_file(&path, &nodes).unwrap();
    let storage = MmapStorage::open(&path, layout).unwrap();
    let limiter = IoLimiter::new(2);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    for epsilon in [0., 1.] {
        let config = BeamSearchConfig {
            search_list_size: 12,
            k: 3,
            window_size: 2,
            epsilon,
            beam_width: 2,
        };
        let query = [5.1; 32];
        let result = runtime.block_on(async_beam_search(
            &storage, &rows, &pq, &codes, 0, &query, &config, &limiter, 12, 12,
        ));
        assert_eq!(result.neighbors, [5, 6, 4]);
        assert!(result.phase1_iters > 0);
        if epsilon > 0. {
            assert!(result.phase2_iters > 0);
            assert!(result.cache_hits > 0);
            assert!(result.cache_misses > 0);
        }
        assert_eq!(limiter.semaphore().available_permits(), 2);
    }
    drop(storage);
    std::fs::remove_file(path).unwrap();
}
