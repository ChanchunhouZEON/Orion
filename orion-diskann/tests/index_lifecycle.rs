/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Real small indices exercise build, persistence, deletion and search together.
mod support;
use orion_ann_diskann::{
    index::{ANNInmemIndex, InmemIndex},
    model::{IndexConfiguration, IndexWriteParametersBuilder},
};
use orion_ann_vector::Metric;
use support::TestDir;

fn index(points: usize) -> InmemIndex<f32, 32> {
    let params = IndexWriteParametersBuilder::new(16, 8)
        .with_num_threads(1)
        .build();
    InmemIndex::new(IndexConfiguration::new(
        Metric::L2,
        32,
        32,
        points,
        false,
        0,
        false,
        0,
        1.5,
        params,
    ))
    .unwrap()
}
fn data(points: usize) -> Vec<f32> {
    (0..points)
        .flat_map(|i| (0..32).map(move |d| ((i * 73 + d * 19 + i * d * 7) % 251) as f32))
        .collect()
}
fn write_vectors(path: &str, values: &[f32], dim: u32) {
    let mut bytes = ((values.len() / dim as usize) as u32)
        .to_le_bytes()
        .to_vec();
    bytes.extend(dim.to_le_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::write(path, bytes).unwrap();
}
fn search(index: &InmemIndex<f32, 32>, query: &[f32]) -> Vec<u32> {
    let mut ids = vec![u32::MAX; 4];
    ANNInmemIndex::search(index, query, 4, 64, &mut ids).unwrap();
    ids
}

#[test]
fn build_save_load_search_and_delete_roundtrip() {
    let dir = TestDir::new();
    let data = data(24);
    let base = dir.file("base.fbin");
    write_vectors(&base, &data, 32);
    let mut original = index(24);
    original.build(&base, 24).unwrap();
    let mut expected: Vec<u32> = (0..24).collect();
    expected.sort_by(|&a, &b| {
        let distance = |id: u32| {
            data[id as usize * 32..][..32]
                .iter()
                .zip(&data[32..64])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        };
        distance(a).total_cmp(&distance(b))
    });
    assert_eq!(search(&original, &data[32..64]), expected[..4]);
    original.soft_delete(vec![1, 17], 2).unwrap();
    expected.retain(|id| *id != 1 && *id != 17);
    assert_eq!(search(&original, &data[32..64]), expected[..4]);
    let cache = dir.file("index.bin");
    original.save(&cache).unwrap();
    assert_eq!(
        std::fs::metadata(format!("{cache}.data")).unwrap().len(),
        8 + 24 * 32 * 4
    );
    let mut restored = index(24);
    restored.load(&cache, 24).unwrap();
    assert_eq!(
        *restored.delete_set.read().unwrap(),
        *original.delete_set.read().unwrap()
    );
    assert_eq!(restored.extract_graph(), original.extract_graph());
    assert_eq!(restored.start_node(), original.start_node());
    assert_eq!(search(&restored, &data[32..64]), expected[..4]);
    // Repeated calls also cover scratch growth followed by reuse.
    assert_eq!(search(&restored, &data[32..64]), expected[..4]);
}

#[test]
fn memory_build_matches_file_build_and_mmap_export() {
    let dir = TestDir::new();
    let data = data(24);
    let mut original = index(24);
    original.build_from_data(&data, 24).unwrap();
    assert_eq!(original.num_active_points(), 24);
    #[cfg(feature = "orion")]
    original.sort_neighbors_by_distance_in_place().unwrap();
    let path = dir.file("graph.mmap");
    original.save_graph_mmap_with_vectors(&path).unwrap();
    let mapped = orion_ann_platform::MmapGraph::open(&path, 0).unwrap();
    assert_eq!(mapped.header.num_nodes, 24);
    assert_eq!(mapped.header.dimension, 32);
    assert_eq!(search(&original, &data[..32])[0], 0);
    #[cfg(feature = "benchmark")]
    {
        let mut sampled = [u32::MAX; 4];
        ANNInmemIndex::search_adsampling(&original, &data[..32], 4, 64, 0.0, &mut sampled).unwrap();
        assert_eq!(sampled.as_slice(), search(&original, &data[..32]));
    }
    original
        .save_graph_mmap(&dir.file("adjacency.mmap"))
        .unwrap();
    let extracted = original.extract_final_graph(24, 8);
    assert_eq!(extracted.size(), 24);
    assert_eq!(original.final_graph.size(), 0);
}

#[test]
fn build_rejects_invalid_metadata_and_short_inputs() {
    let dir = TestDir::new();
    let mut index = index(24);
    assert!(index.build(&dir.file("missing"), 24).is_err());
    let base = dir.file("base");
    write_vectors(&base, &data(24), 32);
    assert!(index.build(&base, 25).is_err());
    write_vectors(&base, &data(25), 32);
    assert!(index.build(&base, 24).is_err());
    write_vectors(&base, &[1.; 48], 2);
    assert!(index.build(&base, 24).is_err());
    assert!(index.build_from_data(&[0.; 31], 1).is_err());
    assert!(index.build_from_data(&[], 0).is_err());
    assert!(ANNInmemIndex::search(&index, &[0.; 31], 1, 2, &mut [0]).is_err());
    assert!(ANNInmemIndex::search(&index, &[0.; 32], 2, 1, &mut [0; 2]).is_err());
    #[cfg(feature = "benchmark")]
    assert!(ANNInmemIndex::search_adsampling(&index, &[0.; 32], 2, 1, 0., &mut [0; 2]).is_err());
}

#[test]
fn graph_header_and_delete_file_failures_are_reported() {
    let dir = TestDir::new();
    let path = dir.file("bad.graph");
    let mut index = index(3);
    for frozen in [1u64, 2] {
        let mut bytes = 24u64.to_le_bytes().to_vec();
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(frozen.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(index.load_graph(&path, 3).is_err());
    }
    let mut bytes = 28u64.to_le_bytes().to_vec();
    bytes.extend(8u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    std::fs::write(&path, bytes).unwrap();
    assert!(index.load_graph(&path, 3).is_err());
    assert_eq!(index.load_delete_list(&dir.file("absent")).unwrap(), 0);
    assert_eq!(index.save_delete_list(&dir.file("empty")).unwrap(), 0);
    std::fs::write(&path, [2, 0, 0, 0, 1]).unwrap();
    assert!(index.load_delete_list(&path).is_err());
}

#[test]
fn deleting_outside_active_range_is_rejected() {
    let mut index = index(24);
    index.build_from_data(&data(24), 24).unwrap();
    assert!(index.soft_delete(vec![24], 1).is_err());
    assert!(index.soft_delete(vec![25], 1).is_err());
    assert!(index.soft_delete(vec![1], 2).is_err());
    assert!(index.delete_set.read().unwrap().is_empty());
}

#[test]
fn insertion_connects_new_vertices_and_persists_them() {
    let dir = TestDir::new();
    let values = data(28);
    let path = dir.file("insert.fbin");
    write_vectors(&path, &values[24 * 32..], 32);
    let mut index = index(24);
    index.build_from_data(&values[..24 * 32], 24).unwrap();
    assert!(index.insert(&dir.file("missing"), 1).is_err());
    assert!(index.insert(&path, 5).is_err());
    index.insert(&path, 4).unwrap();
    assert_eq!(index.num_active_points(), 28);
    for id in 24..28 {
        assert_eq!(search(&index, &values[id * 32..][..32])[0], id as u32);
        assert_eq!(
            index
                .final_graph
                .read_vertex_and_neighbors(id as u32)
                .unwrap()
                .vertex_id,
            id as u32
        );
    }
    let path = dir.file("extended.graph");
    index.save(&path).unwrap();
    assert_eq!(
        std::fs::metadata(format!("{path}.data")).unwrap().len(),
        8 + 28 * 32 * 4
    );
}

#[cfg(feature = "orion")]
#[test]
fn candidate_files_roundtrip_and_reject_truncation() {
    // One candidate slab per process; this is the only test enabling it here.
    let dir = TestDir::new();
    let mut original = index(24);
    original
        .configuration
        .index_write_parameter
        .compute_candidate_sets = true;
    let config = original.configuration.clone();
    let mut original = InmemIndex::<f32, 32>::new(config).unwrap();
    original.build_from_data(&data(24), 24).unwrap();
    let expected = original.extract_candidate_sets().unwrap();
    let path = dir.file("candidates.bin");
    let size = original.save_candidate_sets(&path).unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    assert_eq!(
        InmemIndex::<f32, 32>::load_candidate_sets(&path).unwrap(),
        expected
    );
    assert!(InmemIndex::<f32, 32>::load_candidate_sets(&dir.file("missing")).is_err());
    let good = std::fs::read(&path).unwrap();
    for bytes in [&good[..7], &good[..good.len() - 1]] {
        std::fs::write(&path, bytes).unwrap();
        assert!(InmemIndex::<f32, 32>::load_candidate_sets(&path).is_err());
    }
    for declared in [16u64, size + 4] {
        let mut bytes = good.clone();
        bytes[..8].copy_from_slice(&declared.to_le_bytes());
        std::fs::write(&path, bytes).unwrap();
        assert!(InmemIndex::<f32, 32>::load_candidate_sets(&path).is_err());
    }
    original.drop_candidate_slab();
    let graph = original.extract_graph();
    let partitions = original.extract_graph_and_candidates(2).unwrap();
    for (id, (local, remote, extra)) in partitions.iter().enumerate() {
        let mut actual: Vec<_> = local.iter().chain(remote).copied().collect();
        actual.sort_unstable();
        let mut expected = graph[&(id as u32)].clone();
        expected.sort_unstable();
        assert_eq!(actual, expected);
        assert!(extra.is_empty());
    }
}

#[test]
fn factory_dispatch_preserves_all_supported_dimensions() {
    use orion_ann_diskann::index::create_inmem_index;
    for dimension in [32, 100, 104, 128, 256, 768, 784, 960, 1536] {
        let mut config = index(8).configuration;
        config.dim = dimension;
        config.aligned_dim = dimension;
        let values: Vec<f32> = (0..8).flat_map(|row| vec![row as f32; dimension]).collect();
        let mut index = create_inmem_index::<f32>(config).unwrap();
        index.build_from_data(&values, 8).unwrap();
        assert_eq!(index.num_active_points(), 8);
        let query = vec![3.1; dimension];
        let mut result = [u32::MAX; 2];
        index.search(&query, 2, 16, &mut result).unwrap();
        assert_eq!(result, [3, 4], "dimension={dimension}");
    }
    let mut config = index(8).configuration;
    config.aligned_dim = 33;
    assert!(create_inmem_index::<f32>(config).is_err());
}
