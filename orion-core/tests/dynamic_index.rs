/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime dimensions must change storage, not graph traversal semantics.
#[path = "support/index_graph.rs"]
mod graph_fixture;
use graph_fixture::{GraphFixture, from_fixture_graph};
use orion_ann_core::algorithm::search::stage::{admission::NativeU8Admission, rerank::NoRerank};
use orion_ann_core::{DynamicCascade, DynamicSearchConfig, Orion, StaticOrion};
use orion_ann_diskann::model::{InmemDataset, PaddedDataset};
use orion_ann_vector::{L2F32Distance, L2U8Distance, PaddedLayout};
const ROWS: usize = 16;
fn graph(rows: usize) -> GraphFixture {
    let partitions: Vec<_> = (0..rows)
        .map(|i| {
            (
                (0..rows as u32).filter(|&j| j != i as u32).collect(),
                vec![],
                vec![],
            )
        })
        .collect();
    GraphFixture::build_from_partitions(&partitions, rows as u32, 0)
}
fn config() -> DynamicSearchConfig {
    DynamicSearchConfig {
        k: 5,
        search_list_size: ROWS,
        ..Default::default()
    }
}
fn values(dim: usize) -> Vec<u8> {
    (0..ROWS * dim)
        .map(|i| ((i / dim * 13 + i % dim * 7) % 251) as u8)
        .collect()
}
fn byte_index(dim: usize) -> Orion<u8> {
    from_fixture_graph(
        PaddedDataset::from_packed(
            &values(dim),
            PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap(),
        )
        .unwrap(),
        graph(ROWS),
        0,
    )
}
fn scalar_top(base: &[u8], query: &[u8]) -> Vec<u32> {
    let mut distances: Vec<_> = base
        .chunks_exact(query.len())
        .enumerate()
        .map(|(id, row)| {
            let distance: u64 = row
                .iter()
                .zip(query)
                .map(|(&a, &b)| {
                    let d = i64::from(a) - i64::from(b);
                    (d * d) as u64
                })
                .sum();
            (distance, id as u32)
        })
        .collect();
    distances.sort_unstable();
    distances
        .into_iter()
        .take(config().k)
        .map(|(_, id)| id)
        .collect()
}

#[test]
fn arbitrary_dimensions_match_scalar_and_batch_for_both_storage_types() {
    for dim in [1, 3, 31, 127, 128, 129, 769, 960] {
        let packed = values(dim);
        let bytes = byte_index(dim);
        let data = PaddedDataset::from_packed(
            &packed.iter().map(|&x| f32::from(x)).collect::<Vec<_>>(),
            PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
        )
        .unwrap();
        let pointer = data.as_slice().as_ptr();
        let floats = from_fixture_graph(data, graph(ROWS), 0);
        assert_eq!(
            pointer,
            floats.dataset().as_slice().as_ptr(),
            "index must move, not duplicate base"
        );
        let queries: Vec<u8> = [0, 7, 15]
            .into_iter()
            .flat_map(|i| packed[i * dim..(i + 1) * dim].iter().copied())
            .collect();
        let float_queries: Vec<f32> = queries.iter().map(|&x| f32::from(x)).collect();
        let expected: Vec<_> = queries
            .chunks_exact(dim)
            .map(|q| scalar_top(&packed, q))
            .collect();
        assert_eq!(
            bytes
                .search_batch_l2(&queries, config())
                .unwrap()
                .rows()
                .map(<[u32]>::to_vec)
                .collect::<Vec<_>>(),
            expected,
            "u8 dim={dim}"
        );
        assert_eq!(
            floats
                .search_batch_l2(&float_queries, config())
                .unwrap()
                .rows()
                .map(<[u32]>::to_vec)
                .collect::<Vec<_>>(),
            expected,
            "f32 dim={dim}"
        );
        for (i, q) in queries.chunks_exact(dim).enumerate() {
            assert_eq!(bytes.search_l2(q, config()).unwrap(), expected[i]);
            assert_eq!(
                floats
                    .search_l2(&float_queries[i * dim..(i + 1) * dim], config())
                    .unwrap(),
                expected[i]
            );
        }
    }
}

#[test]
fn dynamic_native_l2_matches_byte_admission_cascade() {
    let packed = values(128);
    let mut data = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    data.data[..packed.len()].copy_from_slice(&packed);
    // Retain the static byte admission as an independent kernel reference.
    // Its recipe can run on a graph with matching f32 rows; admission owns the
    // byte coordinates and NoRerank does not consult the reference base.
    let mut reference_data = InmemDataset::<f32, 128>::new(ROWS, 1.0).unwrap();
    for (out, &value) in reference_data.data.iter_mut().zip(&packed) {
        *out = f32::from(value);
    }
    let typed = from_fixture_graph(reference_data, graph(ROWS), 0);
    let dynamic = byte_index(128);
    for l in [5, 8, 16] {
        for epsilon in [0., 0.5, 1.] {
            for row in [0, 7, 15] {
                let q = &packed[row * 128..(row + 1) * 128];
                let fq: [f32; 128] = std::array::from_fn(|j| f32::from(q[j]));
                let cfg = DynamicSearchConfig {
                    search_list_size: l,
                    epsilon,
                    ..config()
                };
                let expected = typed
                    .search_cascade(
                        &fq,
                        cfg,
                        DynamicCascade {
                            prefilter: None,
                            admission: &NativeU8Admission::new(&data),
                            rerank: &NoRerank,
                        },
                    )
                    .unwrap();
                assert_eq!(dynamic.search_l2(q, cfg).unwrap(), expected);
            }
        }
    }
}

#[test]
fn invalid_queries_controls_and_graph_shapes_are_rejected() {
    let index = byte_index(3);
    assert!(index.search_l2(&[0, 1], config()).is_err());
    assert!(index.search_batch_l2(&[0, 1, 2, 3], config()).is_err());
    assert!(index.search_batch_l2(&[], config()).unwrap().is_empty());
    for cfg in [
        DynamicSearchConfig { k: 0, ..config() },
        DynamicSearchConfig {
            k: ROWS + 1,
            ..config()
        },
        DynamicSearchConfig {
            search_list_size: 4,
            ..config()
        },
        DynamicSearchConfig {
            window_size: 0,
            ..config()
        },
        DynamicSearchConfig {
            window_size: 65,
            ..config()
        },
        DynamicSearchConfig {
            epsilon: f32::NAN,
            ..config()
        },
    ] {
        assert!(index.search_l2(&[0, 1, 2], cfg).is_err());
    }
    let floats = from_fixture_graph(
        PaddedDataset::from_packed(
            &vec![0.0f32; ROWS * 3],
            PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
        )
        .unwrap(),
        graph(ROWS),
        0,
    );
    assert!(floats.search_l2(&[0., f32::NAN, 1.], config()).is_err());
    let wrong = byte_index(3);
    let parts = vec![(vec![], vec![], vec![]); ROWS];
    assert!(Orion::from_graph_partitions(wrong.into_dataset(), &parts, ROWS as u32, 1, 0).is_err());
    let parts = &parts[..ROWS - 1];
    assert!(Orion::from_graph_partitions(byte_index(3).into_dataset(), parts, 0, 1, 0).is_err());
}

fn encoded<D: orion_ann_core::index::compressed_index::IndexData, S: Default>(
    index: &orion_ann_core::CompressedIndex<D, S>,
) -> (Vec<u8>, Vec<u8>) {
    let mut metadata = Vec::new();
    let mut graph = std::io::BufWriter::new(Vec::new());
    index
        .encode_native_graph(&mut metadata, &mut graph)
        .unwrap();
    (metadata, graph.into_inner().unwrap())
}

#[test]
fn cache_checks_logical_shape_but_allows_different_padding() {
    let (meta, graph) = encoded(&byte_index(3));
    let repadded =
        PaddedDataset::from_packed(&values(3), PaddedLayout::new::<u8>(3, 128).unwrap()).unwrap();
    let loaded = Orion::decode_native_graph(
        repadded,
        meta.as_slice(),
        graph.as_slice(),
        graph.len() as u64,
    )
    .unwrap();
    assert_eq!(
        loaded.search_l2(&[1, 2, 3], config()).unwrap(),
        byte_index(3).search_l2(&[1, 2, 3], config()).unwrap()
    );
    let wrong_dim = PaddedDataset::from_packed(
        &values(4),
        PaddedLayout::for_kernel::<L2U8Distance>(4).unwrap(),
    )
    .unwrap();
    assert!(
        Orion::decode_native_graph(
            wrong_dim,
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_err()
    );
    let wrong_type = PaddedDataset::from_packed(
        &vec![0.0f32; ROWS * 3],
        PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
    )
    .unwrap();
    assert!(
        Orion::decode_native_graph(
            wrong_type,
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_err()
    );
    let wrong_count = PaddedDataset::from_packed(
        &[0u8; 3],
        PaddedLayout::for_kernel::<L2U8Distance>(3).unwrap(),
    )
    .unwrap();
    assert!(
        Orion::decode_native_graph(
            wrong_count,
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_err()
    );
}

#[test]
fn legacy_and_corrupt_graph_metadata_are_rejected() {
    let (_, graph) = encoded(&byte_index(128));
    // Version 5 stored only version and entry; no dimension/type provenance.
    let meta = [5u32.to_le_bytes(), 0u32.to_le_bytes()].concat();
    let typed = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    assert!(
        StaticOrion::decode_native_graph(
            typed,
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_err()
    );
    assert!(
        Orion::decode_native_graph(
            byte_index(128).into_dataset(),
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_err()
    );
    for bytes in [
        vec![],
        vec![6, 0, 0, 0],
        [99u32.to_le_bytes(), 0u32.to_le_bytes()].concat(),
    ] {
        let meta = bytes;
        assert!(
            Orion::decode_native_graph(
                byte_index(128).into_dataset(),
                meta.as_slice(),
                graph.as_slice(),
                graph.len() as u64
            )
            .is_err()
        );
    }
}

#[test]
fn new_cache_is_shared_between_typed_and_dynamic_indexes() {
    let data = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    let (meta, graph) = encoded(&from_fixture_graph(data, graph(ROWS), 0));
    let dynamic = Orion::decode_native_graph(
        byte_index(128).into_dataset(),
        meta.as_slice(),
        graph.as_slice(),
        graph.len() as u64,
    )
    .unwrap();
    let (meta, graph) = encoded(&dynamic);
    assert!(
        StaticOrion::decode_native_graph(
            InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap(),
            meta.as_slice(),
            graph.as_slice(),
            graph.len() as u64
        )
        .is_ok()
    );
}

#[test]
fn float_byte_queries_match_native_queries_and_reject_lossy_conversion() {
    for dim in [3, 31, 128, 129] {
        let index = byte_index(dim);
        let bytes = values(dim);
        let floats: Vec<_> = bytes.iter().map(|&v| f32::from(v)).collect();
        assert_eq!(
            index.search_batch_l2(&bytes, config()).unwrap(),
            index
                .search_batch_l2_byte_queries(&floats, config())
                .unwrap()
        );
        assert!(
            index
                .search_batch_l2_byte_queries(&[], config())
                .unwrap()
                .is_empty()
        );
        assert!(
            index
                .search_batch_l2_byte_queries(&floats[..dim - 1], config())
                .is_err()
        );
        for invalid in [-1., 0.5, 256., f32::NAN, f32::INFINITY] {
            let mut query = vec![0.; dim];
            query[dim - 1] = invalid;
            assert!(
                index
                    .search_batch_l2_byte_queries(&query, config())
                    .is_err()
            );
        }
    }
}
