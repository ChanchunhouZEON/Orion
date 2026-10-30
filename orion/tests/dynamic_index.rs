//! Runtime dimensions must change storage, not graph traversal semantics.
use diskann::model::{InmemDataset, PaddedDataset};
use orion::algorithm::search::stage::{
    admission::NativeU8Admission, prefilter::NoPrefilter, rerank::NoRerank,
};
use orion::{DynamicSearchConfig, Orion, PhasedGraph, StaticOrion};
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
use vector::{L2F32Distance, L2U8Distance, PaddedLayout};

const ROWS: usize = 16;
fn graph(rows: usize) -> PhasedGraph {
    let partitions: Vec<_> = (0..rows)
        .map(|i| {
            (
                (0..rows as u32).filter(|&j| j != i as u32).collect(),
                vec![],
                vec![],
            )
        })
        .collect();
    PhasedGraph::build_from_partitions(&partitions, rows as u32, 0)
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
    Orion::from_phased_graph(
        PaddedDataset::from_packed(
            &values(dim),
            PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap(),
        )
        .unwrap(),
        graph(ROWS),
        0,
        None,
        None,
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
        let floats = Orion::from_phased_graph(data, graph(ROWS), 0, None, None);
        assert_eq!(
            pointer,
            floats.dataset.as_slice().as_ptr(),
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
            bytes.search_batch_l2(&queries, config()).unwrap(),
            expected,
            "u8 dim={dim}"
        );
        assert_eq!(
            floats.search_batch_l2(&float_queries, config()).unwrap(),
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
fn dynamic_and_typed_native_u8_share_search_behavior() {
    let packed = values(128);
    let mut data = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    data.data[..packed.len()].copy_from_slice(&packed);
    let typed = StaticOrion::from_phased_graph(data, graph(ROWS), 0, None, None);
    let dynamic = byte_index(128);
    for l in [5, 8, 16] {
        for epsilon in [0., 0.5, 1.] {
            for row in [0, 7, 15] {
                let q = &packed[row * 128..(row + 1) * 128];
                let fq = std::array::from_fn(|j| f32::from(q[j]));
                let cfg = DynamicSearchConfig {
                    search_list_size: l,
                    epsilon,
                    ..config()
                };
                let expected = typed
                    .search_unified(
                        &fq,
                        cfg.k,
                        l,
                        cfg.window_size,
                        epsilon,
                        cfg.early_exit_limit,
                        None::<&NoPrefilter>,
                        &NativeU8Admission::new(&typed.dataset),
                        &NoRerank,
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
    let floats = Orion::from_phased_graph(
        PaddedDataset::from_packed(
            &vec![0.0f32; ROWS * 3],
            PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
        )
        .unwrap(),
        graph(ROWS),
        0,
        None,
        None,
    );
    assert!(floats.search_l2(&[0., f32::NAN, 1.], config()).is_err());
    let mut wrong = byte_index(3);
    wrong.entry = ROWS as u32;
    assert!(wrong.search_l2(&[0, 1, 2], config()).is_err());
    wrong.entry = 0;
    wrong.graph = graph(ROWS - 1);
    assert!(wrong.search_l2(&[0, 1, 2], config()).is_err());
}

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!(
            "orion-dynamic-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn cache(&self) -> PathBuf {
        self.0.join("index.bin")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn cache_checks_logical_shape_but_allows_different_padding() {
    let tmp = Temp::new();
    let path = tmp.cache();
    byte_index(3).save(&path).unwrap();
    let repadded =
        PaddedDataset::from_packed(&values(3), PaddedLayout::new::<u8>(3, 128).unwrap()).unwrap();
    let loaded = Orion::load_from_cache(&path, repadded).unwrap();
    assert_eq!(
        loaded.search_l2(&[1, 2, 3], config()).unwrap(),
        byte_index(3).search_l2(&[1, 2, 3], config()).unwrap()
    );
    let wrong_dim = PaddedDataset::from_packed(
        &values(4),
        PaddedLayout::for_kernel::<L2U8Distance>(4).unwrap(),
    )
    .unwrap();
    assert!(Orion::load_from_cache(&path, wrong_dim).is_err());
    let wrong_type = PaddedDataset::from_packed(
        &vec![0.0f32; ROWS * 3],
        PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
    )
    .unwrap();
    assert!(Orion::load_from_cache(&path, wrong_type).is_err());
    let wrong_count = PaddedDataset::from_packed(
        &[0u8; 3],
        PaddedLayout::for_kernel::<L2U8Distance>(3).unwrap(),
    )
    .unwrap();
    assert!(Orion::load_from_cache(&path, wrong_count).is_err());
}

#[test]
fn legacy_cache_is_typed_only_and_corrupt_metadata_is_rejected() {
    let tmp = Temp::new();
    let path = tmp.cache();
    byte_index(128).save(&path).unwrap();
    // Version 5 stored only version and entry; no dimension/type provenance.
    std::fs::write(&path, [5u32.to_le_bytes(), 0u32.to_le_bytes()].concat()).unwrap();
    let typed = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    assert!(StaticOrion::load_from_cache(&path, typed).is_ok());
    assert!(Orion::load_from_cache(&path, byte_index(128).dataset).is_err());
    for bytes in [
        vec![],
        vec![6, 0, 0, 0],
        [99u32.to_le_bytes(), 0u32.to_le_bytes()].concat(),
    ] {
        std::fs::write(&path, bytes).unwrap();
        assert!(Orion::load_from_cache(&path, byte_index(128).dataset).is_err());
    }
}

#[test]
fn new_cache_is_shared_between_typed_and_dynamic_indexes() {
    let tmp = Temp::new();
    let path = tmp.cache();
    let data = InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap();
    StaticOrion::from_phased_graph(data, graph(ROWS), 0, None, None)
        .save(&path)
        .unwrap();
    let dynamic = Orion::load_from_cache(&path, byte_index(128).dataset).unwrap();
    dynamic.save(&path).unwrap();
    assert!(
        StaticOrion::load_from_cache(&path, InmemDataset::<u8, 128>::new(ROWS, 1.0).unwrap())
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
        assert!(index
            .search_batch_l2_byte_queries(&[], config())
            .unwrap()
            .is_empty());
        assert!(index
            .search_batch_l2_byte_queries(&floats[..dim - 1], config())
            .is_err());
        for invalid in [-1., 0.5, 256., f32::NAN, f32::INFINITY] {
            let mut query = vec![0.; dim];
            query[dim - 1] = invalid;
            assert!(index
                .search_batch_l2_byte_queries(&query, config())
                .is_err());
        }
    }
}
