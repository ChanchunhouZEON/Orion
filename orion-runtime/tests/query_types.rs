/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(feature = "parlay")]
use orion_ann::{
    BaseInput, GraphConfig, Index, IndexOptions, OpenOptions, QueryBatch, QueryBuffer,
    SearchMetric, SearchOptions, StorageMode, StorageOptions,
};

#[test]
fn typed_calibration_search_snapshot_and_concurrency_share_one_path() {
    let dim = 13;
    let bytes: Vec<u8> = (0..32 * dim).map(|i| ((i * 7) % 97 + 1) as u8).collect();
    let floats: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();
    let cases = [
        QueryBuffer::F32((0..2 * dim).map(|i| i as f32 - 6.5).collect()),
        QueryBuffer::U8(bytes[..2 * dim].to_vec()),
        QueryBuffer::I8((0..2 * dim).map(|i| i as i8 - 13).collect()),
        QueryBuffer::U16((0..2 * dim).map(|i| 1000 + i as u16).collect()),
        QueryBuffer::U32((0..2 * dim).map(|i| 16_777_217 + i as u32).collect()),
    ];
    for native in [false, true] {
        for metric in [
            SearchMetric::SquaredL2,
            SearchMetric::InnerProduct,
            SearchMetric::Cosine,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let base = || {
                if native {
                    BaseInput::from_u8(&bytes, dim).unwrap()
                } else {
                    BaseInput::from_f32(&floats, dim).unwrap()
                }
            };
            let mut index = Index::new(
                base(),
                IndexOptions {
                    metric,
                    threads: 2,
                    auto_pin: false,
                    cache_dir: dir.path().join("cache"),
                    graph: GraphConfig {
                        degree: 8,
                        build_l: 16,
                        max_extra: 4,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            let controls = SearchOptions::new(3, 24).with_scores(true);
            for input in &cases {
                let batch = QueryBatch::from_input(input.as_input(), dim).unwrap();
                index.calibrate_batch(batch, 3, Some(24)).unwrap();
                let expected = index.search(batch, &controls).unwrap();
                assert_eq!(expected.counts, [3, 3]);
                assert!(expected
                    .scores
                    .as_ref()
                    .unwrap()
                    .values
                    .iter()
                    .all(|s| s.is_finite()));
                std::thread::scope(|scope| {
                    let a = scope.spawn(|| index.search(batch, &controls).unwrap());
                    let b = scope.spawn(|| index.search(batch, &controls).unwrap());
                    assert_eq!(expected, a.join().unwrap());
                    assert_eq!(expected, b.join().unwrap());
                });
            }
            let snapshot = index.save(&dir.path().join("snapshot")).unwrap();
            let restored = OpenOptions {
                storage_options: Some(StorageOptions {
                    mode: StorageMode::Mapped,
                    directory: Some(dir.path().to_owned()),
                }),
                ..Default::default()
            }
            .open(snapshot, base())
            .unwrap();
            for input in &cases {
                let batch = QueryBatch::from_input(input.as_input(), dim).unwrap();
                assert_eq!(
                    index.search(batch, &controls).unwrap(),
                    restored.search(batch, &controls).unwrap()
                );
            }
            assert!(index
                .calibrate_batch(
                    QueryBatch::from_f32(&[f32::NAN; 13], 13).unwrap(),
                    3,
                    Some(24)
                )
                .is_err());
            if metric == SearchMetric::Cosine {
                assert!(index
                    .search(QueryBatch::from_u32(&[0; 13], 13).unwrap(), &controls)
                    .is_err());
            }
        }
    }
}

#[test]
fn generic_constructor_borrows_every_supported_dtype() {
    assert_eq!(QueryBatch::from_slice(&[1i8, 2, 3, 4], 2).unwrap().len(), 2);
    assert_eq!(
        QueryBatch::from_slice(&[1u16, 2], 2).unwrap().dimension(),
        2
    );
    assert!(QueryBatch::from_slice(&[u32::MAX], 2).is_err());
    assert!(QueryBatch::from_slice(&[] as &[u32], 0).is_err());
}

#[test]
fn typed_queries_share_quantized_preparation_and_exact_reranking() {
    use orion_ann::CascadeConfig;
    let dim = 25;
    let bytes: Vec<u8> = (0..48 * dim).map(|i| ((i * 31) % 251 + 1) as u8).collect();
    let floats: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();
    let signed: Vec<i8> = (0..2 * dim).map(|i| i as i8 - 25).collect();
    let words: Vec<u16> = (0..2 * dim).map(|i| 1000 + i as u16).collect();
    let cases = [
        QueryBuffer::I8(signed),
        QueryBuffer::U16(words),
        QueryBuffer::U32((0..2 * dim).map(|i| 10_000 + i as u32).collect()),
    ];
    for native in [false, true] {
        for (metric, admission, rerank) in [
            (SearchMetric::SquaredL2, "l2-u8", "f32"),
            (SearchMetric::SquaredL2, "pq-l2", "f32"),
            (SearchMetric::InnerProduct, "pq-ip", "ip-f32"),
            (SearchMetric::Cosine, "pq-cosine", "cosine-f32"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let base = if native {
                BaseInput::from_u8(&bytes, dim).unwrap()
            } else {
                BaseInput::from_f32(&floats, dim).unwrap()
            };
            let mut index = Index::new(
                base,
                IndexOptions {
                    metric,
                    threads: 2,
                    auto_pin: false,
                    cache_dir: dir.path().join("cache"),
                    graph: GraphConfig {
                        degree: 8,
                        build_l: 16,
                        max_extra: 4,
                        ..Default::default()
                    },
                    cascade: CascadeConfig {
                        prefilter: Some("jl".into()),
                        admission: Some(admission.into()),
                        rerank: Some(rerank.into()),
                    },
                    ..Default::default()
                },
            )
            .unwrap();
            for input in &cases {
                let q = QueryBatch::from_input(input.as_input(), dim).unwrap();
                index.calibrate_batch(q, 3, Some(24)).unwrap();
                let values = input.as_input().to_f32();
                let controls = SearchOptions::new(3, 24).with_scores(true);
                assert_eq!(
                    index.search(q, &controls).unwrap(),
                    index
                        .search(QueryBatch::from_f32(&values, dim).unwrap(), &controls)
                        .unwrap(),
                    "{native} {metric:?} {admission}"
                );
            }
        }
    }
}
