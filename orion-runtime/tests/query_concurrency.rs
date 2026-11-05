/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Public query ownership and shared-session contracts, independent of builders.
use orion_ann::{
    BaseInput, ErrorCode, GraphConfig, GraphSourcePolicy, Index, IndexOptions, QueryBatch,
    SearchMetric, SearchOptions,
};
use std::{io::Write, sync::Barrier};

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new(label: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "orion-query-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn staged(&self, points: usize) -> std::path::PathBuf {
        let path = self.0.join("graph.staged");
        let mut file = std::fs::File::create(&path).unwrap();
        for value in [0x53544147u32, 3, points as u32, 12, 4, 0] {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
        for node in 0..points {
            for value in [6u32, 6, 4] {
                file.write_all(&value.to_le_bytes()).unwrap();
            }
            for offset in 1..=16 {
                file.write_all(&(((node + offset) % points) as u32).to_le_bytes())
                    .unwrap();
            }
        }
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn borrowed_queries_and_mixed_requests_share_one_index() {
    const DIM: usize = 13;
    let bytes: Vec<u8> = (0..96 * DIM)
        .map(|i| ((i * 31 + i / DIM * 17) % 251) as u8)
        .collect();
    let values: Vec<f32> = bytes.iter().map(|&v| f32::from(v)).collect();
    let queries = &values[..DIM * 16];
    let original = queries.to_vec();
    let controls = [
        SearchOptions::new(2, 8),
        SearchOptions::new(5, 24),
        SearchOptions::new(7, 40),
    ];
    for (native, metric) in [
        (false, SearchMetric::L2),
        (false, SearchMetric::InnerProduct),
        (false, SearchMetric::Cosine),
        (true, SearchMetric::L2),
    ] {
        let fixture = Fixture::new("shared");
        let staged = fixture.staged(96);
        let input = if native {
            BaseInput::from_u8(&bytes, DIM)
        } else {
            BaseInput::from_f32(&values, DIM)
        }
        .unwrap();
        let mut index = Index::new(
            input,
            IndexOptions {
                metric,
                threads: 2,
                cache_dir: fixture.0.join("cache"),
                graph: GraphConfig {
                    source: GraphSourcePolicy::Parlayann,
                    staged_file: Some(staged.clone()),
                    degree: 12,
                    max_extra: 4,
                    build_l: 24,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        for options in &controls {
            index
                .calibrate(queries.to_vec(), options.k, Some(40))
                .unwrap();
        }
        let batch = QueryBatch::from_f32(queries, DIM).unwrap();
        let byte_batch = QueryBatch::from_u8(&bytes[..DIM * 16], DIM).unwrap();
        let start = Barrier::new(8);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|caller| {
                    let (index, start, controls) = (&index, &start, &controls);
                    scope.spawn(move || {
                        start.wait();
                        index
                            .search(
                                if caller % 2 == 0 { batch } else { byte_batch },
                                &controls[caller % controls.len()],
                            )
                            .unwrap()
                    })
                })
                .collect();
            for (caller, handle) in handles.into_iter().enumerate() {
                assert_eq!(
                    handle.join().unwrap(),
                    index
                        .search(batch, &controls[caller % controls.len()])
                        .unwrap()
                );
            }
        });
        let expected: Vec<_> = controls
            .iter()
            .map(|options| index.search(batch, options).unwrap())
            .collect();
        let before = index.info();
        let first = index.save(&fixture.0.join("before")).unwrap();
        // Native core and runtime share the actual result type and row semantics.
        if native {
            let base = orion_ann_core::PaddedDataset::from_packed(
                &bytes,
                orion_ann_core::PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(DIM)
                    .unwrap(),
            )
            .unwrap();
            let core = orion_ann_core::Orion::from_staged_graph(
                base,
                std::io::BufReader::new(std::fs::File::open(&staged).unwrap()),
                std::fs::metadata(&staged).unwrap().len(),
                12,
                0,
            )
            .unwrap();
            let calibration = index.calibrate(queries.to_vec(), 5, Some(40)).unwrap();
            let core_result: orion_ann::SearchResult = core
                .search_batch_l2(
                    &bytes[..DIM * 16],
                    orion_ann_core::DynamicSearchConfig {
                        k: 5,
                        search_list_size: 24,
                        window_size: 5,
                        epsilon: calibration.threshold,
                        early_exit_limit: calibration.early_exit_limit,
                    },
                )
                .unwrap();
            assert_eq!(core_result, expected[1]);
        }
        // More callers than workers, mixed k/beam, and two query representations.
        // An immutable reference makes accidental request mutation a compile error.
        let shared: &Index = &index;
        let start = Barrier::new(8);
        std::thread::scope(|scope| {
            for caller in 0..8 {
                let (start, expected, controls) = (&start, &expected, &controls);
                scope.spawn(move || {
                    start.wait();
                    for repeat in 0..12 {
                        let i = (caller + repeat) % controls.len();
                        let input = if repeat % 2 == 0 { batch } else { byte_batch };
                        assert_eq!(shared.search(input, &controls[i]).unwrap(), expected[i]);
                    }
                });
            }
        });
        assert_eq!(
            queries, original,
            "normalization must not modify borrowed input"
        );
        assert_eq!(index.info(), before);
        let after = index.save(&fixture.0.join("after")).unwrap();
        assert_eq!(
            std::fs::read(first.path().join("index.json")).unwrap(),
            std::fs::read(after.path().join("index.json")).unwrap()
        );
        assert!(index
            .search(QueryBatch::from_f32(&[], DIM).unwrap(), &controls[0])
            .unwrap()
            .is_empty());
        assert_eq!(
            index
                .search(
                    QueryBatch::from_u8(&[], DIM).unwrap(),
                    &SearchOptions::new(3, 8)
                )
                .unwrap_err()
                .code(),
            ErrorCode::InvalidState
        );
        assert_eq!(
            index
                .search(QueryBatch::from_f32(&[0.; 12], 12).unwrap(), &controls[0])
                .unwrap_err()
                .code(),
            ErrorCode::DimensionMismatch
        );
        for bad in [
            SearchOptions::new(0, 8),
            SearchOptions::new(5, 4),
            SearchOptions::new(2, usize::MAX),
        ] {
            assert_eq!(
                index.search(batch, &bad).unwrap_err().code(),
                ErrorCode::InvalidArgument
            );
        }
        for value in [f32::NAN, f32::INFINITY] {
            assert!(index
                .search(
                    QueryBatch::from_f32(&[value; DIM], DIM).unwrap(),
                    &controls[0]
                )
                .is_err());
        }
        if native {
            for value in [-1., 256., 1.5] {
                assert!(index
                    .search(
                        QueryBatch::from_f32(&[value; DIM], DIM).unwrap(),
                        &controls[0]
                    )
                    .is_err());
            }
        }
        if metric == SearchMetric::Cosine {
            assert!(index
                .search(QueryBatch::from_u8(&[0; DIM], DIM).unwrap(), &controls[0])
                .is_err());
        }
        assert_eq!(index.search(batch, &controls[0]).unwrap(), expected[0]);
    }
}

#[test]
fn query_views_validate_shape_without_reading_or_owning_coordinates() {
    assert!(QueryBatch::from_f32(&[], 0).is_err());
    assert!(QueryBatch::from_u8(&[1, 2, 3], 2).is_err());
    let values = [12345.; 6];
    let batch = QueryBatch::from_f32(&values, 3).unwrap();
    assert_eq!(batch.dimension(), 3);
    assert_eq!(batch.len(), 2);
    assert!(!format!("{batch:?}").contains("12345"));
    let _: QueryBatch<'_> = batch; // Copyable views do not consume caller storage.
    assert_eq!(values, [12345.; 6]);
}
