/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(feature = "parlay")]
use orion_ann::{
    BaseInput, GraphConfig, Index, IndexOptions, OpenOptions, QueryBatch, SearchMetric,
    SearchOptions, StorageMode, StorageOptions,
};

fn options(path: &std::path::Path, metric: SearchMetric, mode: StorageMode) -> IndexOptions {
    IndexOptions {
        metric,
        threads: 2,
        auto_pin: false,
        cache_dir: path.join("cache"),
        storage_options: StorageOptions {
            mode,
            directory: Some(path.to_owned()),
        },
        graph: GraphConfig {
            degree: 4,
            build_l: 8,
            max_extra: 2,
            ..Default::default()
        },
        ..Default::default()
    }
}
#[test]
fn native_metrics_snapshot_mmap_and_concurrent_search() {
    for metric in [SearchMetric::InnerProduct, SearchMetric::Cosine] {
        let d = tempfile::tempdir().unwrap();
        let base: Vec<_> = (0..32 * 33).map(|i| ((i * 31) % 255 + 1) as u8).collect();
        let input = || BaseInput::from_u8(&base, 33).unwrap();
        let queries = &base[..33 * 5];
        let floats: Vec<_> = queries.iter().map(|&v| v as f32).collect();
        let mut index =
            Index::new(input(), options(d.path(), metric, StorageMode::Resident)).unwrap();
        let expected = index.calibrate(floats.clone(), 3, Some(16)).unwrap();
        let search = SearchOptions::new(3, 16).with_scores(true);
        let q = QueryBatch::from_u8(queries, 33).unwrap();
        let result = index.search(q, &search).unwrap();
        assert_eq!(
            result,
            index
                .search(QueryBatch::from_f32(&floats, 33).unwrap(), &search)
                .unwrap()
        );
        let snapshot = index.save(&d.path().join("snapshot")).unwrap();
        let original = std::fs::read(snapshot.path().join("index.json")).unwrap();
        let manifest: serde_json::Value = serde_json::from_slice(&original).unwrap();
        let norm = manifest["payloads"]["sidecars"]["u8-norms"].clone();
        if metric == SearchMetric::Cosine {
            assert_eq!(norm[0], 24 + 32 * 4);
        } else {
            assert!(norm.is_null());
        }
        // Calibration-only save preserves the exact immutable norm object.
        index.calibrate(floats.clone(), 4, Some(16)).unwrap();
        let second = index.save(&d.path().join("snapshot2")).unwrap();
        let m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(second.path().join("index.json")).unwrap())
                .unwrap();
        assert_eq!(m["payloads"]["sidecars"]["u8-norms"], norm);
        let mapped = OpenOptions {
            storage_options: Some(StorageOptions {
                mode: StorageMode::Mapped,
                directory: Some(d.path().to_owned()),
            }),
            ..Default::default()
        }
        .open(&snapshot, input())
        .unwrap();
        assert_eq!(mapped.calibration(3).unwrap(), &expected);
        assert_eq!(
            mapped.info()["mapped_sidecar_bytes"],
            if metric == SearchMetric::Cosine {
                32 * 4
            } else {
                0
            }
        );
        assert_eq!(mapped.search(q, &search).unwrap(), result);
        if metric == SearchMetric::Cosine {
            let store = std::path::PathBuf::from(manifest["object_store"].as_str().unwrap());
            let object = store.join(format!("{}.u8-norms", norm[1].as_str().unwrap()));
            let mut corrupt = std::fs::read(&object).unwrap();
            corrupt[24] ^= 1;
            std::fs::write(&object, corrupt).unwrap();
            assert!(Index::open(&snapshot, input()).is_err());
            std::fs::remove_file(object).unwrap();
            let zero = [0u8; 33];
            assert!(
                mapped
                    .search(QueryBatch::from_u8(&zero, 33).unwrap(), &search)
                    .is_err()
            );
        }
        // Restored mappings own their backing independently of cache paths.
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let mapped = &mapped;
                let result = &result;
                scope.spawn(move || assert_eq!(&mapped.search(q, &search).unwrap(), result));
            }
        });
    }
}

#[test]
fn high_dimension_u64_norms_and_zero_rejection() {
    let dimension = 66052;
    let d = tempfile::tempdir().unwrap();
    let mut base = vec![255u8; dimension * 4];
    base[dimension..2 * dimension].fill(128);
    base[2 * dimension..3 * dimension].fill(1);
    base[3 * dimension..].fill(64);
    let mut index = Index::new(
        BaseInput::from_u8(&base, dimension).unwrap(),
        options(d.path(), SearchMetric::Cosine, StorageMode::Mapped),
    )
    .unwrap();
    assert_eq!(index.info()["mapped_sidecar_bytes"], 4 * 8);
    let query = vec![255.; dimension];
    index.calibrate(query.clone(), 2, Some(4)).unwrap();
    let result = index
        .search(
            QueryBatch::from_f32(&query, dimension).unwrap(),
            &SearchOptions::new(2, 4).with_scores(true),
        )
        .unwrap();
    for score in result.scores.unwrap().values {
        assert_eq!(score, 1.);
    }
    let mut bad = base.clone();
    bad[..dimension].fill(0);
    assert!(
        Index::new(
            BaseInput::from_u8(&bad, dimension).unwrap(),
            options(d.path(), SearchMetric::Cosine, StorageMode::Resident)
        )
        .is_err()
    );
}

#[test]
fn byte_cascades_restore_normalization_and_mapped_payloads() {
    use orion_ann::CascadeConfig;
    for (metric, admission, rerank) in [
        (SearchMetric::SquaredL2, "pq-l2", "f32"),
        (SearchMetric::InnerProduct, "pq-ip", "ip-f32"),
        (SearchMetric::Cosine, "pq-cosine", "cosine-f32"),
        (SearchMetric::SquaredL2, "l2-kt", "f32"),
        (SearchMetric::Cosine, "cosine-i8", "cosine-f32"),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let dimension = 25;
        let base: Vec<u8> = (0..48 * dimension)
            .map(|i| ((i * 31 + i / dimension * 7) % 251 + 1) as u8)
            .collect();
        let input = || BaseInput::from_u8(&base, dimension).unwrap();
        let mut settings = options(directory.path(), metric, StorageMode::Resident);
        settings.cascade = CascadeConfig {
            prefilter: Some("jl".into()),
            admission: Some(admission.into()),
            rerank: Some(rerank.into()),
        };
        let mut index = Index::new(input(), settings).unwrap();
        let q = &base[..3 * dimension];
        let floats: Vec<f32> = q.iter().map(|&x| f32::from(x)).collect();
        index.calibrate(floats.clone(), 3, Some(24)).unwrap();
        let search = SearchOptions::new(3, 24).with_scores(true);
        let query = QueryBatch::from_u8(q, dimension).unwrap();
        let expected = index.search(query, &search).unwrap();
        assert_eq!(
            expected,
            index
                .search(QueryBatch::from_f32(&floats, dimension).unwrap(), &search)
                .unwrap()
        );
        // Returned scores must be exact against original bytes, independent of
        // the normalized sidecar coordinates used for cosine admission.
        for (qi, ids) in expected.ids.chunks(3).enumerate() {
            for (j, &id) in ids.iter().enumerate() {
                let row = &base[id as usize * dimension..(id as usize + 1) * dimension];
                let q = &q[qi * dimension..(qi + 1) * dimension];
                let mut dot = 0f64;
                let mut q2 = 0f64;
                let mut x2 = 0f64;
                let mut l2 = 0f64;
                for (&a, &b) in q.iter().zip(row) {
                    let (a, b) = (f64::from(a), f64::from(b));
                    dot += a * b;
                    q2 += a * a;
                    x2 += b * b;
                    l2 += (a - b) * (a - b);
                }
                let exact = match metric {
                    SearchMetric::SquaredL2 => l2,
                    SearchMetric::InnerProduct => dot,
                    SearchMetric::Cosine => (dot / (q2.sqrt() * x2.sqrt())).clamp(0., 1.),
                } as f32;
                assert_eq!(expected.scores.as_ref().unwrap().values[qi * 3 + j], exact);
            }
        }
        let snapshot = index.save(&directory.path().join("snapshot")).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(snapshot.path().join("index.json")).unwrap())
                .unwrap();
        assert!(manifest["payloads"]["sidecars"].as_object().unwrap().len() >= 2);
        let mapped = OpenOptions {
            storage_options: Some(StorageOptions {
                mode: StorageMode::Mapped,
                directory: Some(directory.path().to_owned()),
            }),
            ..Default::default()
        }
        .open(&snapshot, input())
        .unwrap();
        assert!(mapped.info()["mapped_sidecar_bytes"].as_u64().unwrap() > 0);
        assert_eq!(mapped.search(query, &search).unwrap(), expected);
        // Private mappings must survive removal of their source objects. No
        // second fitting or unauthenticated reread is allowed during search.
        drop(index);
        let store = std::path::PathBuf::from(manifest["object_store"].as_str().unwrap());
        std::fs::remove_dir_all(store).unwrap();
        std::thread::scope(|scope| {
            for _ in 0..3 {
                let (mapped, expected) = (&mapped, &expected);
                scope.spawn(move || assert_eq!(&mapped.search(query, &search).unwrap(), expected));
            }
        });
    }
}
