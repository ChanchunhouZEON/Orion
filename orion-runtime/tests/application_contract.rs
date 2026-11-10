/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Application-level scores and preparation, independent of benchmark features.
use orion_ann::{
    BaseInput, ErrorCode, Index, IndexOptions, QueryBatch, ScoreMetric, SearchMetric, SearchOptions,
};

fn data() -> Vec<f32> {
    (0..64 * 13)
        .map(|i| ((i * 31 + i / 13 * 7) % 251) as f32 + 1.)
        .collect()
}

#[test]
fn exact_scores_preserve_ids_for_all_metrics_and_native_bytes() {
    for (metric, native) in [
        (SearchMetric::L2, false),
        (SearchMetric::InnerProduct, false),
        (SearchMetric::Cosine, false),
        (SearchMetric::L2, true),
    ] {
        if native && !cfg!(feature = "parlay") {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let base = data();
        let bytes: Vec<u8> = base.iter().map(|&v| v as u8).collect();
        let input = if native {
            BaseInput::from_u8(&bytes, 13)
        } else {
            BaseInput::from_f32(&base, 13)
        }
        .unwrap();
        let mut index = Index::new(
            input,
            IndexOptions {
                metric,
                cache_dir: dir.path().join("cache"),
                threads: 2,
                auto_pin: false,
                ..Default::default()
            },
        )
        .unwrap();
        let queries = &base[..13 * 8];
        assert_eq!(
            index
                .search(
                    QueryBatch::from_f32(queries, 13).unwrap(),
                    &SearchOptions::new(3, 32)
                )
                .unwrap_err()
                .code(),
            ErrorCode::InvalidState
        );
        index
            .calibrate_many(queries.to_vec(), &[3, 5], Some(32))
            .unwrap();
        assert_eq!(index.calibrated_ks().collect::<Vec<_>>(), [3, 5]);
        let before = index.calibration(3).unwrap().clone();
        assert!(index
            .calibrate_many(queries.to_vec(), &[3, 0], Some(32))
            .is_err());
        assert!(index
            .calibrate_many(queries.to_vec(), &[3, 3], Some(32))
            .is_err());
        assert_eq!(index.calibration(3).unwrap(), &before);
        let plain = SearchOptions::new(3, 32);
        let ids = index
            .search(QueryBatch::from_f32(queries, 13).unwrap(), &plain)
            .unwrap();
        let scored = index
            .search(
                QueryBatch::from_f32(queries, 13).unwrap(),
                &plain.with_scores(true),
            )
            .unwrap();
        assert_eq!(ids.ids, scored.ids);
        assert_eq!(ids.counts, scored.counts);
        assert!(ids.scores.is_none());
        let scores = scored.scores.unwrap();
        assert_eq!(
            scores.metric,
            match metric {
                SearchMetric::L2 => ScoreMetric::SquaredL2,
                SearchMetric::InnerProduct => ScoreMetric::InnerProduct,
                SearchMetric::Cosine => ScoreMetric::CosineSimilarity,
            }
        );
        for (row, query) in queries.chunks(13).enumerate() {
            for column in 0..scored.counts[row] as usize {
                let slot = row * 3 + column;
                let start = scored.ids[slot] as usize * 13;
                let vector = &base[start..start + 13];
                let dot = query
                    .iter()
                    .zip(vector)
                    .map(|(&q, &x)| f64::from(q) * f64::from(x))
                    .sum::<f64>();
                let expected = match metric {
                    SearchMetric::L2 => query
                        .iter()
                        .zip(vector)
                        .map(|(&q, &x)| (f64::from(q) - f64::from(x)).powi(2))
                        .sum(),
                    SearchMetric::InnerProduct => dot,
                    SearchMetric::Cosine => {
                        dot / (query.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                            * vector.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>())
                        .sqrt()
                    }
                };
                assert!(
                    (f64::from(scores.values[slot]) - expected).abs()
                        <= expected.abs().max(1.) * 2e-6
                );
            }
        }
        if native {
            let result = index
                .search(
                    QueryBatch::from_u8(&bytes[..13 * 8], 13).unwrap(),
                    &plain.with_scores(true),
                )
                .unwrap();
            assert_eq!(result.ids, ids.ids);
            assert_eq!(result.scores.unwrap(), scores);
        }
        let empty = index
            .search(
                QueryBatch::from_f32(&[], 13).unwrap(),
                &plain.with_scores(true),
            )
            .unwrap();
        assert!(empty.is_empty());
        assert!(empty.scores.unwrap().values.is_empty());
    }
}

fn mapped_index(root: &std::path::Path, offset: u64, calibrated: bool) -> Index {
    let values = data();
    let mut ids: Vec<u64> = (0..64).map(|n| offset + n).collect();
    ids[63] = u64::MAX;
    let mut index = Index::new_with_ids(
        BaseInput::from_f32(&values, 13).unwrap(),
        IndexOptions {
            cache_dir: root.into(),
            threads: 2,
            auto_pin: false,
            ..Default::default()
        },
        ids,
    )
    .unwrap();
    if calibrated {
        index
            .calibrate(values[..13 * 8].to_vec(), 3, Some(32))
            .unwrap();
    }
    index
}

#[test]
fn external_identity_roundtrip_preserves_objects_and_rejects_corruption() {
    let root = tempfile::tempdir().unwrap();
    let mut index = mapped_index(&root.path().join("cache"), 1 << 40, true);
    let queries = data();
    let search = |i: &Index| {
        i.search(
            QueryBatch::from_f32(&queries[..13 * 8], 13).unwrap(),
            &SearchOptions::new(3, 32).with_scores(true),
        )
        .unwrap()
    };
    let first = search(&index);
    for (id, business) in first.ids.iter().zip(first.external_ids.as_ref().unwrap()) {
        assert_eq!(
            *business,
            if *id == 63 {
                u64::MAX
            } else {
                (1 << 40) + *id as u64
            }
        );
    }
    let saved = index.save(&root.path().join("snapshot")).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(saved.path().join("index.json")).unwrap()).unwrap();
    let store = std::path::PathBuf::from(manifest["object_store"].as_str().unwrap());
    let before: Vec<_> = std::fs::read_dir(&store)
        .unwrap()
        .map(|e| {
            let e = e.unwrap();
            (e.path(), e.metadata().unwrap().modified().unwrap())
        })
        .collect();
    index
        .calibrate(queries[..13 * 8].to_vec(), 5, Some(32))
        .unwrap();
    let second = index.save(&root.path().join("second")).unwrap();
    for (path, modified) in before {
        assert_eq!(
            std::fs::metadata(path).unwrap().modified().unwrap(),
            modified
        );
    }
    std::fs::remove_dir_all(root.path().join("cache")).unwrap();
    let reopened = Index::open(&second, BaseInput::from_f32(&queries, 13).unwrap()).unwrap();
    assert_eq!(reopened.calibrated_ks().collect::<Vec<_>>(), [3, 5]);
    assert_eq!(search(&reopened), first);
    let id_path = store.join(format!(
        "{}.ids-u64-le",
        manifest["payloads"]["external_ids"]["sha256"]
            .as_str()
            .unwrap()
    ));
    let mut bytes = std::fs::read(&id_path).unwrap();
    bytes[0] ^= 1;
    std::fs::write(id_path, bytes).unwrap();
    assert!(Index::open(saved, BaseInput::from_f32(&queries, 13).unwrap()).is_err());
}

#[test]
fn external_ids_validate_shape_uniqueness_and_budget() {
    let root = tempfile::tempdir().unwrap();
    for ids in [vec![1; 64], vec![1, 2]] {
        let result = Index::new_with_ids(
            BaseInput::from_f32(&data(), 13).unwrap(),
            IndexOptions {
                cache_dir: root.path().join("cache"),
                ..Default::default()
            },
            ids,
        );
        assert_eq!(result.unwrap_err().code(), ErrorCode::InvalidArgument);
    }
    let result = Index::new_with_ids(
        BaseInput::from_f32(&data(), 13).unwrap(),
        IndexOptions {
            memory_budget_gib: Some(1e-9),
            ..Default::default()
        },
        (0..64).collect(),
    );
    assert_eq!(result.unwrap_err().code(), ErrorCode::ResourceExhausted);
}

#[test]
fn portable_bundle_survives_move_and_rejects_version_or_content_changes() {
    use std::io::Write;
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("base.fbin");
    let values = data();
    let mut file = std::fs::File::create(&base).unwrap();
    file.write_all(&64u32.to_le_bytes()).unwrap();
    file.write_all(&13u32.to_le_bytes()).unwrap();
    for value in &values {
        file.write_all(&value.to_le_bytes()).unwrap();
    }
    drop(file);
    let mut index = Index::new_with_ids(
        BaseInput::from_file(&base),
        IndexOptions {
            cache_dir: root.path().join("cache"),
            threads: 2,
            auto_pin: false,
            ..Default::default()
        },
        (0..64).map(|i| i + 1000).collect(),
    )
    .unwrap();
    index
        .calibrate(values[..104].to_vec(), 3, Some(32))
        .unwrap();
    let search = |i: &Index| {
        i.search(
            QueryBatch::from_f32(&values[..104], 13).unwrap(),
            &SearchOptions::new(3, 32).with_scores(true),
        )
        .unwrap()
    };
    let expected = search(&index);
    let reference = index
        .export_bundle(&root.path().join("bundle"), true)
        .unwrap();
    drop(index);
    std::fs::remove_file(&base).unwrap();
    std::fs::remove_dir_all(root.path().join("cache")).unwrap();
    std::fs::remove_dir_all(root.path().join("cache.objects")).unwrap();
    std::fs::rename(root.path().join("bundle"), root.path().join("moved")).unwrap();
    let relocated = reference.relocated(root.path().join("moved/snapshot"));
    let restored = orion_ann::OpenOptions::default()
        .open_bundle(&relocated)
        .unwrap();
    assert_eq!(search(&restored), expected);
    let manifest = relocated.path().join("index.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).unwrap()).unwrap();
    value["format_version"] = 999.into();
    std::fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(orion_ann::OpenOptions::default()
        .open_bundle(relocated.path())
        .is_err());
    assert!(orion_ann::OpenOptions::default()
        .open_bundle(&relocated)
        .is_err());
}

#[test]
fn generation_switch_keeps_complete_old_and_new_identities() {
    let root = tempfile::tempdir().unwrap();
    let slot =
        orion_ann::IndexSlot::new(mapped_index(&root.path().join("a"), 1000, true), &[3]).unwrap();
    let old = slot.current().unwrap();
    let failed = mapped_index(&root.path().join("b"), 2000, false);
    assert!(slot.replace(failed).is_err());
    assert!(std::sync::Arc::ptr_eq(&old, &slot.current().unwrap()));
    let next = mapped_index(&root.path().join("c"), 3000, true);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            let values = data();
            for _ in 0..20 {
                let current = slot.current().unwrap();
                let result = current
                    .search(
                        QueryBatch::from_f32(&values[..104], 13).unwrap(),
                        &SearchOptions::new(3, 32),
                    )
                    .unwrap();
                let external = result.external_ids.unwrap();
                let offsets: std::collections::BTreeSet<_> = result
                    .ids
                    .iter()
                    .zip(external)
                    .filter(|(id, _)| **id != 63)
                    .map(|(id, ext)| ext - *id as u64)
                    .collect();
                assert_eq!(offsets.len(), 1);
                assert!(offsets.contains(&1000) || offsets.contains(&3000));
            }
        });
        let retired = slot.replace(next).unwrap();
        assert!(std::sync::Arc::ptr_eq(&old, &retired));
        worker.join().unwrap();
    });
    assert!(!std::sync::Arc::ptr_eq(&old, &slot.current().unwrap()));
    let values = data();
    assert!(old
        .search(
            QueryBatch::from_f32(&values[..104], 13).unwrap(),
            &SearchOptions::new(3, 32)
        )
        .is_ok());
}

#[test]
fn array_bundle_and_explicit_store_relocation_preserve_calibration() {
    let root = tempfile::tempdir().unwrap();
    let index = mapped_index(&root.path().join("cache"), 1234, true);
    assert!(index
        .export_bundle(&root.path().join("invalid"), true)
        .is_err());
    assert!(!root.path().join("invalid").exists());
    let snapshot = index
        .export_bundle(&root.path().join("array-bundle"), false)
        .unwrap();
    assert!(orion_ann::OpenOptions::default()
        .open_bundle(&snapshot)
        .is_err());
    std::fs::rename(
        root.path().join("array-bundle/objects"),
        root.path().join("relocated-store"),
    )
    .unwrap();
    let restored = orion_ann::OpenOptions {
        object_store: Some(root.path().join("relocated-store")),
        ..Default::default()
    }
    .open(&snapshot, BaseInput::from_f32(&data(), 13).unwrap())
    .unwrap();
    let values = data();
    let query = QueryBatch::from_f32(&values[..104], 13).unwrap();
    let options = SearchOptions::new(3, 32).with_scores(true);
    assert_eq!(
        index.search(query, &options).unwrap(),
        restored.search(query, &options).unwrap()
    );
    assert_eq!(restored.calibrated_ks().collect::<Vec<_>>(), [3]);
}

#[test]
fn approximate_selection_still_reports_exact_candidate_scores() {
    let root = tempfile::tempdir().unwrap();
    let base = data();
    for rerank in ["none", "u16"] {
        let mut index = Index::new(
            BaseInput::from_f32(&base, 13).unwrap(),
            IndexOptions {
                cascade: orion_ann::CascadeConfig {
                    prefilter: Some("none".into()),
                    admission: Some("l2-u8".into()),
                    rerank: Some(rerank.into()),
                },
                cache_dir: root.path().join(rerank),
                threads: 2,
                auto_pin: false,
                ..Default::default()
            },
        )
        .unwrap();
        index.calibrate(base[..104].to_vec(), 3, Some(32)).unwrap();
        let query = QueryBatch::from_f32(&base[..104], 13).unwrap();
        let plain = index.search(query, &SearchOptions::new(3, 32)).unwrap();
        let scored = index
            .search(query, &SearchOptions::new(3, 32).with_scores(true))
            .unwrap();
        assert_eq!(plain.ids, scored.ids);
        for (row, ids) in scored.rows().enumerate() {
            for (column, &id) in ids.iter().enumerate() {
                let expected: f32 = base[row * 13..(row + 1) * 13]
                    .iter()
                    .zip(&base[id as usize * 13..(id as usize + 1) * 13])
                    .map(|(q, x)| (q - x).powi(2))
                    .sum();
                assert_eq!(
                    scored.scores.as_ref().unwrap().values[row * 3 + column],
                    expected
                );
            }
        }
    }
}
