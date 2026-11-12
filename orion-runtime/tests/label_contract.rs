/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! The first independently mutable index state: labels, never graph edges.
use orion_ann::{
    BaseInput, ErrorCode, ExternalLabelUpdate, FilterSearchOptions, Index, IndexOptions,
    LabelFilter, LabelKey, LabelUpdate, QueryBatch, SearchMetric,
};
use std::sync::{Arc, Barrier};

fn key(value: &str) -> LabelKey {
    LabelKey::new("category", value).unwrap()
}
fn change(value: &str, add: &[u32], remove: &[u32]) -> LabelUpdate {
    LabelUpdate {
        label: key(value),
        add: add.to_vec(),
        remove: remove.to_vec(),
    }
}
fn filter(value: &str) -> LabelFilter {
    LabelFilter {
        all: vec![key(value)],
        ..Default::default()
    }
}
fn base() -> Vec<f32> {
    (0..48)
        .flat_map(|i| [i as f32 + 1., 1., (i % 7) as f32 + 1.])
        .collect()
}
fn index(dir: &std::path::Path, metric: SearchMetric, native: bool, mapped: bool) -> Index {
    let data = base();
    let bytes: Vec<_> = data.iter().map(|&v| v as u8).collect();
    let input = if native {
        BaseInput::from_u8(&bytes, 3)
    } else {
        BaseInput::from_f32(&data, 3)
    }
    .unwrap();
    let mut options = IndexOptions {
        cache_dir: dir.join("cache"),
        metric,
        auto_pin: false,
        threads: 2,
        ..Default::default()
    };
    if mapped {
        options.storage_options.mode = orion_ann::StorageMode::Mapped;
    }
    Index::new_with_ids(input, options, (0..48).map(|i| 1000 + i).collect()).unwrap()
}
fn search(index: &Index, label: &str) -> orion_ann::FilteredSearchResult {
    index
        .search_filtered(
            QueryBatch::from_f32(&[1., 1., 1., 1., 1., 1.], 3).unwrap(),
            &filter(label),
            &FilterSearchOptions::new(3).with_scores(true),
        )
        .unwrap()
}
#[test]
fn batches_are_atomic_idempotent_and_conflicts_or_invalid_inputs_roll_back() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path(), SearchMetric::L2, false, false);
    assert_eq!(index.label_revision().unwrap(), 0);
    assert_eq!(
        index
            .update_labels(0, &[change("book", &[0, 2, 2], &[])])
            .unwrap(),
        1
    );
    assert_eq!(
        index
            .update_labels(1, &[change("book", &[0], &[])])
            .unwrap(),
        1
    );
    assert_eq!(index.update_labels(1, &[]).unwrap(), 1);
    let before = search(&index, "book");
    assert_eq!(before.result.counts, [2, 2]);
    assert_eq!(
        &before.result.external_ids.as_ref().unwrap()[..2],
        &[1000, 1002]
    );
    for changes in [
        vec![change("book", &[1], &[]), change("movie", &[48], &[])],
        vec![change("book", &[1], &[1])],
        vec![change("book", &[1], &[]), change("book", &[2], &[])],
        vec![LabelUpdate {
            label: LabelKey {
                namespace: String::new(),
                value: "x".into(),
            },
            add: vec![1],
            remove: vec![],
        }],
    ] {
        assert_eq!(
            index.update_labels(1, &changes).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(index.label_revision().unwrap(), 1);
        assert_eq!(search(&index, "book").result.ids, before.result.ids);
    }
    assert_eq!(
        index.update_labels(0, &[]).unwrap_err().code(),
        ErrorCode::InvalidState
    );
    assert_eq!(
        index.resolve_external_ids(&[1047, 1000, 1047]).unwrap(),
        [47, 0, 47]
    );
    assert!(index.resolve_external_ids(&[999]).is_err());
    assert_eq!(
        index
            .update_labels(
                1,
                &[change("book", &[], &[0, 2]), change("movie", &[0, 2], &[])]
            )
            .unwrap(),
        2
    );
    assert_eq!(search(&index, "book").result.counts, [0, 0]);
    assert_eq!(search(&index, "movie").result.counts, [2, 2]);
}
#[test]
fn boolean_filters_empty_sets_and_budgets_have_explicit_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path(), SearchMetric::L2, false, false);
    index
        .update_labels(
            0,
            &[change("a", &[0, 1, 2], &[]), change("b", &[2, 3], &[])],
        )
        .unwrap();
    let q = QueryBatch::from_f32(&[1., 1., 1.], 3).unwrap();
    let opts = FilterSearchOptions::new(5);
    for (f, ids) in [
        (
            LabelFilter {
                all: vec![key("a"), key("b")],
                ..Default::default()
            },
            vec![2],
        ),
        (
            LabelFilter {
                any: vec![key("a"), key("b")],
                none: vec![key("b")],
                ..Default::default()
            },
            vec![0, 1],
        ),
        (filter("unknown"), vec![]),
        (
            LabelFilter {
                any: vec![key("unknown")],
                ..Default::default()
            },
            vec![],
        ),
        (LabelFilter::default(), vec![0, 1, 2, 3, 4]),
    ] {
        let r = index.search_filtered(q, &f, &opts).unwrap();
        assert_eq!(&r.result.ids[..r.result.counts[0] as usize], ids.as_slice());
        assert!(r.result.scores.is_none());
    }
    let mut limited = opts;
    limited.max_candidates = 2;
    assert!(index.search_filtered(q, &filter("a"), &limited).is_err());
    assert!(index
        .search_filtered(q, &LabelFilter::default(), &limited)
        .is_err());
    assert_eq!(
        index
            .search_filtered(q, &filter("unknown"), &limited)
            .unwrap()
            .result
            .counts,
        [0]
    );
    assert!(index
        .search_filtered(
            QueryBatch::from_f32(&[f32::NAN, 0., 0.], 3).unwrap(),
            &filter("unknown"),
            &opts
        )
        .is_err());
    assert!(index
        .search_filtered(
            QueryBatch::from_f32(&[1., 2.], 2).unwrap(),
            &filter("a"),
            &opts
        )
        .is_err());
    assert!(index
        .search_filtered(q, &filter("a"), &FilterSearchOptions::new(0))
        .is_err());
    assert!(index
        .search_filtered(QueryBatch::from_f32(&[], 3).unwrap(), &filter("a"), &opts)
        .unwrap()
        .result
        .is_empty());
}
#[test]
fn exhaustive_filter_scores_match_reference_for_all_metrics_and_storage() {
    for (metric, native, mapped) in [
        (SearchMetric::L2, false, false),
        (SearchMetric::InnerProduct, false, false),
        (SearchMetric::Cosine, false, true),
        (SearchMetric::L2, true, true),
    ] {
        if native && !cfg!(feature = "parlay") {
            continue;
        }
        let dir = tempfile::tempdir().unwrap();
        let index = index(dir.path(), metric, native, mapped);
        index
            .update_labels(0, &[change("a", &[2, 7, 11, 19], &[])])
            .unwrap();
        let q = [3., 2., 4.];
        let result = index
            .search_filtered(
                QueryBatch::from_f32(&q, 3).unwrap(),
                &filter("a"),
                &FilterSearchOptions::new(3).with_scores(true),
            )
            .unwrap();
        let data = base();
        let mut expected: Vec<_> = [2, 7, 11, 19]
            .into_iter()
            .map(|id| {
                let row = &data[id as usize * 3..id as usize * 3 + 3];
                let dot: f64 = q.iter().zip(row).map(|(&a, &b)| a as f64 * b as f64).sum();
                let score = match metric {
                    SearchMetric::L2 => q
                        .iter()
                        .zip(row)
                        .map(|(&a, &b)| (a as f64 - b as f64).powi(2))
                        .sum(),
                    SearchMetric::InnerProduct => dot,
                    SearchMetric::Cosine => {
                        dot / (q.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt()
                            * row.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt())
                    }
                };
                (id, score as f32)
            })
            .collect();
        expected.sort_by(|a, b| {
            let c = a.1.total_cmp(&b.1);
            if metric == SearchMetric::L2 {
                c
            } else {
                c.reverse()
            }
        });
        assert_eq!(
            result.result.ids,
            expected[..3].iter().map(|v| v.0).collect::<Vec<_>>()
        );
        for (&actual, (_, wanted)) in result.result.scores.unwrap().values.iter().zip(&expected) {
            assert!((actual - wanted).abs() < 1e-4 * wanted.abs().max(1.));
        }
        assert!(index.calibrated_ks().next().is_none());
    }
}
#[test]
fn concurrent_queries_hold_one_revision_and_writers_have_compare_and_publish() {
    let dir = tempfile::tempdir().unwrap();
    let index = Arc::new(index(dir.path(), SearchMetric::L2, false, false));
    index
        .update_labels(0, &[change("a", &[0, 1, 2], &[])])
        .unwrap();
    let barrier = Arc::new(Barrier::new(5));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let index = Arc::clone(&index);
            let barrier = Arc::clone(&barrier);
            scope.spawn(move || {
                barrier.wait();
                for _ in 0..100 {
                    let r = search(&index, "a");
                    let expected = if r.label_revision % 2 == 1 {
                        vec![0, 1, 2, 0, 1, 2]
                    } else {
                        vec![9, 10, 11, 9, 10, 11]
                    };
                    assert_eq!(r.result.ids, expected);
                }
            });
        }
        barrier.wait();
        for rev in 1..60 {
            let (add, remove) = if rev % 2 == 1 {
                (&[9, 10, 11][..], &[0, 1, 2][..])
            } else {
                (&[0, 1, 2][..], &[9, 10, 11][..])
            };
            assert_eq!(
                index
                    .update_labels(rev, &[change("a", add, remove)])
                    .unwrap(),
                rev + 1
            );
        }
    });
    let barrier = Barrier::new(2);
    let outcomes = std::thread::scope(|scope| {
        let handles: Vec<_> = ["b", "c"]
            .into_iter()
            .map(|label| {
                let barrier = &barrier;
                let index = &index;
                scope.spawn(move || {
                    barrier.wait();
                    index.update_labels(60, &[change(label, &[4], &[])])
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap().is_ok())
            .collect::<Vec<_>>()
    });
    assert_eq!(outcomes.iter().filter(|&&ok| ok).count(), 1);
}
#[test]
fn snapshots_bundles_failure_and_corruption_preserve_label_contract() {
    let dir = tempfile::tempdir().unwrap();
    let index = index(dir.path(), SearchMetric::L2, false, false);
    index
        .update_labels(0, &[change("a", &[2, 3, 9], &[])])
        .unwrap();
    let snapshot = index.save(&dir.path().join("saved")).unwrap();
    let before = search(&index, "a");
    index.update_labels(1, &[change("a", &[20], &[2])]).unwrap();
    assert!(index.save(snapshot.path()).is_err());
    assert_eq!(index.label_revision().unwrap(), 2);
    let restored = Index::open(&snapshot, BaseInput::from_f32(&base(), 3).unwrap()).unwrap();
    assert_eq!(search(&restored, "a"), before);
    let bundle = index
        .export_bundle(&dir.path().join("bundle"), false)
        .unwrap();
    let moved = dir.path().join("moved");
    std::fs::rename(dir.path().join("bundle"), &moved).unwrap();
    let bundled = Index::open(
        bundle.relocated(moved.join("snapshot")),
        BaseInput::from_f32(&base(), 3).unwrap(),
    )
    .unwrap();
    assert_eq!(search(&bundled, "a"), search(&index, "a"));
    let manifest_path = snapshot.path().join("index.json");
    let mut wire: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(wire["format_version"], 2);
    let store = std::path::PathBuf::from(wire["object_store"].as_str().unwrap());
    let hash = wire["payloads"]["labels"]["sha256"].as_str().unwrap();
    let object = store.join(format!("{hash}.labels.json"));
    let original = std::fs::read(&object).unwrap();
    std::fs::write(&object, b"{}").unwrap();
    assert!(Index::open(snapshot.path(), BaseInput::from_f32(&base(), 3).unwrap()).is_err());
    std::fs::write(object, original).unwrap();
    wire["payloads"]["labels"]["revision"] = 999.into();
    std::fs::write(&manifest_path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(Index::open(snapshot.path(), BaseInput::from_f32(&base(), 3).unwrap()).is_err());
    wire["payloads"]["labels"]["revision"] = 1.into();
    wire["format_version"] = 1.into();
    std::fs::write(&manifest_path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert!(Index::open(snapshot.path(), BaseInput::from_f32(&base(), 3).unwrap()).is_err());
    wire["payloads"].as_object_mut().unwrap().remove("labels");
    std::fs::write(&manifest_path, serde_json::to_vec(&wire).unwrap()).unwrap();
    assert_eq!(
        Index::open(snapshot.path(), BaseInput::from_f32(&base(), 3).unwrap())
            .unwrap()
            .label_revision()
            .unwrap(),
        0
    );
}

#[test]
fn labels_are_bound_to_business_mapping_and_failed_save_leaves_no_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let idx = index(dir.path(), SearchMetric::L2, false, false);
    idx.update_labels(0, &[change("a", &[2, 3, 9], &[])])
        .unwrap();
    let first = idx.save(&dir.path().join("first")).unwrap();
    let one: serde_json::Value =
        serde_json::from_slice(&std::fs::read(first.path().join("index.json")).unwrap()).unwrap();
    idx.update_labels(1, &[change("a", &[4], &[])]).unwrap();
    let second = idx.save(&dir.path().join("second")).unwrap();
    let two: serde_json::Value =
        serde_json::from_slice(&std::fs::read(second.path().join("index.json")).unwrap()).unwrap();
    for key in [
        "graph_sha256",
        "base_sha256",
        "sidecars",
        "metadata_sha256",
        "external_ids",
    ] {
        assert_eq!(one["payloads"][key], two["payloads"][key]);
    }
    assert_ne!(one["payloads"]["labels"], two["payloads"]["labels"]);
    let other = Index::new_with_ids(
        BaseInput::from_f32(&base(), 3).unwrap(),
        IndexOptions {
            cache_dir: dir.path().join("cache"),
            threads: 2,
            auto_pin: false,
            ..Default::default()
        },
        (0..48).map(|i| 9000 + i).collect(),
    )
    .unwrap();
    let wrong = other.save(&dir.path().join("wrong")).unwrap();
    let mut wire: serde_json::Value =
        serde_json::from_slice(&std::fs::read(wrong.path().join("index.json")).unwrap()).unwrap();
    wire["payloads"]["labels"] = one["payloads"]["labels"].clone();
    std::fs::write(
        wrong.path().join("index.json"),
        serde_json::to_vec(&wire).unwrap(),
    )
    .unwrap();
    assert!(Index::open(wrong.path(), BaseInput::from_f32(&base(), 3).unwrap()).is_err());
    // Force an object-store I/O failure after the resident index was prepared.
    let store = std::path::PathBuf::from(one["object_store"].as_str().unwrap());
    let hidden = dir.path().join("hidden-objects");
    std::fs::rename(&store, &hidden).unwrap();
    std::fs::write(&store, b"not a directory").unwrap();
    let failed = dir.path().join("failed");
    assert!(idx.save(&failed).is_err());
    assert!(!failed.exists());
    assert_eq!(idx.label_revision().unwrap(), 2);
    assert_eq!(search(&idx, "a").result.counts, [3, 3]);
    std::fs::remove_file(&store).unwrap();
    std::fs::rename(hidden, store).unwrap();
    idx.save(&failed).unwrap();
}

#[test]
fn multi_label_batches_and_concurrent_checkpoints_never_mix_memberships() {
    let dir = tempfile::tempdir().unwrap();
    let idx = index(dir.path(), SearchMetric::L2, false, false);
    idx.update_labels(
        0,
        &[change("a", &[0, 1, 2], &[]), change("b", &[9, 10, 11], &[])],
    )
    .unwrap();
    let start = Barrier::new(2);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            start.wait();
            for rev in 1..40 {
                let (a, b) = if rev % 2 == 1 {
                    (&[9, 10, 11][..], &[0, 1, 2][..])
                } else {
                    (&[0, 1, 2][..], &[9, 10, 11][..])
                };
                idx.update_labels(rev, &[change("a", a, b), change("b", b, a)])
                    .unwrap();
                std::thread::yield_now();
            }
        });
        start.wait();
        for i in 0..8 {
            let both = LabelFilter {
                all: vec![key("a"), key("b")],
                ..Default::default()
            };
            let result = idx
                .search_filtered(
                    QueryBatch::from_f32(&[1., 1., 1.], 3).unwrap(),
                    &both,
                    &FilterSearchOptions::new(3),
                )
                .unwrap();
            assert_eq!(result.result.counts, [0]);
            let saved = idx
                .save(&dir.path().join(format!("checkpoint-{i}")))
                .unwrap();
            let restored = Index::open(saved, BaseInput::from_f32(&base(), 3).unwrap()).unwrap();
            let a = search(&restored, "a");
            let b = search(&restored, "b");
            assert_eq!(a.label_revision, b.label_revision);
            let (expected_a, expected_b) = if a.label_revision % 2 == 1 {
                (vec![0, 1, 2], vec![9, 10, 11])
            } else {
                (vec![9, 10, 11], vec![0, 1, 2])
            };
            assert_eq!(&a.result.ids[..3], expected_a);
            assert_eq!(&b.result.ids[..3], expected_b);
        }
    });
}

#[test]
fn external_batches_convert_once_and_publish_atomically_with_full_u64_ids() {
    let dir = tempfile::tempdir().unwrap();
    let data = base();
    // Deliberately reverse business order and exceed u32: never cast external IDs.
    let business: Vec<_> = (0..48).map(|i| u64::MAX - i * 3).collect();
    let idx = Index::new_with_ids(
        BaseInput::from_f32(&data, 3).unwrap(),
        IndexOptions {
            cache_dir: dir.path().join("cache"),
            auto_pin: false,
            threads: 2,
            ..Default::default()
        },
        business.clone(),
    )
    .unwrap();
    let external = |label: &str, add: Vec<u64>, remove: Vec<u64>| ExternalLabelUpdate {
        label: key(label),
        add,
        remove,
    };
    assert_eq!(idx.update_labels_external(0, &[]).unwrap(), 0);
    assert_eq!(
        idx.update_labels_external(
            0,
            &[external(
                "book",
                vec![business[2], business[0], business[2]],
                vec![]
            ),]
        )
        .unwrap(),
        1
    );
    let before = search(&idx, "book");
    assert_eq!(&before.result.ids[..2], &[0, 2]);
    assert_eq!(
        &before.result.external_ids.as_ref().unwrap()[..2],
        &[business[0], business[2]]
    );
    assert_eq!(
        idx.update_labels_external(1, &[external("book", vec![business[0]], vec![]),])
            .unwrap(),
        1
    );
    for changes in [
        // A valid first operation must not survive any invalid later operation.
        vec![
            external("book", vec![business[1]], vec![]),
            external("movie", vec![0], vec![]),
        ],
        vec![
            external("book", vec![business[1]], vec![]),
            external("movie", vec![], vec![0]),
        ],
        vec![
            external("book", vec![business[1]], vec![]),
            external("book", vec![], vec![business[0]]),
        ],
        vec![
            external("book", vec![business[1]], vec![]),
            external("movie", vec![business[2]], vec![business[2]]),
        ],
        vec![
            external("book", vec![business[1]], vec![]),
            ExternalLabelUpdate {
                label: LabelKey {
                    namespace: String::new(),
                    value: "invalid".into(),
                },
                add: vec![business[2]],
                remove: vec![],
            },
        ],
        vec![external("book", vec![business[1]; 1_000_001], vec![])],
    ] {
        assert_eq!(
            idx.update_labels_external(1, &changes).unwrap_err().code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(idx.label_revision().unwrap(), 1);
        assert_eq!(search(&idx, "book").result.ids, before.result.ids);
        assert_eq!(search(&idx, "movie").result.counts, [0, 0]);
    }
    assert_eq!(
        idx.update_labels_external(0, &[external("book", vec![business[1]], vec![]),])
            .unwrap_err()
            .code(),
        ErrorCode::InvalidState
    );
    // Both entry points use one publication lock and one revision sequence.
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let first = scope.spawn(|| {
            barrier.wait();
            idx.update_labels_external(1, &[external("movie", vec![business[4]], vec![])])
        });
        let second = scope.spawn(|| {
            barrier.wait();
            idx.update_labels(1, &[change("movie", &[5], &[])])
        });
        let results = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .find_map(|r| r.as_ref().err())
                .unwrap()
                .code(),
            ErrorCode::InvalidState
        );
    });
    assert_eq!(idx.label_revision().unwrap(), 2);
    assert_eq!(search(&idx, "movie").result.counts, [1, 1]);
    assert_eq!(
        idx.update_labels_external(
            2,
            &[
                external("book", vec![], vec![business[0], business[2]]),
                external("moved", vec![business[0], business[2]], vec![]),
            ]
        )
        .unwrap(),
        3
    );
    assert_eq!(search(&idx, "book").result.counts, [0, 0]);
    let saved = idx.save(&dir.path().join("saved")).unwrap();
    let restored = Index::open(&saved, BaseInput::from_f32(&data, 3).unwrap()).unwrap();
    assert_eq!(restored.label_revision().unwrap(), 3);
    assert_eq!(search(&restored, "moved").result.ids, before.result.ids);
    // Restored mapping also accepts subsequent business-ID updates.
    restored
        .update_labels_external(3, &[external("moved", vec![business[1]], vec![])])
        .unwrap();
    assert_eq!(search(&restored, "moved").result.counts, [3, 3]);
}

#[test]
fn external_updates_require_a_mapping_without_changing_internal_api() {
    let dir = tempfile::tempdir().unwrap();
    let idx = Index::new(
        BaseInput::from_f32(&base(), 3).unwrap(),
        IndexOptions {
            cache_dir: dir.path().join("cache"),
            auto_pin: false,
            threads: 2,
            ..Default::default()
        },
    )
    .unwrap();
    for changes in [
        vec![],
        vec![ExternalLabelUpdate {
            label: key("book"),
            add: vec![0],
            remove: vec![],
        }],
    ] {
        assert_eq!(
            idx.update_labels_external(0, &changes).unwrap_err().code(),
            ErrorCode::InvalidState
        );
        assert_eq!(idx.label_revision().unwrap(), 0);
    }
    idx.update_labels(0, &[change("book", &[0], &[])]).unwrap();
    assert_eq!(search(&idx, "book").result.counts, [1, 1]);
}
