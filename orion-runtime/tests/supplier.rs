/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(feature = "experimental-graph")]
use orion_ann::{
    supplier::*, BaseInput, Index, IndexOptions, QueryBatch, SearchOptions, StorageMode,
};
fn spec() -> PartitionSpec {
    PartitionSpec {
        origin: PartitionOrigin {
            name: "fixture-supplier".into(),
            revision: Some("v1".into()),
        },
        points: 16,
        dimension: 3,
        element: PartitionElement::F32,
        metric: PartitionMetric::SquaredL2,
        entry: 7,
        max_degree: 12,
        max_extra: 3,
        base_sha256: None,
    }
}
fn rows() -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    (0..16)
        .map(|i| {
            let ids: Vec<u32> = (1..16).map(|n| (i + n) % 16).collect();
            (ids[..6].to_vec(), ids[6..12].to_vec(), ids[12..].to_vec())
        })
        .collect()
}
fn values() -> Vec<f32> {
    (0..48).map(|i| ((i * 19) % 37) as f32).collect()
}
fn base(native: bool) -> BaseInput {
    if native {
        BaseInput::from_u8(&values().iter().map(|v| *v as u8).collect::<Vec<_>>(), 3)
    } else {
        BaseInput::from_f32(&values(), 3)
    }
    .unwrap()
}
fn options(path: &std::path::Path, mapped: bool) -> IndexOptions {
    let mut o = IndexOptions {
        cache_dir: path.to_owned(),
        threads: 2,
        auto_pin: false,
        ..Default::default()
    };
    if mapped {
        o.storage_options.mode = StorageMode::Mapped;
    }
    o
}
#[test]
fn staged_import_search_snapshot_and_origin_are_preserved() {
    for native in [false, true] {
        for mapped in [false, true] {
            let d = tempfile::tempdir().unwrap();
            let mut s = spec();
            if native {
                s.element = PartitionElement::U8;
            }
            let path = d.path().join("input.staged");
            write_staged_file(&path, &s, &mut SlicePartitions::new(&rows())).unwrap();
            let mut index = Index::from_staged(
                base(native),
                &path,
                options(&d.path().join("cache"), mapped),
            )
            .unwrap();
            assert_eq!(index.info()["graph_source"], "supplied");
            assert_eq!(index.info()["supplier"]["name"], s.origin.name);
            index.calibrate(values(), 3, Some(16)).unwrap();
            let query = values();
            let q = QueryBatch::from_f32(&query, 3).unwrap();
            let search = SearchOptions::new(3, 16).with_scores(true);
            let result = index.search(q, &search).unwrap();
            let snapshot = index.save(&d.path().join("snapshot")).unwrap();
            std::fs::remove_file(path).unwrap();
            if d.path().join("cache").exists() {
                std::fs::remove_dir_all(d.path().join("cache")).unwrap();
            }
            let reopened = Index::open(&snapshot, base(native)).unwrap();
            assert_eq!(reopened.info()["supplier"], index.info()["supplier"]);
            assert_eq!(reopened.search(q, &search).unwrap().ids, result.ids);
            assert_eq!(reopened.search(q, &search).unwrap().scores, result.scores);
        }
    }
}
#[test]
fn duplicate_failure_never_publishes_and_writer_never_overwrites() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("graph.staged");
    let s = spec();
    let digest = write_staged_file(&path, &s, &mut SlicePartitions::new(&rows())).unwrap();
    assert_eq!(digest.len(), 64);
    let before = std::fs::read(&path).unwrap();
    assert!(write_staged_file(&path, &s, &mut SlicePartitions::new(&rows())).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let mut invalid = rows();
    invalid[4].2[0] = invalid[4].0[0];
    let failed = d.path().join("bad.staged");
    assert!(write_staged_file(&failed, &s, &mut SlicePartitions::new(&invalid)).is_err());
    assert!(!failed.exists());
    assert!(!std::fs::read_dir(d.path()).unwrap().any(|p| p
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with(".tmp")));
}
#[test]
fn names_do_not_change_topology_identity_and_base_binding_is_checked() {
    let d = tempfile::tempdir().unwrap();
    let path = d.path().join("graph.staged");
    let mut s = spec();
    write_staged_file(&path, &s, &mut SlicePartitions::new(&rows())).unwrap();
    let first =
        Index::from_staged(base(false), &path, options(&d.path().join("cache"), false)).unwrap();
    s.origin.name = "../different/display/name".into();
    let path2 = d.path().join("other.staged");
    write_staged_file(&path2, &s, &mut SlicePartitions::new(&rows())).unwrap();
    let second =
        Index::from_staged(base(false), &path2, options(&d.path().join("cache"), false)).unwrap();
    assert_eq!(
        first.info()["cache_namespace"],
        second.info()["cache_namespace"]
    );
    s.base_sha256 = Some("0".repeat(64));
    let bad = d.path().join("mismatch.staged");
    write_staged_file(&bad, &s, &mut SlicePartitions::new(&rows())).unwrap();
    assert!(Index::from_staged(
        base(false),
        bad,
        options(&d.path().join("othercache"), false)
    )
    .is_err());
    let mut low = options(&d.path().join("tinycache"), false);
    low.memory_budget_gib = Some(1e-12);
    assert!(Index::from_staged(base(false), &path, low).is_err());
}

#[test]
fn memory_path_matches_staged_and_supplier_failure_releases_source() {
    use std::sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    };
    struct Failing {
        dropped: Arc<AtomicBool>,
    }
    impl PartitionSource for Failing {
        fn next_partition(&mut self) -> orion_ann::Result<Option<NodePartitions<'_>>> {
            Err(orion_ann::Error::invalid_argument("supplier stopped"))
        }
    }
    impl Drop for Failing {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }
    let d = tempfile::tempdir().unwrap();
    let mut memory = Index::from_partitions(
        base(false),
        spec(),
        OwnedPartitions::new(rows()),
        options(&d.path().join("cache"), false),
    )
    .unwrap();
    let path = d.path().join("staged");
    write_staged_file(&path, &spec(), &mut SlicePartitions::new(&rows())).unwrap();
    let mut staged =
        Index::from_staged(base(false), &path, options(&d.path().join("cache"), false)).unwrap();
    memory.calibrate(values(), 3, Some(16)).unwrap();
    staged.calibrate(values(), 3, Some(16)).unwrap();
    let query = values();
    let q = QueryBatch::from_f32(&query, 3).unwrap();
    let o = SearchOptions::new(3, 16);
    assert_eq!(
        memory.search(q, &o).unwrap().ids,
        staged.search(q, &o).unwrap().ids
    );
    assert_eq!(
        memory.info()["cache_namespace"],
        staged.info()["cache_namespace"]
    );
    let count = std::fs::read_dir(d.path().join("cache.objects"))
        .unwrap()
        .count();
    let dropped = Arc::new(AtomicBool::new(false));
    assert!(Index::from_partitions(
        base(false),
        spec(),
        Failing {
            dropped: dropped.clone()
        },
        options(&d.path().join("cache"), true)
    )
    .is_err());
    assert!(dropped.load(Ordering::SeqCst));
    assert_eq!(
        count,
        std::fs::read_dir(d.path().join("cache.objects"))
            .unwrap()
            .count()
    );
    assert_eq!(
        memory.search(q, &o).unwrap().ids,
        staged.search(q, &o).unwrap().ids
    );
}

#[test]
fn native_supplier_metrics_keep_raw_bytes_and_validate_metric() {
    use orion_ann::SearchMetric;
    for (metric, partition) in [
        (SearchMetric::InnerProduct, PartitionMetric::InnerProduct),
        (SearchMetric::Cosine, PartitionMetric::Cosine),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let mut spec = spec();
        spec.element = PartitionElement::U8;
        spec.metric = partition;
        let mut opts = options(&dir.path().join("cache"), true);
        opts.metric = metric;
        let path = dir.path().join("native.staged");
        write_staged_file(&path, &spec, &mut SlicePartitions::new(&rows())).unwrap();
        let mut index = Index::from_staged(base(true), &path, opts.clone()).unwrap();
        index.calibrate(values(), 3, Some(16)).unwrap();
        let q = values();
        let result = index
            .search(
                QueryBatch::from_f32(&q, 3).unwrap(),
                &SearchOptions::new(3, 16).with_scores(true),
            )
            .unwrap();
        assert!(result.scores.is_some());
        let saved = index.save(&dir.path().join("saved")).unwrap();
        assert_eq!(
            Index::open(saved, base(true))
                .unwrap()
                .search(
                    QueryBatch::from_f32(&q, 3).unwrap(),
                    &SearchOptions::new(3, 16).with_scores(true)
                )
                .unwrap(),
            result
        );
        opts.metric = SearchMetric::SquaredL2;
        assert!(Index::from_staged(base(true), &path, opts).is_err());
    }
}
