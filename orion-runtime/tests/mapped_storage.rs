/* Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License. */
use orion_ann::{
    BaseInput, CascadeConfig, GraphConfig, GraphSourcePolicy, Index, IndexOptions, OpenOptions,
    QueryBatch, SearchOptions, StorageMode, StorageOptions,
};

#[test]
fn mapped_calibration_snapshot_and_concurrent_search_match_resident() {
    let root = std::env::temp_dir().join(format!("orion-mmap-lifecycle-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let values: Vec<_> = (0..96 * 25).map(|i| ((i * 17) % 251) as f32).collect();
    let input = || BaseInput::from_f32(&values, 25).unwrap();
    let options = IndexOptions {
        threads: 2,
        auto_pin: false,
        cache_dir: root.join("cache"),
        graph: GraphConfig {
            source: GraphSourcePolicy::Rust,
            degree: 8,
            build_l: 16,
            max_extra: 4,
            ..Default::default()
        },
        cascade: CascadeConfig {
            prefilter: Some("none".into()),
            admission: Some("l2-u8".into()),
            rerank: Some("f32".into()),
        },
        ..Default::default()
    };
    let mut owned = Index::new(input(), options.clone()).unwrap();
    let queries = &values[..8 * 25];
    let expected_cal = owned.calibrate(queries.to_vec(), 3, Some(16)).unwrap();
    let search = |index: &Index| {
        index
            .search(
                QueryBatch::from_f32(queries, 25).unwrap(),
                &SearchOptions::new(3, 16),
            )
            .unwrap()
    };
    let expected = search(&owned);
    let snapshot = owned.save(&root.join("snapshot")).unwrap();
    let mapped_options = StorageOptions {
        mode: StorageMode::Mapped,
        directory: Some(root.clone()),
    };
    let mut mapped = Index::new(
        input(),
        IndexOptions {
            storage_options: mapped_options.clone(),
            ..options
        },
    )
    .unwrap();
    assert_eq!(mapped.info()["mapped_sidecar_bytes"], 0);
    assert_eq!(
        mapped.calibrate(queries.to_vec(), 3, Some(16)).unwrap(),
        expected_cal
    );
    assert_eq!(mapped.info()["mapped_sidecar_bytes"], 96 * 32);
    assert_eq!(search(&mapped), expected);
    // Repeated preparation reuses the installed backing rather than planning
    // against free memory after its own allocations have already consumed it.
    mapped.calibrate(queries.to_vec(), 4, Some(16)).unwrap();
    assert_eq!(mapped.info()["mapped_sidecar_bytes"], 96 * 32);
    assert_eq!(search(&mapped), expected);
    assert_eq!(
        owned.info()["cache_namespace"],
        mapped.info()["cache_namespace"]
    );
    let reopened = OpenOptions {
        storage_options: Some(mapped_options),
        auto_pin: Some(true),
        ..Default::default()
    }
    .open(&snapshot, input())
    .unwrap();
    assert_eq!(reopened.info()["mapped_sidecar_bytes"], 96 * 32);
    // Every payload is mapped. Automatic locking must leave all of them pageable.
    assert_eq!(reopened.info()["residency"]["requested_datasets"], 0);
    assert!(reopened.info()["mapped_base_bytes"].as_u64().unwrap() >= 96 * 32 * 4);
    assert!(reopened.info()["mapped_graph_bytes"].as_u64().unwrap() > 0);
    assert_eq!(search(&reopened), expected);
    // All external cache objects can disappear after open. Stable private
    // backing for base, graph and sidecars remain valid throughout shared search.
    std::fs::remove_dir_all(root.join("cache.objects")).unwrap();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| assert_eq!(search(&reopened), expected));
        }
    });
    drop(mapped);
    drop(owned);
    drop(reopened);
}

#[test]
fn insufficient_mapping_reserve_fails_before_base_allocation() {
    let root = std::env::temp_dir().join(format!("orion-mmap-budget-{}", std::process::id()));
    let values: Vec<_> = (0..64 * 8).map(|i| (i % 17) as f32).collect();
    let result = Index::new(
        BaseInput::from_f32(&values, 8).unwrap(),
        IndexOptions {
            cache_dir: root.clone(),
            threads: 1,
            auto_pin: false,
            memory_budget_gib: Some(0.001),
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                degree: 4,
                build_l: 8,
                ..Default::default()
            },
            storage_options: StorageOptions {
                mode: StorageMode::Mapped,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("ORION_RESOURCE_EXHAUSTED"),
        "{error}"
    );
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(root.with_extension("objects"));
}

#[test]
fn backing_errors_leave_index_unpublished() {
    let root = std::env::temp_dir().join(format!("orion-mmap-errors-{}", std::process::id()));
    let values: Vec<_> = (0..64 * 8).map(|i| (i % 17) as f32).collect();
    let result = Index::new(
        BaseInput::from_f32(&values, 8).unwrap(),
        IndexOptions {
            cache_dir: root.clone(),
            threads: 1,
            auto_pin: false,
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                degree: 4,
                build_l: 8,
                ..Default::default()
            },
            storage_options: StorageOptions {
                mode: StorageMode::Mapped,
                directory: Some(root.join("does-not-exist")),
            },
            ..Default::default()
        },
    );
    assert!(result.is_err());
    let _ = std::fs::remove_dir_all(&root);
    let _ = std::fs::remove_dir_all(root.with_extension("objects"));
}

#[test]
fn mapped_native_base_staged_import_cache_and_snapshot_survive_source_removal() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().join("base.u8bin");
    let staged = root.path().join("graph.staged");
    let rows = 16u32;
    let dim = 25u32;
    let mut bytes = Vec::new();
    bytes.extend(rows.to_le_bytes());
    bytes.extend(dim.to_le_bytes());
    for row in 0..rows {
        bytes.extend(std::iter::repeat_n(row as u8, dim as usize));
    }
    std::fs::write(&base, bytes).unwrap();
    let mut graph = Vec::new();
    for word in [0x53544147u32, 3, rows, 2, 1, 0] {
        graph.extend(word.to_le_bytes());
    }
    for n in 0..rows {
        for word in [1, 1, 1, (n + 1) % rows, (n + 2) % rows, (n + 3) % rows] {
            graph.extend(word.to_le_bytes());
        }
    }
    std::fs::write(&staged, graph).unwrap();
    let mut options = IndexOptions {
        threads: 2,
        auto_pin: true,
        cache_dir: root.path().join("cache"),
        graph: GraphConfig {
            staged_file: Some(staged),
            degree: 2,
            max_extra: 1,
            ..Default::default()
        },
        storage_options: StorageOptions {
            mode: StorageMode::Mapped,
            directory: Some(root.path().into()),
        },
        ..Default::default()
    };
    let mut index = Index::new(BaseInput::from_file(&base), options.clone()).unwrap();
    let queries = vec![0f32; 25];
    index.calibrate(queries.clone(), 1, Some(16)).unwrap();
    let search = |i: &Index| {
        i.search(
            QueryBatch::from_f32(&queries, 25).unwrap(),
            &SearchOptions::new(1, 16),
        )
        .unwrap()
    };
    let expected = search(&index);
    assert!(index.info()["mapped_base_bytes"].as_u64().unwrap() >= 16 * 32);
    assert!(index.info()["mapped_graph_bytes"].as_u64().unwrap() > 0);
    assert_eq!(index.info()["residency"]["requested_datasets"], 0);
    let snapshot = index.save(&root.path().join("snapshot")).unwrap();
    options.storage_options.mode = StorageMode::Resident;
    let mut owned = Index::new(BaseInput::from_file(&base), options).unwrap();
    owned.calibrate(queries.clone(), 1, Some(16)).unwrap();
    assert_eq!(search(&owned), expected);
    let reopened = OpenOptions::default()
        .open(snapshot, BaseInput::from_file(&base))
        .unwrap();
    std::fs::write(base, b"truncated external input").unwrap();
    std::fs::remove_dir_all(root.path().join("cache.objects")).unwrap();
    assert_eq!(search(&reopened), expected);
    assert_eq!(search(&index), expected);
}
