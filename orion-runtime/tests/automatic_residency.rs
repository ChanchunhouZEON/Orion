/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! One process-local lifecycle test so registry assertions cannot race other indexes.
use orion_ann::{
    BaseInput, CascadeConfig, GraphConfig, GraphSourcePolicy, Index, IndexOptions, OpenOptions,
    PqCodeStorage, QueryBatch, SearchOptions,
};

#[test]
fn automatic_locks_follow_calibration_relocation_snapshot_and_owner_lifetime() {
    let root = std::env::temp_dir().join(format!("orion-auto-lifecycle-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let _cleanup = Cleanup(root.clone());
    let values: Vec<_> = (0..64 * 32).map(|i| ((i * 17) % 251) as f32).collect();
    let input = || BaseInput::from_f32(&values, 32).unwrap();
    let start = orion_ann_platform::residency::registered_bytes();
    let mut index = Index::new(
        input(),
        IndexOptions {
            cache_dir: root.join("cache"),
            threads: 2,
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                degree: 8,
                build_l: 16,
                max_extra: 4,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    index
        .calibrate(values[..32 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let first = index.info();
    let pin = &first["residency"];
    assert!(pin["automatic"]["enabled"].as_bool().unwrap());
    assert!(
        pin["pinned_bytes"].as_u64().unwrap() <= pin["automatic"]["budget_bytes"].as_u64().unwrap()
    );
    let search = |i: &Index| {
        i.search(
            QueryBatch::from_f32(&values[..32 * 8], 32).unwrap(),
            &SearchOptions::new(3, 16),
        )
        .unwrap()
    };
    let expected = search(&index);
    // Query-time search neither refreshes the lock budget nor replaces guards.
    assert_eq!(index.info()["residency"], first["residency"]);
    // Moving the Index changes handles, not the backing heap addresses.
    let mut moved = Box::new(index);
    moved
        .calibrate(values[..32 * 8].to_vec(), 3, Some(16))
        .unwrap();
    assert_eq!(expected, search(&moved));
    let snapshot = moved.save(&root.join("snapshot")).unwrap();
    drop(moved);
    assert_eq!(orion_ann_platform::residency::registered_bytes(), start);
    let off = OpenOptions {
        auto_pin: Some(false),
        ..Default::default()
    }
    .open(&snapshot, input())
    .unwrap();
    assert_eq!(off.info()["residency"]["pinned_bytes"], 0);
    assert_eq!(off.info()["residency"]["automatic"]["enabled"], false);
    assert_eq!(expected, search(&off));
    drop(off);
    let on = OpenOptions {
        auto_pin: Some(true),
        ..Default::default()
    }
    .open(&snapshot, input())
    .unwrap();
    assert_eq!(on.info()["residency"]["automatic"]["enabled"], true);
    assert_eq!(expected, search(&on));
    drop(on);
    assert_eq!(orion_ann_platform::residency::registered_bytes(), start);

    let mut disabled = Index::new(
        input(),
        IndexOptions {
            cache_dir: root.join("disabled-cache"),
            auto_pin: false,
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                degree: 8,
                build_l: 16,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    disabled
        .calibrate(values[..32 * 8].to_vec(), 3, Some(16))
        .unwrap();
    assert_eq!(disabled.info()["residency"]["automatic"]["enabled"], false);
    assert_eq!(disabled.info()["residency"]["pinned_bytes"], 0);
    assert_eq!(orion_ann_platform::residency::registered_bytes(), start);
    drop(disabled);

    // Replacing PQ code placement must release automatic locks before unmapping.
    let mut pq = Index::new(
        input(),
        IndexOptions {
            cache_dir: root.join("pq-cache"),
            threads: 2,
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                degree: 8,
                build_l: 16,
                ..Default::default()
            },
            cascade: CascadeConfig {
                prefilter: Some("none".into()),
                admission: Some("pq".into()),
                rerank: Some("f32".into()),
            },
            ..Default::default()
        },
    )
    .unwrap();
    pq.calibrate(values[..32 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let before = search(&pq);
    let requests = pq.info()["residency"]["requested_datasets"]
        .as_u64()
        .unwrap();
    pq.set_pq_code_storage(PqCodeStorage::OnDemand).unwrap();
    assert_eq!(before, search(&pq));
    assert_eq!(
        pq.info()["residency"]["requested_datasets"]
            .as_u64()
            .unwrap()
            + 1,
        requests
    );
    drop(pq);
    assert_eq!(orion_ann_platform::residency::registered_bytes(), start);
}
