/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Exercise accounting at the public runtime boundary, using small real objects.
use orion_ann::{
    BaseInput, CascadeConfig, GraphConfig, GraphSourcePolicy, Index, IndexOptions, OpenOptions,
    PqCodeStorage, QueryBatch, SearchOptions, StorageMode, StorageOptions,
};

fn values() -> Vec<f32> {
    (0..96 * 25)
        .map(|i| ((i * 31 + i / 25 * 17) % 251) as f32)
        .collect()
}
fn options(root: &std::path::Path, admission: &str, mode: StorageMode) -> IndexOptions {
    IndexOptions {
        cache_dir: root.join("cache"),
        threads: 2,
        auto_pin: false,
        graph: GraphConfig {
            source: GraphSourcePolicy::Rust,
            degree: 8,
            build_l: 16,
            max_extra: 4,
            ..Default::default()
        },
        cascade: CascadeConfig {
            prefilter: Some("none".into()),
            admission: Some(admission.into()),
            rerank: Some("none".into()),
        },
        storage_options: StorageOptions {
            mode,
            directory: Some(root.to_owned()),
            ..Default::default()
        },
        ..Default::default()
    }
}
fn accounting(index: &Index, key: &str) -> u64 {
    index.info()["storage_accounting"][key].as_u64().unwrap()
}
fn search(index: &Index, data: &[f32]) -> orion_ann::SearchResult {
    index
        .search(
            QueryBatch::from_f32(&data[..25 * 8], 25).unwrap(),
            &SearchOptions::new(3, 16),
        )
        .unwrap()
}

#[test]
fn calibration_counts_once_and_snapshot_reports_actual_placement() {
    let root = tempfile::tempdir().unwrap();
    let data = values();
    let mut index = Index::new(
        BaseInput::from_f32(&data, 25).unwrap(),
        options(root.path(), "l2-u8", StorageMode::Resident),
    )
    .unwrap();
    let before = accounting(&index, "resident_payload_bytes");
    index
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let after = accounting(&index, "resident_payload_bytes");
    let codes = orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(25)
        .unwrap()
        .allocation_bytes(96)
        .unwrap() as u64;
    assert_eq!(after - before, codes);
    index
        .calibrate(data[..25 * 8].to_vec(), 4, Some(16))
        .unwrap();
    assert_eq!(accounting(&index, "resident_payload_bytes"), after);
    let expected = search(&index, &data);
    let snapshot = index.save(&root.path().join("snapshot")).unwrap();
    let mapped = OpenOptions {
        auto_pin: Some(false),
        storage_options: Some(StorageOptions {
            mode: StorageMode::Mapped,
            directory: Some(root.path().to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }
    .open(snapshot, BaseInput::from_f32(&data, 25).unwrap())
    .unwrap();
    assert_eq!(accounting(&mapped, "resident_payload_bytes"), 0);
    assert_eq!(accounting(&mapped, "mapped_payload_bytes"), after);
    assert!(mapped.info()["storage_accounting"]
        .get("reserved_bytes")
        .is_none());
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| assert_eq!(search(&mapped, &data), expected));
        }
    });
    assert_eq!(accounting(&mapped, "mapped_payload_bytes"), after);
}

#[test]
fn pq_codes_and_codebook_are_not_double_charged_across_replacement() {
    let root = tempfile::tempdir().unwrap();
    let data = values();
    let mut index = Index::new(
        BaseInput::from_f32(&data, 25).unwrap(),
        options(root.path(), "pq-l2", StorageMode::Resident),
    )
    .unwrap();
    index
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let expected = search(&index, &data);
    let before = accounting(&index, "resident_payload_bytes");
    index.set_pq_code_storage(PqCodeStorage::OnDemand).unwrap();
    let mapped = accounting(&index, "mapped_payload_bytes");
    assert!(mapped > 0);
    // A mapped codebook remains mapped when only codes move into RAM. Budget
    // exactly the code bytes, without another codebook or a synthetic allowance.
    let tight_root = tempfile::tempdir().unwrap();
    let mut tight_options = options(tight_root.path(), "pq-l2", StorageMode::Mapped);
    tight_options.memory_budget_gib = Some(mapped as f64 / (1u64 << 30) as f64);
    let mut tight = Index::new(BaseInput::from_f32(&data, 25).unwrap(), tight_options).unwrap();
    tight
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    // Compare against this index's own graph, not a separately built graph.
    let tight_expected = search(&tight, &data);
    tight.set_pq_code_storage(PqCodeStorage::Resident).unwrap();
    assert_eq!(accounting(&tight, "resident_payload_bytes"), mapped);
    assert_eq!(search(&tight, &data), tight_expected);

    assert_eq!(
        accounting(&index, "resident_payload_bytes") + mapped,
        before
    );
    assert_eq!(search(&index, &data), expected);
    index.set_pq_code_storage(PqCodeStorage::OnDemand).unwrap();
    assert_eq!(
        accounting(&index, "resident_payload_bytes") + accounting(&index, "mapped_payload_bytes"),
        before
    );
    index.set_pq_code_storage(PqCodeStorage::Resident).unwrap();
    assert_eq!(accounting(&index, "resident_payload_bytes"), before);
    assert_eq!(accounting(&index, "mapped_payload_bytes"), 0);
    assert!(index.info()["storage_accounting"]
        .get("reserved_bytes")
        .is_none());
    assert_eq!(search(&index, &data), expected);
}

#[test]
fn insufficient_replacement_budget_preserves_mapping_and_search() {
    let root = tempfile::tempdir().unwrap();
    let data = values();
    let mut config = options(root.path(), "pq-l2", StorageMode::Mapped);
    config.memory_budget_gib = Some(100f64 / (1u64 << 30) as f64);
    let mut index = Index::new(BaseInput::from_f32(&data, 25).unwrap(), config).unwrap();
    index
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let expected = search(&index, &data);
    let before = index.info()["storage_accounting"].clone();
    let error = index
        .set_pq_code_storage(PqCodeStorage::Resident)
        .unwrap_err();
    assert_eq!(error.code(), orion_ann::ErrorCode::ResourceExhausted);
    assert_eq!(index.info()["storage_accounting"], before);
    assert_eq!(search(&index, &data), expected);
}

#[test]
fn pq_only_search_keeps_base_mapped_and_hot_objects_resident() {
    let root = tempfile::tempdir().unwrap();
    let data = values();
    let mut resident = Index::new(
        BaseInput::from_f32(&data, 25).unwrap(),
        options(root.path(), "pq-l2", StorageMode::Resident),
    )
    .unwrap();
    resident
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    let expected = search(&resident, &data);
    let base_bytes =
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(25)
            .unwrap()
            .allocation_bytes(96)
            .unwrap() as u64;
    // Give graph + PQ exactly enough RAM; the original base must remain pageable.
    let hot_bytes = accounting(&resident, "resident_payload_bytes") - base_bytes;
    let mut config = options(root.path(), "pq-l2", StorageMode::Auto);
    config.memory_budget_gib = Some(hot_bytes as f64 / (1u64 << 30) as f64);
    // Reuse the same graph cache, so this compares storage, not build randomness.
    let mut mixed = Index::new(BaseInput::from_f32(&data, 25).unwrap(), config).unwrap();
    mixed
        .calibrate(data[..25 * 8].to_vec(), 3, Some(16))
        .unwrap();
    assert_eq!(mixed.info()["mapped_base_bytes"], base_bytes);
    assert_eq!(mixed.info()["mapped_graph_bytes"], 0);
    assert_eq!(mixed.info()["mapped_sidecar_bytes"], 0);
    assert_eq!(accounting(&mixed, "mapped_payload_bytes"), base_bytes);
    assert_eq!(accounting(&mixed, "resident_payload_bytes"), hot_bytes);
    assert_eq!(accounting(&mixed, "available_bytes"), 0);
    assert_eq!(search(&mixed, &data), expected);
    // Repeated preparation must neither double-charge nor promote the base.
    mixed
        .calibrate(data[..25 * 8].to_vec(), 4, Some(16))
        .unwrap();
    assert_eq!(accounting(&mixed, "mapped_payload_bytes"), base_bytes);
    assert_eq!(accounting(&mixed, "resident_payload_bytes"), hot_bytes);
    assert_eq!(search(&mixed, &data), expected);
}

fn object_usage(index: &Index, object: &str) -> serde_json::Value {
    index.info()["storage_accounting"]["objects"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["object"] == object)
        .unwrap()
        .clone()
}

#[test]
fn mixed_overrides_survive_snapshot_keep_results_and_release_failed_preparation() {
    use orion_ann::StorageObject;
    let root = tempfile::tempdir().unwrap();
    let mapping_dir = root.path().join("mappings");
    std::fs::create_dir(&mapping_dir).unwrap();
    let data = values();
    let mut config = options(root.path(), "l2-u8", StorageMode::Resident);
    config.storage_options.directory = Some(mapping_dir.clone());
    config.storage_options.objects = [
        (StorageObject::Base, StorageMode::Mapped),
        (StorageObject::L2U8, StorageMode::Mapped),
        (StorageObject::Pq, StorageMode::Resident),
    ]
    .into();
    let mut index = Index::new(BaseInput::from_f32(&data, 25).unwrap(), config).unwrap();
    let before = index.info()["storage_accounting"].clone();
    // Fail after the base and graph are live, while creating a mapped sidecar.
    std::fs::remove_dir(&mapping_dir).unwrap();
    assert!(index.calibrate(data[..200].to_vec(), 3, Some(16)).is_err());
    assert_eq!(index.info()["storage_accounting"], before);
    std::fs::create_dir(&mapping_dir).unwrap();
    index.calibrate(data[..200].to_vec(), 3, Some(16)).unwrap();
    assert!(
        object_usage(&index, "base")["mapped_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        object_usage(&index, "graph")["resident_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(object_usage(&index, "l2-u8")["mapped_bytes"], 96 * 32);
    assert!(!index.info()["storage_accounting"]["objects"]
        .as_array()
        .unwrap()
        .iter()
        .any(|o| o["object"] == "pq"));
    let expected = search(&index, &data);
    let before = index.info()["storage_accounting"].clone();
    index.calibrate(data[..200].to_vec(), 4, Some(16)).unwrap();
    assert_eq!(index.info()["storage_accounting"], before);
    let snapshot = index.save(&root.path().join("snapshot")).unwrap();
    let inherited = Index::open(&snapshot, BaseInput::from_f32(&data, 25).unwrap()).unwrap();
    assert_eq!(
        inherited.info()["storage_options"],
        index.info()["storage_options"]
    );
    assert_eq!(search(&inherited, &data), expected);
    let swapped = OpenOptions {
        storage_options: Some(StorageOptions {
            mode: StorageMode::Mapped,
            directory: Some(mapping_dir.clone()),
            objects: [
                (StorageObject::Base, StorageMode::Resident),
                (StorageObject::L2U8, StorageMode::Resident),
            ]
            .into(),
        }),
        ..Default::default()
    }
    .open(&snapshot, BaseInput::from_f32(&data, 25).unwrap())
    .unwrap();
    assert_eq!(
        swapped.info()["cache_namespace"],
        index.info()["cache_namespace"]
    );
    assert_eq!(swapped.calibration(3), index.calibration(3));
    assert_eq!(swapped.info()["mapped_base_bytes"], 0);
    assert_eq!(swapped.info()["mapped_sidecar_bytes"], 0);
    assert!(swapped.info()["mapped_graph_bytes"].as_u64().unwrap() > 0);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| assert_eq!(search(&swapped, &data), expected));
        }
    });
    drop(swapped);
    drop(inherited);
    drop(index);
    assert_eq!(std::fs::read_dir(&mapping_dir).unwrap().count(), 0);
}

#[test]
fn resident_override_cannot_silently_fall_back_to_mapping() {
    use orion_ann::{ErrorCode, StorageObject};
    let root = tempfile::tempdir().unwrap();
    let data = values();
    let mut config = options(root.path(), "l2-u8", StorageMode::Mapped);
    config.memory_budget_gib = Some(1f64 / (1u64 << 30) as f64);
    config
        .storage_options
        .objects
        .insert(StorageObject::Graph, StorageMode::Resident);
    let error = Index::new(BaseInput::from_f32(&data, 25).unwrap(), config).unwrap_err();
    assert_eq!(error.code(), ErrorCode::ResourceExhausted);
    let mut config = options(root.path(), "l2-u8", StorageMode::Mapped);
    config
        .storage_options
        .objects
        .insert(StorageObject::Labels, StorageMode::Mapped);
    let error = Index::new(BaseInput::from_f32(&data, 25).unwrap(), config).unwrap_err();
    assert_eq!(error.code(), ErrorCode::UnsupportedConfiguration);
}
