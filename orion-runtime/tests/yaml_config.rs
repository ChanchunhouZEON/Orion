/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann::{BaseInput, ErrorCode, Index, IndexOptions, StorageMode, StorageObject};

#[test]
fn yaml_and_json_share_schema_defaults_and_object_overrides() {
    let yaml = IndexOptions::from_yaml(
        r#"
dataset_name: example
metric: squared-l2
threads: 8
auto_pin: false
storage_options:
  mode: auto
  objects:
    graph: resident
    base: mapped
"#,
    )
    .unwrap();
    let json: IndexOptions = serde_json::from_value(serde_json::json!({
        "dataset_name": "example", "metric": "squared-l2", "threads": 8,
        "auto_pin": false, "storage_options": { "mode": "auto",
            "objects": {"graph": "resident", "base": "mapped"} }
    }))
    .unwrap();
    assert_eq!(yaml, json);
    assert_eq!(
        yaml.storage_options.objects[&StorageObject::Graph],
        StorageMode::Resident
    );
    assert_eq!(
        IndexOptions::from_yaml("{}").unwrap(),
        IndexOptions::default()
    );
    assert_eq!(
        IndexOptions::from_yaml("metric: l2").unwrap().metric,
        yaml.metric
    );
}

#[test]
fn ambiguous_or_invalid_yaml_is_rejected_with_shared_errors() {
    for source in [
        "unknown: true",
        "graph: {unknown: 1}",
        "threads: [8]",
        "threads: 8\nthreads: 16",
        "storage_options: {objects: {graph: resident, graph: mapped}}",
        "storage_options: {objects: {unknown: auto}}",
        "---\n{}\n---\n{}",
        "threads: [",
        "",
        "null",
        "[]",
    ] {
        let error = IndexOptions::from_yaml(source).expect_err(source);
        assert_eq!(error.code(), ErrorCode::InvalidArgument, "{source}");
        assert_eq!(error.operation(), Some("parse_config"));
        assert_eq!(error.details()["field"], "yaml");
        assert!(!error.message().is_empty());
    }
}

#[test]
fn semantic_validation_remains_shared_with_rust_callers() {
    let values = vec![0.0; 16 * 32];
    for source in ["threads: 0", "memory_budget_gib: -1", "window_size: 0"] {
        let options = IndexOptions::from_yaml(source).unwrap();
        let decoded: IndexOptions =
            serde_json::from_value(serde_json::to_value(&options).unwrap()).unwrap();
        let input = || BaseInput::from_f32(&values, 32).unwrap();
        let yaml_error = Index::new(input(), options).unwrap_err();
        let rust_error = Index::new(input(), decoded).unwrap_err();
        assert_eq!(yaml_error.code(), rust_error.code());
        assert_eq!(yaml_error.message(), rust_error.message());
    }
}
