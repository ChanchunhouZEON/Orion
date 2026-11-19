/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Offline and execution preflight contracts.
use super::*;
use crate::config::VectorStorageKind;
use crate::VectorFormat;

fn native_config() -> ResolvedRunConfig {
    let mut config = crate::IndexOptions::default()
        .resolve(
            "missing.bvecs".into(),
            VectorFormat::Bvecs,
            128,
            1_000_000_000,
            None,
        )
        .unwrap();
    config.prepare_only = false;
    config.replay_queries = true;
    config.query = Some("missing-query.bvecs".into());
    config.groundtruth = Some("missing.ivecs".into());
    config.sweep.k = 10;
    config
}
#[test]
fn offline_sweep_keeps_unknown_query_buffers_explicit() {
    let directory =
        std::env::temp_dir().join(format!("orion-offline-sweep-{}", std::process::id()));
    let mut config = native_config();
    config.base = directory.join("base.bvecs");
    config.query = Some(directory.join("query.bvecs"));
    config.staged_file = Some(directory.join("graph.staged"));
    config.cache_dir = directory.join("cache");
    config.num_queries = None;
    let report = preflight_report(&config).unwrap();
    assert_eq!(report["unknown_replay_buffers"], true);
    for name in ["query buffers", "result IDs", "ground truth IDs"] {
        let part = report["components"]
            .as_array()
            .unwrap()
            .iter()
            .find(|part| part["name"] == name)
            .unwrap();
        assert!(part["bytes"].is_null());
    }
    // The runtime path cannot substitute declarations for missing query files.
    assert!(estimate_resident_memory(&config, 1_000_000_000).is_err());
    config.num_queries = Some(10_000);
    let report = preflight_report(&config).unwrap();
    assert_eq!(report["unknown_replay_buffers"], false);
    assert_eq!(report["num_queries"], 10_000);
    config.prepare_only = true;
    let report = preflight_report(&config).unwrap();
    assert_eq!(report["num_queries"], 0);
}

#[test]
fn native_billion_estimate_counts_one_byte_base_and_no_admission_copy() {
    let directory =
        std::env::temp_dir().join(format!("orion-native-estimate-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let staged = directory.join("graph.staged");
    let header: Vec<u8> = [0x53544147u32, 3, 1_000_000_000, 64, 16, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    std::fs::write(&staged, header).unwrap();
    let mut config = native_config();
    config.replay_queries = false;
    config.query = Some("-".into());
    config.staged_file = Some(staged);
    config.cache_dir = directory.clone();
    let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
    assert_eq!(report["base_f32_bytes"], 0);
    assert_eq!(report["base_u8_bytes"], 128_000_000_000u64);
    assert_eq!(report["sidecar_bytes"], 0);
    assert_eq!(report["resident_lower_bound_bytes"], 512_000_000_000u64);
    std::fs::remove_dir_all(directory).unwrap();
}

#[cfg(feature = "parlay")]
#[test]
fn slab_build_counts_graph_once() {
    let mut config = native_config();
    config.cache_dir =
        std::env::temp_dir().join(format!("orion-slab-budget-{}", std::process::id()));
    config.staged_file = None;
    config.staged_file_explicit = false;
    let report = preflight_report(&config).unwrap();
    assert_eq!(report["graph_action"], "build ParlayANN graph");
    assert_eq!(report["build_lower_bound_bytes"], 512_000_000_000u64);
    let working = report["components"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "ParlayANN construction norms")
        .unwrap();
    assert_eq!(working["bytes"], 0);
}

#[test]
fn billion_lower_bound_and_budget_guard() {
    let mut config = crate::IndexOptions {
        storage: Some(VectorStorageKind::F32),
        graph: crate::GraphConfig {
            source: crate::GraphSourcePolicy::Rust,
            ..Default::default()
        },
        cache_dir: "/tmp/orion-resource-unit-test-unused".into(),
        ..Default::default()
    }
    .resolve(
        "missing-large.bvecs".into(),
        VectorFormat::Bvecs,
        128,
        1_000_000_000,
        None,
    )
    .unwrap();
    config.query = Some("-".into());
    config.groundtruth = Some("missing.ivecs".into());
    config.prepare_only = false;
    let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
    assert_eq!(report["base_f32_bytes"], 512_000_000_000u64);
    assert_eq!(report["graph_slab_bytes"], 384_000_000_000u64);
    assert_eq!(report["sidecar_bytes"], 128_000_000_000u64);
    config.memory_budget_gib = Some(128.);
    // Owned-only remains a strict guard. Auto can now move base and graph.
    config.storage_options.mode = crate::StorageMode::Resident;
    let report = estimate_resident_memory(&config, 1_000_000_000).unwrap();
    let error = enforce_budget(&report).unwrap_err();
    for component in [
        "base vectors",
        "graph slab",
        "node reader counters",
        "query buffers",
        "worker scratch",
        "Budget:",
        "excess:",
        "unknown",
    ] {
        assert!(
            error.to_string().contains(component),
            "missing {component}: {error}"
        );
    }
    assert!(report["components"].as_array().unwrap().len() >= 10);
}
