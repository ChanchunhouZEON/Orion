/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resource planning must work before downloading a dataset or building its graph.
use serde_json::Value;
use std::{
    path::PathBuf,
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "orion-offline-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        Self(root)
    }
    fn run(&self, args: &[&str]) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_orion"));
        cmd.current_dir(&self.0).args([
            "--base",
            "data/sift1b/bigann_base.bvecs",
            "--dimension",
            "128",
            "--query",
            "data/sift1b/bigann_query.bvecs",
        ]);
        if !args.contains(&"--max-points") {
            cmd.args(["--max-points", "1000000000"]);
        }
        cmd.args(args).output().unwrap()
    }
    fn report(&self, args: &[&str]) -> Value {
        let output = self.run(args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
    fn base_path(&self) -> PathBuf {
        self.0.join("data/sift1b/bigann_base.bvecs")
    }
    fn write_base(&self, rows: usize) {
        let path = self.base_path();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = vec![];
        for _ in 0..rows {
            bytes.extend(128u32.to_le_bytes());
            bytes.extend([0; 128]);
        }
        std::fs::write(path, bytes).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn explicit_metadata_supports_billion_scale_offline_planning() {
    let fixture = Fixture::new();
    for count in [10_000_000u64, 100_000_000, 1_000_000_000] {
        let report = fixture.report(&["--max-points", &count.to_string(), "--preflight"]);
        assert_eq!(report["num_points"], count);
        assert_eq!(report["base_u8_bytes"], count * 128);
        assert_eq!(report["graph_slab_bytes"], count * 384);
        assert_eq!(report["sidecar_bytes"], 0);
    }
    assert!(!fixture.0.join("cache").exists());
}

#[test]
fn point_limit_and_budget_are_independent_of_declared_source_size() {
    let fixture = Fixture::new();
    let report = fixture.report(&[
        "--preflight",
        "--max-points",
        "1000000",
        "--memory-budget-gib",
        "1",
    ]);
    assert_eq!(report["num_points"], 1_000_000);
    assert_eq!(report["budget_exceeded"], false);
    // A resident-only plan must reject an oversized payload. Auto may instead
    // place the graph in a mapping, so it has a different budget contract.
    let output = fixture.run(&[
        "--preflight",
        "--memory-budget-gib",
        "128",
        "--storage-mode",
        "resident",
    ]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["budget_exceeded"], true);
    assert!(String::from_utf8_lossy(&output.stderr).contains("memory budget exceeded"));
}

#[test]
fn readable_headers_override_offline_point_limit() {
    let fixture = Fixture::new();
    fixture.write_base(12);
    let report = fixture.report(&["--preflight"]);
    assert_eq!(report["num_points"], 12);
    assert_eq!(report["metadata_sources"]["base"], "file");
    let out = fixture.run(&["--dimension", "64", "--preflight"]);
    assert!(!out.status.success());
}

#[test]
fn malformed_inputs_and_exports_do_not_fall_back_to_configuration() {
    let fixture = Fixture::new();
    fixture.write_base(12);
    std::fs::write(fixture.base_path(), [0, 1]).unwrap();
    assert!(!fixture.run(&["--preflight"]).status.success());
    std::fs::remove_file(fixture.base_path()).unwrap();
    std::fs::write(fixture.0.join("bad.staged"), [0, 1]).unwrap();
    assert!(!fixture
        .run(&["--preflight", "--staged-file", "bad.staged"])
        .status
        .success());
}

#[test]
fn ordinary_execution_still_requires_files() {
    let fixture = Fixture::new();
    for args in [vec![], vec!["--prepare-only"]] {
        assert!(!fixture.run(&args).status.success());
    }
}

#[test]
fn readable_query_and_staged_headers_determine_budget() {
    let fixture = Fixture::new();
    fixture.write_base(100);
    let row = [128u32.to_le_bytes().as_slice(), &[0u8; 128]].concat();
    std::fs::write(
        fixture.0.join("data/sift1b/bigann_query.bvecs"),
        row.repeat(2),
    )
    .unwrap();
    let staged: Vec<u8> = [0x53544147u32, 3, 100, 64, 8, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    std::fs::write(fixture.0.join("graph.staged"), staged).unwrap();
    let report = fixture.report(&["--preflight", "--staged-file", "graph.staged"]);
    assert_eq!(report["num_queries"], 2);
    assert_eq!(report["metadata_sources"]["graph"], "staged header");
    assert_eq!(report["graph_slab_bytes"], 100 * 320);
}

#[test]
fn table_and_json_formats_report_the_same_budget_result() {
    let fixture = Fixture::new();
    let table = fixture.run(&[
        "--preflight",
        "--preflight-format",
        "table",
        "--memory-budget-gib",
        "900",
    ]);
    assert!(table.status.success());
    let text = String::from_utf8(table.stdout).unwrap();
    for label in [
        "Memory preflight — anonymous",
        "Component / phase",
        "Source",
        "unknown",
        "119.209 GiB",
        "357.628 GiB",
        "known components within budget",
    ] {
        assert!(text.contains(label), "missing {label}: {text}");
    }
    assert!(!text.contains("\\u"));
    assert!(
        table.stderr.is_empty(),
        "preflight should not dump resolved config before the table"
    );
    let json = fixture.report(&[
        "--preflight",
        "--preflight-format",
        "json",
        "--memory-budget-gib",
        "900",
    ]);
    assert_eq!(json["budget_exceeded"], false);
    assert_eq!(json["node_reader_bytes"], 0);
    let resident = json["resident_lower_bound_gib"].as_f64().unwrap();
    assert!(text.contains(&format!("{resident:.3} GiB")));
    let exceeded = fixture.run(&[
        "--preflight",
        "--preflight-format",
        "table",
        "--memory-budget-gib",
        "128",
        "--storage-mode",
        "resident",
    ]);
    assert!(!exceeded.status.success());
    assert!(String::from_utf8_lossy(&exceeded.stdout).contains("EXCEEDED"));
}

#[test]
fn default_auto_maps_the_graph_to_fit_an_offline_budget() {
    let fixture = Fixture::new();
    let args = ["--preflight", "--memory-budget-gib", "128"];
    let report = fixture.report(&args);
    assert_eq!(report["budget_exceeded"], false);
    assert_eq!(report["resident_lower_bound_bytes"], 128_000_000_000u64);
    assert!(report.get("mapping_reserve_bytes").is_none());
    let graph = report["components"]
        .as_array()
        .unwrap()
        .iter()
        .find(|component| component["name"] == "graph slab")
        .unwrap();
    assert_eq!(graph["phase"], "mapped");
    assert_eq!(graph["bytes"], 384_000_000_000u64);
    let table = fixture.run(&[
        "--preflight",
        "--memory-budget-gib",
        "128",
        "--preflight-format",
        "table",
    ]);
    assert!(table.status.success());
    let text = String::from_utf8(table.stdout).unwrap();
    assert!(text.contains("mapped"));
    assert!(!text.contains("Mapping working-set reserve"));
    assert!(text.contains("known components within budget"));
    // Offline planning must not allocate a mapping or require the source files.
    assert!(!fixture.0.join("cache").exists());
}

#[test]
fn mapped_mode_budgets_real_working_buffers_without_fixed_allowance() {
    let fixture = Fixture::new();
    let report = fixture.report(&[
        "--preflight",
        "--storage-mode",
        "mapped",
        "--memory-budget-gib",
        "1",
    ]);
    assert_eq!(report["resident_lower_bound_bytes"], 0);
    assert!(report.get("mapping_reserve_bytes").is_none());
    let working_bytes: u64 = report["components"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|part| {
            matches!(
                part["name"].as_str(),
                Some("query buffers" | "result IDs" | "ground truth IDs")
            )
        })
        .map(|part| part["bytes"].as_u64().unwrap())
        .sum();
    assert!(working_bytes > 0);
    assert_eq!(report["accounted_lower_bound_bytes"], working_bytes);
    let insufficient_budget = ((working_bytes - 1) as f64 / (1u64 << 30) as f64).to_string();
    for mode in ["auto", "mapped"] {
        for format in ["json", "table"] {
            // A budget below the old allowance can now succeed. Known query
            // buffers still count, even when every persistent payload is mapped.
            let enough = fixture.run(&[
                "--preflight",
                "--storage-mode",
                mode,
                "--memory-budget-gib",
                "0.01",
                "--preflight-format",
                format,
            ]);
            assert!(
                enough.status.success(),
                "{mode}/{format}: {}",
                String::from_utf8_lossy(&enough.stderr)
            );
            let output = fixture.run(&[
                "--preflight",
                "--storage-mode",
                mode,
                "--memory-budget-gib",
                &insufficient_budget,
                "--preflight-format",
                format,
            ]);
            assert!(!output.status.success(), "{mode}/{format}");
            assert!(String::from_utf8_lossy(&output.stderr).contains("memory budget exceeded"));
            if format == "json" {
                let report: Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(report["budget_exceeded"], true);
                assert_eq!(report["resident_lower_bound_bytes"], 0);
            } else {
                assert!(String::from_utf8_lossy(&output.stdout).contains("EXCEEDED"));
            }
        }
    }
}

#[test]
fn dataset_name_labels_preflight_without_changing_cache_identity() {
    let fixture = Fixture::new();
    let anonymous = fixture.report(&["--preflight"]);
    let named = fixture.report(&["--preflight", "--dataset-name", "SIFT 1B"]);
    assert_eq!(anonymous["dataset"], "anonymous");
    assert_eq!(named["dataset"], "SIFT 1B");
    let original_config = fixture.report(&["--preflight", "--print-config"]);
    let named_config =
        fixture.report(&["--preflight", "--print-config", "--dataset-name", "SIFT 1B"]);
    assert_eq!(
        original_config["cache_namespace"],
        named_config["cache_namespace"]
    );
    assert!(named_config["cache_namespace"].is_string());
    let table = fixture.run(&[
        "--preflight",
        "--preflight-format",
        "table",
        "--dataset-name",
        "SIFT 1B",
    ]);
    assert!(table.status.success());
    assert!(String::from_utf8_lossy(&table.stdout).contains("Memory preflight — SIFT 1B"));
    let invalid = fixture.run(&["--preflight", "--dataset-name", "   "]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("dataset_name"));
}

#[test]
fn object_overrides_have_visible_decisions_and_reject_invalid_input() {
    let fixture = Fixture::new();
    let report = fixture.report(&[
        "--preflight",
        "--storage-mode",
        "mapped",
        "--object-storage",
        "graph=resident",
        "--object-storage",
        "pq=mapped",
    ]);
    let objects = report["placement_objects"].as_array().unwrap();
    let graph = objects.iter().find(|o| o["object"] == "graph").unwrap();
    assert_eq!(graph["placement"], "resident");
    assert_eq!(graph["overridden"], true);
    assert_eq!(report["base_mapped"], true);
    assert_eq!(
        report["inactive_placement_overrides"],
        serde_json::json!(["pq"])
    );
    let low = fixture.run(&[
        "--preflight",
        "--storage-mode",
        "mapped",
        "--object-storage",
        "graph=resident",
        "--memory-budget-gib",
        "0.1",
    ]);
    assert!(!low.status.success());
    let budget: Value = serde_json::from_slice(&low.stdout).unwrap();
    assert_eq!(budget["budget_exceeded"], true);
    for args in [
        vec!["--object-storage", "graph=unknown"],
        vec!["--object-storage", "unknown=resident"],
        vec!["--object-storage", "graph"],
        vec![
            "--object-storage",
            "graph=resident",
            "--object-storage",
            "graph=mapped",
        ],
        vec!["--object-storage", "labels=mapped"],
    ] {
        let mut input = vec!["--preflight"];
        input.extend(args);
        assert!(!fixture.run(&input).status.success());
    }
    let table = fixture.run(&[
        "--preflight",
        "--preflight-format",
        "table",
        "--object-storage",
        "graph=resident",
    ]);
    assert!(table.status.success());
    assert!(String::from_utf8_lossy(&table.stdout).contains("Storage placement"));
}

#[test]
fn partial_placement_and_phase_reports_distinguish_ram_from_mapping_capacity() {
    let fixture = Fixture::new();
    let args = [
        "--preflight",
        "--max-points",
        "1000000",
        "--vector-storage",
        "f32",
        "--prefilter",
        "none",
        "--admission",
        "l2-u8",
        "--rerank",
        "none",
        "--memory-budget-gib",
        "0.4",
        "--storage-mode",
        "auto",
        "--object-storage",
        "graph=resident",
    ];
    let report = fixture.report(&args);
    assert_eq!(report["planned_phase"], "search-ready");
    assert_eq!(report["budget_exceeded"], false);
    assert_eq!(report["graph_mapped"], false);
    assert_eq!(report["base_mapped"], true);
    assert!(report["mapped_bytes"].as_u64().unwrap() > 0);
    assert!(report["validation_scope"]["mmap_backing_capacity"]
        .as_str()
        .unwrap()
        .contains("not checked"));
    let mut build_args = args.to_vec();
    build_args.push("--prepare-only");
    let graph_only = fixture.report(&build_args);
    assert_eq!(graph_only["planned_phase"], "graph-only");
    assert_eq!(graph_only["sidecar_bytes"], 0);
    assert!(report["sidecar_bytes"].as_u64().unwrap() > 0);
    assert!(
        graph_only["mapped_bytes"].as_u64().unwrap() < report["mapped_bytes"].as_u64().unwrap()
    );
    let mut table_args = args.to_vec();
    table_args.extend(["--preflight-format", "table"]);
    let output = fixture.run(&table_args);
    assert!(output.status.success());
    let table = String::from_utf8(output.stdout).unwrap();
    assert!(table.contains("Planned phase: search-ready"));
    assert!(table.contains("Mapped payload limit: not configured"));
    assert!(table.contains("not a peak-RSS guarantee"));
}

#[test]
fn yaml_preflight_matches_flags_and_reports_config_errors() {
    let f = Fixture::new();
    std::fs::write(f.0.join("config.yaml"),
        "dataset_name: demo\nthreads: 8\nmemory_budget_gib: 0.4\nstorage_options: {mode: auto, objects: {graph: resident, base: mapped}}\n"
    ).unwrap();
    let yaml = f.report(&[
        "--config",
        "config.yaml",
        "--preflight",
        "--max-points",
        "1000000",
    ]);
    let flags = f.report(&[
        "--dataset-name",
        "demo",
        "--threads",
        "8",
        "--memory-budget-gib",
        "0.4",
        "--storage-mode",
        "auto",
        "--object-storage",
        "graph=resident",
        "--object-storage",
        "base=mapped",
        "--preflight",
        "--max-points",
        "1000000",
    ]);
    assert_eq!(yaml, flags);
    std::fs::write(f.0.join("config.yaml"), "threads: [8]").unwrap();
    let output = f.run(&[
        "--config",
        "config.yaml",
        "--preflight",
        "--error-format",
        "json",
    ]);
    assert!(!output.status.success());
    let error: Value = serde_json::from_slice(&output.stderr).unwrap();
    assert_eq!(error["operation"], "parse_config");
    assert_eq!(error["details"]["path"], "config.yaml");
}
