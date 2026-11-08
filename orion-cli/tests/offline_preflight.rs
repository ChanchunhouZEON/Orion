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
    assert_eq!(report["mapping_reserve_bytes"], 64 * 1024 * 1024);
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
    assert!(text.contains("known components within budget"));
    // Offline planning must not allocate a mapping or require the source files.
    assert!(!fixture.0.join("cache").exists());
}

#[test]
fn mapped_mode_accounts_for_reserve_and_rejects_insufficient_budgets() {
    let fixture = Fixture::new();
    let report = fixture.report(&[
        "--preflight",
        "--storage-mode",
        "mapped",
        "--memory-budget-gib",
        "1",
    ]);
    assert_eq!(report["resident_lower_bound_bytes"], 0);
    assert!(report["accounted_lower_bound_bytes"].as_u64().unwrap() >= 64 * 1024 * 1024);
    for mode in ["auto", "mapped"] {
        for format in ["json", "table"] {
            // Even mapping every payload cannot fit the 64 MiB reserve plus
            // query buffers into this budget. Mappings are not free memory.
            let output = fixture.run(&[
                "--preflight",
                "--storage-mode",
                mode,
                "--memory-budget-gib",
                "0.01",
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
