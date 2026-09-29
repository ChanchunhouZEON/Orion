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
        Command::new(env!("CARGO_BIN_EXE_orion"))
            .current_dir(&self.0)
            .env_remove("ORION_SWEEP_CONFIG")
            .env_remove("ORION_STAGED_FILE")
            .env_remove("ORION_GRAPH")
            .env("PA_ROOT", self.0.join("missing-parlay"))
            .args(args)
            .output()
            .unwrap()
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
fn large_presets_need_no_files_or_additional_parameters() {
    let fixture = Fixture::new();
    for (preset, count) in [
        ("sift10m", 10_000_000u64),
        ("sift100m", 100_000_000),
        ("sift1b", 1_000_000_000),
    ] {
        let report = fixture.report(&[preset, "--preflight"]);
        assert_eq!(report["num_points"], count);
        assert_eq!(report["num_queries"], 10_000);
        assert_eq!(report["base_u8_bytes"], count * 128 + 64);
        assert_eq!(report["graph_slab_bytes"], count * 384);
        assert_eq!(report["l2_u8_admission_bytes"], 0);
        assert_eq!(report["metadata_sources"]["base"], "configuration");
        assert_eq!(report["metadata_sources"]["graph"], "configuration");
        assert!(report["memory_budget_gib"].is_null());
    }
    assert!(
        !fixture.0.join("cache").exists(),
        "planning must not create caches"
    );
}

#[test]
fn point_limit_and_budget_are_independent_of_declared_source_size() {
    let fixture = Fixture::new();
    let report = fixture.report(&[
        "sift1b",
        "--preflight",
        "--max-points",
        "1000000",
        "--memory-budget-gib",
        "1",
    ]);
    assert_eq!(report["num_points"], 1_000_000);
    assert_eq!(report["budget_exceeded"], false);
    let output = fixture.run(&["sift1b", "--preflight", "--memory-budget-gib", "128"]);
    assert!(!output.status.success());
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["budget_exceeded"], true);
    assert!(String::from_utf8_lossy(&output.stderr).contains("memory budget exceeded"));
}

#[test]
fn existing_files_must_agree_with_declarations() {
    let fixture = Fixture::new();
    fixture.write_base(12);
    let output = fixture.run(&["sift1b", "--preflight"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("file count 12 != declared count 1000000000"));
    // Explicitly selecting the file discards the preset's source-count claim.
    let report = fixture.report(&[
        "sift1b",
        "--preflight",
        "--base",
        fixture.base_path().to_str().unwrap(),
    ]);
    assert_eq!(report["num_points"], 12);
    assert_eq!(report["metadata_sources"]["base"], "file");
}

#[test]
fn missing_overrides_cannot_borrow_preset_point_counts() {
    let fixture = Fixture::new();
    let output = fixture.run(&["sift1b", "--preflight", "--base", "different.bvecs"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("offline preflight needs preset num_points"));
}

#[test]
fn malformed_inputs_and_exports_do_not_fall_back_to_configuration() {
    let fixture = Fixture::new();
    fixture.write_base(12);
    std::fs::write(fixture.base_path(), [0, 1]).unwrap();
    assert!(!fixture.run(&["sift1b", "--preflight"]).status.success());
    std::fs::remove_file(fixture.base_path()).unwrap();
    std::fs::write(fixture.0.join("bad.staged"), [0, 1]).unwrap();
    assert!(!fixture
        .run(&["sift1b", "--preflight", "--staged-file", "bad.staged"])
        .status
        .success());
}

#[test]
fn ordinary_execution_still_requires_files() {
    let fixture = Fixture::new();
    for args in [vec!["sift1b"], vec!["sift1b", "--prepare-only"]] {
        assert!(!fixture.run(&args).status.success());
    }
}

#[test]
fn custom_declarations_and_existing_graph_query_headers_are_checked() {
    let fixture = Fixture::new();
    let yaml = include_str!("../../benchmark/configs/sweep.yaml")
        .replace("num_points: 1000000000", "num_points: 100")
        .replace("num_queries: 10000", "num_queries: 2");
    std::fs::write(fixture.0.join("small.yaml"), yaml).unwrap();
    fixture.write_base(100);
    let query = fixture.0.join("data/sift1b/bigann_query.bvecs");
    // A readable query file with the wrong count must not be hidden by YAML.
    let row = [128u32.to_le_bytes().as_slice(), &[0u8; 128]].concat();
    std::fs::write(&query, &row).unwrap();
    let output = fixture.run(&["sift1b", "--config", "small.yaml", "--preflight"]);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("file count 1 != declared count 2"));
    std::fs::write(&query, row.repeat(2)).unwrap();
    let staged: Vec<u8> = [0x53544147u32, 3, 100, 64, 8, 0]
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    std::fs::write(fixture.0.join("graph.staged"), staged).unwrap();
    let report = fixture.report(&[
        "sift1b",
        "--config",
        "small.yaml",
        "--preflight",
        "--staged-file",
        "graph.staged",
    ]);
    assert_eq!(report["metadata_sources"]["base"], "file");
    assert_eq!(report["metadata_sources"]["query"], "file");
    assert_eq!(report["metadata_sources"]["graph"], "staged header");
    // The materialized graph's 8 extras override the configured 16 for accounting.
    assert_eq!(report["graph_slab_bytes"], 100 * 320);
}

#[test]
fn table_and_json_formats_report_the_same_budget_result() {
    let fixture = Fixture::new();
    let table = fixture.run(&[
        "sift1b",
        "--preflight",
        "--preflight-format",
        "table",
        "--memory-budget-gib",
        "650",
    ]);
    assert!(table.status.success());
    let text = String::from_utf8(table.stdout).unwrap();
    for label in [
        "Memory preflight — sift1b",
        "Component / phase",
        "Source",
        "unknown",
        "119.209 GiB",
        "357.628 GiB",
        "484.289 GiB",
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
        "sift1b",
        "--preflight",
        "--preflight-format",
        "json",
        "--memory-budget-gib",
        "650",
    ]);
    assert_eq!(json["budget_exceeded"], false);
    let exceeded = fixture.run(&[
        "sift1b",
        "--preflight",
        "--preflight-format",
        "table",
        "--memory-budget-gib",
        "128",
    ]);
    assert!(!exceeded.status.success());
    assert!(String::from_utf8_lossy(&exceeded.stdout).contains("EXCEEDED"));
}
