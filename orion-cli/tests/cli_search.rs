/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use serde_json::Value;
use std::{
    io::Write,
    path::PathBuf,
    process::{Command, Output, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);

struct Fixture {
    root: PathBuf,
    query_bytes: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "orion-cli-e2e-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let mut base = vec![];
        for i in 0..16u8 {
            base.extend(128u32.to_le_bytes());
            base.extend([i * 7; 128]);
        }
        std::fs::write(root.join("base.bvecs"), &base).unwrap();
        let mut query_bytes = vec![];
        let mut gt = vec![];
        // Cross calibration and multiple search batches; every input must appear once.
        for i in 0..205 {
            let row = i % 16;
            query_bytes.extend(&base[row * 132..(row + 1) * 132]);
            gt.extend(1u32.to_le_bytes());
            gt.extend((row as u32).to_le_bytes());
        }
        std::fs::write(root.join("query.bvecs"), &query_bytes).unwrap();
        std::fs::write(root.join("gt.ivecs"), gt).unwrap();
        let mut graph: Vec<u8> = [0x53544147u32, 3, 16, 15, 0, 0]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        for i in 0..16u32 {
            for v in [15, 0, 0].into_iter().chain((0..16).filter(|j| *j != i)) {
                graph.extend(v.to_le_bytes());
            }
        }
        std::fs::write(root.join("graph.staged"), graph).unwrap();
        Self { root, query_bytes }
    }
    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_orion"));
        cmd.current_dir(&self.root)
            .env_remove("ORION_GRAPH")
            .env_remove("ORION_STAGED_FILE")
            .env_remove("ORION_SWEEP_CONFIG");
        cmd.args([
            "--base",
            "base.bvecs",
            "--metric",
            "l2",
            "--graph-degree",
            "15",
            "--max-extra",
            "0",
            "--k",
            "1",
            "--search-l",
            "16",
            "--threads",
            "2",
            "--cache-dir",
            "cache",
            "--query-batch-size",
            "2",
        ]);
        cmd
    }
    fn native(&self) -> Command {
        let mut cmd = self.command();
        cmd.args([
            "--vector-storage",
            "u8",
            "--admission",
            "native-l2-u8",
            "--rerank",
            "none",
            "--staged-file",
            "graph.staged",
        ]);
        cmd
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}
fn successful(out: Output) -> Vec<Value> {
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|line| {
            serde_json::from_str(line).unwrap_or_else(|e| panic!("non-JSON stdout {line:?}: {e}"))
        })
        .collect()
}

#[test]
fn anonymous_without_gt_builds_graph_and_reports_null_recall() {
    let f = Fixture::new();
    let lines = successful(
        f.command()
            .args(["--query", "query.bvecs"])
            .output()
            .unwrap(),
    );
    assert_eq!(lines.len(), 206);
    assert_eq!(lines.last().unwrap()["dataset"], "anonymous");
    assert_eq!(lines.last().unwrap()["queries"], 205);
    assert!(lines.last().unwrap()["recall"].is_null());
}

#[test]
fn stdin_native_queries_preserve_calibration_prefix_and_evaluate_optional_gt() {
    let f = Fixture::new();
    let mut child = f
        .native()
        .args([
            "--query",
            "-",
            "--query-format",
            "bvecs",
            "--groundtruth",
            "gt.ivecs",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Write on another thread so pipe backpressure cannot deadlock large inputs.
    let mut stdin = child.stdin.take().unwrap();
    let input = f.query_bytes.clone();
    let writer = std::thread::spawn(move || stdin.write_all(&input).unwrap());
    let lines = successful(child.wait_with_output().unwrap());
    writer.join().unwrap();
    for (i, line) in lines[..205].iter().enumerate() {
        assert_eq!(line["query_id"], i);
        assert_eq!(line["ids"][0], i % 16);
    }
    assert_eq!(lines[205]["recall"], 1.0);
    assert_eq!(lines[205]["ndc_f32"], 0);
    assert!(!std::fs::read_dir(f.root.join("cache")).unwrap().any(|p| p
        .unwrap()
        .path()
        .extension()
        .is_some_and(|e| e == "qds")));
}

#[test]
fn preparation_needs_no_queries_and_cached_pa_graph_needs_no_export() {
    let f = Fixture::new();
    successful(f.native().arg("--prepare-only").output().unwrap());
    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    successful(
        f.native()
            .args(["--query", "query.bvecs"])
            .output()
            .unwrap(),
    );
    // Same graph request with a fresh cache cannot silently switch to Rust.
    std::fs::rename(f.root.join("cache"), f.root.join("saved-cache")).unwrap();
    assert!(!f
        .native()
        .args(["--query", "query.bvecs"])
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn invalid_explicit_gt_is_not_ignored_and_budget_errors_keep_the_breakdown() {
    let f = Fixture::new();
    let out = f
        .native()
        .args(["--query", "query.bvecs", "--groundtruth", "missing.ivecs"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("missing.ivecs"));
    std::fs::write(
        f.root.join("gt.ivecs"),
        [1u32.to_le_bytes(), 0u32.to_le_bytes()].concat(),
    )
    .unwrap();
    let out = f
        .native()
        .args(["--query", "query.bvecs", "--groundtruth", "gt.ivecs"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("ground truth ended"));
    let out = f
        .native()
        .args([
            "--query",
            "query.bvecs",
            "--memory-budget-gib",
            "0.000001",
            "--preflight",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["budget_exceeded"], true);
    assert!(report["components"].as_array().unwrap().len() >= 10);
    let error = String::from_utf8_lossy(&out.stderr);
    for text in [
        "base vectors",
        "graph slab",
        "query buffers",
        "worker scratch",
        "Budget:",
        "excess:",
    ] {
        assert!(error.contains(text), "{error}");
    }
}

#[test]
fn anonymous_config_uses_defaults_and_preset_evaluation_can_be_disabled() {
    let f = Fixture::new();
    let out = f
        .command()
        .args(["--query", "query.bvecs", "--print-config"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let config: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(config["dataset"], "anonymous");
    assert_eq!(config["graph_source"], "parlayann");
    assert_eq!(config["cascade"]["admission"], "L2U8");
    assert!(config["groundtruth"].is_null());

    let out = f
        .command()
        .args(["sift10m", "--no-groundtruth", "--print-config"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let config: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(config["dataset"], "sift10m");
    assert!(config["groundtruth"].is_null());

    let out = f
        .command()
        .args(["--prepare-only", "--preflight"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["sidecar_bytes"], 0);
    assert_eq!(report["query_buffered_rows"], 0);

    let out = f
        .command()
        .args([
            "misspelled-preset",
            "--query",
            "query.bvecs",
            "--print-config",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("unknown dataset shortcut"));
}

// Selection tests use real tiny imports/builds, so preflight predictions can be
// checked against the cache files actually produced by execution.
impl Fixture {
    fn auto_config(&self) {
        std::fs::write(
            self.root.join("auto.yaml"),
            include_str!("fixtures/search.yaml"),
        )
        .unwrap();
    }
    fn auto_command(&self) -> Command {
        let mut command = self.command();
        command.args(["fixture", "--config", "auto.yaml"]);
        command
    }
    fn preflight(&self) -> Value {
        let out = self.auto_command().arg("--preflight").output().unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        serde_json::from_slice(&out.stdout).unwrap()
    }
    fn cache_manifest(&self, report: &Value) -> Value {
        let path = self.root.join(report["cache_path"].as_str().unwrap());
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }
}

#[test]
fn automatic_selection_prefers_pa_cache_then_export_then_memory_build() {
    let f = Fixture::new();
    f.auto_config();
    // Establish a Rust cache first. A PA export still takes precedence over it.
    successful(
        f.auto_command()
            .args(["--graph-source", "rust", "--prepare-only"])
            .output()
            .unwrap(),
    );
    let pa = f.preflight();
    assert_eq!(pa["graph_source_policy"], "auto");
    assert_eq!(pa["graph_source"], "parlayann");
    assert_eq!(pa["graph_action"], "import PA staged export");
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    assert_eq!(f.cache_manifest(&pa)["graph_source"], "parlayann");
    assert!(f.cache_manifest(&pa)["imported_from"]["path"]
        .as_str()
        .unwrap()
        .ends_with("graph.staged"));

    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    let hit = f.preflight();
    assert_eq!(hit["cache_hit"], true);
    assert_eq!(hit["cache_path"], pa["cache_path"]);
    // Auto and explicit PA use the same physical cache and namespace.
    successful(
        f.auto_command()
            .args(["--graph-source", "parlayann", "--prepare-only"])
            .output()
            .unwrap(),
    );

    let path = f.root.join(pa["cache_path"].as_str().unwrap());
    std::fs::remove_file(path).unwrap();
    let built = f.preflight();
    assert_eq!(built["graph_source"], "parlayann");
    assert_eq!(built["cache_hit"], false);
    assert_eq!(built["graph_action"], "build ParlayANN graph");
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    assert_eq!(
        f.cache_manifest(&built)["builder_recipe"],
        "parlay-memory-v1-unquantized-two-pass"
    );
    assert!(f.cache_manifest(&built)["imported_from"].is_null());

    std::fs::remove_dir_all(f.root.join("cache")).unwrap();
    assert_eq!(f.preflight()["graph_action"], "build ParlayANN graph");
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    assert_eq!(f.preflight()["cache_hit"], true);
}

#[test]
fn explicit_missing_exports_remain_errors_and_native_u8_builds() {
    let f = Fixture::new();
    f.auto_config();
    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    for args in [
        vec!["--staged-file", "graph.staged"],
        vec!["--graph-source", "auto", "--staged-file", "graph.staged"],
    ] {
        let out = f
            .auto_command()
            .args(args)
            .arg("--prepare-only")
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("No Rust graph was substituted"));
    }
    let out = f
        .auto_command()
        .env("ORION_STAGED_FILE", "missing.staged")
        .arg("--prepare-only")
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(!f.root.join("cache").exists());

    // Offline native-u8 planning now identifies the in-memory PA build.
    let out = f
        .auto_command()
        .args([
            "--vector-storage",
            "u8",
            "--admission",
            "native-l2-u8",
            "--rerank",
            "none",
            "--preflight",
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["graph_source"], "parlayann");
    assert_eq!(report["graph_action"], "build ParlayANN graph");
}

#[test]
fn auto_does_not_hide_invalid_exports_or_incomplete_caches() {
    let f = Fixture::new();
    f.auto_config();
    // Honor the historical suffix in both selection and preflight.
    std::fs::rename(
        f.root.join("graph.staged"),
        f.root.join("graph_pct60.staged"),
    )
    .unwrap();
    assert_eq!(f.preflight()["graph_action"], "import PA staged export");
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    let report = f.preflight();
    let digest = f.cache_manifest(&report)["graph_sha256"]
        .as_str()
        .unwrap()
        .to_owned();
    std::fs::remove_file(
        f.root
            .join("cache.objects")
            .join(format!("{digest}.pgraph")),
    )
    .unwrap();
    assert!(!f
        .auto_command()
        .arg("--preflight")
        .output()
        .unwrap()
        .status
        .success());
    std::fs::remove_dir_all(f.root.join("cache")).unwrap();
    std::fs::write(f.root.join("graph_pct60.staged"), b"bad export").unwrap();
    assert!(!f
        .auto_command()
        .arg("--preflight")
        .output()
        .unwrap()
        .status
        .success());
    // Explicit Rust ignores an unrelated broken PA export.
    successful(
        f.auto_command()
            .args(["--graph-source", "rust", "--prepare-only"])
            .output()
            .unwrap(),
    );
}

#[test]
fn cache_source_is_required_and_validated() {
    let f = Fixture::new();
    f.auto_config();
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    let report = f.preflight();
    let path = f.root.join(report["cache_path"].as_str().unwrap());
    let mut manifest = f.cache_manifest(&report);
    manifest.as_object_mut().unwrap().remove("graph_source");
    std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(!f
        .auto_command()
        .arg("--preflight")
        .output()
        .unwrap()
        .status
        .success());
    manifest["graph_source"] = "rust".into();
    std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(!f
        .auto_command()
        .arg("--preflight")
        .output()
        .unwrap()
        .status
        .success());
}

#[test]
fn auto_uses_source_specific_default_directories_and_yaml_can_be_strict() {
    let f = Fixture::new();
    f.auto_config();
    let run = |extra: &[&str]| {
        Command::new(env!("CARGO_BIN_EXE_orion"))
            .current_dir(&f.root)
            .env_remove("ORION_GRAPH")
            .env_remove("ORION_STAGED_FILE")
            .env_remove("ORION_SWEEP_CONFIG")
            .args([
                "fixture",
                "--config",
                "auto.yaml",
                "--graph-degree",
                "15",
                "--max-extra",
                "0",
                "--k",
                "1",
                "--search-l",
                "16",
                "--threads",
                "2",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let report: Value = serde_json::from_slice(&run(&["--preflight"]).stdout).unwrap();
    assert!(report["cache_path"]
        .as_str()
        .unwrap()
        .starts_with("cache/orion_parlayann/"));
    successful(run(&["--prepare-only"]));
    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&run(&["--preflight"]).stdout).unwrap()["cache_hit"],
        true
    );
    std::fs::remove_dir_all(f.root.join("cache")).unwrap();
    let report: Value = serde_json::from_slice(&run(&["--preflight"]).stdout).unwrap();
    assert!(report["cache_path"]
        .as_str()
        .unwrap()
        .starts_with("cache/orion_parlayann/"));

    let yaml_path = f.root.join("auto.yaml");
    let mut yaml: serde_yaml::Value =
        serde_yaml::from_slice(&std::fs::read(&yaml_path).unwrap()).unwrap();
    yaml["datasets"]["fixture"]["base_graph"]["source"] = "parlayann".into();
    std::fs::write(yaml_path, serde_yaml::to_string(&yaml).unwrap()).unwrap();
    successful(run(&["--prepare-only"]));
    successful(run(&["--graph-source", "auto", "--prepare-only"]));
}
