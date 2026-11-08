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
fn production_defaults_infer_native_bytes_and_reject_presets() {
    let f = Fixture::new();
    let out = f
        .command()
        .args(["--query", "query.bvecs", "--print-config"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let config: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(config["dataset"], "anonymous");
    assert_eq!(config["cascade"]["admission"], "NativeL2U8");
    assert!(config["groundtruth"].is_null());
    for extra in [
        vec!["sift10m"],
        vec!["anonymous"],
        vec!["--config", "anything.yaml"],
    ] {
        assert!(!f
            .command()
            .args(extra)
            .arg("--prepare-only")
            .output()
            .unwrap()
            .status
            .success());
    }
    // No environment/YAML parsing occurs in the production executable.
    let poisoned = f
        .command()
        .env("ORION_SWEEP_CONFIG", "missing.yaml")
        .env("ORION_GRAPH", "invalid")
        .env("ORION_STAGED_FILE", "missing.staged")
        .args(["--query", "query.bvecs", "--print-config"])
        .output()
        .unwrap();
    assert!(poisoned.status.success());
    assert_eq!(
        serde_json::from_slice::<Value>(&poisoned.stdout).unwrap(),
        config
    );
}

// Selection tests use real tiny imports/builds, so preflight predictions can be
// checked against the cache files actually produced by execution.
impl Fixture {
    fn auto_command(&self) -> Command {
        let mut command = self.command();
        command.args(["--vector-storage", "f32"]);
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
fn automatic_selection_reuses_import_then_builds_without_export() {
    let f = Fixture::new();
    successful(
        f.auto_command()
            .args(["--staged-file", "graph.staged", "--prepare-only"])
            .output()
            .unwrap(),
    );
    let pa = f.preflight();
    assert_eq!(pa["cache_hit"], true);
    assert_eq!(f.cache_manifest(&pa)["graph_source"], "parlayann");
    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    assert_eq!(f.preflight()["cache_hit"], true);
    successful(
        f.auto_command()
            .args(["--graph-source", "parlayann", "--prepare-only"])
            .output()
            .unwrap(),
    );
    std::fs::remove_file(f.root.join(pa["cache_path"].as_str().unwrap())).unwrap();
    let built = f.preflight();
    assert_eq!(built["graph_action"], "build ParlayANN graph");
    successful(f.auto_command().arg("--prepare-only").output().unwrap());
    assert!(f.cache_manifest(&built)["imported_from"].is_null());
}

#[test]
fn explicit_missing_exports_fail_and_native_u8_builds_without_presets() {
    let f = Fixture::new();
    std::fs::remove_file(f.root.join("graph.staged")).unwrap();
    let out = f.native().arg("--prepare-only").output().unwrap();
    assert!(!out.status.success());
    assert!(!f.root.join("cache").exists());
    successful(f.command().arg("--prepare-only").output().unwrap());
}

#[test]
fn auto_does_not_hide_incomplete_cache_objects() {
    let f = Fixture::new();
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
}

#[test]
fn cache_source_is_required_and_validated() {
    let f = Fixture::new();
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
fn cli_uses_the_service_cache_directory_and_strict_graph_parameters() {
    let f = Fixture::new();
    let out = Command::new(env!("CARGO_BIN_EXE_orion"))
        .current_dir(&f.root)
        .args(["--base", "base.bvecs", "--preflight"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(report["cache_path"]
        .as_str()
        .unwrap()
        .starts_with("cache/orion-ann/"));
    for flags in [["--alpha", "0.98"], ["--build-l", "2"]] {
        assert!(!f
            .auto_command()
            .args(flags)
            .arg("--prepare-only")
            .output()
            .unwrap()
            .status
            .success());
    }
}

#[test]
fn snapshot_cli_restores_calibration_and_matches_rust_search() {
    use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions};
    let f = Fixture::new();
    let mut index = Index::new(
        BaseInput::from_file(f.root.join("base.bvecs")),
        IndexOptions {
            threads: 1,
            cache_dir: f.root.join("snapshot-build"),
            graph: GraphConfig {
                degree: 15,
                build_l: 24,
                max_extra: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let queries: Vec<f32> = f
        .query_bytes
        .chunks_exact(132)
        .flat_map(|row| row[4..].iter().map(|v| f32::from(*v)))
        .collect();
    index
        .calibrate(queries[..128 * 16].to_vec(), 1, Some(24))
        .unwrap();
    let expected = index
        .search(
            QueryBatch::from_f32(&queries, 128).unwrap(),
            &SearchOptions::new(1, 24),
        )
        .unwrap();
    let snapshot = f.root.join("snapshot");
    index.save(&snapshot).unwrap();
    let command = || {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_orion"));
        cmd.current_dir(&f.root)
            .env_remove("ORION_SWEEP_CONFIG")
            .env_remove("ORION_GRAPH")
            .env_remove("ORION_STAGED_FILE");
        cmd.args([
            "--snapshot",
            "snapshot",
            "--base",
            "base.bvecs",
            "--query",
            "query.bvecs",
            "--k",
            "1",
            "--search-l",
            "24",
            "--threads",
            "1",
            "--query-batch-size",
            "7",
        ]);
        cmd
    };
    let lines = successful(
        command()
            .args(["--groundtruth", "gt.ivecs"])
            .output()
            .unwrap(),
    );
    for (line, ids) in lines.iter().zip(expected.rows()) {
        assert_eq!(line["ids"], serde_json::json!(ids));
    }
    assert_eq!(lines.len(), 206);
    assert_eq!(lines.last().unwrap()["dataset"], "snapshot");
    assert_eq!(lines.last().unwrap()["recall"], 1.);
    let summary = successful(
        command()
            .args(["--output", "results.jsonl"])
            .output()
            .unwrap(),
    );
    assert_eq!(summary.len(), 1);
    let output = std::fs::read_to_string(f.root.join("results.jsonl")).unwrap();
    assert_eq!(output.lines().count(), 205);
    let failed = command()
        .args(["--output", "results.jsonl"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert_eq!(
        std::fs::read_to_string(f.root.join("results.jsonl")).unwrap(),
        output
    );
    // Restore a snapshot without calibration: CLI must surface InvalidState,
    // rather than quietly recalibrating or searching a different recipe.
    let uncalibrated = Index::new(
        BaseInput::from_file(f.root.join("base.bvecs")),
        IndexOptions {
            threads: 1,
            cache_dir: f.root.join("uncalibrated-cache"),
            graph: GraphConfig {
                degree: 15,
                build_l: 24,
                max_extra: 0,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    std::fs::rename(&snapshot, f.root.join("calibrated-snapshot")).unwrap();
    uncalibrated.save(&snapshot).unwrap();
    let failed = command().args(["--error-format", "json"]).output().unwrap();
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("no saved calibration"));
}
