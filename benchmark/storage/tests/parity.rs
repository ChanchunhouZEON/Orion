/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use std::{path::Path, process::Command};
fn run(args: &[&str]) {
    let output = Command::new(env!("CARGO_BIN_EXE_orion-storage-bench"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}
#[test]
fn cascades_and_concurrent_storage_arms_return_identical_neighbors() {
    let root = tempfile::tempdir().unwrap();
    run(&["fixture", "--directory", text(root.path())]);
    for dataset in ["sift", "gist"] {
        let config = root.path().join(format!("{dataset}.json"));
        let mut expected = None;
        for concurrency in ["1", "8"] {
            for arm in ["mmap", "mmap-driver", "ssd"] {
                let output = root
                    .path()
                    .join(format!("{dataset}-{concurrency}-{arm}.jsonl"));
                run(&[
                    "run",
                    "--dataset",
                    text(&config),
                    "--calibration",
                    text(&root.path().join("calibration.json")),
                    "--work",
                    text(&root.path().join("work")),
                    "--output",
                    text(&output),
                    "--arm",
                    arm,
                    "--concurrency",
                    concurrency,
                    "--seconds",
                    "0",
                    "--page-cache-mib",
                    "1",
                ]);
                let rows: Vec<serde_json::Value> = std::fs::read_to_string(output)
                    .unwrap()
                    .lines()
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect();
                assert_eq!(rows.len(), 2);
                let signature: Vec<_> = rows
                    .iter()
                    .map(|r| {
                        (
                            r["L"].clone(),
                            r["validation_sha256"].clone(),
                            r["recall"].clone(),
                        )
                    })
                    .collect();
                if let Some(previous) = &expected {
                    assert_eq!(&signature, previous);
                } else {
                    expected = Some(signature);
                }
                for r in rows {
                    assert_eq!(r["queries"], 32);
                    assert!(r["search_seconds"].as_f64().unwrap() > 0.);
                    assert!(r["p50_us"].as_f64().unwrap() <= r["p95_us"].as_f64().unwrap());
                    assert!(r["p95_us"].as_f64().unwrap() <= r["p99_us"].as_f64().unwrap());
                    assert_eq!(r["undersampled"], true);
                }
            }
        }
    }
}
#[test]
fn mismatched_query_shape_is_rejected_before_measurement() {
    let root = tempfile::tempdir().unwrap();
    run(&["fixture", "--directory", text(root.path())]);
    // A complete file with the wrong row width, not just a truncated file.
    std::fs::write(
        root.path().join("query.fvecs"),
        [1u32.to_le_bytes(), 0u32.to_le_bytes()].concat(),
    )
    .unwrap();
    let output = root.path().join("invalid.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_orion-storage-bench"))
        .args([
            "run",
            "--dataset",
            text(&root.path().join("sift.json")),
            "--calibration",
            text(&root.path().join("calibration.json")),
            "--work",
            text(&root.path().join("work")),
            "--output",
            text(&output),
            "--arm",
            "ssd",
        ])
        .output()
        .unwrap();
    assert!(!status.status.success());
    assert!(!output.exists());
}

#[test]
fn load_only_records_stages_and_skips_search() {
    let root = tempfile::tempdir().unwrap();
    run(&["fixture", "--directory", text(root.path())]);
    for arm in ["mmap", "mmap-driver", "ssd"] {
        let path = root.path().join(format!("{arm}.jsonl"));
        let output = Command::new(env!("CARGO_BIN_EXE_orion-storage-bench"))
            .args([
                "run",
                "--dataset",
                text(&root.path().join("gist.json")),
                "--calibration",
                text(&root.path().join("calibration.json")),
                "--work",
                text(&root.path().join("work")),
                "--output",
                text(&path),
                "--arm",
                arm,
                "--load-only",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["kind"], "load_validation");
        assert_eq!(rows[0]["base_placement"], "Mapped");
        assert_eq!(rows[0]["cache_policy"], "writeback_then_reclaim");
        assert_eq!(rows[0]["reclaim_window_bytes"], 32 * 1024 * 1024);
        assert!(rows[0].get("qps").is_none());
        let events: Vec<serde_json::Value> = String::from_utf8_lossy(&output.stderr)
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        for phase in [
            "allocation.begin",
            "allocation.complete",
            "read.begin",
            "read.complete",
            "flush.begin",
            "flush.complete",
        ] {
            assert!(
                events
                    .iter()
                    .any(|e| e["object"] == "graph" && e["phase"] == phase),
                "missing {arm} {phase}"
            );
        }
        let validation = if arm == "mmap" {
            "validation.complete"
        } else {
            "header_validation.complete"
        };
        assert!(events.iter().any(|e| e["phase"] == validation));
        assert!(events.iter().any(|e| e["phase"] == "released"));
        assert_eq!(
            std::fs::read_dir(root.path().join("work")).unwrap().count(),
            0
        );
    }
}
