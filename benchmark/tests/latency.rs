/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use benchmark::latency::{
    config::Protocol,
    evaluation::{evaluate, recall_row, validate_truth},
    report::median,
    runner::measure,
    statistics::Distribution,
};
use orion_ann::{
    BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchMetric, SearchOptions,
};

fn protocol() -> Protocol {
    Protocol {
        min_requests: 128,
        min_seconds: 0.0,
        max_seconds: 10.0,
        warmup_requests: 16,
        warmup_seconds: 0.0,
        cooldown_seconds: 0.0,
        ..Default::default()
    }
}

#[test]
fn histogram_merges_samples_instead_of_averaging_quantiles() {
    let mut a = Distribution::new().unwrap();
    let mut b = Distribution::new().unwrap();
    assert!(a.summary().is_none());
    for _ in 0..1000 {
        a.record(10_000).unwrap();
    }
    for _ in 0..10 {
        b.record(1_000_000).unwrap();
    }
    a.merge(&b).unwrap();
    let s = a.summary().unwrap();
    assert_eq!(s.samples, 1010);
    assert!((s.mean_us - 20_000.0 / 1010.0).abs() < 1e-10);
    assert!((s.p99_us - 10.0).abs() < 0.02);
    assert_eq!(s.max_us, 1000.0);
    assert_eq!(a.buckets().iter().map(|b| b.count).sum::<u64>(), 1010);
    assert!(a.record(3_600_000_000_001).is_err());
    assert_eq!(a.summary().unwrap().samples, 1010);
    a.record(0).unwrap();
    assert_eq!(a.summary().unwrap().zero_clock_samples, 1);
}

#[test]
fn known_distribution_percentiles_and_median() {
    let mut d = Distribution::new().unwrap();
    for value in 1..=1000 {
        d.record(value * 1000).unwrap();
    }
    let s = d.summary().unwrap();
    for (actual, expected) in [(s.p50_us, 500.0), (s.p95_us, 950.0), (s.p99_us, 990.0)] {
        assert!((actual - expected).abs() < expected * 0.001);
    }
    assert_eq!(s.mean_us, 500.5);
    assert_eq!(median(&[]), None);
    assert_eq!(median(&[9.0, 1.0, 3.0]), Some(3.0));
    assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), Some(2.5));
}

#[test]
fn recall_respects_partial_results_and_rejects_bad_truth() {
    assert_eq!(recall_row(&[1], &[1, 2, 3], 3).unwrap(), 1.0 / 3.0);
    assert_eq!(recall_row(&[1, 1, 4], &[1, 2, 3], 3).unwrap(), 1.0 / 3.0);
    assert_eq!(recall_row(&[], &[1, 2], 2).unwrap(), 0.0);
    assert!(recall_row(&[1], &[1, 1], 2).is_err());
    assert!(validate_truth(&[vec![1, 2]], 2, 2).is_err());
    assert!(validate_truth(&[vec![1, 1]], 1, 2).is_err());
}

#[test]
fn protocol_rejects_ambiguous_or_unbounded_runs() {
    assert!(protocol().validate().is_ok());
    for p in [
        Protocol {
            min_seconds: f64::NAN,
            ..protocol()
        },
        Protocol {
            max_seconds: 0.0,
            ..protocol()
        },
        Protocol {
            min_seconds: 20.0,
            ..protocol()
        },
        Protocol {
            concurrency: vec![8, 8],
            ..protocol()
        },
        Protocol {
            concurrency: vec![0],
            ..protocol()
        },
        Protocol {
            concurrency: vec![],
            ..protocol()
        },
        Protocol {
            rounds: 0,
            ..protocol()
        },
        Protocol {
            min_requests: 0,
            ..protocol()
        },
    ] {
        assert!(p.validate().is_err());
    }
}

#[test]
fn requests_match_batch_for_all_metrics_and_byte_storage() {
    let temp = tempfile::tempdir().unwrap();
    let dimension = 7;
    let bytes: Vec<u8> = (0..96 * dimension)
        .map(|i| ((i * 19 + i / dimension * 11) % 251 + 1) as u8)
        .collect();
    let floats: Vec<f32> = bytes.iter().map(|&v| v as f32).collect();
    for (m, metric) in [
        SearchMetric::SquaredL2,
        SearchMetric::InnerProduct,
        SearchMetric::Cosine,
    ]
    .into_iter()
    .enumerate()
    {
        for native in [false, true] {
            let options = IndexOptions {
                metric,
                threads: 2,
                auto_pin: false,
                cache_dir: temp.path().join(format!("{m}-{native}")),
                graph: GraphConfig {
                    degree: 8,
                    build_l: 16,
                    max_extra: 4,
                    ..Default::default()
                },
                ..Default::default()
            };
            let base = if native {
                BaseInput::from_u8(&bytes, dimension)
            } else {
                BaseInput::from_f32(&floats, dimension)
            }
            .unwrap();
            let mut index =
                Index::new_with_ids(base, options, (0..96).map(|x| 1000 + x).collect()).unwrap();
            // A byte query against an f32 base exercises the unified query path too.
            let batch = QueryBatch::from_u8(&bytes[..16 * dimension], dimension).unwrap();
            index.calibrate_batch(batch, 3, Some(16)).unwrap();
            let rows: Vec<_> = bytes[..16 * dimension]
                .chunks_exact(dimension)
                .map(|r| QueryBatch::from_u8(r, dimension).unwrap())
                .collect();
            let order: Vec<_> = (0..rows.len()).rev().collect();
            for scores in [false, true] {
                let options = SearchOptions::new(3, 16).with_scores(scores);
                let expected = index.search(batch, &options).unwrap();
                let truth: Vec<_> = expected.rows().map(|r| r.to_vec()).collect();
                for concurrency in [1, 8] {
                    let e = evaluate(
                        &index,
                        &rows,
                        &options,
                        &expected,
                        Some(&truth),
                        concurrency,
                    )
                    .unwrap();
                    assert_eq!(e.recall, Some(1.0));
                    let measured = measure(
                        &index,
                        &rows,
                        &options,
                        &order,
                        concurrency,
                        &protocol(),
                        true,
                    )
                    .unwrap();
                    assert_eq!(measured.requests, 128);
                    assert_eq!(measured.latency.unwrap().samples, 128);
                    assert!(measured.minimum_reached);
                    let control = measure(
                        &index,
                        &rows,
                        &options,
                        &order,
                        concurrency,
                        &protocol(),
                        false,
                    )
                    .unwrap();
                    assert_eq!(control.requests, 128);
                    assert!(control.latency.is_none());
                }
            }
            let invalid = SearchOptions::new(4, 16); // k=4 was not calibrated.
            assert!(measure(&index, &rows, &invalid, &order, 8, &protocol(), true).is_err());
            let no_warmup = Protocol {
                warmup_requests: 0,
                ..protocol()
            };
            assert!(measure(&index, &rows, &invalid, &order, 8, &no_warmup, true).is_err());
            assert!(measure(
                &index,
                &rows,
                &SearchOptions::new(3, 16),
                &vec![0; rows.len()],
                1,
                &protocol(),
                true
            )
            .is_err());
        }
    }
}

#[test]
fn executable_uses_relative_inputs_and_preserves_existing_results() {
    use std::{io::Write, process::Command};
    let dir = tempfile::tempdir().unwrap();
    let mut base = std::fs::File::create(dir.path().join("base.u8bin")).unwrap();
    for word in [4u32, 7] {
        base.write_all(&word.to_le_bytes()).unwrap();
    }
    base.write_all(&(1..=28).collect::<Vec<u8>>()).unwrap();
    let mut graph = std::fs::File::create(dir.path().join("graph.staged")).unwrap();
    for word in [0x53544147u32, 3, 4, 3, 0, 0] {
        graph.write_all(&word.to_le_bytes()).unwrap();
    }
    for node in 0..4 {
        for word in [2u32, 1, 0] {
            graph.write_all(&word.to_le_bytes()).unwrap();
        }
        for delta in 1..4 {
            graph
                .write_all(&((node + delta) % 4u32).to_le_bytes())
                .unwrap();
        }
    }
    let config = serde_json::json!({
        "dataset":"tiny", "base":"base.u8bin", "query":"base.u8bin",
        "index":{"threads":2,"auto_pin":false,"cache_dir":"cache",
            "graph":{"source":"parlayann","staged_file":"graph.staged","degree":3,"build_l":8,"max_extra":0}},
        "k":2,"beams":[3,4],"scores":true,"calibration_samples":4,"calibration_l":4,
        "protocol":{"rounds":1,"concurrency":[1,8],"min_requests":16,"min_seconds":0.0,
            "max_seconds":5.0,"warmup_requests":8,"warmup_seconds":0.0,"cooldown_seconds":0.0}
    });
    let path = dir.path().join("experiment.json");
    let mut validated: benchmark::latency::config::Experiment =
        serde_json::from_value(config.clone()).unwrap();
    assert!(validated.validate().is_ok());
    for source in [
        orion_ann::GraphSourcePolicy::Auto,
        orion_ann::GraphSourcePolicy::Rust,
    ] {
        validated.index.graph.source = source;
        assert!(validated.validate().is_err());
    }
    validated.snapshot = Some("snapshot".into());
    assert!(validated.validate().is_ok());
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let output = dir.path().join("results");
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_latency-sweep"))
            .args([
                "--config",
                path.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };
    let result = run();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let bytes = std::fs::read(output.join("results.json")).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(report["status"], "complete");
    assert_eq!(report["query_element"], "u8");
    assert!(report.get("checkout_git_head_at_run").is_some());
    assert_eq!(report["points"].as_array().unwrap().len(), 4);
    for point in report["points"].as_array().unwrap() {
        assert_eq!(point["pooled"]["samples"], 16);
        assert!(point["evaluation"]["recall"].is_null());
        assert_eq!(point["evaluation"]["batch_parity"], true);
    }
    assert!(!run().status.success());
    assert_eq!(std::fs::read(output.join("results.json")).unwrap(), bytes);
}
