#![cfg(feature = "parlay")]
use orion_ann_runtime::diagnostics::{
    Candidate, Phase, Score, SearchTrace, TraceEvent, TraceLimits, TraceReplay,
};
use orion_ann_runtime::{BaseInput, CascadeConfig, ErrorCode, GraphConfig, Index, IndexOptions};
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(std::path::PathBuf);
impl Temp {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "orion-trace-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn diagnostics_uses_the_native_public_error_type() {
    let error = SearchTrace::from_json("{}").unwrap_err();
    let report = error.report();
    let same_error: orion_ann::Error = error;
    assert_eq!(same_error.report(), report);
}

#[test]
fn f32_u8_trace_and_truncated_trace_match_normal_search() {
    const DIM: usize = 13;
    let base: Vec<u8> = (0..96 * DIM)
        .map(|i| {
            let row = i / DIM;
            let col = i % DIM;
            ((row * 19 + col * 13 + row * col * 7) % 251) as u8
        })
        .collect();
    for native in [false, true] {
        let dir = Temp::new();
        let values: Vec<f32> = base.iter().map(|&v| v as f32).collect();
        let input = if native {
            BaseInput::from_u8(&base, DIM)
        } else {
            BaseInput::from_f32(&values, DIM)
        }
        .unwrap();
        let mut index = Index::new(
            input,
            IndexOptions {
                threads: 2,
                cache_dir: dir.0.clone(),
                graph: GraphConfig {
                    degree: 12,
                    build_l: 24,
                    max_extra: 4,
                    ..Default::default()
                },
                cascade: if native {
                    CascadeConfig::default()
                } else {
                    CascadeConfig {
                        prefilter: Some("jl".into()),
                        admission: Some("l2-u8".into()),
                        rerank: Some("f32".into()),
                    }
                },
                ..Default::default()
            },
        )
        .unwrap();
        let query = values[3 * DIM..4 * DIM].to_vec();
        let err = index
            .trace_search(query.clone(), 5, 16, true, TraceLimits::default())
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::InvalidState);
        index
            .calibrate(values[..24 * DIM].to_vec(), 5, Some(16))
            .unwrap();
        for adaptive in [false, true] {
            for beam in [8, 16, 32] {
                let result = index.search(query.clone(), 5, beam, adaptive).unwrap();
                let expected = &result.ids[..result.counts[0] as usize];
                let trace = index
                    .trace_search(query.clone(), 5, beam, adaptive, TraceLimits::default())
                    .unwrap();
                assert_eq!(trace.result_ids, expected);
                assert!(!trace.truncated);
                assert!(matches!(
                    trace.events.first(),
                    Some(TraceEvent::Entry { .. })
                ));
                assert!(trace.events.iter().any(|e| matches!(
                    e,
                    TraceEvent::Expansion {
                        phase: Phase::Peel,
                        ..
                    }
                )));
                assert!(trace
                    .events
                    .iter()
                    .any(|e| matches!(e, TraceEvent::Beam { .. })));
                for event in &trace.events {
                    if native {
                        let check = |candidate: &Candidate| {
                            let row = candidate.id as usize * DIM;
                            let distance: f32 = values[row..row + DIM]
                                .iter()
                                .zip(&query)
                                .map(|(x, y)| (x - y) * (x - y))
                                .sum();
                            assert_eq!(candidate.distance, Score::Finite(distance));
                        };
                        match event {
                            TraceEvent::Entry { candidate } => check(candidate),
                            TraceEvent::Admission { admitted, .. } => {
                                admitted.iter().for_each(check)
                            }
                            _ => (),
                        }
                    }
                }
                let encoded = trace.to_json().unwrap();
                assert_eq!(SearchTrace::from_json(&encoded).unwrap(), trace);
                let mut replay = TraceReplay::from_json(&encoded).unwrap();
                assert_eq!(replay.next_events(1024).unwrap(), trace.events);
                let prefix = index
                    .trace_search(
                        query.clone(),
                        5,
                        beam,
                        adaptive,
                        TraceLimits {
                            max_events: 1,
                            max_items: 6,
                        },
                    )
                    .unwrap();
                assert!(prefix.truncated);
                assert_eq!(prefix.result_ids, expected);
                assert_eq!(prefix.events, trace.events[..1]);
            }
        }
        assert!(index
            .trace_search(vec![0.0], 5, 16, false, TraceLimits::default())
            .is_err());
        assert!(index
            .trace_search(
                query.clone(),
                5,
                16,
                false,
                TraceLimits {
                    max_events: 1,
                    max_items: 4
                }
            )
            .is_err());
        assert!(index.search(query, 5, 16, false).is_ok());
    }
}
