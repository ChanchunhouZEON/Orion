/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_diagnostics::*;

fn metadata() -> TraceMetadata {
    TraceMetadata {
        dimension: 3,
        points: 10,
        k: 2,
        beam: 4,
        adaptive: false,
        metric: "L2".into(),
        storage: "U8".into(),
        prefilter: "None".into(),
        admission: "NativeL2U8".into(),
        rerank: "None".into(),
    }
}
fn trace() -> SearchTrace {
    let mut recorder = TraceRecorder::new(metadata(), TraceLimits::default()).unwrap();
    recorder.record(1, || TraceEvent::Entry {
        candidate: Candidate {
            id: 0,
            distance: f32::INFINITY.into(),
        },
    });
    recorder.record(0, || TraceEvent::Rerank { candidates: 2 });
    recorder.finish(vec![0, 1]).unwrap()
}
#[test]
fn json_nonfinite_and_replay_roundtrip() {
    let original = trace();
    let json = original.to_json().unwrap();
    assert!(json.contains("positive_infinity"));
    let mut replay = TraceReplay::from_json(&json).unwrap();
    assert_eq!(replay.trace(), &original);
    assert_eq!(replay.next_events(1).unwrap(), &original.events[..1]);
    assert_eq!(replay.position(), 1);
    replay.seek(0).unwrap();
    assert_eq!(replay.next_events(1024).unwrap(), original.events);
    assert!(replay.next_events(1).unwrap().is_empty());
    assert!(replay.seek(3).is_err());
    assert_eq!(replay.position(), 2);
    assert!(replay.next_events(0).is_err());
    assert!(replay.next_events(1025).is_err());
}
#[test]
fn limits_stop_recording_without_calling_payload_builder() {
    let mut recorder = TraceRecorder::new(
        metadata(),
        TraceLimits {
            max_events: 1,
            max_items: 3,
        },
    )
    .unwrap();
    recorder.record(1, || TraceEvent::Entry {
        candidate: Candidate {
            id: 0,
            distance: 0.0.into(),
        },
    });
    recorder.record(0, || {
        panic!("event budget must be checked before allocation")
    });
    recorder.record(0, || panic!("truncation must retain a prefix"));
    recorder.early_exit();
    let trace = recorder.finish(vec![1, 2]).unwrap();
    assert!(trace.truncated);
    assert_eq!(trace.events.len(), 1);
    assert_eq!(trace.stop_reason, StopReason::EarlyExit);
    assert_eq!(trace.result_ids, [1, 2]);

    let mut recorder = TraceRecorder::new(
        metadata(),
        TraceLimits {
            max_events: 10,
            max_items: 2,
        },
    )
    .unwrap();
    recorder.record(1, || {
        panic!("item budget must be checked before allocation")
    });
    assert!(recorder.finish(vec![0, 1]).unwrap().events.is_empty());
}
#[test]
fn malformed_documents_and_bad_budgets_are_rejected() {
    let mut bad = trace();
    bad.schema_version += 1;
    assert!(SearchTrace::from_json(&serde_json::to_string(&bad).unwrap()).is_err());
    bad = trace();
    bad.result_ids = vec![10];
    assert!(bad.validate().is_err());
    bad = trace();
    bad.limits.max_events = 1;
    assert!(bad.validate().is_err());
    bad = trace();
    bad.events.clear();
    assert!(bad.validate().is_err());
    bad = trace();
    bad.metadata.dimension = 0;
    assert!(bad.validate().is_err());
    bad = trace();
    bad.events[0] = TraceEvent::Entry {
        candidate: Candidate {
            id: 99,
            distance: 1.0.into(),
        },
    };
    assert!(bad.validate().is_err());
    assert!(
        TraceRecorder::new(
            metadata(),
            TraceLimits {
                max_events: 9000,
                max_items: 3
            }
        )
        .is_err()
    );
    assert!(
        TraceRecorder::new(
            metadata(),
            TraceLimits {
                max_events: 1,
                max_items: 1
            }
        )
        .is_err()
    );
    assert!(SearchTrace::from_json(&" ".repeat(MAX_TRACE_JSON_BYTES + 1)).is_err());
    assert!(SearchTrace::from_json("{}").is_err());
}
