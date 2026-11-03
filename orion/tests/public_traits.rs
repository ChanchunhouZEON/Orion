//! Compile-time contracts for root exports, plus bounded diagnostic formatting.
use orion_ann::*;
use std::fmt::Debug;

fn debug<T: Debug>() {}
fn shareable<T: Send + Sync>() {}
fn config<T: Debug + Clone + PartialEq + Default>() {}

#[test]
fn public_facade_trait_contracts() {
    config::<BuildConfig>();
    config::<BuildBackend>();
    config::<BuildMetric>();
    config::<CalibrationConfig>();
    config::<DynamicSearchConfig>();
    debug::<Orion>();
    debug::<Orion<u8>>();
    debug::<StaticOrion<128, f32>>();
    debug::<NaiveOrion<128>>();
    debug::<DynamicCascade<'static>>();
    debug::<DiskANNBuildResult>();
    debug::<PhasedGraph>();
    debug::<PaddedDataset<f32>>();
    debug::<PaddedLayout>();
    debug::<FixedChunkPQTable>();
    debug::<CalibrationDiagnostics>();
    debug::<CalibratedParams>();
    debug::<SearchConvergenceChecker>();
    debug::<SearchProfile>();
    debug::<SearchProfileStats>();
    debug::<BeamSearchConfig>();
    debug::<BeamSearchResult>();
    debug::<IoLimiter>();
    debug::<Error>();
    debug::<ErrorCode>();
    debug::<ErrorReport>();
    shareable::<Orion>();
    shareable::<Orion<u8>>();
    shareable::<Error>();
}

#[test]
fn debug_does_not_require_storage_debug_or_dump_vector_payloads() {
    #[derive(Default)]
    struct Opaque;
    let index: CompressedIndex<Opaque, Opaque> = CompressedIndex::from_phased_graph(
        Opaque,
        PhasedGraph::try_empty(2, 1, 0).unwrap(),
        0,
        None,
        None,
    );
    let summary = format!("{index:#?}");
    assert!(summary.contains("num_nodes: 2"));
    assert!(summary.len() < 1000);

    let layout = PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(13).unwrap();
    let data = PaddedDataset::from_packed(&vec![12345.0_f32; 10_000 * 13], layout).unwrap();
    let summary = format!("{data:#?}");
    assert!(summary.contains("10000"));
    assert!(!summary.contains("12345"));
    assert!(summary.len() < 500);
    let allocation = data.into_allocation();
    let summary = format!("{allocation:#?}");
    assert!(!summary.contains("12345"));
    assert!(summary.len() < 500);
}

#[test]
fn stable_error_codes_support_display_hashing_and_wire_output() {
    let codes = std::collections::HashSet::from([ErrorCode::InvalidArgument, ErrorCode::Backend]);
    assert_eq!(codes.len(), 2);
    let code = ErrorCode::InvalidArgument;
    assert_eq!(code.to_string(), code.as_str());
    assert_eq!(serde_json::to_value(code).unwrap(), code.as_str());
    let error = Orion::<f32>::build_from_slice(&[], 13, BuildConfig::default()).unwrap_err();
    assert_eq!(error.code(), code);
}
