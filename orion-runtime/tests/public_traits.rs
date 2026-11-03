use orion_ann_runtime::{
    BaseInput, Calibration, CascadeConfig, GraphConfig, Index, IndexOptions, SearchResult,
};
use std::fmt::Debug;

#[test]
fn public_service_types_are_inspectable_and_movable() {
    fn debug<T: Debug>() {}
    fn shareable<T: Send + Sync>() {}
    fn config<T: Debug + Clone + PartialEq + Default>() {}
    config::<IndexOptions>();
    config::<GraphConfig>();
    config::<CascadeConfig>();
    debug::<BaseInput>();
    debug::<Calibration>();
    debug::<SearchResult>();
    debug::<Index>();
    shareable::<Index>();
    shareable::<BaseInput>();
    shareable::<SearchResult>();
    struct OpaqueReader;
    debug::<orion_ann_runtime::data::query::VectorReader<OpaqueReader>>();
    debug::<orion_ann_runtime::data::query::GroundTruthReader<OpaqueReader>>();
}

#[test]
fn input_and_result_debug_are_bounded_and_results_compare_by_value() {
    let base = BaseInput::from_f32(&vec![12345.; 10_000 * 13], 13).unwrap();
    let summary = format!("{base:#?}");
    assert!(!summary.contains("12345"));
    assert!(summary.len() < 700);
    let result = SearchResult {
        ids: vec![12345; 20_000],
        counts: vec![10; 2000],
        k: 10,
    };
    assert_eq!(result, result.clone());
    let summary = format!("{result:?}");
    assert!(summary.contains("2000"));
    assert!(!summary.contains("12345"));
    assert!(summary.len() < 200);
}
