/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann::{
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
    debug::<orion_ann::IndexSlot>();
    shareable::<orion_ann::IndexSlot>();
    debug::<orion_ann::SearchScores>();
    debug::<orion_ann::ScoreMetric>();
    debug::<orion_ann::PinOptions>();
    debug::<orion_ann::PinHint>();
    debug::<orion_ann::PinReport>();
    debug::<orion_ann::PinnedDatasets<'_>>();
    shareable::<orion_ann::PinnedDatasets<'_>>();
    debug::<orion_ann::QueryBatch<'_>>();
    debug::<orion_ann::SearchOptions>();
    shareable::<orion_ann::QueryBatch<'_>>();
    shareable::<Index>();
    shareable::<BaseInput>();
    shareable::<SearchResult>();
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
        scores: None,
        external_ids: None,
    };
    assert_eq!(result, result.clone());
    let summary = format!("{result:?}");
    assert!(summary.contains("2000"));
    assert!(!summary.contains("12345"));
    assert!(summary.len() < 200);
}
