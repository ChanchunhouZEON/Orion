/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Preflight regressions; representation arithmetic lives in core datasets.
use crate::sidecars::requirements::SidecarRequirement;
use orion_ann_core::QuantizedDatasetShape;
#[test]
fn mips_buffers_contribute_to_report_and_budget_rejection() {
    use super::{enforce_budget, estimate_resident_memory};
    use crate::{
        CascadeConfig, GraphConfig, GraphSourcePolicy, IndexOptions, SearchMetric, VectorFormat,
    };
    let rows = 1_000_000;
    for (admission, bytes) in [
        (
            "mips-i8",
            SidecarRequirement::MipsI8
                .estimate(QuantizedDatasetShape {
                    rows,
                    dimension: 100,
                })
                .unwrap()
                .total_bytes() as u64,
        ),
        (
            "mips-i16",
            SidecarRequirement::MipsI16
                .estimate(QuantizedDatasetShape {
                    rows,
                    dimension: 100,
                })
                .unwrap()
                .total_bytes() as u64,
        ),
    ] {
        let options = IndexOptions {
            storage_options: crate::StorageOptions {
                mode: crate::StorageMode::Resident,
                ..Default::default()
            },
            metric: SearchMetric::InnerProduct,
            graph: GraphConfig {
                source: GraphSourcePolicy::Rust,
                ..Default::default()
            },
            cascade: CascadeConfig {
                admission: Some(admission.into()),
                ..Default::default()
            },
            cache_dir: std::env::temp_dir().join(format!(
                "orion-mips-budget-{}-{admission}",
                std::process::id()
            )),
            ..Default::default()
        };
        let mut config = options
            .resolve(
                "unused.fbin".into(),
                VectorFormat::Fbin,
                100,
                rows,
                Some("budget-test".into()),
            )
            .unwrap();
        config.prepare_only = false;
        let report = estimate_resident_memory(&config, rows).unwrap();
        assert_eq!(report["sidecar_bytes"], bytes);
        assert_eq!(
            report["resident_lower_bound_bytes"].as_u64().unwrap(),
            report["base_storage_bytes"].as_u64().unwrap()
                + report["graph_slab_bytes"].as_u64().unwrap()
                + bytes
        );

        let required = report["required_lower_bound_bytes"].as_u64().unwrap();
        // This budget fits the old estimate that omitted the IP sidecar.
        config.memory_budget_gib = Some((required - bytes / 2) as f64 / (1u64 << 30) as f64);
        let report = estimate_resident_memory(&config, rows).unwrap();
        assert!(enforce_budget(&report).is_err());
    }
}

#[test]
fn stacked_requirements_and_phase_selection() {
    use crate::config::{
        cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice},
        plan::IndexPlan,
    };
    use crate::sidecars::requirements::{estimate_sidecars, SidecarKind, SidecarRecipe};
    let input_shape = QuantizedDatasetShape {
        rows: 101,
        dimension: 33,
    };
    let cascade = Cascade {
        prefilter: PrefilterChoice::Jl,
        admission: AdmissionChoice::L2U16,
        rerank: RerankChoice::U16,
    };
    let plan = IndexPlan::resolve(
        crate::VectorStorageKind::F32,
        33,
        crate::SearchMetric::SquaredL2,
        crate::VectorFormat::Fbin,
        crate::VectorFormat::Fbin,
        cascade,
        crate::config::GraphSource::Rust,
    )
    .unwrap();
    let sidecar_requirements = plan.search_sidecar_requirements();
    assert_eq!(sidecar_requirements.len(), 3);
    let report = estimate_sidecars(input_shape, &sidecar_requirements).unwrap();
    assert_eq!(report.allocations().len(), 2);
    assert_eq!(
        report.total_bytes(),
        SidecarKind::JlSparse
            .requirement()
            .estimate(input_shape)
            .unwrap()
            .total_bytes()
            + SidecarRequirement::L2U16
                .estimate(input_shape)
                .unwrap()
                .total_bytes()
    );
    assert!(crate::config::RunPhase::GraphOnly
        .sidecar_requirements(plan)
        .is_empty());
    let native = IndexPlan::resolve(
        crate::VectorStorageKind::U8,
        33,
        crate::SearchMetric::SquaredL2,
        crate::VectorFormat::U8bin,
        crate::VectorFormat::U8bin,
        Cascade {
            prefilter: PrefilterChoice::None,
            admission: AdmissionChoice::NativeL2U8,
            rerank: RerankChoice::None,
        },
        crate::config::GraphSource::Parlayann,
    )
    .unwrap();
    assert!(native.search_sidecar_requirements().is_empty());
    let mips = IndexPlan::resolve(
        crate::VectorStorageKind::F32,
        33,
        crate::SearchMetric::InnerProduct,
        crate::VectorFormat::Fbin,
        crate::VectorFormat::Fbin,
        Cascade {
            admission: AdmissionChoice::IpI8,
            rerank: RerankChoice::IpF32,
            ..cascade
        },
        crate::config::GraphSource::Rust,
    )
    .unwrap();
    assert!(mips
        .search_sidecar_requirements()
        .contains(&SidecarKind::JlSparseMips.requirement()));
}

#[test]
fn native_norms_are_budgeted_including_prepare_only_and_mapped() {
    use crate::{IndexOptions, SearchMetric, StorageMode, VectorFormat};
    // Budget arithmetic must also work without a graph builder installed.
    // Resolve offline metadata: this test does not import or construct a graph.
    let estimate = |config: &crate::config::ResolvedRunConfig, rows| {
        let input = super::resolve_memory_inputs(config, rows, true, "configuration").unwrap();
        super::calculate_memory(config, input).unwrap()
    };
    for (dimension, width) in [(128, 4u64), (66051, 4), (66052, 8)] {
        let rows = 1000;
        for mode in [StorageMode::Resident, StorageMode::Mapped] {
            let mut options = IndexOptions {
                metric: SearchMetric::Cosine,
                ..Default::default()
            };
            options.storage_options.mode = mode;
            let mut config = options
                .resolve(
                    "unused.u8bin".into(),
                    VectorFormat::U8bin,
                    dimension,
                    rows,
                    None,
                )
                .unwrap();
            for prepare in [false, true] {
                config.prepare_only = prepare;
                let report = estimate(&config, rows);
                assert_eq!(report["native_norm_bytes"], rows as u64 * width);
                assert_eq!(report["sidecar_bytes"], rows as u64 * width);
                assert_eq!(
                    report[if mode == StorageMode::Resident {
                        "sidecar_owned_bytes"
                    } else {
                        "sidecar_mapped_bytes"
                    }],
                    rows as u64 * width
                );
            }
        }
    }
    let config = IndexOptions {
        metric: SearchMetric::InnerProduct,
        ..Default::default()
    }
    .resolve("unused.u8bin".into(), VectorFormat::U8bin, 128, 1000, None)
    .unwrap();
    assert_eq!(estimate(&config, 1000)["native_norm_bytes"], 0);
}

#[test]
fn phase_selected_sidecars_match_preflight_and_placement() {
    use crate::config::RunPhase;
    use crate::{CascadeConfig, GraphConfig, GraphSourcePolicy, IndexOptions, StorageMode};
    use orion_ann_platform::resources::MemoryResources;

    let rows = 101;
    let shape = QuantizedDatasetShape {
        rows,
        dimension: 33,
    };
    let options = IndexOptions {
        graph: GraphConfig {
            source: GraphSourcePolicy::Rust,
            ..Default::default()
        },
        cascade: CascadeConfig {
            prefilter: Some("jl".into()),
            admission: Some("l2-u16".into()),
            rerank: Some("u16".into()),
        },
        ..Default::default()
    };
    let mut config = options
        .resolve(
            "unused.fbin".into(),
            crate::VectorFormat::Fbin,
            33,
            rows,
            None,
        )
        .unwrap();
    let search_sidecar_requirements = config.index_plan().search_sidecar_requirements();
    assert_eq!(search_sidecar_requirements.len(), 3);
    let expected =
        crate::sidecars::requirements::estimate_sidecars(shape, &search_sidecar_requirements)
            .unwrap();
    assert_eq!(expected.allocations().len(), 2);

    for (prepare_only, phase) in [(true, RunPhase::GraphOnly), (false, RunPhase::SearchReady)] {
        config.prepare_only = prepare_only;
        assert_eq!(config.run_phase(), phase);
        let sidecar_requirements = config.run_phase().sidecar_requirements(config.index_plan());
        let expected_bytes = match phase {
            RunPhase::GraphOnly => {
                assert!(sidecar_requirements.is_empty());
                0
            }
            RunPhase::SearchReady => {
                assert_eq!(sidecar_requirements, search_sidecar_requirements);
                expected.total_bytes() as u64
            }
        };
        // A later search preparation requests its recipe even after a graph-only
        // run. Selecting that target must not mutate config or cache identity.
        let serialized = serde_json::to_value(&config).unwrap();
        assert_eq!(
            RunPhase::SearchReady.sidecar_requirements(config.index_plan()),
            search_sidecar_requirements
        );
        assert_eq!(serde_json::to_value(&config).unwrap(), serialized);

        for mode in [StorageMode::Resident, StorageMode::Mapped] {
            config.storage_options.mode = mode;
            let input = super::resolve_memory_inputs(&config, rows, true, "configuration").unwrap();
            let report = super::calculate_memory(&config, input).unwrap();
            let base = report["base_storage_bytes"].as_u64().unwrap();
            let graph = report["graph_slab_bytes"].as_u64().unwrap();
            let placement = crate::placement::complete_plan(
                &config,
                shape,
                &sidecar_requirements,
                base,
                graph,
                0,
                MemoryResources::default(),
            )
            .unwrap();
            assert_eq!(report["sidecar_bytes"], expected_bytes);
            assert_eq!(
                placement.resident_bytes + placement.mapped_bytes,
                base + graph + expected_bytes
            );
            assert_eq!(
                report["resident_lower_bound_bytes"],
                placement.resident_bytes
            );
            assert!(report.get("mapping_reserve_bytes").is_none());
            let known_working_bytes: u64 = report["components"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|component| {
                    matches!(
                        component["name"].as_str(),
                        Some("query buffers" | "result IDs" | "ground truth IDs")
                    )
                })
                .map(|component| component["bytes"].as_u64().unwrap())
                .sum();
            assert_eq!(
                report["accounted_lower_bound_bytes"],
                placement.resident_bytes + known_working_bytes
            );
            let text = super::report::format_report(&report);
            assert!(!text.contains("Mapping working-set reserve"));
            if mode == StorageMode::Mapped {
                assert!(text.contains("Mapped payload:"));
            }
            if mode == StorageMode::Mapped {
                assert_eq!(report["sidecar_mapped_bytes"], expected_bytes);
                assert_eq!(report["sidecar_owned_bytes"], 0);
            } else {
                assert_eq!(report["sidecar_owned_bytes"], expected_bytes);
                assert_eq!(report["sidecar_mapped_bytes"], 0);
            }
        }
    }
}
