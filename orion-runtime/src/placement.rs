/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Placement planning before payload allocation. No mapping or search occurs here.
use crate::{
    config::ResolvedRunConfig,
    sidecars::requirements::{estimate_sidecars, SidecarKind, SidecarRecipe, SidecarRequirement},
    Error, Result,
};
use orion_ann_core::{index::IndexData, CompressedIndex, QuantizedDatasetShape};
use orion_ann_platform::{
    region::{Placement, RegionOptions},
    resources::MemoryResources,
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

/// Placement for base vectors, graph slabs and quantized payloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum StorageMode {
    #[default]
    Auto,
    Resident,
    Mapped,
}

/// Deployment settings, independent of cascade, cache identity and calibration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageOptions {
    pub mode: StorageMode,
    /// Existing disk-backed directory; None uses the OS temporary directory.
    /// Mapped payloads occupy private files for the lifetime of the index.
    pub directory: Option<PathBuf>,
}
impl StorageOptions {
    pub(crate) fn is_default(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Debug)]
pub(crate) struct StoragePlan {
    pub options: BTreeMap<String, RegionOptions>,
    pub resident_bytes: u64,
    pub mapped_bytes: u64,
    pub reserve_bytes: u64,
    pub limit_bytes: Option<u64>,
}

pub(crate) fn codec(kind: SidecarKind) -> &'static str {
    match kind {
        SidecarKind::L2U8 => "qds",
        SidecarKind::L2U16 => "qds6",
        SidecarKind::MipsI8 => "qdm8",
        SidecarKind::MipsI16 => "qdm16",
        SidecarKind::Pq => "pq-adc",
        SidecarKind::L2Kt => "qdsl2kt",
        SidecarKind::JlSparse => "jls",
        SidecarKind::JlSparseMips => "jls_mips",
        SidecarKind::JlHadamard => "jlh",
        SidecarKind::Rabitq => "qrbq",
        SidecarKind::Rabitq4 => "qrb4",
    }
}
/// `already_resident` is credited only during preparation of a live index. This
/// avoids subtracting base/graph twice from observed available memory.
#[cfg(test)]
fn plan_with_resources(
    config: &ResolvedRunConfig,
    shape: QuantizedDatasetShape,
    requirements: &[SidecarRequirement],
    fixed_bytes: u64,
    already_resident: u64,
    resources: MemoryResources,
) -> Result<StoragePlan> {
    plan_objects(
        config,
        shape,
        requirements,
        fixed_bytes,
        already_resident,
        resources,
        &[],
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn plan_objects(
    config: &ResolvedRunConfig,
    shape: QuantizedDatasetShape,
    requirements: &[SidecarRequirement],
    fixed_bytes: u64,
    already_resident: u64,
    resources: MemoryResources,
    extra: &[(&str, u64)],
    existing_mapping: bool,
) -> Result<StoragePlan> {
    let estimates = estimate_sidecars(shape, requirements)?;
    let norm_bytes = native_norm_bytes(config, shape.rows)?;
    let total = extra.iter().try_fold(
        (estimates.total_bytes() as u64)
            .checked_add(norm_bytes)
            .ok_or_else(|| Error::invalid_argument("storage size overflow"))?,
        |sum, (_, bytes)| {
            sum.checked_add(*bytes)
                .ok_or_else(|| Error::invalid_argument("storage size overflow"))
        },
    )?;
    let mut result = StoragePlan {
        options: BTreeMap::new(),
        resident_bytes: total,
        mapped_bytes: 0,
        reserve_bytes: 0,
        limit_bytes: None,
    };
    // A reserve is not a bound on growing search scratch or PQ training memory.
    // Keep page-cache working space and the bounded copy blocks out of heap plans.
    let reserve = 64 * 1024 * 1024u64;
    let user = config
        .memory_budget_gib
        .map(|g| (g * (1u64 << 30) as f64) as u64);
    let observed = resources
        .available_bytes
        .map(|a| a.saturating_add(already_resident));
    let limit = match (user, observed) {
        (Some(u), Some(a)) if config.storage_options.mode == StorageMode::Auto => Some(u.min(a)),
        (Some(u), _) => Some(u),
        (None, a) if config.storage_options.mode == StorageMode::Auto => a,
        _ => None,
    };
    let mut candidates = Vec::new();
    for item in estimates.allocations() {
        let bytes = item.memory.total_bytes() as u64;
        if bytes != 0 {
            candidates.push((codec(item.requirement.kind()), bytes));
        }
    }
    if norm_bytes != 0 {
        candidates.push((
            orion_ann_core::model::dataset::native_u8::NORM_CODEC,
            norm_bytes,
        ));
    }
    candidates.extend(extra.iter().copied());
    // Largest allocations first gives a deterministic, stage-independent policy.
    candidates.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    for (name, bytes) in candidates {
        let needs_space = limit.is_some_and(|l| {
            fixed_bytes
                .saturating_add(result.resident_bytes)
                .saturating_add(if result.mapped_bytes == 0 && !existing_mapping {
                    0
                } else {
                    reserve
                })
                > l
        });
        if config.storage_options.mode == StorageMode::Mapped
            || (config.storage_options.mode == StorageMode::Auto && needs_space)
        {
            result.resident_bytes -= bytes;
            result.mapped_bytes += bytes;
            result.options.insert(
                name.into(),
                RegionOptions {
                    placement: Placement::Mapped,
                    directory: config.storage_options.directory.clone(),
                },
            );
        }
    }
    if result.mapped_bytes != 0 || existing_mapping {
        result.reserve_bytes = reserve;
    }
    result.limit_bytes = limit;
    Ok(result)
}

pub(crate) fn configure<D: IndexData>(
    index: &CompressedIndex<D>,
    config: &ResolvedRunConfig,
    extra_codecs: impl Iterator<Item = String>,
) -> Result<()> {
    if index.sidecar_storage_plan().is_some() {
        return Ok(());
    }
    let shape = QuantizedDatasetShape {
        rows: index.dataset().point_count(),
        dimension: index.dataset().dimension(),
    };
    let mut requirements = config.index_plan().sidecar_requirements(false);
    // Snapshots may contain prepared sidecars outside the active cascade.
    for name in extra_codecs {
        if name == orion_ann_core::model::dataset::native_u8::NORM_CODEC {
            if native_norm_bytes(config, shape.rows)? == 0 {
                return Err(Error::cache("native norm object requires u8 cosine"));
            }
            continue;
        }
        let kind = [
            SidecarKind::L2U8,
            SidecarKind::L2U16,
            SidecarKind::MipsI8,
            SidecarKind::MipsI16,
            SidecarKind::Pq,
            SidecarKind::L2Kt,
            SidecarKind::JlSparse,
            SidecarKind::JlSparseMips,
            SidecarKind::JlHadamard,
            SidecarKind::Rabitq,
            SidecarKind::Rabitq4,
        ]
        .into_iter()
        .find(|&k| codec(k) == name)
        .ok_or_else(|| Error::cache("unknown sidecar codec"))?;
        requirements.push(kind.requirement());
    }
    let layout = if index.dataset().element_type() == "u8" {
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpU8Distance>(
            shape.dimension,
        )?
    } else {
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(
            shape.dimension,
        )?
    };
    let base = layout.allocation_bytes(shape.rows)? as u64;
    let fixed = if index.dataset().placement() == Placement::Resident {
        base
    } else {
        0
    } + if index.graph().placement() == Placement::Resident {
        index.graph().buffer_bytes().len() as u64
    } else {
        0
    };
    let result = plan_objects(
        config,
        shape,
        &requirements,
        fixed,
        fixed,
        MemoryResources::detect(),
        &[],
        index.dataset().placement() == Placement::Mapped
            || index.graph().placement() == Placement::Mapped,
    )?;
    let required = fixed
        .checked_add(result.resident_bytes)
        .and_then(|n| n.checked_add(result.reserve_bytes))
        .ok_or_else(|| Error::invalid_argument("storage plan overflow"))?;
    if result.limit_bytes.is_some_and(|limit| required > limit) {
        return Err(Error::new(
            crate::ErrorCode::ResourceExhausted,
            "base, graph, remaining sidecars and mapping reserve exceed the placement budget",
        )
        .with_detail("required_bytes", required)
        .with_detail("mapped_bytes", result.mapped_bytes)
        .with_detail("budget_bytes", result.limit_bytes.unwrap())
        .at("plan_storage"));
    }
    log::info!(
        "Sidecar storage: {} owned bytes, {} mapped bytes, {} mapping reserve bytes",
        result.resident_bytes,
        result.mapped_bytes,
        result.reserve_bytes
    );
    index.configure_sidecar_storage(result.options)
}

/// Plan every persistent payload together; callers retain only the decision for
/// the next allocation. Live objects are subsequently charged at their actual placement.
pub(crate) fn complete_plan(
    config: &ResolvedRunConfig,
    shape: QuantizedDatasetShape,
    base: u64,
    graph: u64,
    other: u64,
    resources: MemoryResources,
) -> Result<StoragePlan> {
    plan_objects(
        config,
        shape,
        &config
            .index_plan()
            .sidecar_requirements(config.prepare_only),
        other,
        0,
        resources,
        &[("base", base), ("graph", graph)],
        false,
    )
}
fn enforce(plan: &StoragePlan, fixed: u64) -> Result<()> {
    let required = fixed
        .saturating_add(plan.resident_bytes)
        .saturating_add(plan.reserve_bytes);
    if let Some(limit) = plan.limit_bytes.filter(|&l| required > l) {
        return Err(Error::new(
            crate::ErrorCode::ResourceExhausted,
            "remaining owned payloads and mapping reserve exceed the placement budget",
        )
        .with_detail("required_bytes", required)
        .with_detail("budget_bytes", limit)
        .at("plan_storage"));
    }
    Ok(())
}
fn base_size(config: &ResolvedRunConfig, rows: usize) -> Result<u64> {
    let layout = if config.storage() == crate::config::VectorStorageKind::U8 {
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::IpU8Distance>(
            config.dimension,
        )?
    } else {
        orion_ann_vector::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(
            config.dimension,
        )?
    };
    Ok(layout.allocation_bytes(rows)? as u64)
}
pub(crate) fn base_options(
    config: &ResolvedRunConfig,
    rows: usize,
    already_resident: bool,
) -> Result<RegionOptions> {
    let base = base_size(config, rows)?;
    let graph = crate::preflight::graph_bytes(config, rows)?;
    let mut resources = MemoryResources::detect();
    if already_resident {
        resources.available_bytes = resources.available_bytes.map(|n| n.saturating_add(base));
    }
    let mut search = config.clone();
    search.prepare_only = false;
    let plan = complete_plan(
        &search,
        QuantizedDatasetShape {
            rows,
            dimension: config.dimension,
        },
        base,
        graph,
        0,
        resources,
    )?;
    enforce(&plan, 0)?;
    Ok(plan.options.get("base").cloned().unwrap_or_default())
}
pub(crate) fn graph_options<D: IndexData>(
    config: &ResolvedRunConfig,
    data: &D,
    bytes: u64,
) -> Result<RegionOptions> {
    let fixed = if data.placement() == Placement::Resident {
        base_size(config, data.point_count())?
    } else {
        0
    };
    let plan = plan_objects(
        config,
        QuantizedDatasetShape {
            rows: data.point_count(),
            dimension: data.dimension(),
        },
        &config.index_plan().sidecar_requirements(false),
        fixed,
        fixed,
        MemoryResources::detect(),
        &[("graph", bytes)],
        data.placement() == Placement::Mapped,
    )?;
    enforce(&plan, fixed)?;
    Ok(plan.options.get("graph").cloned().unwrap_or_default())
}

/// Native cosine normalization is a base auxiliary object, not a quantizer.
pub(crate) fn native_norm_bytes(config: &ResolvedRunConfig, rows: usize) -> Result<u64> {
    if config.storage() == crate::VectorStorageKind::U8
        && config.metric == crate::SearchMetric::Cosine
    {
        Ok(
            orion_ann_core::model::dataset::native_u8::NativeU8Norms::estimated_bytes(
                rows,
                config.dimension,
            )? as u64,
        )
    } else {
        Ok(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> ResolvedRunConfig {
        crate::IndexOptions::default()
            .resolve(
                "unused.fbin".into(),
                crate::VectorFormat::Fbin,
                128,
                1_000_000,
                Some("fixture".into()),
            )
            .unwrap()
    }
    #[test]
    fn auto_moves_only_enough_unique_payloads_and_preserves_fixed_storage() {
        let mut c = config();
        let shape = QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        };
        let requirements = [
            SidecarRequirement::L2U8,
            SidecarRequirement::L2U8,
            SidecarRequirement::L2U16,
        ];
        c.memory_budget_gib = Some(0.9);
        let resources = MemoryResources {
            available_bytes: Some(1 << 30),
            ..Default::default()
        };
        let p = plan_with_resources(&c, shape, &requirements, 700_000_000, 0, resources).unwrap();
        assert_eq!(p.mapped_bytes, 256_000_000);
        assert_eq!(p.resident_bytes, 128_000_000);
        assert_eq!(p.options.len(), 1);
        assert!(p.options.contains_key("qds6"));
        c.storage_options.mode = StorageMode::Resident;
        let p = plan_with_resources(&c, shape, &requirements, 700_000_000, 0, resources).unwrap();
        assert_eq!(p.resident_bytes, 384_000_000);
        assert!(p.options.is_empty());
        assert!(700_000_000 + p.resident_bytes > p.limit_bytes.unwrap());
    }
    #[test]
    fn unknown_observations_do_not_invent_pressure_and_live_base_is_not_counted_twice() {
        let c = config();
        let shape = QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        };
        let req = [SidecarRequirement::L2U8];
        let p = plan_with_resources(&c, shape, &req, 700_000_000, 0, MemoryResources::default())
            .unwrap();
        assert!(p.options.is_empty());
        let p = plan_with_resources(
            &c,
            shape,
            &req,
            700_000_000,
            700_000_000,
            MemoryResources {
                available_bytes: Some(300_000_000),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.options.is_empty());
    }
    #[test]
    fn fitting_owned_payload_does_not_reserve_unused_mapping_buffers() {
        let c = config();
        let p = plan_with_resources(
            &c,
            QuantizedDatasetShape {
                rows: 1_000_000,
                dimension: 128,
            },
            &[SidecarRequirement::L2U8],
            700_000_000,
            0,
            MemoryResources {
                available_bytes: Some(830_000_000),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(p.options.is_empty());
        assert_eq!(p.reserve_bytes, 0);
    }
    #[test]
    fn all_sidecar_payloads_including_codebooks_can_be_mapped() {
        let mut c = config();
        c.storage_options.mode = StorageMode::Mapped;
        let shape = QuantizedDatasetShape {
            rows: 1000,
            dimension: 128,
        };
        let req = [SidecarKind::Pq.requirement(), SidecarRequirement::L2Kt];
        let p = plan_with_resources(&c, shape, &req, 1000, 0, MemoryResources::default()).unwrap();
        assert_eq!(
            p.mapped_bytes,
            estimate_sidecars(shape, &req).unwrap().total_bytes() as u64
        );
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.options.len(), 2);
        assert_eq!(p.reserve_bytes, 64 << 20);
    }
    #[test]
    fn whole_index_plan_maps_base_and_graph_below_their_owned_size() {
        let mut c = config();
        c.memory_budget_gib = Some(0.1);
        let shape = QuantizedDatasetShape {
            rows: 1_000_000,
            dimension: 128,
        };
        let p = complete_plan(
            &c,
            shape,
            512_000_000,
            384_000_000,
            0,
            MemoryResources::default(),
        )
        .unwrap();
        assert!(p.options.contains_key("base"));
        assert!(p.options.contains_key("graph"));
        assert_eq!(p.resident_bytes, 0);
        assert_eq!(p.reserve_bytes, 64 << 20);
        enforce(&p, 0).unwrap();
        let p = plan_objects(&c, shape, &[], 0, 0, MemoryResources::default(), &[], true).unwrap();
        assert_eq!(p.reserve_bytes, 64 << 20); // Existing mappings still need working space.
        c.memory_budget_gib = Some(0.01);
        let p = complete_plan(
            &c,
            shape,
            512_000_000,
            384_000_000,
            0,
            MemoryResources::default(),
        )
        .unwrap();
        assert!(enforce(&p, 0).is_err());
    }
}
