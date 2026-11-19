/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Runtime storage policy and allocation-scoped payload accounting.
//! The scheduler forecasts placement; buffers themselves own budget leases.
mod accounting;
pub(crate) use accounting::{budget, report};
mod object;
mod options;
pub use object::StorageObject;
pub use options::{StorageMode, StorageOptions};
mod installed;
mod scheduler;
#[cfg(test)]
mod tests;
use crate::{
    config::ResolvedRunConfig,
    sidecars::requirements::{estimate_sidecars, SidecarKind, SidecarRecipe, SidecarRequirement},
    Error, Result,
};
use orion_ann_core::{index::IndexData, QuantizedDatasetShape};
use orion_ann_platform::{
    region::{Placement, RegionOptions},
    resources::MemoryResources,
};
use std::collections::BTreeMap;

/// Resolved per-object placement and payload budget used by allocation and preflight.
#[derive(Debug)]
pub(crate) struct StoragePlan {
    /// Per-object placement decisions shared by preflight and preparation.
    pub objects: Vec<object::ObjectDecision>,

    /// Overrides whose objects are not required in the selected preparation phase.
    pub inactive_overrides: Vec<StorageObject>,

    /// Resolved allocation options keyed by managed object name.
    pub options: BTreeMap<String, RegionOptions>,

    /// Managed payload bytes planned for resident allocation; not process RSS.
    pub resident_bytes: u64,

    /// Addressable mapped payload bytes; not a physical-residency guarantee.
    pub mapped_bytes: u64,

    /// Optional resident payload budget applied during placement.
    pub limit_bytes: Option<u64>,
}

const KINDS: [SidecarKind; 11] = [
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
];

/// Maps a sidecar kind to the storage key shared by allocation policy and persistence.
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
    sidecar_requirements: &[SidecarRequirement],
    fixed_bytes: u64,
    already_resident: u64,
    resources: MemoryResources,
) -> Result<StoragePlan> {
    plan_objects(
        config,
        shape,
        sidecar_requirements,
        fixed_bytes,
        already_resident,
        resources,
        &[],
    )
}

/// Combines sidecar and fixed-object estimates, explicit overrides and resource observations into a placement plan.
#[allow(clippy::too_many_arguments)]
fn plan_objects(
    config: &ResolvedRunConfig,
    shape: QuantizedDatasetShape,
    sidecar_requirements: &[SidecarRequirement],
    fixed_bytes: u64,
    already_resident: u64,
    resources: MemoryResources,
    extra: &[(&'static str, u64)],
) -> Result<StoragePlan> {
    config.storage_options.validate()?;
    let estimates = estimate_sidecars(shape, sidecar_requirements)?;
    let norm_bytes = native_norm_bytes(config, shape.rows)?;
    let limit = effective_limit(config, resources, already_resident);
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
    let descriptors = candidates
        .into_iter()
        .map(|(name, bytes)| {
            let object = StorageObject::from_codec(name)?;
            Ok(object::ObjectDescriptor {
                name,
                object,
                bytes,
                frequency: frequency(config, name),
                mode: config.storage_options.requested(object),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let selected = scheduler::select(descriptors, limit, fixed_bytes, &config.storage_options)?;
    Ok(StoragePlan {
        objects: selected.objects,
        inactive_overrides: selected.inactive_overrides,
        options: selected
            .placements
            .into_iter()
            .filter(|(_, p)| *p == Placement::Mapped)
            .map(|(name, placement)| {
                (
                    name.into(),
                    RegionOptions {
                        placement,
                        directory: config.storage_options.directory.clone(),
                        accounting: None,
                        reclaim_after_write: false,
                    },
                )
            })
            .collect(),
        resident_bytes: selected.resident,
        mapped_bytes: selected.mapped,
        limit_bytes: limit,
    })
}

/// Chooses the resident-payload ceiling from an existing ledger or the user and observed memory limits.
///
/// Already-owned resident input is credited once; this is not an operating-system RSS limit.
fn effective_limit(
    config: &ResolvedRunConfig,
    resources: MemoryResources,
    already_resident: u64,
) -> Option<u64> {
    if let Some(budget) = config.allocation_budget.get() {
        return budget.usage().limit_bytes;
    }
    let user = config
        .memory_budget_gib
        .map(|g| (g * (1u64 << 30) as f64) as u64);
    let observed = resources
        .available_bytes
        .map(|a| a.saturating_add(already_resident));
    match (user, observed) {
        (Some(u), Some(a)) if config.storage_options.uses_auto() => Some(u.min(a)),
        (Some(u), _) => Some(u),
        (None, a) if config.storage_options.uses_auto() => a,
        _ => None,
    }
}

/// Assigns a placement priority from an object's role in the active search recipe.
///
/// This is a static access-frequency heuristic, not runtime telemetry or cache residency measurement.
fn frequency(config: &ResolvedRunConfig, name: &str) -> scheduler::Frequency {
    use crate::config::cascade::{AdmissionChoice, Cascade, PrefilterChoice, RerankChoice};
    use scheduler::Frequency;
    if name == "graph" {
        return Frequency::Graph;
    }
    let cascade = config.cascade();
    let admission = Cascade {
        prefilter: PrefilterChoice::None,
        rerank: RerankChoice::None,
        ..cascade
    };
    if admission
        .sidecar_requirements()
        .iter()
        .any(|r| codec(r.kind()) == name)
    {
        return Frequency::Admission;
    }
    let native = matches!(
        cascade.admission,
        AdmissionChoice::NativeL2U8 | AdmissionChoice::NativeIpU8 | AdmissionChoice::NativeCosineU8
    );
    #[cfg(feature = "benchmark")]
    let native = native || cascade.admission == AdmissionChoice::AdsF32;
    if name == "base" || name == orion_ann_core::model::dataset::native_u8::NORM_CODEC {
        return if native {
            Frequency::Admission
        } else if matches!(
            cascade.rerank,
            RerankChoice::F32 | RerankChoice::IpF32 | RerankChoice::CosineF32
        ) {
            Frequency::Rerank
        } else {
            Frequency::Cold
        };
    }
    if cascade.rerank == RerankChoice::U16 && name == codec(SidecarKind::L2U16) {
        return Frequency::Rerank;
    }
    if cascade
        .sidecar_requirements()
        .iter()
        .any(|r| codec(r.kind()) == name)
    {
        Frequency::Prefilter
    } else {
        Frequency::Cold
    }
}

/// Prioritize lazy initialization by the same frequency hints as placement.
/// This changes only preparation order, not distance/session dispatch.
pub(crate) fn ordered_sidecar_requirements(config: &ResolvedRunConfig) -> Vec<SidecarRequirement> {
    let mut sidecar_requirements = config.index_plan().search_sidecar_requirements();
    sidecar_requirements.sort_by(|a, b| {
        frequency(config, codec(b.kind()))
            .cmp(&frequency(config, codec(a.kind())))
            .then(codec(a.kind()).cmp(codec(b.kind())))
    });
    sidecar_requirements.dedup();
    sidecar_requirements
}

/// Install immutable allocation options before any lazy sidecars are created.
/// Restoration includes inactive snapshot sidecars in this same planning step.
pub(crate) fn configure_storage_policy<T: crate::data::BaseElement>(
    index: &orion_ann_core::Orion<T>,
    config: &ResolvedRunConfig,
    extra_codecs: impl Iterator<Item = String>,
) -> Result<()> {
    let installed = index.sidecar_storage_plan();
    let shape = QuantizedDatasetShape {
        rows: index.dataset().point_count(),
        dimension: index.dataset().dimension(),
    };
    let mut sidecar_requirements = config.index_plan().search_sidecar_requirements();
    // Snapshots may contain prepared sidecars outside the active cascade.
    // Include every existing binding, even if inactive in the current recipe:
    // its allocation policy must still fit and unknown codecs must not bypass checks.
    for name in extra_codecs.chain(installed.into_iter().flat_map(|plan| plan.keys().cloned())) {
        if name == orion_ann_core::model::dataset::native_u8::NORM_CODEC {
            if native_norm_bytes(config, shape.rows)? == 0 {
                return Err(Error::cache("native norm object requires u8 cosine"));
            }
            continue;
        }
        let kind = KINDS
            .into_iter()
            .find(|&k| codec(k) == name)
            .ok_or_else(|| Error::cache("unknown sidecar codec"))?;
        sidecar_requirements.push(kind.requirement());
    }
    // Base rows already exist here. Count their actual backing storage, including
    // row padding and the SIMD tail, rather than inferring a layout from a distance
    // kernel. Storage accounting is independent of the selected search metric.
    let base = std::mem::size_of_val(index.dataset().as_slice()) as u64;
    let fixed = if index.dataset().placement() == Placement::Resident {
        base
    } else {
        0
    } + if index.graph().placement() == Placement::Resident {
        index.graph().buffer_bytes().len() as u64
    } else {
        0
    };
    let mut result = plan_objects(
        config,
        shape,
        &sidecar_requirements,
        fixed,
        budget(config, 0).usage().resident_bytes,
        MemoryResources::detect(),
        &[],
    )?;
    if let Some(installed) = installed {
        installed::validate(&mut result, installed)?;
        return enforce(&result, fixed);
    }
    enforce(&result, fixed)?;
    let options = result
        .objects
        .iter()
        .map(|object| {
            let options = result.options.get(object.name).cloned().unwrap_or_default();
            Ok((
                object.name.to_owned(),
                accounting::bind(config, object.name, options)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    index.configure_sidecar_storage(options)
}

/// Plan every persistent payload together; callers retain only the decision for
/// the next allocation. The caller supplies the phase-selected sidecar list so
/// preflight and placement cannot independently select different dependencies.
/// Live objects are subsequently charged at their actual placement.
pub(crate) fn complete_plan(
    config: &ResolvedRunConfig,
    shape: QuantizedDatasetShape,
    sidecar_requirements: &[SidecarRequirement],
    base: u64,
    graph: u64,
    other: u64,
    resources: MemoryResources,
) -> Result<StoragePlan> {
    plan_objects(
        config,
        shape,
        sidecar_requirements,
        other,
        0,
        resources,
        &[("base", base), ("graph", graph)],
    )
}

/// Rejects planned resident objects that exceed the remaining managed-payload budget.
fn enforce(plan: &StoragePlan, fixed: u64) -> Result<()> {
    let required = fixed.saturating_add(plan.resident_bytes);
    if let Some(limit) = plan.limit_bytes.filter(|&l| required > l) {
        return Err(Error::new(
            crate::ErrorCode::ResourceExhausted,
            "remaining resident payloads exceed the placement budget",
        )
        .with_detail("required_bytes", required)
        .with_detail("budget_bytes", limit)
        .at("plan_storage"));
    }
    Ok(())
}

/// Computes padded base allocation bytes using coordinate layout rather than a search metric choice.
fn base_size(config: &ResolvedRunConfig, rows: usize) -> Result<u64> {
    let layout = if config.storage() == crate::config::VectorStorageKind::U8 {
        orion_ann_vector::PaddedLayout::for_base_scalar::<u8>(config.dimension)?
    } else {
        orion_ann_vector::PaddedLayout::for_base_scalar::<f32>(config.dimension)?
    };
    Ok(layout.allocation_bytes(rows)? as u64)
}

/// Plans base placement before allocation while accounting for graph and search-ready sidecar needs.
pub(crate) fn base_options(
    config: &ResolvedRunConfig,
    rows: usize,
    already_resident: bool,
) -> Result<RegionOptions> {
    let base = base_size(config, rows)?;
    budget(config, if already_resident { base } else { 0 });
    let graph = crate::preflight::graph_bytes(config, rows)?;
    let mut resources = MemoryResources::detect();
    if already_resident {
        resources.available_bytes = resources.available_bytes.map(|n| n.saturating_add(base));
    }
    // Preserve space for eventual search even when the current run only builds
    // the graph; placement should not require mutating the caller's run phase.
    let sidecar_requirements =
        crate::config::RunPhase::SearchReady.sidecar_requirements(config.index_plan());
    let plan = complete_plan(
        config,
        QuantizedDatasetShape {
            rows,
            dimension: config.dimension,
        },
        &sidecar_requirements,
        base,
        graph,
        0,
        resources,
    )?;
    enforce(&plan, 0)?;
    accounting::bind(
        config,
        "base",
        plan.options.get("base").cloned().unwrap_or_default(),
    )
}

/// Plans graph placement around the actual base allocation and binds its allocation-budget account.
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
        &crate::config::RunPhase::SearchReady.sidecar_requirements(config.index_plan()),
        fixed,
        fixed,
        MemoryResources::detect(),
        &[("graph", bytes)],
    )?;
    enforce(&plan, fixed)?;
    accounting::bind(
        config,
        "graph",
        plan.options.get("graph").cloned().unwrap_or_default(),
    )
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
