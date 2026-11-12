/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Resolve requirements to resident storage, then apply a bounded OS policy.
//! No cascade, calibration, fitting or cache I/O occurs in this module.
use crate::{
    sidecars::requirements::{DatasetRequirement, SidecarRecipe},
    Error, Result,
};
use orion_ann_core::{
    model::dataset::{quantized_dataset::QuantizedKind, QuantizedDatasetShape},
    CompressedIndex,
};
use orion_ann_platform::residency::PinGuard;
use std::collections::HashSet;

/// A storage allocation selected for RAM residency, independently of its stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum DatasetPin {
    Base,
    /// Immutable navigation graph slab.
    Graph,
    Quantized(QuantizedKind),
}

/// Whether a denied memory lock is acceptable for this deployment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PinMode {
    Off,
    #[default]
    BestEffort,
    Required,
}

/// Deployment policy; never part of a cache key or saved calibration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PinOptions {
    pub mode: PinMode,
    /// Aggregate page-rounded limit, not a resident-memory estimate.
    pub budget_bytes: usize,
    /// None selects the current requirements. Some selects only those named;
    /// an empty selection locks nothing. Unused/unprepared selections are errors.
    pub datasets: Option<Vec<DatasetPin>>,
}
impl Default for PinOptions {
    fn default() -> Self {
        Self {
            mode: PinMode::BestEffort,
            budget_bytes: 64 * 1024 * 1024,
            datasets: None,
        }
    }
}

mod automatic;
pub(crate) use automatic::AutomaticPlan;

/// Prepared dataset's total payload. Smaller allocations are attempted first
/// by the default policy; no quality or CPU-cache-residency claim is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PinHint {
    pub dataset: DatasetPin,
    pub bytes: usize,
}

/// Outcome of a best-effort or required residency request.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct PinReport {
    pub requested_datasets: usize,
    pub pinned_datasets: usize,
    pub pinned_bytes: usize,
    /// Effective automatic budget and resource observations, absent for manual pin.
    pub automatic: Option<serde_json::Value>,
    /// Dataset and reason for each skipped lock (budget or OS restrictions).
    pub skipped: Vec<(DatasetPin, String)>,
}

/// Keeps the selected allocations borrowed and locked until dropped.
/// Drop this guard before recalibrating or replacing the index.
#[derive(Debug)]
#[must_use = "keep the guard alive while searching to retain RAM residency"]
pub struct PinnedDatasets<'a> {
    guard: PinGuard<'a>,
    report: PinReport,
}
impl PinnedDatasets<'_> {
    /// Caller must keep every covered allocation live until the returned token drops.
    pub(crate) unsafe fn into_owned(
        self,
    ) -> (orion_ann_platform::residency::OwnedPinGuard, PinReport) {
        (unsafe { self.guard.into_owned() }, self.report)
    }

    pub fn report(&self) -> &PinReport {
        &self.report
    }
    pub fn pinned_bytes(&self) -> usize {
        self.guard.pinned_bytes()
    }
}

struct PreparedDataset<'a> {
    hint: PinHint,
    regions: Vec<&'a [u8]>,
}
fn prepared<'a, D>(
    index: &'a CompressedIndex<D>,
    base: &'a [u8],
    shape: QuantizedDatasetShape,
    requirements: &[DatasetRequirement],
) -> Result<Vec<PreparedDataset<'a>>> {
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    for &r in requirements {
        // A kind alone is not an identity. Reject a noncanonical recipe rather
        // than pinning another seed's slot. Dedup uses resolved descriptors.
        let r = match r {
            DatasetRequirement::Base => r,
            DatasetRequirement::Sidecar(s) => {
                let resolved = s.resolved(shape)?;
                if resolved != s.kind().requirement().resolved(shape)? {
                    return Err(Error::unsupported(
                        "pinning requires the prepared dataset's exact recipe",
                    ));
                }
                DatasetRequirement::Sidecar(resolved)
            }
        };
        if !seen.insert(r) {
            continue;
        }
        let (dataset, regions) = match r {
            DatasetRequirement::Base => {
                let mut regions = vec![base];
                if let Some((norms, orion_ann_platform::region::Placement::Resident)) =
                    index.native_norms_region()
                {
                    regions.push(norms);
                }
                (DatasetPin::Base, regions)
            }
            DatasetRequirement::Sidecar(s) => (
                DatasetPin::Quantized(s.kind()),
                index.resident_sidecar_regions(s.kind())?,
            ),
        };
        let bytes = regions
            .iter()
            .try_fold(0usize, |sum, r| sum.checked_add(r.len()))
            .ok_or_else(|| Error::invalid_argument("pin payload size overflow"))?;
        result.push(PreparedDataset {
            hint: PinHint { dataset, bytes },
            regions,
        });
    }
    result.sort_by_key(|p| p.hint.bytes);
    Ok(result)
}

pub(crate) fn hints<D>(
    index: &CompressedIndex<D>,
    base: &[u8],
    shape: QuantizedDatasetShape,
    requirements: &[DatasetRequirement],
) -> Result<Vec<PinHint>> {
    Ok(prepared(index, base, shape, requirements)?
        .into_iter()
        .map(|p| p.hint)
        .collect())
}

pub(crate) fn pin<'a, D>(
    index: &'a CompressedIndex<D>,
    base: &'a [u8],
    shape: QuantizedDatasetShape,
    requirements: &[DatasetRequirement],
    options: &PinOptions,
) -> Result<PinnedDatasets<'a>> {
    pin_with(
        index,
        base,
        shape,
        requirements,
        options,
        |guard, regions, budget| guard.try_pin(regions, budget),
    )
}

fn pin_with<'a, D>(
    index: &'a CompressedIndex<D>,
    base: &'a [u8],
    shape: QuantizedDatasetShape,
    requirements: &[DatasetRequirement],
    options: &PinOptions,
    mut lock: impl FnMut(&mut PinGuard<'a>, &[&'a [u8]], usize) -> std::io::Result<()>,
) -> Result<PinnedDatasets<'a>> {
    let mut result = PinnedDatasets {
        guard: PinGuard::default(),
        report: PinReport::default(),
    };
    if options.mode == PinMode::Off {
        return Ok(result);
    }
    let target = |r: &DatasetRequirement| match r {
        DatasetRequirement::Base => DatasetPin::Base,
        DatasetRequirement::Sidecar(s) => DatasetPin::Quantized(s.kind()),
    };
    if let Some(selected) = &options.datasets {
        if selected
            .iter()
            .any(|s| *s != DatasetPin::Graph && !requirements.iter().any(|r| target(r) == *s))
        {
            return Err(Error::invalid_argument(
                "pin selection is not in the search requirements",
            ));
        }
    }
    let selected: Vec<_> = requirements
        .iter()
        .copied()
        .filter(|r| {
            options
                .datasets
                .as_ref()
                .is_none_or(|s| s.contains(&target(r)))
        })
        .collect();
    let mut datasets = prepared(index, base, shape, &selected)?;
    if options
        .datasets
        .as_ref()
        .is_some_and(|s| s.contains(&DatasetPin::Graph))
    {
        datasets.push(PreparedDataset {
            hint: PinHint {
                dataset: DatasetPin::Graph,
                bytes: index.graph().buffer_bytes().len(),
            },
            regions: vec![index.graph().buffer_bytes()],
        });
        datasets.sort_by_key(|d| d.hint.bytes);
    }
    for dataset in datasets {
        result.report.requested_datasets += 1;
        match lock(&mut result.guard, &dataset.regions, options.budget_bytes) {
            Ok(()) => result.report.pinned_datasets += 1,
            Err(error) => {
                if options.mode == PinMode::Required {
                    // Dropping result also releases locks acquired earlier in this request.
                    return Err(Error::invalid_state(format!(
                        "cannot pin {:?}: {error}",
                        dataset.hint.dataset
                    ))
                    .at("pin"));
                }
                result
                    .report
                    .skipped
                    .push((dataset.hint.dataset, error.to_string()));
            }
        }
    }
    result.report.pinned_bytes = result.guard.pinned_bytes();
    log::info!(
        "RAM pin: {} / {} datasets, {} page-rounded bytes locked",
        result.report.pinned_datasets,
        result.report.requested_datasets,
        result.report.pinned_bytes,
    );
    for (dataset, reason) in &result.report.skipped {
        log::warn!("RAM pin skipped {dataset:?}: {reason}; retaining pageable storage");
    }
    Ok(result)
}

/// Add graph storage to the same requirement-driven residency transaction.
pub(crate) fn automatic_pin<'a, D: orion_ann_core::index::IndexData>(
    index: &'a CompressedIndex<D>,
    base: &'a [u8],
    shape: QuantizedDatasetShape,
    requirements: &[DatasetRequirement],
    enabled: bool,
    budget: Option<f64>,
) -> Result<PinnedDatasets<'a>> {
    // Explicit on-demand placement takes precedence over automatic code locks.
    // Owned PQ codebooks retain their independent bounded core policy.
    let requirements: Vec<_> = requirements
        .iter()
        .copied()
        .filter(|r| match r {
            DatasetRequirement::Base => {
                index.dataset().placement() == orion_ann_platform::region::Placement::Resident
            }
            DatasetRequirement::Sidecar(s) => !index.sidecar_is_mapped(s.kind()),
        })
        .collect();
    let mut plan = AutomaticPlan::detect(enabled, budget)?;
    plan.options.datasets = Some(
        requirements
            .iter()
            .map(|r| match r {
                DatasetRequirement::Base => DatasetPin::Base,
                DatasetRequirement::Sidecar(s) => DatasetPin::Quantized(s.kind()),
            })
            .chain(
                (index.graph().placement() == orion_ann_platform::region::Placement::Resident)
                    .then_some(DatasetPin::Graph),
            )
            .collect(),
    );
    let mut result = pin(index, base, shape, &requirements, &plan.options)?;
    log::info!(
        "Automatic RAM pin budget: {} bytes; pinned {} bytes",
        plan.options.budget_bytes,
        result.pinned_bytes()
    );
    result.report.automatic = Some(plan.observations);
    Ok(result)
}

#[cfg(test)]
mod tests;
