/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Allocation requirements for one immutable base dataset.
//!
//! A list is scoped to one base identity and shape. Distinct bases must use
//! separate lists. Equal descriptors share storage; different seeds or transform
//! widths never collapse just because their memory sizes happen to match.
//! Descriptors and report types precede the list-level estimate/prepare operations.
//! Per-kind recipe resolution and dataset dispatch are implementation details below.
use crate::Result;
use orion_ann_core::model::dataset::{
    quantized_dataset::HadamardOptions, rabitq_dataset::RabitQDataset, *,
};

/// Official representations supported by the index's canonical lazy slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SidecarKind {
    L2U8,
    L2U16,
    MipsI8,
    MipsI16,
    L2Kt,
    JlSparse,
    JlSparseMips,
    JlHadamard,
    Rabitq,
    Rabitq4,
}

/// Fully specified representation, without allocating its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum SidecarRequirement {
    L2U8,
    L2U16,
    MipsI8,
    MipsI16,
    L2Kt,
    JlSparse(u64),
    JlSparseMips(u64),
    JlHadamard(HadamardOptions),
    Rabitq(u64),
    Rabitq4(u64),
}

/// Per-allocation detail. Repeated consumers do not create repeated entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SidecarMemory {
    pub(crate) requirement: SidecarRequirement,
    pub(crate) memory: ResidentMemory,
}

/// Deduplicated resident allocation report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SidecarMemoryReport {
    allocations: Vec<SidecarMemory>,
    total_bytes: usize,
}

impl SidecarMemoryReport {
    /// Unique allocations in first-use order.
    pub(crate) fn allocations(&self) -> &[SidecarMemory] {
        &self.allocations
    }
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

/// Sum unique descriptors for a single base dataset, in first-use order.
///
/// # Errors
/// Rejects invalid dataset recipes and overflow in the aggregate byte count.
pub(crate) fn estimate_sidecars(
    input_shape: QuantizedDatasetShape,
    requirements: &[SidecarRequirement],
) -> Result<SidecarMemoryReport> {
    let mut report = SidecarMemoryReport::default();
    let mut seen = std::collections::HashSet::new();
    for &requirement in requirements {
        let requirement = requirement.resolved(input_shape)?;
        if !seen.insert(requirement) {
            continue;
        }
        let memory = requirement.estimate(input_shape)?;
        report.total_bytes = report
            .total_bytes
            .checked_add(memory.total_bytes())
            .ok_or_else(|| crate::Error::invalid_argument("stacked sidecar memory overflow"))?;
        report.allocations.push(SidecarMemory {
            requirement,
            memory,
        });
    }
    Ok(report)
}

/// Prepare canonical sidecars from the same list used by preflight.
///
/// # Errors
/// Rejects noncanonical recipes (build those as standalone datasets) and
/// propagates construction errors. The index has one lazy slot per kind.
pub(crate) fn prepare<D: orion_ann_core::model::dataset::rows::F32Rows>(
    index: &orion_ann_core::CompressedIndex<D>,
    requirements: &[SidecarRequirement],
) -> Result<()> {
    let input_shape = QuantizedDatasetShape {
        rows: index.dataset().len(),
        dimension: index.dataset().dimension(),
    };
    // Validate the entire request before publishing any sidecar.
    for &requirement in requirements {
        if requirement.resolved(input_shape)?
            != requirement.kind().requirement().resolved(input_shape)?
        {
            return Err(crate::Error::unsupported(
                "index lazy slots require canonical sidecar recipes",
            ));
        }
    }
    for requirement in requirements {
        match requirement.kind() {
            SidecarKind::L2U8 => {
                index.try_ensure_quantized_dataset()?;
            }
            SidecarKind::L2U16 => {
                index.try_ensure_quantized_dataset_l2_u16()?;
            }
            SidecarKind::MipsI8 => {
                index.try_ensure_quantized_dataset_mips()?;
            }
            SidecarKind::MipsI16 => {
                index.try_ensure_quantized_dataset_mips_i16()?;
            }
            SidecarKind::L2Kt => {
                index.try_ensure_quantized_dataset_l2_kt()?;
            }
            SidecarKind::JlSparse => {
                index.try_ensure_quantized_dataset_jl()?;
            }
            SidecarKind::JlSparseMips => {
                index.try_ensure_quantized_dataset_jl_mips()?;
            }
            SidecarKind::JlHadamard => {
                index.try_ensure_quantized_dataset_jl_hadamard()?;
            }
            SidecarKind::Rabitq => {
                index.try_ensure_quantized_dataset_rabitq()?;
            }
            SidecarKind::Rabitq4 => {
                index.try_ensure_quantized_dataset_rabitq_b4()?;
            }
        }
    }
    Ok(())
}

impl SidecarKind {
    /// Recipe shared by lazy construction and budgeting.
    pub(crate) fn requirement(self) -> SidecarRequirement {
        match self {
            Self::L2U8 => SidecarRequirement::L2U8,
            Self::L2U16 => SidecarRequirement::L2U16,
            Self::MipsI8 => SidecarRequirement::MipsI8,
            Self::MipsI16 => SidecarRequirement::MipsI16,
            Self::L2Kt => SidecarRequirement::L2Kt,
            Self::JlSparse => {
                SidecarRequirement::JlSparse(<JLSparseDataset<1024, 9>>::default_build_options())
            }
            Self::JlSparseMips => SidecarRequirement::JlSparseMips(
                <JLSparseDatasetMips<1024, 9>>::default_build_options(),
            ),
            Self::JlHadamard => {
                SidecarRequirement::JlHadamard(<JlHadamardDataset>::default_build_options())
            }
            Self::Rabitq => SidecarRequirement::Rabitq(<RabitQDataset>::default_build_options()),
            Self::Rabitq4 => SidecarRequirement::Rabitq4(<RabitQ4Dataset>::default_build_options()),
        }
    }
}

impl SidecarRequirement {
    pub(crate) fn kind(self) -> SidecarKind {
        match self {
            Self::L2U8 => SidecarKind::L2U8,
            Self::L2U16 => SidecarKind::L2U16,
            Self::MipsI8 => SidecarKind::MipsI8,
            Self::MipsI16 => SidecarKind::MipsI16,
            Self::L2Kt => SidecarKind::L2Kt,
            Self::JlSparse(_) => SidecarKind::JlSparse,
            Self::JlSparseMips(_) => SidecarKind::JlSparseMips,
            Self::JlHadamard(_) => SidecarKind::JlHadamard,
            Self::Rabitq(_) => SidecarKind::Rabitq,
            Self::Rabitq4(_) => SidecarKind::Rabitq4,
        }
    }
    // Normalize defaults before comparing identities. Explicitly requesting the
    // canonical Hadamard width and leaving it implicit must share one buffer.
    pub(crate) fn resolved(self, input_shape: QuantizedDatasetShape) -> Result<Self> {
        self.estimate(input_shape)?;
        Ok(match self {
            Self::JlHadamard(options) => Self::JlHadamard(HadamardOptions {
                padded_dimension: Some(match options.padded_dimension {
                    Some(width) => width,
                    None => input_shape
                        .dimension
                        .max(1024)
                        .checked_next_power_of_two()
                        .ok_or_else(|| {
                            crate::Error::invalid_argument("Hadamard dimension overflow")
                        })?,
                }),
                ..options
            }),
            other => other,
        })
    }

    /// Dispatch only the allocation estimate; no cascade or base reads.
    ///
    /// # Errors
    /// Propagates shape, recipe and allocation-size errors from the dataset.
    pub(crate) fn estimate(self, input_shape: QuantizedDatasetShape) -> Result<ResidentMemory> {
        match self {
            Self::L2U8 => {
                <ScalarQuantizedDataset<L2U8>>::estimate_resident_memory(input_shape, &())
            }
            Self::L2U16 => {
                <ScalarQuantizedDataset<L2U16>>::estimate_resident_memory(input_shape, &())
            }
            Self::MipsI8 => {
                <ScalarQuantizedDataset<MipsI8>>::estimate_resident_memory(input_shape, &())
            }
            Self::MipsI16 => {
                <ScalarQuantizedDataset<MipsI16>>::estimate_resident_memory(input_shape, &())
            }
            Self::L2Kt => <L2KTDataset>::estimate_resident_memory(input_shape, &()),
            Self::JlSparse(options) => {
                <JLSparseDataset<1024, 9>>::estimate_resident_memory(input_shape, &options)
            }
            Self::JlSparseMips(options) => {
                <JLSparseDatasetMips<1024, 9>>::estimate_resident_memory(input_shape, &options)
            }
            Self::JlHadamard(options) => {
                <JlHadamardDataset>::estimate_resident_memory(input_shape, &options)
            }
            Self::Rabitq(options) => {
                <RabitQDataset>::estimate_resident_memory(input_shape, &options)
            }
            Self::Rabitq4(options) => {
                <RabitQ4Dataset>::estimate_resident_memory(input_shape, &options)
            }
        }
    }
}

#[cfg(test)]
mod tests;
