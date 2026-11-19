/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! One logical storage vocabulary for forecasting, preparation and reporting.
//! Descriptors describe complete payloads, not allocation ownership. Actual
//! allocation-owned leases live in platform regions, independently of descriptors.
use super::{scheduler::Frequency, StorageMode};
use crate::{Error, Result};
use orion_ann_platform::region::Placement;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};

/// Independently selectable storage objects. Sidecars are keyed by their
/// representation, so sharing one across stages cannot create conflicting rules.
/// PQ covers both its codebook and codes. Labels currently support resident only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum StorageObject {
    Base,
    Graph,
    Norms,
    Labels,
    L2U8,
    L2U16,
    MipsI8,
    MipsI16,
    Pq,
    L2Kt,
    JlSparse,
    JlSparseMips,
    JlHadamard,
    Rabitq,
    Rabitq4,
}

impl StorageObject {
    /// Reports whether this object can use file-backed storage under the current implementation.
    pub(super) fn supports_mapping(self) -> bool {
        self != Self::Labels
    }

    /// Maps an internal persistence codec to its public placement object, rejecting unknown codecs.
    pub(super) fn from_codec(name: &str) -> Result<Self> {
        Ok(match name {
            "base" => Self::Base,
            "graph" => Self::Graph,
            orion_ann_core::model::dataset::native_u8::NORM_CODEC => Self::Norms,
            "qds" => Self::L2U8,
            "qds6" => Self::L2U16,
            "qdm8" => Self::MipsI8,
            "qdm16" => Self::MipsI16,
            "pq-adc" => Self::Pq,
            "qdsl2kt" => Self::L2Kt,
            "jls" => Self::JlSparse,
            "jls_mips" => Self::JlSparseMips,
            "jlh" => Self::JlHadamard,
            "qrbq" => Self::Rabitq,
            "qrb4" => Self::Rabitq4,
            _ => return Err(Error::invalid_argument("unknown storage codec")),
        })
    }
}
impl FromStr for StorageObject {
    type Err = Error;

    /// Parses the same stable kebab-case object names accepted by serialized configuration.
    fn from_str(value: &str) -> Result<Self> {
        serde_json::from_value(serde_json::Value::String(value.into()))
            .map_err(|e| Error::invalid_argument(format!("unknown storage object: {e}")))
    }
}
impl fmt::Display for StorageObject {
    /// Writes the canonical configuration name used by placement reports and overrides.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = serde_json::to_value(self).map_err(|_| fmt::Error)?;
        f.write_str(name.as_str().ok_or(fmt::Error)?)
    }
}

/// One deduplicated storage demand before placement has been selected.
#[derive(Debug)]
pub(super) struct ObjectDescriptor {
    /// Internal codec key used to share a single allocation decision across stages.
    pub name: &'static str,

    /// Public representation identity used to match user overrides.
    pub object: StorageObject,

    /// Complete payload span to forecast, including representation padding.
    pub bytes: u64,

    /// Expected access priority used only during automatic placement.
    pub frequency: Frequency,

    /// Effective requested mode after applying the per-object override.
    pub mode: StorageMode,
}

/// Reportable placement decision for one logical object, not an allocation lease.
#[derive(Debug, serde::Serialize)]
pub(crate) struct ObjectDecision {
    /// Internal codec key used to apply this decision during preparation.
    #[serde(skip)]
    pub name: &'static str,

    /// Public representation identity shown to callers.
    pub object: StorageObject,

    /// Forecast payload bytes; mapped bytes do not represent measured RSS.
    pub bytes: u64,

    /// Effective user policy before automatic placement was resolved.
    pub requested: StorageMode,

    /// Resolved resident or mapped backing selected by the planner.
    pub placement: StorageMode,

    /// Whether a per-object entry supplied the requested policy.
    pub overridden: bool,
}
impl ObjectDecision {
    /// Copies a demand into a report entry with its resolved backing and override provenance.
    pub(super) fn new(d: &ObjectDescriptor, p: Placement, overridden: bool) -> Self {
        Self {
            name: d.name,
            object: d.object,
            bytes: d.bytes,
            requested: d.mode,
            placement: if p == Placement::Resident {
                StorageMode::Resident
            } else {
                StorageMode::Mapped
            },
            overridden,
        }
    }
}
