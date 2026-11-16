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
    pub(super) fn supports_mapping(self) -> bool {
        self != Self::Labels
    }

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
    fn from_str(value: &str) -> Result<Self> {
        serde_json::from_value(serde_json::Value::String(value.into()))
            .map_err(|e| Error::invalid_argument(format!("unknown storage object: {e}")))
    }
}
impl fmt::Display for StorageObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = serde_json::to_value(self).map_err(|_| fmt::Error)?;
        f.write_str(name.as_str().ok_or(fmt::Error)?)
    }
}

#[derive(Debug)]
pub(super) struct ObjectDescriptor {
    pub name: &'static str,
    pub object: StorageObject,
    pub bytes: u64,
    pub frequency: Frequency,
    pub mode: StorageMode,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct ObjectDecision {
    #[serde(skip)]
    pub name: &'static str,
    pub object: StorageObject,
    pub bytes: u64,
    pub requested: StorageMode,
    pub placement: StorageMode,
    pub overridden: bool,
}
impl ObjectDecision {
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
