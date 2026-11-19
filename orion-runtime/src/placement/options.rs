/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Public deployment policy. Names identify representations, never cache codecs
//! or cascade positions. Parsing and validation are shared by all frontends.
use super::object::StorageObject;
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf, str::FromStr};

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

impl FromStr for StorageMode {
    type Err = Error;

    /// Parses the shared serialized storage-mode vocabulary and rejects unknown values.
    fn from_str(value: &str) -> Result<Self> {
        serde_json::from_value(serde_json::Value::String(value.into()))
            .map_err(|_| Error::invalid_argument("storage mode must be auto, resident or mapped"))
    }
}

/// Deployment settings, independent of cascade, cache identity and calibration.
///
/// Per-object entries override `mode`. Explicit resident allocations take budget
/// priority over automatic choices; exceeding the budget never silently maps them.
/// An override does not build an unused sidecar. Labels remain resident-only.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StorageOptions {
    /// Default placement for mappable objects without a per-object override.
    pub mode: StorageMode,
    /// Existing disk-backed directory; None uses the OS temporary directory.
    /// Mapped payloads occupy private files for the lifetime of the index.
    pub directory: Option<PathBuf>,
    /// Stable representation names; PQ currently selects its entire sidecar.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub objects: BTreeMap<StorageObject, StorageMode>,
}
impl StorageOptions {
    /// Checks whether deployment policy can be omitted from serialized configuration.
    pub(crate) fn is_default(&self) -> bool {
        self == &Self::default()
    }

    /// Rejects explicit mapped placement for objects without mapping support.
    pub(crate) fn validate(&self) -> Result<()> {
        for (&object, &mode) in &self.objects {
            if mode == StorageMode::Mapped && !object.supports_mapping() {
                return Err(
                    Error::unsupported("storage object does not support mapped placement")
                        .with_detail("object", object.to_string())
                        .with_detail("requested", "mapped"),
                );
            }
        }
        Ok(())
    }

    /// Resolves one override, keeping resident-only objects resident under a global mapped policy.
    pub(super) fn requested(&self, object: StorageObject) -> StorageMode {
        self.objects.get(&object).copied().unwrap_or_else(|| {
            // Global mapped applies to payloads capable of mapping. Mutable
            // label containers retain resident backing; an explicit request is
            // rejected by validate rather than being silently ignored.
            if object.supports_mapping() {
                self.mode
            } else {
                StorageMode::Resident
            }
        })
    }

    /// Reports whether any configured placement still requires automatic resource planning.
    pub(super) fn uses_auto(&self) -> bool {
        self.mode == StorageMode::Auto || self.objects.values().any(|m| *m == StorageMode::Auto)
    }
}
