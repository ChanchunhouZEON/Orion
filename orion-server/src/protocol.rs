/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! JSON request shapes. Numerical validation remains in the native runtime.
use orion_ann::{BaseInput, IndexOptions, QueryBatch, Result, StorageOptions};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Base {
    File { path: PathBuf },
    F32 { values: Vec<f32>, dimension: usize },
    U8 { values: Vec<u8>, dimension: usize },
}
impl Base {
    pub fn prepare(self, root: &crate::paths::Paths) -> Result<BaseInput> {
        match self {
            Self::File { path } => Ok(BaseInput::from_file(root.input(&path)?)),
            Self::F32 { values, dimension } => BaseInput::from_f32(&values, dimension),
            Self::U8 { values, dimension } => BaseInput::from_u8(&values, dimension),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Queries {
    F32 { values: Vec<f32>, dimension: usize },
    U8 { values: Vec<u8>, dimension: usize },
}
impl Queries {
    pub fn batch(&self) -> Result<QueryBatch<'_>> {
        match self {
            Self::F32 { values, dimension } => QueryBatch::from_f32(values, *dimension),
            Self::U8 { values, dimension } => QueryBatch::from_u8(values, *dimension),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CalibrationRequest {
    pub queries: Vec<f32>,
    pub k: usize,
    pub search_list_size: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct BuildRequest {
    pub base: Base,
    #[serde(default)]
    pub options: IndexOptions,
    #[serde(default)]
    pub calibrations: Vec<CalibrationRequest>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct OpenOverrides {
    pub threads: Option<usize>,
    pub memory_budget_gib: Option<f64>,
    pub auto_pin: Option<bool>,
    pub storage_options: Option<StorageOptions>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LoadRequest {
    /// A snapshot ID created by this service, never an arbitrary filesystem path.
    pub snapshot_id: String,
    pub base: Base,
    #[serde(default)]
    pub options: OpenOverrides,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SearchRequest {
    pub queries: Queries,
    pub k: usize,
    pub search_list_size: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct SearchResponse {
    pub ids: Vec<u32>,
    pub counts: Vec<u32>,
    pub k: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TraceRequest {
    pub query: Vec<f32>,
    pub k: usize,
    pub search_list_size: usize,
}
