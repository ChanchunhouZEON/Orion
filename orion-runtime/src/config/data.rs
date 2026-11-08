/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Metric and resident representation, independent of graph/search policy.
use crate::data::VectorFormat;
use serde::{Deserialize, Serialize};

/// Distance / similarity objective used by the search pipeline.
///
/// The selected metric determines:
///
/// - which admission and rerank implementations are valid,
/// - how calibration thresholds are interpreted,
/// - whether vector normalization is required by the data path.
///
/// `InnerProduct` also accepts the common CLI/config aliases `mips` and `ip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SearchMetric {
    /// Squared Euclidean-distance search.
    L2,

    /// Maximum inner-product search (MIPS).
    #[serde(alias = "mips", alias = "ip")]
    #[cfg_attr(feature = "cli", value(alias = "mips", alias = "ip"))]
    InnerProduct,

    /// Cosine-similarity search.
    ///
    /// The search stages use the inner-product-oriented pipeline after the
    /// corresponding vector normalization has been applied.
    Cosine,
}

impl SearchMetric {
    pub fn calibration_metric(self) -> orion_ann_core::CalibrationMetric {
        match self {
            Self::L2 => orion_ann_core::CalibrationMetric::L2,
            Self::InnerProduct => orion_ann_core::CalibrationMetric::InnerProduct,
            Self::Cosine => orion_ann_core::CalibrationMetric::Cosine,
        }
    }
}

/// Element representation used for resident base vectors.
///
/// This is independent of the on-disk [`VectorFormat`]. For example, vectors read
/// from a `.bvecs` file may either remain as native `u8` values or be expanded into
/// `f32` values after loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum VectorStorageKind {
    /// Store each coordinate as `f32`.
    F32,

    /// Keep each coordinate in its native unsigned 8-bit representation if the base vector
    /// is in `u8` stored form.
    U8,
}

impl VectorStorageKind {
    /// Service input default. Benchmark presets intentionally choose their own default.
    pub(crate) fn for_input(format: VectorFormat) -> Self {
        if matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin) {
            Self::U8
        } else {
            Self::F32
        }
    }
}
