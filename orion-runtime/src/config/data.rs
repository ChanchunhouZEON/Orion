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
/// `SquaredL2` uses the canonical config name `squared-l2`; `l2` and
/// `squared_l2` remain accepted input aliases. Scores never take a square root.
///
/// `InnerProduct` also accepts the common CLI/config aliases `mips` and `ip`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "cli", derive(clap::ValueEnum))]
pub enum SearchMetric {
    /// Squared Euclidean-distance search.
    #[serde(alias = "l2", alias = "squared_l2")]
    #[cfg_attr(feature = "cli", value(alias = "l2", alias = "squared_l2"))]
    SquaredL2,

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
    /// Stable identity token for persisted recipes, independent of public naming.
    /// Keep the original L2 token so spelling changes do not invalidate graph,
    /// sidecar or snapshot identities; this is not a user-facing metric name.
    pub(crate) fn identity_key(self) -> &'static str {
        match self {
            Self::SquaredL2 => "l2",
            Self::InnerProduct => "inner-product",
            Self::Cosine => "cosine",
        }
    }

    pub fn calibration_metric(self) -> orion_ann_core::CalibrationMetric {
        match self {
            Self::SquaredL2 => orion_ann_core::CalibrationMetric::SquaredL2,
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

#[cfg(test)]
mod tests {
    use super::SearchMetric;

    #[test]
    fn squared_l2_names_have_one_canonical_representation() {
        for name in ["squared-l2", "squared_l2", "l2"] {
            let metric: SearchMetric = serde_json::from_value(serde_json::json!(name)).unwrap();
            assert_eq!(metric, SearchMetric::SquaredL2);
            assert_eq!(serde_json::to_value(metric).unwrap(), "squared-l2");
            assert_eq!(
                metric.calibration_metric(),
                orion_ann_core::CalibrationMetric::SquaredL2
            );
            #[cfg(feature = "cli")]
            assert_eq!(
                <SearchMetric as clap::ValueEnum>::from_str(name, false).unwrap(),
                metric
            );
        }
    }
}
