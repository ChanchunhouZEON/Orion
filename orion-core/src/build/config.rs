/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Graph construction controls and capability validation.
use crate::{Error, ErrorCode, Result};
use serde::{Deserialize, Serialize};
/// Graph construction backend. Explicit choices never silently fall back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuildBackend {
    /// Prefer the bundled ParlayANN feature; otherwise use Rust DiskANN for f32.
    #[default]
    Auto,
    /// Bundled C++ builder, available with the default `parlay` feature.
    ParlayAnn,
    /// Rust Vamana builder; supports f32 squared L2 only.
    DiskAnn,
}

/// Construction distance, independent of the search cascade.
///
/// For f32 cosine data, normalize rows before building with squared L2. Native u8
/// supports norm-adjusted cosine directly. This API does not
/// normalize, quantize, or silently reinterpret coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuildMetric {
    #[default]
    /// Sum of squared coordinate differences; smaller is better.
    #[serde(alias = "l2", alias = "squared_l2")]
    SquaredL2,
    /// Supported by ParlayANN on f32 and native u8 input.
    InnerProduct,
    /// Cosine over native u8 with wide, exact squared norms.
    Cosine,
}

/// Construction controls shared by standalone Rust and the service runtime.
///
/// # Example
/// ```
/// use orion_ann_core::{BuildConfig, Orion};
/// let base: Vec<f32> = (0..32 * 13).map(|i| (i % 71) as f32).collect();
/// let index = Orion::<f32>::build_from_slice(&base, 13, BuildConfig {
///     degree: 4, build_l: 8, max_extra: 2, threads: 2,
///     ..Default::default()
/// })?;
/// assert_eq!(index.len(), 32);
/// // Optional: index.save(path)?;
/// # Ok::<(), orion_ann_core::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BuildConfig {
    pub backend: BuildBackend,
    pub metric: BuildMetric,
    /// Maximum navigation degree, positive and fitting a graph row header.
    pub degree: u32,
    /// Construction candidate width, at least `degree` and fitting u32.
    pub build_l: usize,
    /// Finite pruning multiplier, at least 1.
    pub alpha: f32,
    /// Maximum retained extra candidates per node.
    pub max_extra: usize,
    /// Worker limit for both the backend and sampled entry selection (1..=64).
    /// Does not configure or replace the global Rayon pool.
    pub threads: usize,
    /// ParlayANN local-neighbor percentage (1..=100). Rust uses its fixed split
    /// and requires this to remain at the default 60.
    pub local_pct: usize,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            backend: BuildBackend::Auto,
            metric: BuildMetric::SquaredL2,
            degree: 64,
            build_l: 100,
            alpha: 1.2,
            max_extra: 16,
            threads: 8,
            local_pct: 60,
        }
    }
}

impl BuildConfig {
    /// Validate common parameter bounds without requiring a backend or dataset.
    /// Cache/import callers can use this even with the builder feature disabled.
    ///
    /// # Errors
    /// Returns `InvalidArgument`, identifying the offending field in `details`.
    pub fn validate(&self) -> Result<()> {
        let invalid = |field, message| Error::configuration(field, message).at("build_graph");
        if self.degree == 0 || self.degree > u32::MAX - 4 {
            return Err(invalid(
                "degree",
                "must be positive with room for graph row headers",
            ));
        }
        if self.build_l < self.degree as usize || self.build_l > u32::MAX as usize {
            return Err(invalid("build_l", "must be at least degree and fit u32"));
        }
        if !self.alpha.is_finite() || self.alpha < 1.0 {
            return Err(invalid("alpha", "must be finite and at least 1"));
        }
        if self.max_extra > (u32::MAX - 4 - self.degree) as usize {
            return Err(invalid(
                "max_extra",
                "degree plus extras and headers must fit u32",
            ));
        }
        if !(1..=crate::search::MAX_WORKERS).contains(&self.threads) {
            return Err(invalid("threads", "must be in 1..=64"));
        }
        if !(1..=100).contains(&self.local_pct) {
            return Err(invalid("local_pct", "must be in 1..=100"));
        }
        Ok(())
    }

    pub(super) fn backend(
        &self,
        count: usize,
        dimension: usize,
        native_u8: bool,
    ) -> Result<BuildBackend> {
        self.validate()?;
        if count == 0 || count >= u32::MAX as usize || dimension == 0 {
            return Err(Error::invalid_argument(
                "base must have 1..u32::MAX rows and nonzero dimension",
            )
            .at("build_graph"));
        }
        let backend = match self.backend {
            BuildBackend::Auto if cfg!(feature = "parlay") => BuildBackend::ParlayAnn,
            BuildBackend::Auto => BuildBackend::DiskAnn,
            explicit => explicit,
        };
        match backend {
            BuildBackend::ParlayAnn => {
                if !cfg!(feature = "parlay") {
                    return Err(Error::unsupported(
                        "ParlayANN builder requires the parlay feature",
                    )
                    .at("build_graph"));
                }
                // Match the C++ bridge's signed ID/accumulator bounds before
                // allocating a potentially very large output slab.
                if count > i32::MAX as usize
                    || dimension > i32::MAX as usize / 4
                    || self.degree > i32::MAX as u32
                    || self.build_l > i32::MAX as usize
                    || self.max_extra > i32::MAX as usize
                {
                    return Err(Error::invalid_argument(
                        "ParlayANN shape or graph capacity exceeds signed 32-bit bounds",
                    )
                    .at("build_graph"));
                }
                if !native_u8 && self.metric == BuildMetric::Cosine {
                    return Err(Error::unsupported(
                        "f32 cosine construction requires normalized rows and L2",
                    ));
                }
                if native_u8
                    && self.metric == BuildMetric::SquaredL2
                    && dimension > i32::MAX as usize / (255 * 255)
                {
                    // Keep the established FFI error contract, while rejecting
                    // the shape before allocating the graph slab.
                    return Err(Error::invalid_argument(
                        "u8 dimension exceeds int32 distance accumulator range",
                    )
                    .at("build_graph")
                    .with_detail("backend", "parlayann"));
                }
            }
            BuildBackend::DiskAnn => {
                if native_u8
                    || self.metric != BuildMetric::SquaredL2
                    || dimension > 1536
                    || self.local_pct != 60
                {
                    return Err(Error::unsupported("Rust DiskANN construction requires f32 L2, dimension <= 1536 and local_pct=60; use ParlayANN for other supported inputs").at("build_graph"));
                }
            }
            BuildBackend::Auto => unreachable!("resolved above"),
        }
        Ok(backend)
    }

    pub(super) fn install<R: Send>(&self, build: impl FnOnce() -> Result<R> + Send) -> Result<R> {
        // The runtime already installs its worker pool. Reuse it only when its
        // size matches; standalone callers get an isolated, bounded pool.
        if rayon::current_thread_index().is_some() && rayon::current_num_threads() == self.threads {
            return build();
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .build()
            .map_err(|e| {
                Error::new(
                    ErrorCode::ResourceExhausted,
                    "could not create build worker pool",
                )
                .at("build_graph")
                .with_source(e)
            })?;
        pool.install(build)
    }
}
