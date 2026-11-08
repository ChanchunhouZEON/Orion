/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Standalone, in-memory graph construction. No runtime, YAML, or cache is required.
//!
//! ParlayANN borrows the base and fills the final slab, including extras. The
//! Rust DiskANN backend retains its existing temporary graph and base allocation.
//! Neither backend writes a file. Calibration and saving remain explicit.

mod diskann;
mod entry;
#[cfg(feature = "parlay")]
mod parlay;

use crate::{Error, ErrorCode, Orion, PaddedDataset, PhasedGraph, Result};
pub use entry::{sampled_medoid, sampled_medoid_rows};
use serde::{Deserialize, Serialize};

/// Cache recipe identity; relocation of the builder does not change its output.
pub const PARLAY_RECIPE: &str = "parlay-memory-v1-unquantized-two-pass";

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
/// For cosine data, normalize rows before building with L2. This API does not
/// normalize, quantize, or silently reinterpret coordinates.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BuildMetric {
    #[default]
    L2,
    /// Supported by ParlayANN on f32 input only.
    InnerProduct,
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
            metric: BuildMetric::L2,
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
        if !(1..=crate::algorithm::search::MAX_WORKERS).contains(&self.threads) {
            return Err(invalid("threads", "must be in 1..=64"));
        }
        if !(1..=100).contains(&self.local_pct) {
            return Err(invalid("local_pct", "must be in 1..=100"));
        }
        Ok(())
    }

    fn backend(&self, count: usize, dimension: usize, native_u8: bool) -> Result<BuildBackend> {
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
                if native_u8 && self.metric != BuildMetric::L2 {
                    return Err(Error::unsupported("native u8 construction requires L2")
                        .at("build_graph")
                        .with_detail("backend", "parlayann"));
                }
                if native_u8 && dimension > i32::MAX as usize / (255 * 255) {
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
                    || self.metric != BuildMetric::L2
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

    fn install<R: Send>(&self, build: impl FnOnce() -> Result<R> + Send) -> Result<R> {
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

macro_rules! build_api {
    ($ty:ty, $native:expr, $kernel:ty, $pa:ident, $rust:expr) => {
        impl Orion<$ty> {
            /// Build an index in memory, taking ownership of the padded base.
            ///
            /// Uses the configured worker count and preserves the base allocation.
            /// ParlayANN writes directly into the final slab, including extras;
            /// DiskANN uses its existing temporary storage. No cache file or
            /// calibration sidecar is created. Search calibration remains explicit.
            ///
            /// # Errors
            /// Invalid controls, shape or coordinates return `InvalidArgument`;
            /// unavailable backends/metrics return `UnsupportedConfiguration`. Backend and
            /// worker-pool failures retain their source. C++ exceptions are caught
            /// by the FFI bridge; Rust allocation/invariant panics follow the host
            /// panic policy rather than being converted into recoverable errors.
            pub fn build(dataset: PaddedDataset<$ty>, config: BuildConfig) -> Result<Self> {
                Self::build_with_storage(dataset, config, &Default::default())
            }

            /// Construct the final graph directly in the selected private backing.
            pub fn build_with_storage(
                dataset: PaddedDataset<$ty>,
                config: BuildConfig,
                storage: &orion_ann_platform::region::RegionOptions,
            ) -> Result<Self> {
                let backend =
                    config.backend(dataset.len(), dataset.layout().logical_dim(), $native)?;
                // Inspect only logical coordinates; SIMD padding is not input data.
                if (0..dataset.len()).any(|id| {
                    dataset.row(id).unwrap().iter().any(|&x| {
                        !<$ty as crate::index::compressed_index::DynamicElement>::valid_coordinate(
                            x,
                        )
                    })
                }) {
                    return Err(
                        Error::invalid_argument("base contains a non-finite coordinate")
                            .at("build_graph"),
                    );
                }
                config.install(|| match backend {
                    #[cfg(feature = "parlay")]
                    BuildBackend::ParlayAnn => parlay::$pa(dataset, &config, storage),
                    BuildBackend::DiskAnn => ($rust)(dataset, &config, storage),
                    _ => Err(
                        Error::unsupported("ParlayANN builder requires the parlay feature")
                            .at("build_graph"),
                    ),
                })
            }

            /// Build from packed, row-major input, copying once into padded storage.
            /// Use [`Self::build`] to transfer an existing allocation instead.
            ///
            /// # Errors
            /// Rejects empty/ragged input, zero dimension and invalid options before
            /// allocation; also propagates the errors documented by [`Self::build`].
            pub fn build_from_slice(
                values: &[$ty],
                dimension: usize,
                config: BuildConfig,
            ) -> Result<Self> {
                if dimension == 0 || values.is_empty() || values.len() % dimension != 0 {
                    return Err(Error::invalid_argument(
                        "base must be a nonempty rectangular matrix",
                    )
                    .at("build_graph"));
                }
                config.backend(values.len() / dimension, dimension, $native)?;
                let layout = crate::PaddedLayout::for_kernel::<$kernel>(dimension)
                    .map_err(Error::invalid_argument)?;
                let dataset =
                    PaddedDataset::from_packed(values, layout).map_err(Error::invalid_argument)?;
                Self::build(dataset, config)
            }
        }
    };
}
build_api!(
    f32,
    false,
    orion_ann_vector::L2F32Distance,
    f32,
    diskann::build
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_limit_is_local_and_matching_pools_are_reused() {
        let config = BuildConfig {
            threads: 2,
            ..Default::default()
        };
        let outer = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        outer.install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            config
                .install(|| {
                    assert_eq!(rayon::current_num_threads(), 2);
                    let worker = std::thread::current().id();
                    config.install(|| {
                        assert_eq!(std::thread::current().id(), worker);
                        Ok(())
                    })
                })
                .unwrap();
            assert_eq!(rayon::current_num_threads(), 1);
        });
    }

    #[cfg(feature = "parlay")]
    #[test]
    fn ffi_limits_are_checked_without_allocating_the_slab() {
        let config = BuildConfig::default();
        assert!(config.backend(i32::MAX as usize + 1, 128, false).is_err());
        assert!(
            config
                .backend(1, i32::MAX as usize / (255 * 255) + 1, true)
                .is_err()
        );
        assert!(config.backend(1_000_000_000, 128, true).is_ok());
    }
}
build_api!(
    u8,
    true,
    orion_ann_vector::L2U8Distance,
    u8,
    |_: PaddedDataset<u8>, _: &BuildConfig, _: &orion_ann_platform::region::RegionOptions| Err(
        Error::unsupported("native u8 construction requires ParlayANN")
    )
);
