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
pub mod diskann_builder;
mod entry;
#[cfg(feature = "parlay")]
mod parlay;
mod partitions;
#[cfg(feature = "experimental-graph")]
pub mod supplier;

use crate::{Error, Orion, PaddedDataset, PhasedGraph, Result};
pub use entry::{sampled_medoid, sampled_medoid_rows};
mod config;
pub use config::{BuildBackend, BuildConfig, BuildMetric};

/// Cache recipe identity; relocation of the builder does not change its output.
pub const PARLAY_RECIPE: &str = "parlay-memory-v1-unquantized-two-pass";

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
                    dataset
                        .row(id)
                        .unwrap()
                        .iter()
                        .any(|&x| !<$ty as crate::index::DynamicElement>::valid_coordinate(x))
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
    orion_ann_vector::IpU8Distance,
    u8,
    |_: PaddedDataset<u8>, _: &BuildConfig, _: &orion_ann_platform::region::RegionOptions| Err(
        Error::unsupported("native u8 construction requires ParlayANN")
    )
);
