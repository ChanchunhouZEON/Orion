/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! ParlayANN borrows padded input and writes navigation/extras into the final slab.
use super::*;
use orion_ann_parlay::{BuildOptions, Builder, Metric};
pub fn convert_error(error: orion_ann_parlay::Error) -> Error {
    let code = match error.code {
        1 => ErrorCode::InvalidArgument,
        2 => ErrorCode::ResourceExhausted,
        _ => ErrorCode::Backend,
    };
    Error::new(code, error.message.clone())
        .at("build_graph")
        .with_detail("backend", "parlayann")
        .with_source(error)
}

fn options(config: &BuildConfig) -> BuildOptions {
    BuildOptions {
        degree: config.degree,
        build_l: config.build_l as u32,
        max_extra: config.max_extra as u32,
        threads: config.threads as u32,
        local_pct: config.local_pct as u32,
        alpha: config.alpha as f64,
        metric: match config.metric {
            BuildMetric::L2 => Metric::L2,
            BuildMetric::InnerProduct => Metric::InnerProduct,
        },
        ..BuildOptions::default()
    }
}
macro_rules! build {
    ($name:ident, $ty:ty, $constructor:ident) => {
        pub(super) fn $name(
            dataset: PaddedDataset<$ty>,
            config: &BuildConfig,
            storage: &orion_ann_platform::region::RegionOptions,
        ) -> Result<Orion<$ty>> {
            let recipe = options(config);
            let graph = PhasedGraph::try_build_with_storage(
                dataset.len(),
                config.degree,
                config.max_extra,
                |slab, stride| {
                    Builder::$constructor(
                        dataset.as_slice(),
                        dataset.len(),
                        dataset.layout().logical_dim(),
                        dataset.layout().stride_elements(),
                        recipe,
                        slab,
                        stride,
                    )
                    .map_err(convert_error)
                },
                storage,
            )?;
            let entry = sampled_medoid_rows(dataset.len(), |id| dataset.row(id).unwrap());
            Ok(Orion::from_phased_graph(dataset, graph, entry))
        }
    };
}
build!(f32, f32, f32_into);
build!(u8, u8, u8_into);
