//! Borrow padded base rows during C++ construction and fill the final Rust slab.
use crate::{config::ResolvedRunConfig, search::cascade::SearchMetric, Error, ErrorCode, Result};
use orion_ann::{Orion, PhasedGraph};
use orion_ann_diskann::model::PaddedDataset;
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

pub fn options(config: &ResolvedRunConfig) -> Result<BuildOptions> {
    Ok(BuildOptions {
        degree: config.graph_degree,
        build_l: config
            .build_l
            .try_into()
            .map_err(|_| Error::invalid_argument("build_l exceeds u32"))?,
        max_extra: config
            .max_extra
            .try_into()
            .map_err(|_| Error::invalid_argument("max_extra exceeds u32"))?,
        threads: config
            .sweep
            .threads
            .try_into()
            .map_err(|_| Error::invalid_argument("threads exceeds u32"))?,
        local_pct: config
            .legacy_local_pct
            .try_into()
            .map_err(|_| Error::invalid_argument("local_pct exceeds u32"))?,
        alpha: config.alpha as f64,
        metric: if config.metric == SearchMetric::InnerProduct {
            Metric::InnerProduct
        } else {
            Metric::L2
        },
        ..BuildOptions::default()
    })
}

pub fn collect_graph(
    builder: &mut Builder<'_>,
    n: usize,
    degree: u32,
    extra: usize,
) -> Result<PhasedGraph> {
    let mut graph = PhasedGraph::try_empty(n, degree, extra)?;
    for node in 0..n {
        let part = builder
            .next_partition()
            .map_err(convert_error)?
            .ok_or_else(|| Error::invalid_data("ParlayANN returned too few partitions"))?;
        if part.extra.len() > extra {
            return Err(Error::invalid_data(
                "ParlayANN extras exceed configured capacity",
            ));
        }
        graph.set_partition(node, part.local, part.remote, part.extra)?;
    }
    if builder.next_partition().map_err(convert_error)?.is_some() {
        return Err(Error::invalid_data(
            "ParlayANN returned too many partitions",
        ));
    }
    Ok(graph)
}

macro_rules! build {
    ($name:ident, $ty:ty, $constructor:ident) => {
        pub fn $name(
            config: &ResolvedRunConfig,
            dataset: PaddedDataset<$ty>,
        ) -> Result<Orion<$ty>> {
            let recipe = options(config)?;
            let graph = PhasedGraph::try_build_with(
                dataset.len(),
                config.graph_degree,
                config.max_extra,
                |slab, slab_stride| {
                    Builder::$constructor(
                        dataset.as_slice(),
                        dataset.len(),
                        dataset.layout().logical_dim(),
                        dataset.layout().stride_elements(),
                        recipe,
                        slab,
                        slab_stride,
                    )
                    .map_err(convert_error)
                },
            )?;
            let entry =
                super::parlayann::sampled_medoid_rows(dataset.len(), |id| dataset.row(id).unwrap());
            Ok(Orion::from_phased_graph(dataset, graph, entry, None, None))
        }
    };
}
build!(build_f32, f32, f32_into);
build!(build_u8, u8, u8_into);
