/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Rust Vamana adapter. Its working graph is released before creating the slab.
use super::*;
use crate::build_diskann_index;

pub(super) fn build(dataset: PaddedDataset<f32>, config: &BuildConfig) -> Result<Orion> {
    let num_points = dataset.len();
    // DiskANN's builder still owns typed storage. Pack only when SIMD row
    // padding differs from its input; cached/STAG paths never make this copy.
    let dimension = dataset.layout().logical_dim();
    let build_dimension = [32, 100, 104, 128, 256, 768, 784, 960, 1536].into_iter()
        .find(|&d| d >= dimension)
        .ok_or("Rust/DiskANN builder supports at most 1536 dimensions; import a ParlayANN STAG graph for larger inputs")?;
    let packed;
    let input = if dataset.layout().stride_elements() == build_dimension {
        dataset.as_slice()
    } else {
        let count = num_points
            .checked_mul(build_dimension)
            .ok_or("builder input size overflow")?;
        packed = {
            let mut values = vec![0.; count];
            for (id, row) in values.chunks_exact_mut(build_dimension).enumerate() {
                row[..dimension].copy_from_slice(dataset.row(id).unwrap());
            }
            values
        };
        &packed
    };
    let build = build_diskann_index(
        input,
        num_points,
        build_dimension,
        config.alpha,
        config.degree,
        config.build_l as u32,
        false,
        None,
        None,
        true,
        config.max_extra,
    )
    .map_err(Error::from)?;

    // Release the builder's working index before allocating the final graph slab.
    let entry = build.entry_point;
    drop(build.index);

    // Saving and sidecar calibration are separate operations.
    let graph =
        PhasedGraph::try_from_partitions(&build.partitions, config.degree, config.max_extra)?;
    Ok(Orion::from_phased_graph(dataset, graph, entry))
}
