/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::visualization::utils::*;
use diskann::common::{ANNError, ANNResult};
use ndarray::ArcArray2;
use plotters::prelude::*;
use std::collections::HashMap;

pub fn draw_graph(
    data: &ArcArray2<f32>,
    graph: &HashMap<u32, Vec<u32>>,
    output_path: &str,
) -> ANNResult<()> {
    let reduced = reduce_to_2d(data);
    let root = BitMapBackend::new(output_path, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE).map_err(|_| {
        ANNError::log_visualization_error(
            "Could not build the basic information of the clustered graph".to_string(),
        )
    })?;

    let (xmin, xmax) = get_minimax(&reduced, 0);
    let (ymin, ymax) = get_minimax(&reduced, 1);
    let (xmin, xmax) = cal_padding_range(xmin, xmax);
    let (ymin, ymax) = cal_padding_range(ymin, ymax);

    let mut chart = ChartBuilder::on(&root)
        .margin(40) // Global margin for the drawing area
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|_| {
            ANNError::log_visualization_error(
                "Could not build the basic information of the graph".to_string(),
            )
        })?;

    for (&idx, neighbors) in graph.iter() {
        for &neighbor in neighbors.iter() {
            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![
                        (reduced[[idx as usize, 0]], reduced[[idx as usize, 1]]),
                        (
                            reduced[[neighbor as usize, 0]],
                            reduced[[neighbor as usize, 1]],
                        ),
                    ],
                    BLACK.mix(0.5).stroke_width(1),
                )))
                .map_err(|_| {
                    ANNError::log_visualization_error(
                        "Failed to draw edges of the graph".to_string(),
                    )
                })?;
        }
    }

    chart
        .draw_series((0..reduced.nrows() - 1).map(|i| {
            Circle::new(
                (reduced[[i, 0]], reduced[[i, 1]]),
                5,
                BLACK.mix(0.6).filled(),
            )
        }))
        .map_err(|_| {
            ANNError::log_visualization_error("Failed to draw nodes of the graph".to_string())
        })?;

    root.present().map_err(|_| {
        ANNError::log_visualization_error("Could not represent the image".to_string())
    })?;
    Ok(())
}
