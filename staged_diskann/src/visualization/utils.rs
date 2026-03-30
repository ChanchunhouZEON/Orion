/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use ndarray::{ArcArray2, Array2, Axis};
use plotters::prelude::*;
use std::sync::LazyLock;

#[cfg(target_os = "macos")]
extern crate accelerate_src;

pub const IMAGE_SIZE: (u32, u32) = (2400, 2400);

pub const MARGIN: u32 = 40;

pub const CAPTION_FONT_SIZE: u32 = 24;

pub const PADDING: f32 = 0.1;

pub static EDGE_PATTERN: LazyLock<ShapeStyle> = LazyLock::new(|| BLACK.mix(0.5).stroke_width(1));

/// PCA-based dimensionality reduction to 2D for visualization.
pub fn reduce_to_2d(data: &ArcArray2<f32>) -> ArcArray2<f32> {
    let (_n, dim) = data.dim();
    if dim <= 2 {
        return data.clone();
    }

    use ndarray_linalg::{Eigh, UPLO};

    let mean = data.mean_axis(Axis(0)).expect("failed to compute mean");
    let data_view = data.view();
    let centered: Array2<f32> = &data_view - &mean;
    let cov = centered.t().dot(&centered) / centered.nrows() as f32;
    let (eigenvalues, eigenvectors) = cov.eigh(UPLO::Upper).expect("eigen decomposition failed");

    let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

    let components = eigenvectors.select(Axis(1), &indices[..2]);
    let reduced = centered.dot(&components);
    ArcArray2::from(reduced)
}

pub fn cal_padding_range(min: f32, max: f32) -> (f32, f32) {
    let range = (max - min).max(1e-6);
    (min - range * PADDING, max + range * PADDING)
}

pub fn get_minimax(data: &ArcArray2<f32>, col: usize) -> (f32, f32) {
    data.column(col)
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)))
}

pub const VISUALIZATION_DIMENSION: usize = 2;
