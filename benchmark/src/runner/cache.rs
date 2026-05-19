/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Unified cache-path naming for runners that persist built graphs to disk.
//!
//! Cache files live under `cache/<variant>/` and are keyed by the full set
//! of build parameters so variants can't collide across runs.

use std::path::{Path, PathBuf};

const CACHE_ROOT: &str = "cache";

fn alpha_tag(alpha: f32) -> String {
    format!("{:.2}", alpha).replace('.', "_")
}

fn ensure_dir(path: &Path) -> &Path {
    std::fs::create_dir_all(path).ok();
    path
}

/// Cache path for a DiskANN (Vamana) build.
///
/// Produces `cache/diskann/{dataset}_n{N}_r{R}_l{L}_a{alpha}.bin`.
/// The `.bin.data` / `.bin.delete` sidecars are written alongside by
/// `ANNInmemIndex::save`.
pub fn diskann_path(
    dataset: &str,
    num_points: usize,
    r: u32,
    build_l: usize,
    alpha: f32,
) -> PathBuf {
    let dir = PathBuf::from(CACHE_ROOT).join("diskann");
    ensure_dir(&dir);
    dir.join(format!(
        "{dataset}_n{num_points}_r{r}_l{build_l}_a{}.bin",
        alpha_tag(alpha)
    ))
}

/// Cache path for a StagedDiskANN PhasedGraph build.
///
/// Produces `cache/staged/{dataset}_n{N}_r{R}_l{L}_a{alpha}_ex{max_extra}.bin`.
/// The companion `.pgraph` file is written by `PhasedGraph::save`.
pub fn staged_path(
    dataset: &str,
    num_points: usize,
    r: u32,
    build_l: usize,
    alpha: f32,
    max_extra: usize,
) -> PathBuf {
    let dir = PathBuf::from(CACHE_ROOT).join("staged");
    ensure_dir(&dir);
    dir.join(format!(
        "{dataset}_n{num_points}_r{r}_l{build_l}_a{}_ex{max_extra}.bin",
        alpha_tag(alpha)
    ))
}
