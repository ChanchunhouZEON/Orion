/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::visualization::utils::*;
use diskann::common::{ANNError, ANNResult};
use ndarray::ArcArray2;
use plotters::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

const NORMAL_POINT_SIZE: f32 = 5f32;
static NORMAL_POINT_PATTERN: LazyLock<ShapeStyle> = LazyLock::new(|| BLACK.mix(0.6).filled());

const CENTROID_SIZE: f32 = 5f32;
static CENTROID_PATTERN: LazyLock<ShapeStyle> = LazyLock::new(|| RED.filled());

/// Draw the full clustered graph with color-coded clusters, centroids, and graph edges.
pub fn draw_clustered_graph(
    data: &ArcArray2<f32>,
    point_affiliation: &[i32],
    storage_layout: &HashMap<u32, HashSet<u32>>,
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
        .margin(MARGIN)
        .caption("Clustered Graph", ("monospace", CAPTION_FONT_SIZE))
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|_| {
            ANNError::log_visualization_error(
                "Could not build the basic information of the clustered graph".to_string(),
            )
        })?;

    // Draw edges (light gray)
    for (&point, neighbors) in graph.iter() {
        if (point as usize) >= reduced.nrows() {
            continue;
        }
        for &neighbor in neighbors.iter() {
            if (neighbor as usize) >= reduced.nrows() {
                continue;
            }
            chart
                .draw_series(std::iter::once(PathElement::new(
                    vec![
                        (reduced[[point as usize, 0]], reduced[[point as usize, 1]]),
                        (
                            reduced[[neighbor as usize, 0]],
                            reduced[[neighbor as usize, 1]],
                        ),
                    ],
                    EDGE_PATTERN.clone(),
                )))
                .map_err(|_| {
                    ANNError::log_visualization_error(
                        "Failed to draw edges of the clustered graph".to_string(),
                    )
                })?;
        }
    }

    // Draw points color-coded by cluster
    for (&cluster_id, members) in storage_layout.iter() {
        let color = Palette99::pick(cluster_id as usize);
        chart
            .draw_series(members.iter().filter_map(|&pid| {
                if (pid as usize) >= reduced.nrows() {
                    return None;
                }
                Some(Circle::new(
                    (reduced[[pid as usize, 0]], reduced[[pid as usize, 1]]),
                    10,
                    color.mix(0.6).filled(),
                ))
            }))
            .map_err(|_| {
                ANNError::log_visualization_error(
                    "Failed to draw nodes of the clustered graph".to_string(),
                )
            })?;
    }

    // Draw unaffiliated points in gray
    for (i, &aff) in point_affiliation.iter().enumerate() {
        if aff < 0 && i < reduced.nrows() {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (reduced[[i, 0]], reduced[[i, 1]]),
                    10,
                    RGBColor(180, 180, 180).mix(0.6).filled(),
                )))
                .map_err(|_| {
                    ANNError::log_visualization_error(
                        "Failed to draw unaffiliated nodes of the clustered graph".to_string(),
                    )
                })?;
        }
    }

    root.present().map_err(|_| {
        ANNError::log_visualization_error("Could not represent the image".to_string())
    })?;
    Ok(())
}

/// Draw a single cluster with its internal edges and centroid.
pub fn draw_single_cluster(
    data: &ArcArray2<f32>,
    cluster_id: u32,
    members: &HashSet<u32>,
    centroid: Option<u32>,
    graph: &HashMap<u32, Vec<u32>>,
    output_path: &str,
) -> ANNResult<()> {
    let reduced = reduce_to_2d(data);
    let root = BitMapBackend::new(output_path, (1200, 1200)).into_drawing_area();
    root.fill(&WHITE).map_err(|_| {
        ANNError::log_visualization_error(
            "Could not build the basic information of the in-cluster graph".to_string(),
        )
    })?;

    // Compute bounds from cluster points only
    let xs: Vec<f32> = members
        .iter()
        .filter_map(|&p| {
            if (p as usize) < reduced.nrows() {
                Some(reduced[[p as usize, 0]])
            } else {
                None
            }
        })
        .collect();
    let ys: Vec<f32> = members
        .iter()
        .filter_map(|&p| {
            if (p as usize) < reduced.nrows() {
                Some(reduced[[p as usize, 1]])
            } else {
                None
            }
        })
        .collect();

    if xs.is_empty() {
        root.present().map_err(|_| {
            ANNError::log_visualization_error("Could not represent the image".to_string())
        })?;
        return Ok(());
    }

    let xmin = xs.iter().cloned().fold(f32::MAX, f32::min);
    let xmax = xs.iter().cloned().fold(f32::MIN, f32::max);
    let ymin = ys.iter().cloned().fold(f32::MAX, f32::min);
    let ymax = ys.iter().cloned().fold(f32::MIN, f32::max);
    let (xmin, xmax) = cal_padding_range(xmin, xmax);
    let (ymin, ymax) = cal_padding_range(ymin, ymax);

    let mut chart = ChartBuilder::on(&root)
        .margin(MARGIN)
        .caption(
            format!("Cluster {} ({} pts)", cluster_id, members.len()),
            ("monospace", CAPTION_FONT_SIZE),
        )
        .build_cartesian_2d(xmin..xmax, ymin..ymax)
        .map_err(|_| {
            ANNError::log_visualization_error(
                "Could not build the basic information of the in-cluster graph".to_string(),
            )
        })?;

    // Draw intra-cluster edges
    for &point in members {
        if (point as usize) >= reduced.nrows() {
            continue;
        }
        if let Some(neighbors) = graph.get(&point) {
            for &neighbor in neighbors {
                if members.contains(&neighbor) && (neighbor as usize) < reduced.nrows() {
                    chart
                        .draw_series(std::iter::once(PathElement::new(
                            vec![
                                (reduced[[point as usize, 0]], reduced[[point as usize, 1]]),
                                (
                                    reduced[[neighbor as usize, 0]],
                                    reduced[[neighbor as usize, 1]],
                                ),
                            ],
                            EDGE_PATTERN.clone(),
                        )))
                        .map_err(|_| {
                            ANNError::log_visualization_error(
                                "Failed to draw edges of the in-cluster graph".to_string(),
                            )
                        })?;
                }
            }
        }
    }

    // Draw points
    chart
        .draw_series(members.iter().filter_map(|&pid| {
            if (pid as usize) >= reduced.nrows() {
                return None;
            }
            Some(Circle::new(
                (reduced[[pid as usize, 0]], reduced[[pid as usize, 1]]),
                NORMAL_POINT_SIZE,
                NORMAL_POINT_PATTERN.clone(),
            ))
        }))
        .map_err(|_| {
            ANNError::log_visualization_error(
                "Failed to draw nodes of the in-cluster graph".to_string(),
            )
        })?;

    // Draw centroid
    if let Some(c) = centroid {
        if (c as usize) < reduced.nrows() {
            chart
                .draw_series(std::iter::once(Circle::new(
                    (reduced[[c as usize, 0]], reduced[[c as usize, 1]]),
                    CENTROID_SIZE,
                    CENTROID_PATTERN.clone(),
                )))
                .map_err(|_| {
                    ANNError::log_visualization_error(
                        "Failed to draw centroid of the in-cluster graph".to_string(),
                    )
                })?;
        }
    }

    root.present().map_err(|_| {
        ANNError::log_visualization_error("Could not represent the image".to_string())
    })?;
    Ok(())
}

/// Draw the compressed graph (inter-cluster edges only).
pub fn draw_compressed_graph(
    data: &ArcArray2<f32>,
    storage_layout: &HashMap<u32, HashSet<u32>>,
    compressed_graph: &HashMap<u32, Vec<u32>>,
    output_path: &str,
) -> anyhow::Result<()> {
    let reduced = reduce_to_2d(data);
    let root = BitMapBackend::new(output_path, IMAGE_SIZE).into_drawing_area();
    root.fill(&WHITE)?;

    let (xmin, xmax) = get_minimax(&reduced, 0);
    let (ymin, ymax) = get_minimax(&reduced, 1);
    let (xmin, xmax) = cal_padding_range(xmin, xmax);
    let (ymin, ymax) = cal_padding_range(ymin, ymax);

    let mut chart = ChartBuilder::on(&root)
        .margin(MARGIN)
        .caption("Compressed Graph (inter-cluster edges)", ("sans-serif", 30))
        .build_cartesian_2d(xmin..xmax, ymin..ymax)?;

    // Draw compressed graph edges (cross-cluster)
    for (&point, neighbors) in compressed_graph.iter() {
        if (point as usize) >= reduced.nrows() {
            continue;
        }
        for &neighbor in neighbors.iter() {
            if (neighbor as usize) >= reduced.nrows() {
                continue;
            }
            // let src_cluster = point_affiliation[point as usize];
            // let dst_cluster = point_affiliation[neighbor as usize];
            // let color = if src_cluster != dst_cluster {
            //     RGBColor(0, 100, 200).mix(0.3)
            // } else {
            //     RGBColor(200, 100, 0).mix(0.2)
            // };
            chart.draw_series(std::iter::once(PathElement::new(
                vec![
                    (reduced[[point as usize, 0]], reduced[[point as usize, 1]]),
                    (
                        reduced[[neighbor as usize, 0]],
                        reduced[[neighbor as usize, 1]],
                    ),
                ],
                BLACK.mix(0.5).stroke_width(1),
            )))?;
        }
    }

    // Draw points color-coded by cluster
    for (&_, members) in storage_layout.iter() {
        chart.draw_series(members.iter().filter_map(|&pid| {
            if (pid as usize) >= reduced.nrows() {
                return None;
            }
            Some(Circle::new(
                (reduced[[pid as usize, 0]], reduced[[pid as usize, 1]]),
                5,
                BLACK.mix(0.6).filled(),
            ))
        }))?;
    }

    root.present()?;
    Ok(())
}

// ─── Clustering Quality Metrics ───

/// Comprehensive clustering quality report.
#[derive(Debug, Clone)]
pub struct ClusteringMetrics {
    pub num_clusters: usize,
    pub num_points: usize,
    pub num_unaffiliated: usize,
    pub avg_cluster_size: f32,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub std_cluster_size: f32,
    /// Average intra-cluster distance (lower = tighter clusters)
    pub avg_intra_cluster_distance: f32,
    /// Average inter-cluster distance (higher = better separated)
    pub avg_inter_cluster_distance: f32,
    /// Ratio of inter/intra distance (higher = better)
    pub separation_ratio: f32,
    /// Average graph connectivity within clusters (fraction of edges staying in-cluster)
    pub avg_intra_cluster_edge_ratio: f32,
    /// Number of clusters with no inter-cluster compressed edges (isolated)
    pub num_isolated_clusters: usize,
    /// Average compressed graph degree
    pub avg_compressed_degree: f32,
}

impl std::fmt::Display for ClusteringMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Clustering Quality Metrics ===")?;
        writeln!(f, "  Clusters:            {}", self.num_clusters)?;
        writeln!(f, "  Total points:        {}", self.num_points)?;
        writeln!(
            f,
            "  Unaffiliated:        {} ({:.1}%)",
            self.num_unaffiliated,
            self.num_unaffiliated as f32 / self.num_points as f32 * 100.0
        )?;
        writeln!(
            f,
            "  Cluster sizes:       avg={:.1} min={} max={} std={:.1}",
            self.avg_cluster_size,
            self.min_cluster_size,
            self.max_cluster_size,
            self.std_cluster_size
        )?;
        writeln!(
            f,
            "  Intra-cluster dist:  {:.4} (lower=tighter)",
            self.avg_intra_cluster_distance
        )?;
        writeln!(
            f,
            "  Inter-cluster dist:  {:.4} (higher=better separated)",
            self.avg_inter_cluster_distance
        )?;
        writeln!(
            f,
            "  Separation ratio:    {:.4} (inter/intra, higher=better)",
            self.separation_ratio
        )?;
        writeln!(
            f,
            "  Intra-edge ratio:    {:.4} (fraction of edges within cluster)",
            self.avg_intra_cluster_edge_ratio
        )?;
        writeln!(f, "  Isolated clusters:   {}", self.num_isolated_clusters)?;
        writeln!(
            f,
            "  Avg compressed deg:  {:.2}",
            self.avg_compressed_degree
        )?;
        Ok(())
    }
}

/// Compute comprehensive clustering quality metrics.
pub fn compute_clustering_metrics(
    data: &ArcArray2<f32>,
    point_affiliation: &[i32],
    storage_layout: &HashMap<u32, HashSet<u32>>,
    graph: &HashMap<u32, Vec<u32>>,
    compressed_graph: &HashMap<u32, Vec<u32>>,
) -> ClusteringMetrics {
    let num_points = point_affiliation.len();
    let num_clusters = storage_layout.len();
    let num_unaffiliated = point_affiliation.iter().filter(|&&a| a < 0).count();

    // Cluster size statistics
    let sizes: Vec<usize> = storage_layout.values().map(|s| s.len()).collect();
    let avg_size = if sizes.is_empty() {
        0.0
    } else {
        sizes.iter().sum::<usize>() as f32 / sizes.len() as f32
    };
    let min_size = sizes.iter().copied().min().unwrap_or(0);
    let max_size = sizes.iter().copied().max().unwrap_or(0);
    let std_size = if sizes.len() > 1 {
        let variance = sizes
            .iter()
            .map(|&s| (s as f32 - avg_size).powi(2))
            .sum::<f32>()
            / sizes.len() as f32;
        variance.sqrt()
    } else {
        0.0
    };

    // Compute centroids for each cluster
    let mut centroids: HashMap<u32, Vec<f32>> = HashMap::new();
    for (&cluster_id, members) in storage_layout.iter() {
        let dim = data.ncols();
        let mut centroid = vec![0.0f32; dim];
        let mut count = 0;
        for &pid in members {
            if (pid as usize) < data.nrows() {
                let row = data.row(pid as usize);
                for (j, &v) in row.iter().enumerate() {
                    centroid[j] += v;
                }
                count += 1;
            }
        }
        if count > 0 {
            for v in &mut centroid {
                *v /= count as f32;
            }
        }
        centroids.insert(cluster_id, centroid);
    }

    // Average intra-cluster distance (mean distance of points to their centroid)
    let mut total_intra = 0.0f64;
    let mut intra_count = 0usize;
    for (&cluster_id, members) in storage_layout.iter() {
        if let Some(centroid) = centroids.get(&cluster_id) {
            for &pid in members {
                if (pid as usize) < data.nrows() {
                    let row = data.row(pid as usize);
                    let dist: f32 = row
                        .iter()
                        .zip(centroid.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    total_intra += dist as f64;
                    intra_count += 1;
                }
            }
        }
    }
    let avg_intra = if intra_count > 0 {
        (total_intra / intra_count as f64) as f32
    } else {
        0.0
    };

    // Average inter-cluster distance (mean pairwise centroid distances)
    let cluster_ids: Vec<u32> = centroids.keys().copied().collect();
    let mut total_inter = 0.0f64;
    let mut inter_count = 0usize;
    for i in 0..cluster_ids.len() {
        for j in (i + 1)..cluster_ids.len() {
            let c1 = &centroids[&cluster_ids[i]];
            let c2 = &centroids[&cluster_ids[j]];
            let dist: f32 = c1
                .iter()
                .zip(c2.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            total_inter += dist as f64;
            inter_count += 1;
        }
    }
    let avg_inter = if inter_count > 0 {
        (total_inter / inter_count as f64) as f32
    } else {
        0.0
    };

    let separation_ratio = if avg_intra > 1e-9 {
        avg_inter / avg_intra
    } else {
        f32::MAX
    };

    // Intra-cluster edge ratio
    let mut total_edges = 0usize;
    let mut intra_edges = 0usize;
    for (&point, neighbors) in graph.iter() {
        if (point as usize) >= point_affiliation.len() {
            continue;
        }
        let src_cluster = point_affiliation[point as usize];
        for &neighbor in neighbors {
            if (neighbor as usize) >= point_affiliation.len() {
                continue;
            }
            total_edges += 1;
            if point_affiliation[neighbor as usize] == src_cluster && src_cluster >= 0 {
                intra_edges += 1;
            }
        }
    }
    let avg_intra_edge_ratio = if total_edges > 0 {
        intra_edges as f32 / total_edges as f32
    } else {
        0.0
    };

    // Isolated clusters (no compressed graph edges leaving the cluster)
    let mut cluster_has_external = HashSet::new();
    for (&point, neighbors) in compressed_graph.iter() {
        if (point as usize) >= point_affiliation.len() {
            continue;
        }
        let src = point_affiliation[point as usize];
        for &neighbor in neighbors {
            if (neighbor as usize) >= point_affiliation.len() {
                continue;
            }
            let dst = point_affiliation[neighbor as usize];
            if src != dst && src >= 0 {
                cluster_has_external.insert(src as u32);
            }
            if src != dst && dst >= 0 {
                cluster_has_external.insert(dst as u32);
            }
        }
    }
    let num_isolated = storage_layout
        .keys()
        .filter(|&&cid| !cluster_has_external.contains(&cid))
        .count();

    // Average compressed graph degree
    let mut total_degree = 0usize;
    let mut degree_count = 0usize;
    for neighbors in compressed_graph.values() {
        total_degree += neighbors.len();
        degree_count += 1;
    }
    let avg_compressed_degree = if degree_count > 0 {
        total_degree as f32 / degree_count as f32
    } else {
        0.0
    };

    ClusteringMetrics {
        num_clusters,
        num_points,
        num_unaffiliated,
        avg_cluster_size: avg_size,
        min_cluster_size: min_size,
        max_cluster_size: max_size,
        std_cluster_size: std_size,
        avg_intra_cluster_distance: avg_intra,
        avg_inter_cluster_distance: avg_inter,
        separation_ratio,
        avg_intra_cluster_edge_ratio: avg_intra_edge_ratio,
        num_isolated_clusters: num_isolated,
        avg_compressed_degree,
    }
}
