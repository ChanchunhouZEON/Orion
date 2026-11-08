/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::model::dataset::rows::F32Rows;
use crate::{Error, Result};
use orion_ann_platform::region::{ArrayView, RegionOptions};
use rand::{SeedableRng, seq::SliceRandom};
use rayon::prelude::*;

pub const NUM_PQ_CENTROIDS: usize = 256;

/// Reproducible PQ recipe. None uses ceil(dimension/8), capped at 64 chunks.
/// Each chunk has 256 centers and contributes one byte per encoded row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PqBuildOptions {
    /// Number of subspaces; None resolves from the logical dimension.
    pub chunks: Option<usize>,
    /// Positive upper bound for uniform sampling without replacement.
    pub training_rows: usize,
    /// Positive Lloyd iteration count. Empty clusters retain their previous center.
    pub iterations: usize,
    /// Reproducible sample and initialization, independent of the worker count.
    pub seed: u64,
}
impl Default for PqBuildOptions {
    fn default() -> Self {
        Self {
            chunks: None,
            training_rows: 4096,
            iterations: 12,
            seed: 42,
        }
    }
}
impl PqBuildOptions {
    /// Resolve the dimension-dependent chunk count to an explicit recipe.
    /// # Errors
    /// Rejects invalid dimension, chunk count or training budget.
    pub fn resolved(self, dimension: usize) -> Result<Self> {
        Ok(Self {
            chunks: Some(self.chunks(dimension)?),
            ..self
        })
    }

    pub(crate) fn chunks(self, dim: usize) -> Result<usize> {
        let m = self.chunks.unwrap_or(dim.div_ceil(8).min(64));
        if dim == 0 || m == 0 || m > dim || self.training_rows == 0 || self.iterations == 0 {
            return Err(Error::invalid_argument(
                "PQ needs 1..=dimension chunks, positive sample size and iterations",
            ));
        }
        dim.checked_mul(NUM_PQ_CENTROIDS * 4)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| Error::invalid_argument("PQ codebook size overflow"))?;
        Ok(m)
    }
}

/// Immutable 8-bit codebook with independent, possibly unequal subspaces.
/// Centers are chunk-major, then centroid-major; each subvector is contiguous.
#[derive(Clone)]
pub struct FixedChunkPQTable {
    pub(super) centers: ArrayView<f32>,
    pub(super) offsets: ArrayView<usize>,
    pub(super) mean: ArrayView<f32>,
}
impl std::fmt::Debug for FixedChunkPQTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FixedChunkPQTable")
            .field("dimension", &self.dim())
            .field("chunks", &self.get_num_chunks())
            .finish_non_exhaustive()
    }
}

pub(super) struct PackedRows<'a> {
    pub data: &'a [f32],
    pub dim: usize,
}
impl F32Rows for PackedRows<'_> {
    fn dimension(&self) -> usize {
        self.dim
    }
    fn len(&self) -> usize {
        self.data.len() / self.dim
    }
    fn row(&self, id: usize) -> &[f32] {
        &self.data[id * self.dim..(id + 1) * self.dim]
    }
}
pub(super) fn check_rows(data: &[f32], rows: usize, dim: usize) -> Result<()> {
    if rows == 0
        || dim == 0
        || rows.checked_mul(dim) != Some(data.len())
        || data.iter().any(|x| !x.is_finite())
    {
        return Err(Error::invalid_argument(
            "PQ requires finite, nonempty rows matching the declared shape",
        ));
    }
    Ok(())
}

impl FixedChunkPQTable {
    pub(super) fn with_storage(mut self, storage: &RegionOptions) -> Result<Self> {
        if self.centers.placement() != storage.placement {
            self.centers = ArrayView::copy_from(&self.centers, storage)?;
            self.mean = ArrayView::copy_from(&self.mean, storage)?;
            self.offsets = ArrayView::copy_from(&self.offsets, storage)?;
        }
        Ok(self)
    }

    /// Import centroid-major centers and a per-coordinate mean translation.
    /// Supply an all-zero mean to disable translation.
    /// # Errors
    /// Rejects inconsistent lengths, non-finite values or nonincreasing offsets.
    pub fn new(
        dim: usize,
        chunks: usize,
        centers: Vec<f32>,
        mean: Vec<f32>,
        offsets: Vec<usize>,
    ) -> Result<Self> {
        PqBuildOptions {
            chunks: Some(chunks),
            ..Default::default()
        }
        .chunks(dim)?;
        if centers.len() != dim * NUM_PQ_CENTROIDS
            || mean.len() != dim
            || offsets.len() != chunks + 1
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&dim)
            || offsets.windows(2).any(|x| x[0] >= x[1])
            || centers.iter().chain(&mean).any(|x| !x.is_finite())
        {
            return Err(Error::invalid_argument("invalid PQ codebook layout"));
        }
        let mut reordered = Vec::with_capacity(centers.len());
        for range in offsets.windows(2) {
            for c in 0..NUM_PQ_CENTROIDS {
                reordered.extend_from_slice(&centers[c * dim + range[0]..c * dim + range[1]]);
            }
        }
        Ok(Self {
            centers: ArrayView::copy_from(&reordered, &Default::default())?,
            mean: ArrayView::copy_from(&mean, &Default::default())?,
            offsets: ArrayView::copy_from(&offsets, &Default::default())?,
        })
    }

    /// Train using the canonical sample/iteration budget and an explicit chunk count.
    /// # Errors
    /// Rejects invalid shape/options and non-finite coordinates.
    pub fn train(data: &[f32], rows: usize, dim: usize, chunks: usize) -> Result<Self> {
        check_rows(data, rows, dim)?;
        Self::train_rows(
            &PackedRows { data, dim },
            PqBuildOptions {
                chunks: Some(chunks),
                ..Default::default()
            },
        )
    }

    pub(super) fn train_rows(base: &impl F32Rows, options: PqBuildOptions) -> Result<Self> {
        let dim = base.dimension();
        let m = options.chunks(dim)?;
        if base.is_empty() {
            return Err(Error::invalid_argument("PQ training base is empty"));
        }
        // Uniform sample without replacement: bounded memory, deterministic independent of
        // Rayon worker count. The sample alone is packed, never the whole base.
        let count = base.len().min(options.training_rows);
        let mut rng = rand::rngs::StdRng::seed_from_u64(options.seed);
        let mut sample_ids = rand::seq::index::sample(&mut rng, base.len(), count).into_vec();
        sample_ids.shuffle(&mut rng);
        let mut sample = Vec::with_capacity(
            count
                .checked_mul(dim)
                .ok_or_else(|| Error::invalid_argument("PQ training sample overflow"))?,
        );
        for id in sample_ids {
            let row = base.row(id);
            if row.len() != dim || row.iter().any(|x| !x.is_finite()) {
                return Err(Error::invalid_argument("invalid PQ training row"));
            }
            sample.extend_from_slice(row);
        }
        let offsets: Vec<_> = (0..=m).map(|i| i * (dim / m) + i.min(dim % m)).collect();
        let chunks: Vec<Vec<f32>> = offsets
            .par_windows(2)
            .map(|r| {
                let width = r[1] - r[0];
                let rows: Vec<f32> = sample
                    .chunks_exact(dim)
                    .flat_map(|v| v[r[0]..r[1]].iter().copied())
                    .collect();
                let mut centers = Vec::with_capacity(256 * width);
                for c in 0..256 {
                    centers.extend_from_slice(&rows[(c % count) * width..(c % count + 1) * width]);
                }
                // Small fixtures retain every training row exactly, including duplicate centers.
                if count <= 256 {
                    return centers;
                }
                let mut sums = vec![0.0f64; 256 * width];
                let mut counts = [0usize; 256];
                for _ in 0..options.iterations {
                    sums.fill(0.0);
                    counts.fill(0);
                    for row in rows.chunks_exact(width) {
                        let c = nearest(row, &centers, width);
                        counts[c] += 1;
                        for j in 0..width {
                            sums[c * width + j] += row[j] as f64;
                        }
                    }
                    for c in 0..256 {
                        if counts[c] > 0 {
                            for j in 0..width {
                                centers[c * width + j] =
                                    (sums[c * width + j] / counts[c] as f64) as f32;
                            }
                        }
                    }
                }
                centers
            })
            .collect();
        Ok(Self {
            centers: ArrayView::copy_from(
                &chunks.into_iter().flatten().collect::<Vec<_>>(),
                &Default::default(),
            )?,
            offsets: ArrayView::copy_from(&offsets, &Default::default())?,
            mean: ArrayView::copy_from(&vec![0.0; dim], &Default::default())?,
        })
    }

    /// Logical dimension before subspace partitioning.
    pub fn dim(&self) -> usize {
        self.mean.len()
    }
    /// Encoded bytes per vector (one byte per subspace).
    pub fn get_num_chunks(&self) -> usize {
        self.offsets.len() - 1
    }
    pub(super) fn bytes(&self) -> usize {
        self.centers.len() * 4 + self.mean.len() * 4 + self.offsets.len() * size_of::<usize>()
    }
    pub(super) fn regions(&self) -> Vec<&[u8]> {
        vec![
            bytemuck::cast_slice(&self.centers),
            bytemuck::cast_slice(&self.mean),
            bytemuck::cast_slice(&self.offsets),
        ]
    }

    /// Encode finite packed vectors.
    ///
    /// # Errors
    /// Returns an error for mismatched shape or non-finite coordinates.
    pub fn encode(&self, data: &[f32], rows: usize) -> Result<Vec<u8>> {
        check_rows(data, rows, self.dim())?;
        self.encode_rows(&PackedRows {
            data,
            dim: self.dim(),
        })
    }
    pub(super) fn encode_rows(&self, base: &impl F32Rows) -> Result<Vec<u8>> {
        let m = self.get_num_chunks();
        let mut codes = vec![
            0;
            base.len()
                .checked_mul(m)
                .ok_or_else(|| Error::invalid_argument("PQ codes size overflow"))?
        ];
        self.encode_into(base, 0, &mut codes)?;
        Ok(codes)
    }
    pub(super) fn encode_into(
        &self,
        base: &impl F32Rows,
        start: usize,
        codes: &mut [u8],
    ) -> Result<()> {
        let m = self.get_num_chunks();
        codes.par_chunks_mut(m).enumerate().try_for_each_init(
            || vec![0.0; self.dim()],
            |centered, (id, out)| -> Result<()> {
                let row = base.row(start + id);
                if row.len() != self.dim() || row.iter().any(|x| !x.is_finite()) {
                    return Err(Error::invalid_argument("invalid PQ encoding row"));
                }
                for ((out, x), m) in centered.iter_mut().zip(row).zip(self.mean.iter()) {
                    *out = x - m;
                }
                for (c, range) in self.offsets.windows(2).enumerate() {
                    let width = range[1] - range[0];
                    out[c] = nearest(
                        &centered[range[0]..range[1]],
                        &self.centers[range[0] * 256..range[1] * 256],
                        width,
                    ) as u8;
                }
                Ok(())
            },
        )?;
        Ok(())
    }
    /// Translate the query for the legacy ADC adapter.
    /// # Panics
    /// Panics on wrong dimension; new callers should use `PqDataset::prepare_query`.
    pub fn preprocess_query(&self, query: &mut [f32]) {
        assert_eq!(query.len(), self.dim());
        for (q, m) in query.iter_mut().zip(self.mean.iter()) {
            *q -= m;
        }
    }
    /// Lookup table for an already translated L2 query.
    /// # Panics
    /// Panics on wrong dimension. Safe indexing never reads beyond input.
    pub fn populate_chunk_distances(&self, query: &[f32]) -> Vec<f32> {
        assert_eq!(query.len(), self.dim());
        self.lookup(query, false, true)
    }
    pub(super) fn lookup(&self, query: &[f32], ip: bool, centered: bool) -> Vec<f32> {
        let mut table = vec![0.0; self.get_num_chunks() * 256];
        for (chunk, r) in self.offsets.windows(2).enumerate() {
            let width = r[1] - r[0];
            for c in 0..256 {
                let center = &self.centers[r[0] * 256 + c * width..r[0] * 256 + (c + 1) * width];
                table[chunk * 256 + c] = center
                    .iter()
                    .enumerate()
                    .map(|(j, x)| {
                        let axis = r[0] + j;
                        if ip {
                            -query[axis] * (x + self.mean[axis])
                        } else {
                            let delta =
                                query[axis] - x - if centered { 0.0 } else { self.mean[axis] };
                            delta * delta
                        }
                    })
                    .sum();
            }
        }
        table
    }
    /// Sum a complete code against a lookup table.
    /// # Panics
    /// Panics for inconsistent code/table lengths.
    pub fn adc_distance(&self, codes: &[u8], lookup: &[f32]) -> f32 {
        assert_eq!(codes.len(), self.get_num_chunks());
        assert_eq!(lookup.len(), codes.len() * 256);
        codes
            .iter()
            .enumerate()
            .map(|(i, &c)| lookup[i * 256 + c as usize])
            .sum()
    }
    /// Squared-L2 ADC for an already mean-translated query and one encoded row.
    /// # Panics
    /// Panics if query or code lengths do not match the fitted model.
    pub fn l2_distance(&self, centered_query: &[f32], codes: &[u8]) -> f32 {
        self.adc_distance(codes, &self.populate_chunk_distances(centered_query))
    }
}
#[inline]
fn nearest(row: &[f32], centers: &[f32], width: usize) -> usize {
    let mut best = (0, f32::INFINITY);
    for (c, center) in centers.chunks_exact(width).enumerate() {
        let d = crate::utils::distance::l2_sq_slice(row, center);
        if d < best.1 {
            best = (c, d);
        }
    }
    best.0
}
