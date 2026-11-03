//! Historical sampled-medoid entry selection shared by builders and STAG import.
use rayon::prelude::*;

fn dist_l2<T: Copy + Into<f32>>(a: &[T], b: &[T]) -> f32 {
    let mut s = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x).into() - (*y).into();
        s += d * d;
    }
    s
}

/// Preserve the historical PA importer entry selection without owning partitions.
///
/// # Panics
/// Panics when `dim` is zero. Callers must supply complete rows of equal dimension.
pub fn sampled_medoid<T: Copy + Sync + Into<f32>>(base_flat: &[T], dim: usize) -> u32 {
    sampled_medoid_rows(base_flat.len() / dim, |id| {
        &base_flat[id * dim..(id + 1) * dim]
    })
}

/// Select an entry using the historical stride sample and L2 medoid rule.
/// `row(id)` must return a valid, equally sized vector for each ID in `0..n`.
/// Empty input returns the conventional entry hint 0; builders reject empty bases.
pub fn sampled_medoid_rows<'a, T: Copy + Sync + Into<f32> + 'a>(
    n: usize,
    row: impl Fn(usize) -> &'a [T] + Sync,
) -> u32 {
    if n == 0 {
        return 0;
    }
    // Recompute medoid: smallest sum-of-distances to a stride sample.
    let sample_size = 1024.min(n);
    let step = (n / sample_size).max(1);
    let sample: Vec<usize> = (0..sample_size).map(|k| k * step).collect();
    let candidate_stride = (n / 10_000).max(1);
    (0..n)
        .into_par_iter()
        .step_by(candidate_stride)
        .map(|i| {
            let i_vec = row(i);
            let sum: f32 = sample.iter().map(|&s| dist_l2(i_vec, row(s))).sum();
            (sum, i as u32)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, i)| i)
        .unwrap_or(0)
}
