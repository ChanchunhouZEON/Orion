use vector::{FullPrecisionDistance, Metric};

/// Calculate the medoid (closest point to the centroid) of the dataset.
/// Returns the index of the medoid point.
pub fn calculate_medoid<T, const N: usize>(data: &[[T; N]], metric: Metric) -> u32
where
    T: Default + Copy + Sync + Send + Into<f32>,
    [T; N]: FullPrecisionDistance<T, N>,
{
    let n = data.len();
    if n == 0 {
        return 0;
    }

    // Calculate centroid.
    let mut centroid = [0.0f32; N];
    for point in data.iter() {
        for (j, val) in point.iter().enumerate() {
            centroid[j] += (*val).into();
        }
    }
    let n_f32 = n as f32;
    for val in centroid.iter_mut() {
        *val /= n_f32;
    }

    // Find the point closest to the centroid.
    let mut best_id = 0u32;
    let mut best_dist = f32::MAX;

    for (i, point) in data.iter().enumerate() {
        let dist = <[f32; N]>::distance_compare(&centroid, &point.map(|v| v.into()), metric);
        if dist < best_dist {
            best_dist = dist;
            best_id = i as u32;
        }
    }

    best_id
}
