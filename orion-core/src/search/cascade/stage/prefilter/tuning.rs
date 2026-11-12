/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Multiplicative slack on the prefilter threshold (mean over the PQ
/// of per-entry prefilter distances). Mirrors `jl_slack()` in the
/// legacy `in_mem_search_l2_q`; resolved once per process via
/// `OnceLock` so the env-var lookup stays off the inner loop.
///
/// Default `1.05` is the calibration sweep optimum for JL Sparse on
/// GIST 1M (0.1-1.4 pp recall loss, +35-53% iso-recall QPS). Other
/// prefilter implementations (e.g. RaBitQ-as-prefilter) may want
/// different defaults — when they land, refactor this into a per-
/// recipe constant.
#[inline]
#[cfg(feature = "benchmark")]
pub(super) fn prefilter_slack() -> f32 {
    use std::sync::OnceLock;
    static SLACK: OnceLock<f32> = OnceLock::new();
    *SLACK.get_or_init(|| {
        std::env::var("ORION_JL_SLACK")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|&v| v > 0.0)
            .unwrap_or(1.05)
    })
}

#[cfg(not(feature = "benchmark"))]
pub(super) fn prefilter_slack() -> f32 {
    1.05
}
