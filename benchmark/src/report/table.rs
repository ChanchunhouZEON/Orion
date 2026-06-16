/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::time::Duration;

/// Timing breakdown returned by `AlgorithmRunner::build`. Pure
/// data-shape struct, kept after the legacy `run_benchmark` harness
/// was removed because every runner still reports its build time
/// through this triple — the values are populated for parity with the
/// other harnesses (`build-profile`, `memory-profile`) where the
/// breakdown is consumed, even though `--algorithms ads-comparison`
/// and the other in-process diagnostics in `main.rs` don't read the
/// fields back today.
#[allow(dead_code)]
pub struct BuildTiming {
    /// Pure Vamana graph construction time.
    pub graph_build: Duration,
    /// Extra overhead (extract, clustering, compression, etc.). Zero for DiskANN.
    pub overhead: Duration,
}
