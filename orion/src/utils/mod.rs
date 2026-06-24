pub mod distance;

pub use distance::*;

#[cfg(feature = "indicatif")]
use indicatif::ProgressStyle;
#[cfg(feature = "indicatif")]
use std::sync::LazyLock;

pub const DELIMITER_LENGTH: usize = 64;

#[cfg(feature = "indicatif")]
pub static NODES_PROGRESS_STYLE: LazyLock<ProgressStyle> = LazyLock::new(|| {
    ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.green/22} {pos}/{len} nodes (ETA: {eta_precise})",
    )
    .unwrap()
    .progress_chars("━╾─")
});
