//! Configuration, construction, persistence and search contracts.
//! Examples are compiled by `cargo test --doc`; small array examples also run.
#[doc = include_str!("guide/configuration.md")]
pub mod configuration {}
#[doc = include_str!("guide/build.md")]
pub mod build {}
#[doc = include_str!("guide/cache.md")]
pub mod cache {}
#[doc = include_str!("guide/search.md")]
pub mod search {}

/// End-to-end service usage from the package README.
#[doc = include_str!("../README.md")]
pub mod quick_start {}
