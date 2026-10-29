//! Shared Orion runtime: configuration, input, graph preparation and search.
//! CLI and Python use the same runtime without depending on each other.
pub mod config;
pub mod data;
pub mod graph;
pub mod index;
pub mod search;

pub use config::options::{CascadeConfig, GraphConfig, IndexOptions};
pub use index::{BaseInput, Calibration, Index, SearchResult};
