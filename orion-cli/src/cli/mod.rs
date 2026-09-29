//! Shared setup for standalone sweep and diagnostic executables.
//!
//! Flow: `config` resolves settings through a validated `plan`, `data` loads
//! the selected storage, and `index`
//! transfers the base allocation before loading or preparing a graph. `cache`
//! owns cache selection/provenance; `resources` uses the plan for preflight.
//! The standalone sweep imports `execution` for its typed search adapters.
//! CLI flags, serialized config fields, and cache identities are compatibility
//! boundaries even when internal helper names change.

pub mod cache;
pub mod config;
pub mod data;
pub mod index;
pub mod resources;
pub mod plan;

pub mod execution;
pub mod query;
