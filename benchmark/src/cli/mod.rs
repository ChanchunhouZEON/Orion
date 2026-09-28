//! Shared setup for standalone sweep and diagnostic executables.
//!
//! Flow: `config` resolves settings, `data` validates/loads inputs, and `index`
//! transfers the base allocation before loading or preparing a graph. `cache`
//! owns cache selection/provenance; `resources` inspects headers for preflight.
//! CLI flags, serialized config fields, and cache identities are compatibility
//! boundaries even when internal helper names change.

pub mod cache;
pub mod config;
pub mod data;
pub mod index;
pub mod resources;
