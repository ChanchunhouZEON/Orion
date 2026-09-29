//! Shared configuration, input, graph loading, and search execution for Orion.
//! This crate depends on the engine, never on the benchmark harness.
pub mod cascade;
pub mod cli;
pub mod config;
pub mod parlayann_bridge;
pub mod utils;
