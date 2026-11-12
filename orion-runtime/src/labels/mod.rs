/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Versioned label membership, independent of vector and graph mutation.
mod persistence;
mod snapshot;
mod store;
pub(crate) mod update;
pub(crate) use persistence::{identity, LabelObject};
pub(crate) use snapshot::LabelSnapshot;
pub(crate) use store::LabelStore;
pub use update::{ExternalLabelUpdate, LabelKey, LabelUpdate};

// Bounded prototype metadata. These limits apply to the current generation;
// overlapping readers may retain older generations until their queries finish.
pub(crate) const MAX_LABELS: usize = 4096;
pub(crate) const MAX_MEMBERSHIPS: usize = 1_000_000;
pub(crate) const MAX_KEY_BYTES: usize = 1024;
