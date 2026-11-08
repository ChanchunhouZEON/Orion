/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::Result;
use std::io::Read;

/// Code placement, independent of the codebook and cascade stage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PqCodeStorage {
    /// Owned heap allocation. Pages remain pageable unless explicitly pinned.
    #[default]
    Resident,
    /// Private temporary-file mapping; the OS faults code pages on demand.
    /// Requires temporary disk space equal to the code payload. Never maps an
    /// externally mutable cache file, so truncating that file cannot cause SIGBUS.
    OnDemand,
}

pub(super) struct Codes(pub(super) orion_ann_platform::region::ArrayView<u8>);
impl Codes {
    pub(super) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
    pub(super) fn from_reader(
        reader: &mut dyn Read,
        len: usize,
        policy: PqCodeStorage,
    ) -> Result<Self> {
        Self::read_with_options(reader, len, &policy.options())
    }
    pub(super) fn read_with_options(
        reader: &mut dyn Read,
        len: usize,
        options: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        Ok(Self(orion_ann_platform::region::ArrayView::build_rows::<
            crate::Error,
        >(len, 1, 64, options, |_, out| {
            reader.read_exact(out)?;
            Ok(())
        })?))
    }
    pub(super) fn policy(&self) -> PqCodeStorage {
        match self.0.placement() {
            orion_ann_platform::region::Placement::Resident => PqCodeStorage::Resident,
            orion_ann_platform::region::Placement::Mapped => PqCodeStorage::OnDemand,
        }
    }
}
impl PqCodeStorage {
    pub(super) fn options(self) -> orion_ann_platform::region::RegionOptions {
        orion_ann_platform::region::RegionOptions {
            placement: match self {
                Self::Resident => orion_ann_platform::region::Placement::Resident,
                Self::OnDemand => orion_ann_platform::region::Placement::Mapped,
            },
            directory: None,
        }
    }
}
