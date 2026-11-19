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

/// Owned or mapped PQ code storage, independent of the resident codebook.
pub(super) struct Codes(
    /// Encoded row bytes sharing their backing allocation or mapping.
    pub(super) orion_ann_platform::region::ArrayView<u8>,
);
impl Codes {
    /// Borrows contiguous row codes; mapped storage may fault pages in during access.
    pub(super) fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
    #[cfg(test)]
    pub(super) fn from_reader(
        reader: &mut dyn Read,
        len: usize,
        policy: PqCodeStorage,
    ) -> Result<Self> {
        Self::read_with_options(reader, len, &policy.options())
    }

    /// Reads exactly the encoded code payload into resident or mapped allocation-owned storage.
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

    /// Reports the actual code allocation placement rather than its original requested policy.
    pub(super) fn policy(&self) -> PqCodeStorage {
        match self.0.placement() {
            orion_ann_platform::region::Placement::Resident => PqCodeStorage::Resident,
            orion_ann_platform::region::Placement::Mapped => PqCodeStorage::OnDemand,
        }
    }
}
impl PqCodeStorage {
    /// Translates the PQ code policy into common region placement options.
    pub(super) fn options(self) -> orion_ann_platform::region::RegionOptions {
        orion_ann_platform::region::RegionOptions {
            placement: match self {
                Self::Resident => orion_ann_platform::region::Placement::Resident,
                Self::OnDemand => orion_ann_platform::region::Placement::Mapped,
            },
            directory: None,
            accounting: None,
            reclaim_after_write: false,
        }
    }
}
