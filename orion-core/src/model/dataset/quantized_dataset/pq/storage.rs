/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::Result;
use std::io::Read;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;

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

pub(super) enum Codes {
    Resident(Box<[u8]>),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}
impl Codes {
    pub(super) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Resident(x) => x,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(x) => x,
        }
    }
    pub(super) fn from_reader(
        reader: &mut dyn Read,
        len: usize,
        policy: PqCodeStorage,
    ) -> Result<Self> {
        match policy {
            PqCodeStorage::Resident => {
                let mut v = vec![0; len];
                reader.read_exact(&mut v)?;
                Ok(Self::Resident(v.into_boxed_slice()))
            }
            #[cfg(target_arch = "wasm32")]
            PqCodeStorage::OnDemand => Err(crate::Error::unsupported(
                "on-demand PQ code storage requires a native filesystem",
            )),
            #[cfg(not(target_arch = "wasm32"))]
            PqCodeStorage::OnDemand => {
                let mut file = tempfile::tempfile()?;
                let mut writer = std::io::BufWriter::with_capacity(
                    orion_ann_platform::buffered_io::FILE_WRITE_BYTES,
                    &mut file,
                );
                let copied = std::io::copy(&mut reader.take(len as u64), &mut writer)?;
                writer.flush()?;
                drop(writer);
                if copied != len as u64 {
                    return Err(crate::Error::cache("truncated PQ codes"));
                }
                file.flush()?;
                // SAFETY: unnamed file is private, no writable handle escapes;
                // closing our last handle leaves only this immutable mapping.
                let map = unsafe { memmap2::Mmap::map(&file)? };
                #[cfg(unix)]
                map.advise(memmap2::Advice::Random)?;
                Ok(Self::Mapped(map))
            }
        }
    }
    pub(super) fn policy(&self) -> PqCodeStorage {
        match self {
            Self::Resident(_) => PqCodeStorage::Resident,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(_) => PqCodeStorage::OnDemand,
        }
    }
}
