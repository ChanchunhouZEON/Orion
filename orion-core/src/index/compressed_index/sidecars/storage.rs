/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Stream boundary for the runtime's lazy working cache and immutable snapshots.
use super::{CompressedIndex, DynamicSidecars, Error, F32Rows, Result, SidecarCell, SidecarCodec};
use crate::index::compressed_index::IndexData;
use std::io::{Read, Write};

pub type DecodeSidecar<'a> = dyn FnMut(&mut dyn Read, u64) -> Result<()> + 'a;
pub type EncodeSidecar<'a> = dyn Fn(&mut dyn Write) -> Result<()> + 'a;

/// Optional runtime service. A miss skips the decoder. On a hit, the service
/// must verify the bytes consumed by the decoder before returning success.
pub trait SidecarCache: Send + Sync {
    fn load(
        &self,
        base: &dyn F32Rows,
        codec: &'static str,
        decode: &mut DecodeSidecar<'_>,
    ) -> Result<()>;
    fn store(
        &self,
        base: &dyn F32Rows,
        codec: &'static str,
        encode: &EncodeSidecar<'_>,
    ) -> Result<()>;
}

impl<T: SidecarCodec> SidecarCell<T> {
    pub(super) fn ensure(
        &self,
        dataset: &impl F32Rows,
        cache: Option<&dyn SidecarCache>,
    ) -> Result<&T> {
        self.get_or_try_init(|| {
            let Some(cache) = cache else {
                return T::build(dataset, &T::default_build_options());
            };
            let shape = (dataset.dimension(), dataset.len());
            match decode_checked::<T>(shape, |decode| cache.load(dataset, T::EXTENSION, decode)) {
                Ok(Some(value)) => return Ok(value),
                Ok(None) => {}
                Err(error) => log::warn!("rebuilding {} sidecar: {error}", T::EXTENSION),
            }
            let value = T::build(dataset, &T::default_build_options())?;
            if let Err(error) = cache.store(dataset, T::EXTENSION, &|writer| value.encode(writer)) {
                log::warn!(
                    "could not persist optional {} sidecar: {error}",
                    T::EXTENSION
                );
            }
            Ok(value)
        })
    }
}

// Keep the candidate private until the caller finishes verifying its input.
fn decode_checked<T: SidecarCodec>(
    shape: (usize, usize),
    load: impl FnOnce(&mut DecodeSidecar<'_>) -> Result<()>,
) -> Result<Option<T>> {
    let mut candidate = None;
    load(&mut |reader, bytes| {
        let value = T::decode(reader, bytes)?;
        if value.shape() != shape {
            return Err(Error::cache("sidecar shape mismatch").with_detail("codec", T::EXTENSION));
        }
        candidate = Some(value);
        Ok(())
    })?;
    Ok(candidate)
}

trait Slot {
    fn codec(&self) -> &'static str;
    fn visit(
        &self,
        visit: &mut dyn FnMut(&'static str, &EncodeSidecar<'_>) -> Result<()>,
    ) -> Result<()>;
    fn restore(
        &self,
        shape: (usize, usize),
        load: &mut dyn FnMut(&mut DecodeSidecar<'_>) -> Result<()>,
    ) -> Result<()>;
}

impl<T: SidecarCodec> Slot for SidecarCell<T> {
    fn codec(&self) -> &'static str {
        T::EXTENSION
    }
    fn visit(
        &self,
        visit: &mut dyn FnMut(&'static str, &EncodeSidecar<'_>) -> Result<()>,
    ) -> Result<()> {
        if let Some(value) = self.get() {
            visit(T::EXTENSION, &|writer| value.encode(writer))?;
        }
        Ok(())
    }
    fn restore(
        &self,
        shape: (usize, usize),
        load: &mut dyn FnMut(&mut DecodeSidecar<'_>) -> Result<()>,
    ) -> Result<()> {
        let value = decode_checked::<T>(shape, load)?
            .ok_or_else(|| Error::cache("missing snapshot sidecar"))?;
        self.value
            .set(value)
            .map_err(|_| Error::invalid_state("sidecar already initialized"))
    }
}

impl DynamicSidecars {
    fn slots(&self) -> [&dyn Slot; 11] {
        [
            &self.q_dataset,
            &self.q_dataset_l2_u16,
            &self.q_dataset_mips,
            &self.q_dataset_mips_i16,
            &self.q_dataset_rabitq,
            &self.q_dataset_rabitq_b4,
            &self.q_dataset_jl,
            &self.q_dataset_jl_mips,
            &self.q_dataset_jl_hadamard,
            &self.q_dataset_l2_kt,
            &self.q_dataset_pq,
        ]
    }
}

impl<D: IndexData> CompressedIndex<D, DynamicSidecars> {
    /// Associate a working cache for future lazy initializations.
    #[doc(hidden)]
    pub fn set_sidecar_cache(&mut self, cache: Box<dyn SidecarCache>) {
        self.sidecars.cache = Some(cache);
    }

    /// Encode only sidecars that have already been built.
    #[doc(hidden)]
    pub fn visit_sidecars(
        &self,
        mut visit: impl FnMut(&'static str, &EncodeSidecar<'_>) -> Result<()>,
    ) -> Result<()> {
        for slot in self.sidecars.slots() {
            slot.visit(&mut visit)?;
        }
        Ok(())
    }

    /// Resolve the codec before opening the stream. Publish only after the
    /// service returns success and the decoded shape matches this dataset.
    #[doc(hidden)]
    pub fn restore_sidecar(
        &mut self,
        codec: &str,
        mut load: impl FnMut(&mut DecodeSidecar<'_>) -> Result<()>,
    ) -> Result<()> {
        let slot = self
            .sidecars
            .slots()
            .into_iter()
            .find(|slot| slot.codec() == codec)
            .ok_or_else(|| {
                Error::cache("unsupported snapshot sidecar").with_detail("codec", codec)
            })?;
        slot.restore(
            (self.dataset.dimension(), self.dataset.point_count()),
            &mut load,
        )
    }
}
