/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Stream boundary for the runtime's lazy working cache and immutable snapshots.
use super::{CompressedIndex, DynamicSidecars, Error, Result, Rows, SidecarCell, SidecarCodec};
use crate::index::IndexData;
use orion_ann_platform::region::RegionOptions;
use std::io::{Read, Write};
type PlacementPlan = std::collections::BTreeMap<String, RegionOptions>;

pub type DecodeSidecar<'a> = dyn FnMut(&mut dyn Read, u64) -> Result<()> + 'a;
pub type EncodeSidecar<'a> = dyn Fn(&mut dyn Write) -> Result<()> + 'a;

/// Optional runtime service. A miss skips the decoder. On a hit, the service
/// must verify the bytes consumed by the decoder before returning success.
pub trait SidecarCache: Send + Sync {
    fn load(
        &self,
        base: &dyn Rows,
        codec: &'static str,
        decode: &mut DecodeSidecar<'_>,
    ) -> Result<()>;
    fn store(&self, base: &dyn Rows, codec: &'static str, encode: &EncodeSidecar<'_>)
    -> Result<()>;
}

impl<T: SidecarCodec> SidecarCell<T> {
    pub(super) fn ensure(
        &self,
        dataset: &impl Rows,
        cache: Option<&dyn SidecarCache>,
        plan: Option<&PlacementPlan>,
    ) -> Result<&T> {
        let options = plan
            .and_then(|p| p.get(T::EXTENSION))
            .cloned()
            .unwrap_or_default();
        self.get_or_try_init(|| {
            let Some(cache) = cache else {
                return T::build_placed(dataset, &options);
            };
            let shape = (dataset.dimension(), dataset.len());
            match decode_checked::<T>(shape, &options, |decode| {
                cache.load(dataset, T::EXTENSION, decode)
            }) {
                Ok(Some(value)) => return Ok(value),
                Ok(None) => {}
                Err(error) => log::warn!("rebuilding {} sidecar: {error}", T::EXTENSION),
            }
            let value = T::build_placed(dataset, &options)?;
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
    options: &RegionOptions,
    load: impl FnOnce(&mut DecodeSidecar<'_>) -> Result<()>,
) -> Result<Option<T>> {
    let mut candidate = None;
    load(&mut |reader, bytes| {
        let value = T::decode_placed(reader, bytes, options)?;
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
    fn initialized(&self) -> bool;
    fn visit(
        &self,
        visit: &mut dyn FnMut(&'static str, &EncodeSidecar<'_>) -> Result<()>,
    ) -> Result<()>;
    fn restore(
        &self,
        shape: (usize, usize),
        options: &RegionOptions,
        load: &mut dyn FnMut(&mut DecodeSidecar<'_>) -> Result<()>,
    ) -> Result<()>;
}

impl<T: SidecarCodec> Slot for SidecarCell<T> {
    fn initialized(&self) -> bool {
        self.get().is_some()
    }
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
        options: &RegionOptions,
        load: &mut dyn FnMut(&mut DecodeSidecar<'_>) -> Result<()>,
    ) -> Result<()> {
        let value = decode_checked::<T>(shape, options, load)?
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
    /// Bind fitted sidecars to logical input coordinates before fitting/restoring.
    /// # Errors
    /// Rejects a different transform after the generation has been bound. Repeated
    /// preparation with the same transform is harmless, including concurrent calls.
    #[doc(hidden)]
    pub fn set_sidecar_row_transform(
        &self,
        transform: crate::model::dataset::rows::RowTransform,
    ) -> Result<()> {
        let actual = self.sidecars.row_transform.get_or_init(|| transform);
        if *actual != transform {
            return Err(Error::invalid_state(
                "sidecar input transform is already fixed",
            ));
        }
        Ok(())
    }

    /// Install a deployment plan before any sidecar allocation. It is immutable
    /// for this generation, so concurrent searches never mutate backing storage.
    #[doc(hidden)]
    pub fn configure_sidecar_storage(&self, plan: PlacementPlan) -> Result<()> {
        if let Some(current) = self.sidecars.placement.get() {
            return if current == &plan {
                Ok(())
            } else {
                Err(Error::invalid_state(
                    "sidecar placement is already fixed for this generation",
                ))
            };
        }
        if self.sidecars.native_norms.get().is_some()
            || self.sidecars.slots().iter().any(|s| s.initialized())
        {
            return Err(Error::invalid_state(
                "configure storage before materializing sidecars",
            ));
        }
        self.sidecars
            .placement
            .set(plan)
            .map_err(|_| Error::invalid_state("concurrent storage configuration"))
    }
    #[doc(hidden)]
    pub fn sidecar_storage_plan(&self) -> Option<&PlacementPlan> {
        self.sidecars.placement.get()
    }

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
        if let Some(norms) = self.sidecars.native_norms.get() {
            visit(crate::model::dataset::native_u8::NORM_CODEC, &|writer| {
                norms.encode(writer)
            })?;
        }
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
        if codec == crate::model::dataset::native_u8::NORM_CODEC {
            if self.dataset.element_type() != "u8" {
                return Err(Error::cache("native norms require a u8 base"));
            }
            let options = self
                .sidecars
                .placement
                .get()
                .and_then(|p| p.get(codec))
                .cloned()
                .unwrap_or_default();
            let mut candidate = None;
            load(&mut |reader, bytes| {
                candidate = Some(crate::model::dataset::native_u8::NativeU8Norms::decode(
                    reader,
                    bytes,
                    self.dataset.point_count(),
                    self.dataset.dimension(),
                    &options,
                )?);
                Ok(())
            })?;
            let candidate = candidate.ok_or_else(|| Error::cache("missing native u8 norms"))?;
            return self
                .sidecars
                .native_norms
                .value
                .set(candidate)
                .map_err(|_| Error::invalid_state("native norms already initialized"));
        }
        let slot = self
            .sidecars
            .slots()
            .into_iter()
            .find(|slot| slot.codec() == codec)
            .ok_or_else(|| {
                Error::cache("unsupported snapshot sidecar").with_detail("codec", codec)
            })?;
        self.sidecars.row_transform.get_or_init(Default::default);
        slot.restore(
            (self.dataset.dimension(), self.dataset.point_count()),
            &self
                .sidecars
                .placement
                .get()
                .and_then(|p| p.get(codec))
                .cloned()
                .unwrap_or_default(),
            &mut load,
        )
    }
}

impl<D> CompressedIndex<D, DynamicSidecars> {
    #[doc(hidden)]
    pub fn native_norms_region(&self) -> Option<(&[u8], orion_ann_platform::region::Placement)> {
        self.sidecars
            .native_norms
            .get()
            .map(|n| (n.bytes(), n.placement()))
    }
}
impl CompressedIndex<orion_ann_diskann::model::PaddedDataset<u8>, DynamicSidecars> {
    #[doc(hidden)]
    pub fn prepare_native_cosine(&self) -> Result<()> {
        self.native_cosine_norms().map(|_| ())
    }
    pub(crate) fn native_cosine_norms(
        &self,
    ) -> Result<&crate::model::dataset::native_u8::NativeU8Norms> {
        let options = self
            .sidecars
            .placement
            .get()
            .and_then(|p| p.get(crate::model::dataset::native_u8::NORM_CODEC))
            .cloned()
            .unwrap_or_default();
        self.sidecars.native_norms.get_or_try_init(|| {
            crate::model::dataset::native_u8::NativeU8Norms::build(&self.dataset, &options)
        })
    }
}

impl<D: crate::model::dataset::rows::CascadeData> CompressedIndex<D, DynamicSidecars> {
    /// Borrow the base as an exact representation in an ordinary cascade.
    /// # Errors
    /// Rejects unsupported layouts; cosine norm preparation propagates storage
    /// errors and rejects zero vectors before a search session is opened.
    pub fn exact_stage(
        &self,
        metric: crate::ScoreMetric,
    ) -> Result<crate::model::dataset::quantized_dataset::exact::stage::ExactStage<'_>> {
        use crate::model::dataset::quantized_dataset::exact::stage::ExactStage;
        if let Some(base) = self.dataset.padded_f32() {
            return ExactStage::f32(base, metric);
        }
        let base = self
            .dataset
            .padded_u8()
            .ok_or_else(|| Error::unsupported("exact stage requires padded base rows"))?;
        let norms = if metric == crate::ScoreMetric::CosineSimilarity {
            let options = self
                .sidecars
                .placement
                .get()
                .and_then(|p| p.get(crate::model::dataset::native_u8::NORM_CODEC))
                .cloned()
                .unwrap_or_default();
            Some(self.sidecars.native_norms.get_or_try_init(|| {
                crate::model::dataset::native_u8::NativeU8Norms::build(base, &options)
            })?)
        } else {
            None
        };
        ExactStage::u8(base, norms, metric)
    }
    pub(crate) fn sidecar_row_transform(&self) -> crate::model::dataset::rows::RowTransform {
        self.sidecars
            .row_transform
            .get()
            .copied()
            .unwrap_or_default()
    }
}
