/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Prepare and fingerprint the external base before graph selection.
//! Input and prepared-state types come first, followed by their operations,
//! file/array preparation, and the allocation and hashing helpers they use.
use crate::config::{ResolvedRunConfig, VectorStorageKind};
use crate::data::{self, inspect_vector_file, BaseElement, LoadedBase, VectorFormat};
use crate::preflight as resources;
use crate::{Error, IndexOptions, Result};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Array data is copied directly into the final aligned base allocation. Python
/// callers create this while holding the GIL, then release it for graph work.
#[derive(Debug)]
pub struct BaseInput(Input);

#[derive(Debug)]
enum Input {
    File(PathBuf),
    F32 {
        data: LoadedBase<f32>,
        dimension: usize,
    },
    U8 {
        data: LoadedBase<u8>,
        dimension: usize,
    },
}

/// Owned rows plus the original input identity. Graph selection and snapshot
/// verification consume this identity before array normalization changes rows.
pub(super) struct PreparedBase {
    pub rows: Rows,
    pub file: Option<PathBuf>,
    pub sha256: String,
}

pub(super) enum Rows {
    F32(LoadedBase<f32>),
    U8(LoadedBase<u8>),
}

impl BaseInput {
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self(Input::File(path.into()))
    }

    pub fn from_f32(values: &[f32], dimension: usize) -> Result<Self> {
        Ok(Self(Input::F32 {
            data: copy_array(values, dimension)?,
            dimension,
        }))
    }

    pub fn from_u8(values: &[u8], dimension: usize) -> Result<Self> {
        Ok(Self(Input::U8 {
            data: copy_array(values, dimension)?,
            dimension,
        }))
    }

    /// Check shape and storage policy, load or take ownership of the rows, and
    /// fingerprint the original input for graph selection and snapshot verification.
    pub(super) fn validate_and_load(
        self,
        options: &IndexOptions,
        snapshot_graph: Option<&std::path::Path>,
    ) -> Result<(PreparedBase, ResolvedRunConfig)> {
        match self.0 {
            Input::File(path) => prepare_file(path, options, snapshot_graph),
            Input::F32 { data, dimension } => {
                validate_array(&data, dimension)?;
                let digest = f32_digest(&data, dimension)?;
                prepare_array(Rows::F32(data), dimension, digest, options)
            }
            Input::U8 { data, dimension } => {
                validate_array(&data, dimension)?;
                let digest = u8_digest(&data, dimension)?;
                prepare_array(Rows::U8(data), dimension, digest, options)
            }
        }
    }
}

impl PreparedBase {
    /// Apply the metric's coordinate policy to the owned f32 array before it is
    /// used to build a graph, fit sidecars, or search a restored graph:
    ///
    /// - All f32 metrics reject non-finite coordinates.
    /// - Cosine divides each nonzero row by its L2 norm. Its search kernels then
    ///   compare inner products of unit vectors; zero rows are rejected.
    /// - L2 and raw inner product preserve the original coordinates and norms.
    ///
    /// Array fingerprints and snapshot checks use the original coordinates, so
    /// this in-place transformation must follow those checks. It changes only
    /// our owned copy, excluding SIMD padding. File inputs have already applied
    /// this policy during decoding; native u8 is restricted to L2 and stays exact.
    /// Queries apply the same metric policy during calibration and search.
    pub(super) fn validate_and_normalize_array(
        &mut self,
        config: &ResolvedRunConfig,
    ) -> Result<()> {
        if let (None, Rows::F32(data)) = (&self.file, &mut self.rows) {
            let dim = config.dimension;
            for row in data
                .base
                .as_mut()
                .ok_or("missing base")?
                .chunks_exact_mut(data.layout.stride_elements())
            {
                f32::normalize(&mut row[..dim], dim, config.metric)?;
            }
        }
        Ok(())
    }
}

impl Rows {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.num_points,
            Self::U8(data) => data.num_points,
        }
    }
}

fn prepare_file(
    path: PathBuf,
    options: &IndexOptions,
    snapshot_graph: Option<&std::path::Path>,
) -> Result<(PreparedBase, ResolvedRunConfig)> {
    let path = path
        .canonicalize()
        .map_err(|e| Error::io("open", &path, e))?;
    let format = match options.base_format {
        Some(f) => f,
        None => VectorFormat::infer(&path)?,
    };
    let header = inspect_vector_file(&path, format)?;
    let mut config = options.resolve(path.clone(), format, header.dimension, header.count, None)?;
    // Check before allocating the resident base. Its content hash is only
    // available after decoding, and is needed to select the graph cache.
    resources::check_index_budget(&config, header.count.min(config.max_points), snapshot_graph)?;
    let (rows, sha256) = match config.storage() {
        VectorStorageKind::F32 => {
            let (data, hash) = read_file_base(&config)?;
            (Rows::F32(data), hash)
        }
        VectorStorageKind::U8 => {
            let (data, hash) = read_file_base(&config)?;
            (Rows::U8(data), hash)
        }
    };
    config.base_content_id = Some(format!("file-{sha256}"));
    config.cache_namespace = config.compute_cache_namespace();
    Ok((
        PreparedBase {
            rows,
            file: Some(path),
            sha256,
        },
        config,
    ))
}

fn prepare_array(
    rows: Rows,
    dimension: usize,
    sha256: String,
    options: &IndexOptions,
) -> Result<(PreparedBase, ResolvedRunConfig)> {
    let count = rows.len();
    let (format, storage) = match rows {
        Rows::F32(_) => (VectorFormat::Fbin, VectorStorageKind::F32),
        Rows::U8(_) => (VectorFormat::U8bin, VectorStorageKind::U8),
    };
    if options.max_points.is_some_and(|n| n < count) {
        return Err("slice array input before construction instead of using max_points".into());
    }
    if options.storage.is_some_and(|v| v != storage)
        || options.base_format.is_some_and(|v| v != format)
    {
        return Err(
            "array dtype must match storage; base_format applies only to file inputs".into(),
        );
    }
    let config = options.resolve(
        PathBuf::from("<array>"),
        format,
        dimension,
        count,
        Some(sha256.clone()),
    )?;
    Ok((
        PreparedBase {
            rows,
            file: None,
            sha256,
        },
        config,
    ))
}

fn read_file_base<T: BaseElement>(config: &ResolvedRunConfig) -> Result<(LoadedBase<T>, String)> {
    let layout = orion_ann_vector::PaddedLayout::for_kernel::<T::Kernel>(config.dimension)?;
    let (dataset, digest) = data::padded::load_hashed::<T>(
        &config.base,
        config.base_format,
        layout,
        config.max_points,
        config.metric,
    )?;
    let num_points = dataset.len();
    if config.num_points.map(|n| n.min(config.max_points)) != Some(num_points) {
        return Err(Error::invalid_data("base row count changed during load"));
    }
    Ok((
        LoadedBase {
            layout,
            num_points,
            base: Some(dataset.into_allocation()),
        },
        digest,
    ))
}

fn copy_array<T: BaseElement>(values: &[T], dimension: usize) -> Result<LoadedBase<T>> {
    if dimension == 0 || values.is_empty() || values.len() % dimension != 0 {
        return Err("base must be a nonempty rectangular matrix".into());
    }
    let count = values.len() / dimension;
    if count >= u32::MAX as usize {
        return Err("base point count exceeds u32 IDs".into());
    }
    let layout = orion_ann_vector::PaddedLayout::for_kernel::<T::Kernel>(dimension)?;
    let base = orion_ann_diskann::model::PaddedDataset::from_packed(values, layout)?;
    Ok(LoadedBase {
        base: Some(base.into_allocation()),
        num_points: count,
        layout,
    })
}

/// Verify the resident allocation before hashing or slicing its rows.
fn validate_array<T: BaseElement>(data: &LoadedBase<T>, dimension: usize) -> Result<()> {
    if dimension != data.layout.logical_dim() {
        return Err(
            Error::dimension("base.dimension", data.layout.logical_dim(), dimension)
                .at("validate_base"),
        );
    }
    let elements = data
        .num_points
        .checked_mul(data.layout.stride_elements())
        .ok_or_else(|| Error::invalid_argument("base shape overflows address space"))?;
    let allocation = data
        .base
        .as_ref()
        .ok_or_else(|| Error::invalid_state("base allocation has already been consumed"))?;
    if data.num_points == 0 || data.num_points >= u32::MAX as usize || allocation.len() < elements {
        return Err(
            Error::invalid_argument("base allocation does not contain its declared rows")
                .at("validate_base")
                .with_detail("required_elements", elements)
                .with_detail("actual_elements", allocation.len()),
        );
    }
    Ok(())
}

fn f32_digest(data: &LoadedBase<f32>, dimension: usize) -> Result<String> {
    // Hash the owned, unnormalized allocation after the Python GIL
    // has been released. Hashes stay stable across host byte order.
    let base = data.base.as_ref().ok_or("missing array base")?;
    let mut hash = Sha256::new();
    if cfg!(target_endian = "little") && data.layout.stride_elements() == dimension {
        hash.update(bytemuck::cast_slice(base.as_slice()));
    } else {
        for row in base
            .chunks_exact(data.layout.stride_elements())
            .take(data.num_points)
        {
            if cfg!(target_endian = "little") {
                hash.update(bytemuck::cast_slice(&row[..dimension]));
            } else {
                for value in &row[..dimension] {
                    hash.update(value.to_le_bytes());
                }
            }
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn u8_digest(data: &LoadedBase<u8>, dimension: usize) -> Result<String> {
    let base = data.base.as_ref().ok_or("missing array base")?;
    let mut hash = Sha256::new();
    for row in base
        .chunks_exact(data.layout.stride_elements())
        .take(data.num_points)
    {
        hash.update(&row[..dimension]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ErrorCode, Index};

    #[test]
    fn manually_assembled_array_shapes_return_errors_before_slicing() {
        let mut input = BaseInput::from_f32(&[1.0; 24], 3).unwrap();
        if let Input::F32 { dimension, .. } = &mut input.0 {
            *dimension = 4;
        }
        let error = match Index::new(input, IndexOptions::default()) {
            Err(error) => error,
            Ok(_) => panic!("inconsistent array shape accepted"),
        };
        assert_eq!(error.code(), ErrorCode::DimensionMismatch);
        assert_eq!(error.details()["expected"], 3);
        assert_eq!(error.details()["actual"], 4);
    }
}
