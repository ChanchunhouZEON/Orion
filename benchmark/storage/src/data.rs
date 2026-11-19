/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use anyhow::{bail, ensure, Context, Result};
use orion_ann_core::model::dataset::{JLSparseDataset, L2KTDataset, ScalarQuantizedDataset, L2U8};
use orion_ann_core::{Orion, QuantizedDataset, QuantizedDatasetShape};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_platform::region::{Placement, RegionOptions};
use orion_ann_vector::{L2F32Distance, PaddedLayout};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
pub struct Dataset {
    pub name: String,
    pub base: PathBuf,
    pub queries: PathBuf,
    pub ground_truth: PathBuf,
    pub graph: PathBuf,
    pub metadata: PathBuf,
    pub scalar: Option<PathBuf>,
    pub kt: Option<PathBuf>,
    pub jl: Option<PathBuf>,
    pub beams: Vec<usize>,
}
impl Dataset {
    pub fn load(path: &Path) -> Result<Self> {
        let mut d: Self = serde_json::from_reader(File::open(path)?)?;
        let dir = path.parent().context("missing config parent")?;
        for p in [
            &mut d.base,
            &mut d.queries,
            &mut d.ground_truth,
            &mut d.graph,
            &mut d.metadata,
        ] {
            if p.is_relative() {
                *p = dir.join(&*p);
            }
        }
        for p in [&mut d.scalar, &mut d.kt, &mut d.jl].into_iter().flatten() {
            if p.is_relative() {
                *p = dir.join(&*p);
            }
        }
        ensure!(
            !d.beams.is_empty() && d.beams.iter().all(|&l| l >= 10),
            "invalid beams"
        );
        ensure!(
            matches!(
                (&d.scalar, &d.kt, &d.jl),
                (Some(_), None, None) | (None, Some(_), Some(_))
            ),
            "unsupported cascade"
        );
        Ok(d)
    }
    pub fn estimate(&self) -> Result<serde_json::Value> {
        let (n, dim) = shape(&self.base)?;
        let shape = QuantizedDatasetShape {
            rows: n,
            dimension: dim,
        };
        let base = PaddedLayout::for_kernel::<L2F32Distance>(dim)
            .map_err(anyhow::Error::msg)?
            .allocation_bytes(n)
            .map_err(anyhow::Error::msg)?;
        let graph = self
            .graph
            .metadata()?
            .len()
            .checked_sub(16)
            .context("short graph")?;
        let sidecars = if self.scalar.is_some() {
            ScalarQuantizedDataset::<L2U8>::estimate_resident_memory(shape, &())?.total_bytes()
        } else {
            L2KTDataset::estimate_resident_memory(shape, &())?.total_bytes()
                + JLSparseDataset::<1024>::estimate_resident_memory(
                    shape,
                    &JLSparseDataset::<1024>::default_build_options(),
                )?
                .total_bytes()
        };
        let queries = self.queries.metadata()?.len();
        let ground_truth = self.ground_truth.metadata()?.len();
        let counters = n as u64 * 8;
        // Declared common allowance, not a prediction of Linux page residency.
        let runtime_headroom = 256u64 * 1024 * 1024;
        let total = base as u64
            + graph
            + sidecars as u64
            + queries
            + ground_truth
            + counters
            + runtime_headroom;
        Ok(
            serde_json::json!({"dataset":self.name,"rows":n,"dimension":dim,"base_bytes":base,"graph_bytes":graph,"sidecar_bytes":sidecars,"reader_counter_bytes":counters,"query_bytes":queries,"ground_truth_bytes":ground_truth,"runtime_headroom_bytes":runtime_headroom,"estimated_bytes":total}),
        )
    }
}
pub fn reader(p: &Path) -> Result<BufReader<File>> {
    Ok(BufReader::with_capacity(8 * 1024 * 1024, File::open(p)?))
}
pub fn shape(p: &Path) -> Result<(usize, usize)> {
    let mut f = File::open(p)?;
    let mut b = [0; 4];
    f.read_exact(&mut b)?;
    let d = u32::from_le_bytes(b) as usize;
    ensure!(d > 0 && d <= 65536, "invalid vector width");
    let row = 4 + d as u64 * 4;
    let bytes = f.metadata()?.len();
    ensure!(
        bytes % row == 0 && bytes >= row,
        "invalid fvecs/ivecs length"
    );
    Ok(((bytes / row) as usize, d))
}
pub fn mapped_options(dir: &Path) -> RegionOptions {
    RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.to_owned()),
        reclaim_after_write: true,
        ..Default::default()
    }
}
/// Every row is decoded directly into its final private mapped allocation.
pub fn base(p: &Path, opts: &RegionOptions) -> Result<PaddedDataset<f32>> {
    let (n, d) = shape(p)?;
    let layout = PaddedLayout::for_kernel::<L2F32Distance>(d).map_err(anyhow::Error::msg)?;
    crate::load_log::stage("base", "allocation.begin");
    let mut base =
        PaddedDataset::<f32>::new_with_storage(n, layout, opts).map_err(anyhow::Error::msg)?;
    crate::load_log::stage("base", "allocation.complete");
    let mut r = reader(p)?;
    let mut width = [0; 4];
    let window_rows =
        (orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES / layout.stride_bytes()).max(1);
    let mut reclaimed = 0;
    crate::load_log::stage("base", "read_and_reclaim.begin");
    for i in 0..n {
        r.read_exact(&mut width)?;
        ensure!(u32::from_le_bytes(width) as usize == d, "ragged base");
        let row = base.row_mut(i).context("base row")?;
        r.read_exact(bytemuck::cast_slice_mut(row))?;
        ensure!(row.iter().all(|x| x.is_finite()), "non-finite base");
        if opts.reclaim_after_write && (i + 1 - reclaimed >= window_rows || i + 1 == n) {
            base.flush_and_reclaim_rows(reclaimed..i + 1)?;
            reclaimed = i + 1;
        }
    }
    crate::load_log::stage("base", "read_and_reclaim.complete");
    crate::load_log::stage("base", "flush.begin");
    if !opts.reclaim_after_write {
        flush_mapping(base.as_slice())?;
    }
    crate::load_log::stage("base", "flush.complete");
    Ok(base)
}
pub fn vectors<T: bytemuck::Pod>(p: &Path) -> Result<(usize, Vec<T>)> {
    ensure!(
        std::mem::size_of::<T>() == 4,
        "vecs requires 32-bit coordinates"
    );
    let (n, d) = shape(p)?;
    let mut data = vec![T::zeroed(); n * d];
    let mut r = reader(p)?;
    let mut width = [0; 4];
    for row in data.chunks_mut(d) {
        r.read_exact(&mut width)?;
        ensure!(u32::from_le_bytes(width) as usize == d, "ragged vectors");
        r.read_exact(bytemuck::cast_slice_mut(row))?;
    }
    Ok((d, data))
}
pub fn index(d: &Dataset, base: PaddedDataset<f32>, opts: &RegionOptions) -> Result<Orion> {
    crate::load_log::stage("graph", "header.begin");
    let index = Orion::decode_native_graph_with_storage(
        base,
        reader(&d.metadata)?,
        crate::load_log::NativeGraphReader::new(reader(&d.graph)?, d.graph.metadata()?.len()),
        d.graph.metadata()?.len(),
        opts,
    )?;
    crate::load_log::stage("graph", "final_reclaim_and_validation.complete");
    crate::load_log::stage("graph", "validation.complete");
    crate::load_log::stage("graph", "flush.begin");
    flush_mapping(index.graph().buffer_bytes())?;
    crate::load_log::stage("graph", "flush.complete");
    Ok(index)
}
pub fn entry(d: &Dataset) -> Result<u32> {
    let mut r = reader(&d.metadata)?;
    let mut h = [0; 8];
    r.read_exact(&mut h)?;
    ensure!(
        u32::from_le_bytes(h[..4].try_into()?) == 6,
        "metadata must use version 6"
    );
    Ok(u32::from_le_bytes(h[4..].try_into()?))
}
pub enum Sidecars {
    Scalar(ScalarQuantizedDataset<L2U8>),
    KtJl(L2KTDataset, Box<JLSparseDataset<1024>>),
}
impl Sidecars {
    pub fn load(d: &Dataset, o: &RegionOptions) -> Result<Self> {
        if let Some(p) = &d.scalar {
            let s = ScalarQuantizedDataset::<L2U8>::decode_with_storage(
                &mut reader(p)?,
                p.metadata()?.len(),
                o,
            )?;
            flush_mapping(s.data())?;
            Ok(Self::Scalar(s))
        } else if let (Some(k), Some(j)) = (&d.kt, &d.jl) {
            let k = L2KTDataset::decode_with_storage(&mut reader(k)?, k.metadata()?.len(), o)?;
            let j = JLSparseDataset::<1024>::decode_with_storage(
                &mut reader(j)?,
                j.metadata()?.len(),
                o,
            )?;
            flush_mapping(k.data())?;
            flush_mapping(k.norms_sq())?;
            flush_mapping(j.codes())?;
            flush_mapping(j.indices())?;
            Ok(Self::KtJl(k, Box::new(j)))
        } else {
            bail!("missing sidecars")
        }
    }
}
/// Preparation writes shared private mappings. Finish writeback before search,
/// so timed page faults do not include avoidable initialization writeback.
pub fn flush_mapping<T>(slice: &[T]) -> Result<()> {
    if slice.is_empty() {
        return Ok(());
    }
    // SAFETY: the caller only supplies live private mapped spans. Round down
    // for typed views whose payload starts within a mapped page.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let ptr = slice.as_ptr() as usize;
    let start = ptr / page * page;
    let bytes = std::mem::size_of_val(slice) + ptr - start;
    if unsafe { libc::msync(start as *mut libc::c_void, bytes, libc::MS_SYNC) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    Ok(())
}
