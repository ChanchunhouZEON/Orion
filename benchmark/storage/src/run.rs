/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::{
    data::{self, Dataset, Sidecars},
    Calibration,
};
use anyhow::{ensure, Result};
use clap::{Parser, ValueEnum};
use hdrhistogram::Histogram;
use orion_ann_core::{
    model::dataset::rows::RowTransform,
    search::{
        cascade::stage::{
            admission::{L2KTAdmission, L2U8Admission},
            prefilter::jl::JlPrefilter,
            rerank::DynamicF32Rerank,
            NoPrefilter,
        },
        query::QueryInput,
    },
};
use orion_ann_core::{DynamicCascade, DynamicSearchConfig, Orion, SearchResult};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_platform::region::{MutableRegion, ReadRegion};
use orion_ann_ssd::{CachedReader, FileReader, FileReaderOptions, PageCacheOptions, SearchDriver};
use orion_ann_vector::L2F32Distance;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Arm {
    Mmap,
    MmapDriver,
    Ssd,
}
#[derive(Parser)]
pub struct Args {
    /// Load and validate storage, then exit without warmup or timed searches.
    #[arg(long)]
    pub load_only: bool,
    #[arg(long)]
    pub dataset: PathBuf,
    #[arg(long)]
    pub calibration: PathBuf,
    #[arg(long)]
    pub arm: Arm,
    #[arg(long)]
    pub work: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,
    #[arg(long, default_value_t = 128)]
    pub page_cache_mib: usize,
    #[arg(long, default_value_t = 30.0)]
    pub seconds: f64,
    #[arg(long)]
    pub beam: Option<usize>,
}
enum Engine {
    Native(Box<Orion>),
    Mapped(PaddedDataset<f32>, SearchDriver<ReadRegion>),
    File(PaddedDataset<f32>, SearchDriver<CachedReader<FileReader>>),
}
impl Engine {
    fn base(&self) -> &PaddedDataset<f32> {
        match self {
            Self::Native(i) => i.dataset(),
            Self::Mapped(b, _) | Self::File(b, _) => b,
        }
    }
    async fn raw(
        &self,
        q: QueryInput<'_>,
        c: DynamicSearchConfig,
        r: DynamicCascade<'_>,
    ) -> Result<SearchResult> {
        Ok(match self {
            Self::Native(i) => i.search(q, c, r, None)?,
            Self::Mapped(b, d) => d.search(b, q, c, r, RowTransform::Identity, None).await?,
            Self::File(b, d) => d.search(b, q, c, r, RowTransform::Identity, None).await?,
        })
    }
    async fn search(
        &self,
        q: &[f32],
        c: DynamicSearchConfig,
        s: &Sidecars,
    ) -> Result<SearchResult> {
        let rerank = DynamicF32Rerank::<L2F32Distance>::new(self.base());
        match s {
            Sidecars::Scalar(s) => {
                let admission = L2U8Admission::new(s);
                self.raw(
                    QueryInput::F32(q),
                    c,
                    DynamicCascade {
                        prefilter: &NoPrefilter,
                        admission: &admission,
                        rerank: &rerank,
                    },
                )
                .await
            }
            Sidecars::KtJl(k, j) => {
                let admission = L2KTAdmission::new(k);
                let prefilter = JlPrefilter::new(j);
                self.raw(
                    QueryInput::F32(q),
                    c,
                    DynamicCascade {
                        prefilter: &prefilter,
                        admission: &admission,
                        rerank: &rerank,
                    },
                )
                .await
            }
        }
    }
    fn counters(&self) -> serde_json::Value {
        if let Self::File(_, driver) = self {
            let s = driver.graph().reader().stats();
            serde_json::json!({"hits":s.hits,"misses":s.misses,"coalesced":s.coalesced,"evictions":s.evictions,"reserved_bytes":s.reserved_bytes,"page_bytes":4096})
        } else {
            serde_json::Value::Null
        }
    }
}
struct Loaded {
    engine: Engine,
    sidecars: Sidecars,
    queries: Vec<f32>,
    dim: usize,
    gt: Vec<u32>,
}
pub fn execute(a: Args) -> Result<()> {
    ensure!((1..=8).contains(&a.concurrency), "concurrency must be 1..8");
    ensure!(
        a.seconds.is_finite() && a.seconds >= 0. && (1..=1024).contains(&a.page_cache_mib),
        "invalid duration/cache capacity"
    );
    std::fs::create_dir_all(&a.work)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(a.concurrency)
        .max_blocking_threads(8)
        .enable_all()
        .build()?;
    runtime.block_on(run(a))
}
async fn run(a: Args) -> Result<()> {
    let start = Instant::now();
    let d = Dataset::load(&a.dataset)?;
    let cal: Calibration = serde_json::from_reader(File::open(&a.calibration)?)?;
    let private = tempfile::tempdir_in(&a.work)?;
    let opts = data::mapped_options(private.path());
    eprintln!("preparing {} {:?}: base", d.name, a.arm);
    crate::load_log::stage("base", "prepare.begin");
    let base = data::base(&d.base, &opts)?;
    crate::load_log::stage("base", "prepare.complete");
    ensure!(
        base.placement() == orion_ann_platform::region::Placement::Mapped,
        "base must be mapped"
    );
    let (dim, queries) = data::vectors::<f32>(&d.queries)?;
    let (gt_width, gt) = data::vectors::<u32>(&d.ground_truth)?;
    ensure!(
        dim == base.layout().logical_dim()
            && queries.len() / dim == gt.len() / gt_width
            && gt_width >= cal.k,
        "query/GT shape mismatch"
    );
    eprintln!("preparing graph");
    let engine = match a.arm {
        Arm::Mmap => Engine::Native(Box::new(data::index(&d, base, &opts)?)),
        Arm::MmapDriver => {
            crate::load_log::stage("graph", "allocation.begin");
            let mut mapping = MutableRegion::new(d.graph.metadata()?.len() as usize, 64, &opts)?;
            crate::load_log::stage("graph", "allocation.complete");
            crate::load_log::stage("graph", "read.begin");
            let mut reader = data::reader(&d.graph)?;
            let bytes = mapping.as_bytes().len();
            let window = orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES;
            for start in (0..bytes).step_by(window) {
                let end = (start + window).min(bytes);
                orion_ann_platform::buffered_io::read_exact_in_blocks(
                    &mut reader,
                    &mut mapping.as_bytes_mut()[start..end],
                )?;
                mapping.flush_and_reclaim(start..end)?;
            }
            crate::load_log::stage("graph", "read.complete");
            crate::load_log::stage("graph", "flush.begin");
            mapping.flush_and_reclaim(0..mapping.as_bytes().len())?;
            crate::load_log::stage("graph", "flush.complete");
            // The resumable driver validates the header here and records on
            // acquisition. Do not report this as a complete row scan.
            crate::load_log::stage("graph", "header_validation.begin");
            let driver = SearchDriver::open(mapping.freeze()?, data::entry(&d)?).await?;
            crate::load_log::stage("graph", "header_validation.complete");
            Engine::Mapped(base, driver)
        }
        Arm::Ssd => {
            // A byte copy to a fresh inode inside the job prevents another
            // cgroup's warm file-cache charge from bypassing this job's limit.
            // Do not use reflinks, hardlinks or a shared pre-warmed source here.
            crate::load_log::stage("graph", "allocation.begin");
            let path = private.path().join("graph.bin");
            let mut input = data::reader(&d.graph)?;
            let mut output = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)?;
            crate::load_log::stage("graph", "allocation.complete");
            crate::load_log::stage("graph", "read.begin");
            let mut block = vec![0u8; orion_ann_platform::buffered_io::FILE_READ_BYTES];
            let mut copied = 0u64;
            let mut reclaimed = 0u64;
            loop {
                let bytes = input.read(&mut block)?;
                if bytes == 0 {
                    break;
                }
                output.write_all(&block[..bytes])?;
                copied += bytes as u64;
                if copied - reclaimed
                    >= orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES as u64
                {
                    orion_ann_platform::file_cache::sync_and_reclaim_file(
                        &output,
                        reclaimed..copied,
                    )?;
                    reclaimed = copied;
                }
            }
            orion_ann_platform::file_cache::sync_and_reclaim_file(&output, reclaimed..copied)?;
            drop(block);
            crate::load_log::stage("graph", "read.complete");
            crate::load_log::stage("graph", "flush.begin");
            output.flush()?;
            output.sync_all()?;
            drop(output);
            crate::load_log::stage("graph", "flush.complete");
            let cache = CachedReader::new(
                FileReader::open(&path, FileReaderOptions::default())?,
                PageCacheOptions {
                    capacity_bytes: a.page_cache_mib * 1024 * 1024,
                    ..Default::default()
                },
            )?;
            crate::load_log::stage("graph", "header_validation.begin");
            let driver = SearchDriver::open(cache, data::entry(&d)?).await?;
            crate::load_log::stage("graph", "header_validation.complete");
            Engine::File(base, driver)
        }
    };
    eprintln!("preparing sidecars");
    crate::load_log::stage("sidecars", "prepare.begin");
    let sidecars = Sidecars::load(&d, &opts)?;
    crate::load_log::stage("sidecars", "prepare.complete");
    let loaded = Arc::new(Loaded {
        engine,
        sidecars,
        queries,
        dim,
        gt,
    });
    let prep_seconds = start.elapsed().as_secs_f64();
    let mut output = File::create(&a.output)?;
    if a.load_only {
        let record = serde_json::json!({
            "kind": "load_validation", "status": "complete", "dataset": d.name,
            "arm": format!("{:?}", a.arm), "prepare_seconds": prep_seconds,
            "base_placement": format!("{:?}", loaded.engine.base().placement()),
            "graph_validation": if matches!(a.arm, Arm::Mmap) { "all_rows" } else { "header_only_records_checked_on_acquisition" },
            "read_block_bytes": orion_ann_platform::buffered_io::FILE_READ_BYTES,
            "cache_policy": "writeback_then_reclaim",
            "reclaim_window_bytes": orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES
        });
        writeln!(output, "{record}")?;
        println!("{record}");
        drop(loaded);
        private.close()?;
        crate::load_log::stage("all", "released");
        return Ok(());
    }
    let beams = a.beam.map(|l| vec![l]).unwrap_or(d.beams.clone());
    for beam in beams {
        let cfg = DynamicSearchConfig {
            k: cal.k,
            search_list_size: beam,
            window_size: cal.window,
            epsilon: cal.epsilon,
            early_exit_limit: cal.early_exit_limit,
            ..Default::default()
        };
        // Warm up every worker using the same concurrency as measurement.
        // Reorder complete results only after this untimed pass, so comparison
        // catches concurrent-search errors without charging SHA work to QPS.
        let mut validation = Vec::new();
        for lane in 0..a.concurrency {
            let data = loaded.clone();
            let concurrency = a.concurrency;
            validation.push(tokio::spawn(async move {
                let mut rows = Vec::new();
                for i in (lane..data.queries.len() / data.dim).step_by(concurrency) {
                    let q = &data.queries[i * data.dim..(i + 1) * data.dim];
                    let result = data.engine.search(q, cfg, &data.sidecars).await?;
                    rows.push((i, result));
                    tokio::task::yield_now().await;
                }
                Ok::<_, anyhow::Error>(rows)
            }));
        }
        let mut rows = Vec::new();
        for task in validation {
            rows.extend(task.await??);
        }
        rows.sort_unstable_by_key(|(i, _)| *i);
        let mut hash = Sha256::new();
        let mut hits = 0usize;
        for (i, result) in rows {
            let count = result.counts[0] as usize;
            hash.update(result.counts[0].to_le_bytes());
            for &id in &result.ids[..count] {
                hash.update(id.to_le_bytes());
            }
            hits += result.ids[..count]
                .iter()
                .filter(|id| loaded.gt[i * gt_width..i * gt_width + cal.k].contains(id))
                .count();
        }
        let checksum = format!("{:x}", hash.finalize());
        let before = loaded.engine.counters();
        let started = Instant::now();
        let mut histogram = Histogram::<u64>::new(3)?;
        let mut passes = 0;
        let mut count = 0;
        loop {
            let mut tasks = Vec::new();
            for lane in 0..a.concurrency {
                let data = loaded.clone();
                let concurrency = a.concurrency;
                tasks.push(tokio::spawn(async move {
                    let mut h = Histogram::<u64>::new(3)?;
                    for i in (lane..data.queries.len() / data.dim).step_by(concurrency) {
                        let q = &data.queries[i * data.dim..(i + 1) * data.dim];
                        let t = Instant::now();
                        let result = data.engine.search(q, cfg, &data.sidecars).await?;
                        std::hint::black_box(result);
                        h.record(t.elapsed().as_nanos().max(1).min(u64::MAX as u128) as u64)?;
                        tokio::task::yield_now().await;
                    }
                    Ok::<_, anyhow::Error>(h)
                }));
            }
            for task in tasks {
                let h = task.await??;
                count += h.len();
                histogram.add(h)?;
            }
            passes += 1;
            if started.elapsed() >= Duration::from_secs_f64(a.seconds) {
                break;
            }
        }
        let wall = started.elapsed().as_secs_f64();
        let nq = loaded.queries.len() / dim;
        let result = serde_json::json!({"dataset":d.name,"arm":format!("{:?}",a.arm),"concurrency":a.concurrency,"L":beam,"k":cal.k,"recall":hits as f64/(nq*cal.k) as f64,"validation_sha256":checksum,"queries":count,"passes":passes,"search_seconds":wall,"qps":count as f64/wall,"p50_us":histogram.value_at_quantile(0.50) as f64/1000.,"p95_us":histogram.value_at_quantile(0.95) as f64/1000.,"p99_us":histogram.value_at_quantile(0.99) as f64/1000.,"mean_us":histogram.mean()/1000.,"undersampled":count<10000,"prepare_seconds":prep_seconds,"page_cache_mib":a.page_cache_mib,"cache_before":before,"cache_after":loaded.engine.counters(),"calibration":cal});
        writeln!(output, "{}", result)?;
        output.flush()?;
        println!("{}", result);
    }
    // Drop views and reader handles before removing their private directory.
    drop(loaded);
    drop(private);
    Ok(())
}
