/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
#![cfg(not(target_arch = "wasm32"))]
use orion_ann_core::{
    model::dataset::rows::RowTransform,
    search::{
        cascade::stage::{rerank::NoRerank, NoPrefilter},
        query::QueryInput,
    },
    DynamicCascade, DynamicSearchConfig, Orion, ScoreMetric,
};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_platform::{
    reading::{ObjectReader, ReadLease, ReadRequest},
    region::{ArrayView, Placement, ReadRegion, RegionOptions},
};
use orion_ann_ssd::{CachedReader, FileReader, FileReaderOptions, PageCacheOptions, SearchDriver};
use orion_ann_vector::{L2U8Distance, PaddedLayout};
use std::{
    future::Future,
    io::{self, BufWriter, Write},
    pin::Pin,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Semaphore;
fn fixture() -> Orion<u8> {
    let base = PaddedDataset::from_packed(
        &(0..128 * 128)
            .map(|i| ((i * 31 + i / 128 * 7) % 251 + 1) as u8)
            .collect::<Vec<_>>(),
        PaddedLayout::for_kernel::<L2U8Distance>(128).unwrap(),
    )
    .unwrap();
    let parts: Vec<_> = (0..128u32)
        .map(|i| {
            (
                (1..5).map(|j| (i + j * 7) % 128).collect(),
                (5..8).map(|j| (i + j * 7) % 128).collect(),
                vec![(i + 1) % 128],
            )
        })
        .collect();
    Orion::from_graph_partitions(base, &parts, 0, 8, 1).unwrap()
}
fn encoded(index: &Orion<u8>) -> Vec<u8> {
    let mut w = BufWriter::new(Vec::new());
    index.encode_native_graph(Vec::new(), &mut w).unwrap();
    w.into_inner().unwrap()
}
fn cache_options() -> PageCacheOptions {
    PageCacheOptions {
        page_bytes: 64,
        capacity_bytes: 128,
        max_read_bytes: 1024,
        ..Default::default()
    }
}
async fn deadline<T>(f: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), f)
        .await
        .expect("search stalled")
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_pages_and_mapped_graph_match_resident_native_metrics_and_mixed_queries() {
    let index = fixture();
    let bytes = encoded(&index);
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    let file_reader = FileReader::open(file.path(), FileReaderOptions::default()).unwrap();
    let cached = CachedReader::new(file_reader, cache_options()).unwrap();
    // Model preparation and execution can be composed without a second graph.
    let graph = orion_ann_core::model::graph::storage_graph::StorageGraph::open(cached.clone(), 0)
        .await
        .unwrap();
    let driver = SearchDriver::new(graph);
    let mapping = ArrayView::copy_from(
        &bytes,
        &RegionOptions {
            placement: Placement::Mapped,
            ..Default::default()
        },
    )
    .unwrap();
    let mapped = SearchDriver::open(mapping.region().clone(), 0)
        .await
        .unwrap();
    for metric in [
        ScoreMetric::SquaredL2,
        ScoreMetric::InnerProduct,
        ScoreMetric::CosineSimilarity,
    ] {
        let exact = index.exact_stage(metric).unwrap();
        let recipe = DynamicCascade {
            prefilter: &NoPrefilter,
            admission: &exact,
            rerank: &NoRerank,
        };
        let cfg = DynamicSearchConfig {
            k: 10,
            search_list_size: 32,
            epsilon: 0.5,
            early_exit_limit: 4,
            ..Default::default()
        };
        let q = [13u8; 128];
        let floats = [13f32; 128];
        for query in [QueryInput::U8(&q), QueryInput::F32(&floats)] {
            let expected = index.search(query, cfg, recipe, Some(metric)).unwrap();
            let (a, b) = tokio::join!(
                driver.search(
                    index.dataset(),
                    query,
                    cfg,
                    recipe,
                    RowTransform::Identity,
                    Some(metric)
                ),
                mapped.search(
                    index.dataset(),
                    query,
                    cfg,
                    recipe,
                    RowTransform::Identity,
                    Some(metric)
                )
            );
            assert_eq!(expected, a.unwrap());
            assert_eq!(expected, b.unwrap());
        }
    }
    assert!(cached.stats().evictions > 0);
    assert!(cached.stats().reserved_bytes <= 128);
}
#[derive(Clone)]
struct Paused {
    bytes: ReadRegion,
    pause: Arc<AtomicBool>,
    entered: Arc<Semaphore>,
    release: Arc<Semaphore>,
}
impl ObjectReader for Paused {
    type Read<'a> = Pin<Box<dyn Future<Output = io::Result<ReadLease<'a>>> + Send + 'a>>;
    fn byte_len(&self) -> u64 {
        self.bytes.byte_len()
    }
    fn read(&self, r: ReadRequest) -> Self::Read<'_> {
        Box::pin(async move {
            if self.pause.swap(false, Ordering::SeqCst) {
                self.entered.add_permits(1);
                self.release.acquire().await.unwrap().forget();
            }
            self.bytes.read(r).await
        })
    }
}
#[tokio::test]
async fn paused_query_is_resumed_or_cancelled_without_retaining_scratch_or_pages() {
    let index = fixture();
    let source = Paused {
        bytes: ArrayView::copy_from(&encoded(&index), &Default::default())
            .unwrap()
            .region()
            .clone(),
        pause: Arc::new(AtomicBool::new(false)),
        entered: Arc::new(Semaphore::new(0)),
        release: Arc::new(Semaphore::new(0)),
    };
    let cache = CachedReader::new(source.clone(), cache_options()).unwrap();
    let driver = SearchDriver::open(cache.clone(), 0).await.unwrap();
    let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
    let recipe = DynamicCascade {
        prefilter: &NoPrefilter,
        admission: &exact,
        rerank: &NoRerank,
    };
    let cfg = DynamicSearchConfig {
        k: 10,
        search_list_size: 32,
        ..Default::default()
    };
    let query = QueryInput::U8(&[13; 128]);
    let expected = index.search(query, cfg, recipe, None).unwrap();
    for cancel in [false, true] {
        cache.clear().unwrap();
        source.pause.store(true, Ordering::SeqCst);
        let mut search = Box::pin(driver.search(
            index.dataset(),
            query,
            cfg,
            recipe,
            RowTransform::Identity,
            None,
        ));
        // Require a Send future so callers can move a pending query between workers.
        fn is_send<T: Send>(_: &T) {}
        is_send(&search);
        tokio::select! {
            permit = source.entered.acquire() => { permit.unwrap().forget(); }
            result = &mut search => panic!("search did not pause: {result:?}"),
        }
        assert!(cache.stats().reserved_bytes <= 128);
        if cancel {
            drop(search);
            source.release.add_permits(1);
            let result = deadline(driver.search(
                index.dataset(),
                query,
                cfg,
                recipe,
                RowTransform::Identity,
                None,
            ))
            .await
            .unwrap();
            assert_eq!(expected, result);
        } else {
            source.release.add_permits(1);
            assert_eq!(expected, deadline(search).await.unwrap());
        }
    }
    cache.clear().unwrap();
    assert_eq!(cache.stats().reserved_bytes, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_queries_share_a_single_page_slot() {
    let index = Arc::new(fixture());
    let bytes = encoded(&index);
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&bytes).unwrap();
    file.flush().unwrap();
    let cache = CachedReader::new(
        FileReader::open(file.path(), FileReaderOptions::default()).unwrap(),
        PageCacheOptions {
            capacity_bytes: 64,
            ..cache_options()
        },
    )
    .unwrap();
    let driver = Arc::new(SearchDriver::open(cache.clone(), 0).await.unwrap());
    let mut tasks = Vec::new();
    for row in 0..8 {
        let index = index.clone();
        let driver = driver.clone();
        tasks.push(tokio::spawn(async move {
            let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
            let recipe = DynamicCascade {
                prefilter: &NoPrefilter,
                admission: &exact,
                rerank: &NoRerank,
            };
            let cfg = DynamicSearchConfig {
                k: 10,
                search_list_size: 32,
                ..Default::default()
            };
            let q = [row * 13 + 1; 128];
            let query = QueryInput::U8(&q);
            let expected = index.search(query, cfg, recipe, None).unwrap();
            let result = driver
                .search(
                    index.dataset(),
                    query,
                    cfg,
                    recipe,
                    RowTransform::Identity,
                    None,
                )
                .await
                .unwrap();
            assert_eq!(expected, result);
        }));
    }
    for task in tasks {
        deadline(task).await.unwrap();
    }
    assert!(cache.stats().reserved_bytes <= 64);
    cache.clear().unwrap();
    assert_eq!(cache.stats().reserved_bytes, 0);
}
