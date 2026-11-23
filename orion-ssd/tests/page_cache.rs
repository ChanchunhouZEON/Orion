/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(not(target_arch = "wasm32"))]
use orion_ann_platform::{
    allocation_budget::AllocationBudget,
    reading::{ObjectReader, ReadLease, ReadRequest},
    region::{ArrayView, ReadRegion},
};
use orion_ann_ssd::{CachedReader, PageCacheOptions};
use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Semaphore;
fn req(offset: u64, len: usize) -> ReadRequest {
    ReadRequest::new(offset, len).unwrap()
}
fn data() -> ReadRegion {
    ArrayView::copy_from(
        &(0..251u8).cycle().take(1003).collect::<Vec<_>>(),
        &Default::default(),
    )
    .unwrap()
    .region()
    .clone()
}
fn options(pages: usize) -> PageCacheOptions {
    PageCacheOptions {
        page_bytes: 64,
        capacity_bytes: 64 * pages,
        max_read_bytes: 1024,
        ..Default::default()
    }
}
async fn deadline<T>(f: impl Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(5), f)
        .await
        .expect("operation stalled")
}
#[derive(Clone)]
struct Controlled {
    data: ReadRegion,
    gate: Arc<Semaphore>,
    started: Arc<Semaphore>,
    calls: Arc<AtomicUsize>,
    fail: Arc<AtomicUsize>,
}
impl Controlled {
    fn new() -> Self {
        Self {
            data: data(),
            gate: Arc::new(Semaphore::new(0)),
            started: Arc::new(Semaphore::new(0)),
            calls: Arc::new(AtomicUsize::new(0)),
            fail: Arc::new(AtomicUsize::new(0)),
        }
    }
    async fn entered(&self) {
        deadline(self.started.acquire()).await.unwrap().forget();
    }
}
impl ObjectReader for Controlled {
    type Read<'a> = Pin<Box<dyn Future<Output = io::Result<ReadLease<'a>>> + Send + 'a>>;
    fn byte_len(&self) -> u64 {
        self.data.byte_len()
    }
    fn read(&self, request: ReadRequest) -> Self::Read<'_> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.add_permits(1);
            self.gate.acquire().await.unwrap().forget();
            match self.fail.swap(0, Ordering::SeqCst) {
                1 => return Err(io::Error::other("injected read failure")),
                2 => panic!("injected worker panic"),
                3 => return self.data.read(req(0, 1)).await,
                _ => {}
            }
            self.data.read(request).await
        })
    }
}
#[tokio::test]
async fn hits_eviction_and_object_isolation() {
    let source = data();
    let cache = CachedReader::new(source.clone(), options(2)).unwrap();
    for _ in 0..3 {
        assert_eq!(
            cache.read(req(7, 12)).await.unwrap().as_bytes(),
            &source.as_bytes()[7..19]
        );
    }
    assert_eq!(cache.stats().misses, 1);
    assert_eq!(cache.stats().hits, 2);
    drop(cache.read(req(64, 2)).await.unwrap());
    drop(cache.read(req(1, 2)).await.unwrap());
    drop(cache.read(req(128, 2)).await.unwrap());
    assert_eq!(cache.stats().evictions, 1);
    assert_eq!(cache.stats().reserved_bytes, 128);
    let other = CachedReader::new(
        ArrayView::copy_from(&[255u8; 128], &Default::default())
            .unwrap()
            .region()
            .clone(),
        options(1),
    )
    .unwrap();
    assert_eq!(other.read(req(0, 4)).await.unwrap().as_bytes(), &[255; 4]);
    assert_ne!(cache.read(req(0, 4)).await.unwrap().as_bytes(), &[255; 4]);
}
#[tokio::test]
async fn pinned_sublease_blocks_capacity_until_final_drop() {
    let cache = CachedReader::new(data(), options(1)).unwrap();
    let lease = cache.read(req(0, 32)).await.unwrap();
    let child = lease.section(req(1, 5)).unwrap().into_owned().unwrap();
    drop(lease);
    let next = cache.clone();
    let task =
        tokio::spawn(async move { next.read(req(64, 8)).await.unwrap().into_owned().unwrap() });
    tokio::task::yield_now().await;
    assert!(!task.is_finished());
    assert_eq!(cache.stats().reserved_bytes, 64);
    drop(child);
    let lease = deadline(task).await.unwrap();
    cache.clear().unwrap();
    assert_eq!(cache.stats().reserved_bytes, 64);
    drop(lease);
    assert_eq!(cache.stats().reserved_bytes, 0);
}
#[tokio::test]
async fn shared_load_survives_waiter_cancellation() {
    let reader = Controlled::new();
    let cache = CachedReader::new(reader.clone(), options(1)).unwrap();
    let a = cache.clone();
    let one = tokio::spawn(async move { a.read(req(0, 8)).await.unwrap().into_owned() });
    reader.entered().await;
    let b = cache.clone();
    let two = tokio::spawn(async move { b.read(req(2, 8)).await.unwrap().into_owned() });
    tokio::task::yield_now().await;
    assert!(cache.stats().coalesced > 0);
    one.abort();
    let _ = one.await;
    reader.gate.add_permits(1);
    assert_eq!(
        deadline(two).await.unwrap().unwrap().as_bytes(),
        &reader.data.as_bytes()[2..10]
    );
    assert_eq!(reader.calls.load(Ordering::SeqCst), 1);
}
#[tokio::test]
async fn failed_short_and_panicking_loads_release_slots_and_allow_retry() {
    for failure in [1, 2, 3] {
        let reader = Controlled::new();
        reader.fail.store(failure, Ordering::SeqCst);
        reader.gate.add_permits(2);
        let cache = CachedReader::new(reader.clone(), options(1)).unwrap();
        assert!(deadline(cache.read(req(0, 8))).await.is_err());
        assert_eq!(
            deadline(cache.read(req(0, 8))).await.unwrap().as_bytes(),
            &reader.data.as_bytes()[..8]
        );
        assert_eq!(reader.calls.load(Ordering::SeqCst), 2);
        assert_eq!(cache.stats().reserved_bytes, 64);
    }
}
#[tokio::test]
async fn cross_page_requests_need_only_one_slot_and_release_assembly_budget() {
    let budget = AllocationBudget::new(Some(256));
    let opts = PageCacheOptions {
        assembly_accounting: Some(budget.account("assembly").unwrap()),
        ..options(1)
    };
    let source = data();
    let cache = CachedReader::new(source.clone(), opts).unwrap();
    let lease = deadline(cache.read(req(31, 200))).await.unwrap();
    assert_eq!(lease.as_bytes(), &source.as_bytes()[31..231]);
    assert_eq!(budget.usage().resident_bytes, 200);
    let child = lease.section(req(1, 4)).unwrap();
    drop(lease);
    assert_eq!(budget.usage().resident_bytes, 200);
    drop(child);
    assert_eq!(budget.usage().resident_bytes, 0);
    assert_eq!(cache.stats().reserved_bytes, 64);
    assert_eq!(
        cache.read(req(999, 4)).await.unwrap().as_bytes(),
        &source.as_bytes()[999..1003]
    );
    assert!(cache.read(req(1000, 4)).await.is_err());
    assert!(cache
        .read(req(1003, 0))
        .await
        .unwrap()
        .as_bytes()
        .is_empty());
}
#[tokio::test]
async fn cancelling_capacity_waiter_and_many_concurrent_reads_do_not_leak() {
    let cache = CachedReader::new(data(), options(1)).unwrap();
    let pinned = cache.read(req(0, 1)).await.unwrap();
    let c = cache.clone();
    let task = tokio::spawn(async move { c.read(req(64, 8)).await.unwrap().into_owned() });
    tokio::task::yield_now().await;
    task.abort();
    let _ = task.await;
    drop(pinned);
    let mut tasks = Vec::new();
    for i in 0..24u64 {
        let c = cache.clone();
        tasks.push(tokio::spawn(async move {
            let offset = i % 12 * 64;
            let lease = c.read(req(offset, 12)).await.unwrap();
            assert_eq!(lease.as_bytes()[0], (offset % 251) as u8);
            assert!(c.stats().reserved_bytes <= 64);
        }));
    }
    for task in tasks {
        deadline(task).await.unwrap();
    }
    cache.clear().unwrap();
    assert_eq!(cache.stats().reserved_bytes, 0);
}
#[test]
fn runtime_shutdown_cleans_inflight_entry() {
    let reader = Controlled::new();
    let cache = CachedReader::new(reader.clone(), options(1)).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let c = cache.clone();
        tokio::spawn(async move {
            let _ = c.read(req(0, 8)).await;
        });
        reader.entered().await;
    });
    drop(runtime);
    assert_eq!(cache.stats().reserved_bytes, 0);
    reader.gate.add_permits(1);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        deadline(cache.read(req(0, 8))).await.unwrap();
    });
}
#[test]
fn invalid_configuration_rejected() {
    for (page, capacity) in [(0, 64), (3, 64), (64, 0), (64, 65)] {
        assert!(CachedReader::new(
            data(),
            PageCacheOptions {
                page_bytes: page,
                capacity_bytes: capacity,
                ..options(1)
            }
        )
        .is_err());
    }
}

/// A completed load belongs to its subscribers even before they are polled again.
/// Clearing or competing eviction must not turn that completion into another I/O.
#[tokio::test]
async fn completed_load_survives_clear_before_subscriber_resumes() {
    for cancel in [false, true] {
        let source = Controlled::new();
        let cache = CachedReader::new(source.clone(), options(1)).unwrap();
        let mut pending = cache.read(req(0, 8));
        std::future::poll_fn(|cx| {
            assert!(pending.as_mut().poll(cx).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        source.entered().await;
        source.gate.add_permits(1);
        // This second subscriber proves publication has completed, while the
        // original future remains unpolled with its completion queued.
        drop(deadline(cache.read(req(0, 8))).await.unwrap());
        cache.clear().unwrap();
        assert_eq!(cache.stats().reserved_bytes, 64);
        if cancel {
            drop(pending);
        } else {
            let lease = deadline(pending).await.unwrap();
            assert_eq!(lease.as_bytes(), &source.data.as_bytes()[..8]);
            drop(lease);
        }
        assert_eq!(source.calls.load(Ordering::SeqCst), 1);
        assert_eq!(cache.stats().reserved_bytes, 0);
        source.gate.add_permits(1);
        drop(deadline(cache.read(req(64, 8))).await.unwrap());
    }
}
