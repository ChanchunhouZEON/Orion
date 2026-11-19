/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Byte-bounded retained pages with shared in-flight reads and lease-aware LRU.
//! Capacity includes frames held by callers after cache eviction. Cross-page
//! requests use a separately accounted bounded assembly buffer; backend-private
//! read buffers and metadata are not part of the retained-page capacity.

mod loading;
mod state;

use orion_ann_platform::{
    allocation_budget::AllocationAccount,
    reading::{ObjectReader, ReadLease, ReadRequest},
    region::{MutableRegion, RegionOptions},
};
use state::{Entry, Frame, Inner, PageUse, State};
use std::{
    future::Future,
    io,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};
use tokio::sync::{watch, Notify, Semaphore};

/// Retained-page and per-request limits for one immutable object cache.
///
/// These are application-cache bounds, not process RSS or kernel page-cache
/// limits. Configure the backend's buffer account and `assembly_accounting`
/// against a shared budget when aggregate allocations must be constrained.
#[derive(Debug, Clone)]
pub struct PageCacheOptions {
    /// Logical fetch granularity, a power of two of at least 4 bytes.
    /// Misses fetch an aligned page (or the shorter final page). This is separate
    /// from the filesystem block size and the OS virtual-memory page size.
    pub page_bytes: usize,

    /// Retained frame capacity, a positive multiple of `page_bytes`.
    /// In-flight loads and caller-held frames consume slots too. The short final
    /// page reserves a full slot; metadata and assembled results are additional.
    pub capacity_bytes: usize,

    /// Largest logical request accepted by the cache, before page rounding.
    /// Must be nonzero and fit a Rust slice. A request may span more pages than
    /// the cache can retain: cross-page results are assembled sequentially.
    pub max_read_bytes: usize,

    /// Optional account for contiguous outputs assembled across page boundaries.
    /// Single-page results share their backend buffer instead. This account does
    /// not replace the underlying reader's own buffer accounting.
    pub assembly_accounting: Option<AllocationAccount>,
}

impl Default for PageCacheOptions {
    /// Selects bounded cache pages and read assembly limits without allocating cache frames.
    fn default() -> Self {
        Self {
            page_bytes: 4096,
            capacity_bytes: 64 * 1024 * 1024,
            max_read_bytes: 8 * 1024 * 1024,
            assembly_accounting: None,
        }
    }
}

/// Cumulative observations shared by all clones of a [`CachedReader`].
///
/// Counts describe page lookups, not queries or physical device operations.
/// Fields are sampled separately during concurrent activity. Take differences
/// around a phase to exclude warmup; [`CachedReader::clear`] does not reset them.
#[derive(Debug, Clone, Copy)]
pub struct PageCacheStats {
    /// Page lookups served immediately from a ready cache entry.
    pub hits: u64,
    /// Backend page loads started, excluding coalesced waiters.
    pub misses: u64,
    /// Subscriptions to an already pending page load.
    pub coalesced: u64,
    /// Ready frames removed to make room for another page. Explicit `clear`
    /// calls are not counted, and removal is not evidence of kernel-cache eviction.
    pub evictions: u64,
    /// Full page slots reserved by in-flight loads or live frames, including
    /// caller-held frames. A short last page still reserves one full slot.
    pub reserved_bytes: usize,
}

/// Share bounded graph/object pages among concurrent readers of one generation.
///
/// Clones share capacity, LRU state and pending loads. Offset keys belong to the
/// supplied reader for the cache's entire lifetime; open a new cache for a new
/// generation rather than mutating its file underneath existing reads.
///
/// A single-page result borrows the retained frame without copying its bytes.
/// Cross-page results own a contiguous copy assembled one page at a time. A
/// live result can prevent eviction: callers should release unneeded leases
/// before awaiting further reads when capacity is tight.
///
/// # Execution and cancellation
/// Misses require a Tokio runtime and share one load among waiting queries.
/// Cancelling a waiter does not cancel that shared load. Runtime shutdown or a
/// failed loader wakes its waiters with an error instead of leaving a pending
/// entry behind. Clearing the cache is not an I/O cancellation mechanism.
pub struct CachedReader<R> {
    inner: Arc<Inner<R>>,
}

impl<R> Clone for CachedReader<R> {
    /// Shares cache state, in-flight loads and capacity accounting instead of duplicating pages.
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<R> std::fmt::Debug for CachedReader<R> {
    /// Formats cache policy without locking or dumping cached payloads.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedReader")
            .field("options", &self.inner.options)
            .finish_non_exhaustive()
    }
}

impl<R: ObjectReader + 'static> CachedReader<R> {
    /// Create an empty cache without reading or allocating page payloads.
    /// The reader's logical size is captured now; capacity is reserved lazily
    /// as pages are requested. This constructor itself needs no Tokio runtime.
    ///
    /// # Errors
    /// Rejects invalid page geometry, a capacity smaller than one page or too
    /// many slots, and a zero/unrepresentable per-request bound. Backend read
    /// limits are checked by the backend when a miss is serviced.
    pub fn new(reader: R, options: PageCacheOptions) -> io::Result<Self> {
        if options.page_bytes < 4
            || !options.page_bytes.is_power_of_two()
            || options.capacity_bytes < options.page_bytes
            || !options.capacity_bytes.is_multiple_of(options.page_bytes)
            || options.capacity_bytes / options.page_bytes > Semaphore::MAX_PERMITS
            || options.max_read_bytes == 0
            || options.max_read_bytes > isize::MAX as usize
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid page cache bounds",
            ));
        }
        let slots = options.capacity_bytes / options.page_bytes;
        let bytes = reader.byte_len();
        Ok(Self {
            inner: Arc::new(Inner {
                reader,
                bytes,
                options,
                slots: Arc::new(Semaphore::new(slots)),
                state: Mutex::new(State::default()),
                changed: Arc::new(Notify::new()),
                hits: AtomicU64::new(0),
                misses: AtomicU64::new(0),
                coalesced: AtomicU64::new(0),
                evictions: AtomicU64::new(0),
            }),
        })
    }

    /// Inspect activity and currently reserved frame slots without clearing them.
    /// See [`PageCacheStats`] for what these counters include and exclude.
    pub fn stats(&self) -> PageCacheStats {
        let i = &self.inner;
        PageCacheStats {
            hits: i.hits.load(Ordering::Relaxed),
            misses: i.misses.load(Ordering::Relaxed),
            coalesced: i.coalesced.load(Ordering::Relaxed),
            evictions: i.evictions.load(Ordering::Relaxed),
            reserved_bytes: i.options.capacity_bytes
                - i.slots.available_permits() * i.options.page_bytes,
        }
    }

    /// Evict ready entries. Active leases still retain capacity; pending loads
    /// are left intact so this is not cancellation of other queries' work.
    /// A pending load can repopulate the cache after this call returns. Counters
    /// are preserved; reserved capacity need not immediately fall to zero.
    ///
    /// # Errors
    /// Reports a poisoned cache-state mutex.
    pub fn clear(&self) -> io::Result<()> {
        let mut state = self
            .inner
            .state
            .lock()
            .map_err(|_| io::Error::other("page cache poisoned"))?;
        state
            .entries
            .retain(|_, entry| matches!(entry, Entry::Loading { .. }));
        state.lru.clear();
        drop(state);
        self.inner.changed.notify_waiters();
        Ok(())
    }

    /// Acquires one cached page, coalescing misses and waiting for evictable capacity when necessary.
    async fn page(&self, id: u64) -> io::Result<Arc<Frame>> {
        let mut waited_for_page = false;
        'retry: loop {
            // Register before checking capacity, so a final lease drop cannot
            // race between the check and sleeping (including notify_waiters).
            let notified = self.inner.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let pending = {
                let mut state = self
                    .inner
                    .state
                    .lock()
                    .map_err(|_| io::Error::other("page cache poisoned"))?;
                if let Some(Entry::Ready { frame, stamp }) = state.entries.get(&id) {
                    let frame = frame.clone();
                    let stamp = *stamp;
                    state.lru.remove(&(stamp, id));
                    let stamp = state.tick()?;
                    state.lru.insert((stamp, id));
                    state.entries.insert(
                        id,
                        Entry::Ready {
                            frame: frame.clone(),
                            stamp,
                        },
                    );
                    if !waited_for_page {
                        self.inner.hits.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(frame);
                }
                if let Some(Entry::Loading { receiver, .. }) = state.entries.get(&id) {
                    self.inner.coalesced.fetch_add(1, Ordering::Relaxed);
                    Some(receiver.clone())
                } else {
                    let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
                        io::Error::new(
                            io::ErrorKind::Unsupported,
                            "cache misses require a Tokio runtime",
                        )
                    })?;
                    let mut permit = self.inner.slots.clone().try_acquire_owned().ok();
                    if permit.is_none() {
                        // A pinned frame is skipped, never freed or reused. Scan
                        // only LRU entries; metadata remains bounded by page slots.
                        let victim = state.lru.iter().find(|(_, key)| {
                            matches!(
                                state.entries.get(key),
                                Some(Entry::Ready { frame, .. }) if Arc::strong_count(frame) == 1
                            )
                        }).copied();
                        if let Some((stamp, key)) = victim {
                            state.lru.remove(&(stamp, key));
                            state.entries.remove(&key);
                            self.inner.evictions.fetch_add(1, Ordering::Relaxed);
                            permit = self.inner.slots.clone().try_acquire_owned().ok();
                        }
                    }
                    if let Some(permit) = permit {
                        let (sender, receiver) = watch::channel(None);
                        let token = Arc::new(());
                        state.entries.insert(
                            id,
                            Entry::Loading {
                                receiver: receiver.clone(),
                                token: token.clone(),
                            },
                        );
                        self.inner.misses.fetch_add(1, Ordering::Relaxed);
                        let guard =
                            loading::LoadGuard::new(self.inner.clone(), id, token, sender, permit);
                        runtime.spawn(guard.run());
                        Some(receiver)
                    } else {
                        None
                    }
                }
            };
            if let Some(mut receiver) = pending {
                waited_for_page = true;
                loop {
                    if let Some(result) = receiver.borrow_and_update().as_ref() {
                        match result {
                            Ok(()) => continue 'retry,
                            Err(failure) => {
                                return Err(io::Error::new(failure.kind, failure.message.clone()))
                            }
                        }
                    }
                    receiver.changed().await.map_err(|_| {
                        io::Error::new(io::ErrorKind::Interrupted, "page load was cancelled")
                    })?;
                }
            }
            notified.await;
        }
    }

    /// Validates a logical range and leases a page slice or assembles a cross-page result.
    async fn acquire(&self, request: ReadRequest) -> io::Result<ReadLease<'static>> {
        request.validate(self.inner.bytes)?;
        if request.byte_len() > self.inner.options.max_read_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cache read exceeds operation bound",
            ));
        }
        let storage = RegionOptions {
            accounting: self.inner.options.assembly_accounting.clone(),
            ..Default::default()
        };
        if request.byte_len() == 0 {
            return Ok(ReadLease::from_region(
                MutableRegion::new(0, 64, &storage)?.freeze()?,
            ));
        }
        let page_bytes = self.inner.options.page_bytes;
        let first = request.offset() / page_bytes as u64;
        let within = (request.offset() % page_bytes as u64) as usize;
        if request.byte_len() <= page_bytes - within {
            let frame = self.page(first).await?;
            let lease = frame
                .data
                .as_ref()
                .expect("live frame")
                .section(ReadRequest::new(within as u64, request.byte_len())?)?;
            return Ok(lease.retain(Arc::new(PageUse {
                frame: Some(frame),
                changed: self.inner.changed.clone(),
            })));
        }
        // Assemble crossing records one page at a time. Holding all their page
        // leases at once would deadlock a one-page cache on a two-page record.
        let mut out = MutableRegion::new(request.byte_len(), 64, &storage)?;
        let mut copied = 0;
        while copied < request.byte_len() {
            let offset = request.offset() + copied as u64;
            let frame = self.page(offset / page_bytes as u64).await?;
            let start = (offset % page_bytes as u64) as usize;
            let take = (request.byte_len() - copied).min(frame.as_ref().as_ref().len() - start);
            out.as_bytes_mut()[copied..copied + take]
                .copy_from_slice(&frame.as_ref().as_ref()[start..start + take]);
            copied += take;
            drop(frame);
            self.inner.changed.notify_waiters();
        }
        Ok(ReadLease::from_region(out.freeze()?))
    }
}

impl<R: ObjectReader + 'static> ObjectReader for CachedReader<R> {
    type Read<'a>
        = Pin<Box<dyn Future<Output = io::Result<ReadLease<'a>>> + Send + 'a>>
    where
        Self: 'a;

    /// Returns the backing object length captured by this cache.
    fn byte_len(&self) -> u64 {
        self.inner.bytes
    }

    /// Produces a lease through the bounded cache while preserving the underlying read contract.
    fn read(&self, request: ReadRequest) -> Self::Read<'_> {
        Box::pin(async move { self.acquire(request).await })
    }
}
