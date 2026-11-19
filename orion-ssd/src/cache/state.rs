/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
use std::collections::{BTreeSet, HashMap};
use tokio::sync::OwnedSemaphorePermit;

/// Shared ownership boundary for reader clones and detached page-load tasks.
/// Payload capacity and allocation accounting have separate lifetimes.
pub(super) struct Inner<R> {
    /// Underlying asynchronous reader shared by cache callers and load tasks.
    pub reader: R,

    /// Logical size of the file addressed by the page cache.
    pub bytes: u64,

    /// Validated page size and cache capacity policy.
    pub options: PageCacheOptions,
    // One permit per frame, retained until both cache and callers release it.
    /// One capacity permit per frame, retained until cache and reader owners release it.
    pub slots: Arc<Semaphore>,

    /// Mutex-protected entry map and eviction-order state.
    pub state: Mutex<State>,
    // Wake capacity waiters on both final release and becoming evictable.
    /// Notification for capacity waiters when frames are released or become evictable.
    pub changed: Arc<Notify>,

    /// Reads satisfied by a ready cached frame.
    pub hits: AtomicU64,

    /// Reads that initiated a new page load.
    pub misses: AtomicU64,

    /// Reads joined to an already pending load of the same page.
    pub coalesced: AtomicU64,

    /// Ready frames removed from cache ownership to admit other pages.
    pub evictions: AtomicU64,
}
/// Index and eviction order changed together under Inner::state's mutex.
#[derive(Default)]
pub(super) struct State {
    /// Page-ID lookup containing both loading and ready states.
    pub entries: HashMap<u64, Entry>,
    // Ordered by (access stamp, page ID); loading pages have no LRU entry.
    /// Ready-frame eviction order keyed by access stamp and page ID.
    pub lru: BTreeSet<(u64, u64)>,
    clock: u64,
}

impl State {
    /// Issue a unique access stamp; wrapping would invalidate LRU ordering.
    pub fn tick(&mut self) -> io::Result<u64> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or_else(|| io::Error::other("page-cache LRU clock exhausted"))?;
        Ok(self.clock)
    }
}

/// In-flight page load or ready cached frame with its eviction-order stamp.
pub(super) enum Entry {
    Loading {
        receiver: watch::Receiver<Option<LoadResult>>,
        // Identity distinguishes this load from a later retry at the same ID.
        token: Arc<()>,
    },
    Ready {
        frame: Arc<Frame>,
        stamp: u64,
    },
}

pub(super) type LoadResult = Result<(), Failure>;

/// Cloneable I/O outcome delivered to every waiter on a shared page load.
#[derive(Debug, Clone)]
pub(super) struct Failure {
    /// Portable I/O error category shared with every waiting caller.
    pub kind: io::ErrorKind,

    /// Owned diagnostic text for the failed shared page load.
    pub message: String,
}

impl From<io::Error> for Failure {
    /// Copies an I/O failure into a clonable outcome for all waiters on a shared page load.
    fn from(error: io::Error) -> Self {
        Self {
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}
/// Page payload plus its reservation. Options allow explicit drop order:
/// bytes first, then capacity, then notification of sleeping readers.
#[derive(Debug)]
pub(super) struct Frame {
    /// Initialized read lease released before the frame capacity permit.
    pub data: Option<ReadLease<'static>>,

    /// Capacity reservation released after the associated frame bytes.
    pub permit: Option<OwnedSemaphorePermit>,

    /// Notification for capacity waiters when frames are released or become evictable.
    pub changed: Arc<Notify>,
}

impl AsRef<[u8]> for Frame {
    /// Borrows a live frame's immutable bytes without releasing its capacity permit.
    fn as_ref(&self) -> &[u8] {
        self.data.as_ref().expect("live cache frame").as_bytes()
    }
}

impl Drop for Frame {
    /// Frees page storage before returning capacity and waking blocked readers.
    fn drop(&mut self) {
        // Release bytes before capacity, and capacity before waking waiters.
        drop(self.data.take());
        drop(self.permit.take());
        self.changed.notify_waiters();
    }
}

/// Notify when a caller releases a page even if the cache still owns the frame.
/// Frame::drop alone is insufficient: becoming evictable must wake capacity waiters.
#[derive(Debug)]
pub(super) struct PageUse {
    /// Shared frame kept alive while a caller holds this page view.
    pub frame: Option<Arc<Frame>>,

    /// Notification for capacity waiters when frames are released or become evictable.
    pub changed: Arc<Notify>,
}

impl Drop for PageUse {
    /// Releases one active page reference and wakes readers that may now evict it.
    fn drop(&mut self) {
        drop(self.frame.take());
        self.changed.notify_waiters();
    }
}
