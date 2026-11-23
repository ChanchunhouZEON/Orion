/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::state::{Failure, LoadResult};
use super::*;
use tokio::sync::OwnedSemaphorePermit;

/// Detached from any one waiter, but supervised by RAII. Runtime shutdown or a
/// panicking backend removes the in-flight entry and wakes all subscribed readers.
pub(super) struct LoadGuard<R> {
    inner: Arc<Inner<R>>,
    id: u64,
    token: Arc<()>,
    sender: watch::Sender<Option<LoadResult>>,
    permit: Option<OwnedSemaphorePermit>,
    // Drop publishes cancellation only until an explicit result is delivered.
    published: bool,
}

impl<R> LoadGuard<R> {
    /// Transfer the already-reserved slot into a loader before spawning it.
    /// The token ties cleanup/publication to this precise in-flight entry.
    pub fn new(
        inner: Arc<Inner<R>>,
        id: u64,
        token: Arc<()>,
        sender: watch::Sender<Option<LoadResult>>,
        permit: OwnedSemaphorePermit,
    ) -> Self {
        Self {
            inner,
            id,
            token,
            sender,
            permit: Some(permit),
            published: false,
        }
    }

    /// Completes a page-load token and installs the frame only if that token still owns the entry.
    fn publish(&mut self, result: Result<Arc<Frame>, Failure>) {
        if let Ok(mut state) = self.inner.state.lock() {
            if matches!(state.entries.get(&self.id),Some(Entry::Loading { token,.. }) if Arc::ptr_eq(token,&self.token))
            {
                state.entries.remove(&self.id);
                if let Ok(frame) = &result {
                    if let Ok(stamp) = state.tick() {
                        state.lru.insert((stamp, self.id));
                        state.entries.insert(
                            self.id,
                            Entry::Ready {
                                frame: frame.clone(),
                                stamp,
                            },
                        );
                    }
                }
            }
        }
        // Deliver ownership, not just readiness: another page must not evict
        // this result before subscribed readers resume and acquire it.
        self.sender.send_replace(Some(result.map(|frame| {
            Arc::new(PageUse {
                frame: Some(frame),
                changed: self.inner.changed.clone(),
            })
        })));
        self.published = true;
        self.inner.changed.notify_waiters();
    }
}

impl<R: ObjectReader + 'static> LoadGuard<R> {
    /// Load one page, transfer its slot to the frame, then notify subscribers.
    /// The guard remains responsible for wakeup if polling ends before publish.
    pub async fn run(mut self) {
        let offset = self.id * self.inner.options.page_bytes as u64;
        let len = (self.inner.bytes - offset).min(self.inner.options.page_bytes as u64) as usize;
        let result = async {
            let lease = self
                .inner
                .reader
                .read(ReadRequest::new(offset, len)?)
                .await?;
            if lease.as_bytes().len() != len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "backend returned a short page",
                ));
            }
            lease.into_owned()
        }
        .await;
        let result = result
            .map(|data| {
                Arc::new(Frame {
                    data: Some(data),
                    permit: self.permit.take(),
                    changed: self.inner.changed.clone(),
                })
            })
            .map_err(Failure::from);
        self.publish(result);
    }
}

impl<R> Drop for LoadGuard<R> {
    /// Publishes an interrupted outcome for unfinished loads and wakes capacity waiters.
    fn drop(&mut self) {
        if !self.published {
            self.publish(Err(Failure {
                kind: io::ErrorKind::Interrupted,
                message: "page load aborted".into(),
            }));
        }
        drop(self.permit.take());
        self.inner.changed.notify_waiters();
    }
}
