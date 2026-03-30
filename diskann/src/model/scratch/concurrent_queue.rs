/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::common::{ANNError, ANNResult};

#[derive(Debug)]
pub struct ConcurrentQueue<T> {
    q: Mutex<VecDeque<T>>,
    c: Mutex<bool>,
    push_cv: Condvar,
}

impl Default for ConcurrentQueue<usize> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> ConcurrentQueue<T> {
    pub fn new() -> Self {
        Self {
            q: Mutex::new(VecDeque::new()),
            c: Mutex::new(false),
            push_cv: Condvar::new(),
        }
    }

    pub fn reserve(&self, size: usize) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        guard.reserve(size);
        Ok(())
    }

    pub fn size(&self) -> ANNResult<usize> {
        let guard = lock(&self.q)?;
        Ok(guard.len())
    }

    pub fn is_empty(&self) -> ANNResult<bool> {
        Ok(self.size()? == 0)
    }

    pub fn push(&self, new_val: T) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        self.push_internal(&mut guard, new_val);
        self.push_cv.notify_all();
        Ok(())
    }

    fn push_internal(&self, guard: &mut MutexGuard<VecDeque<T>>, new_val: T) {
        guard.push_back(new_val);
    }

    pub fn insert<I>(&self, iter: I) -> ANNResult<()>
    where
        I: IntoIterator<Item = T>,
    {
        let mut guard = lock(&self.q)?;
        for item in iter {
            self.push_internal(&mut guard, item);
        }
        self.push_cv.notify_all();
        Ok(())
    }

    pub fn pop(&self) -> ANNResult<Option<T>> {
        let mut guard = lock(&self.q)?;
        Ok(guard.pop_front())
    }

    pub fn empty_queue(&self) -> ANNResult<()> {
        let mut guard = lock(&self.q)?;
        while !guard.is_empty() {
            let _ = guard.pop_front();
        }
        Ok(())
    }

    pub fn wait_for_push_notify(&self, wait_time: Duration) -> ANNResult<()> {
        let guard_lock = lock(&self.c)?;
        let _ = self
            .push_cv
            .wait_timeout(guard_lock, wait_time)
            .map_err(|err| {
                ANNError::log_lock_poison_error(format!(
                    "ConcurrentQueue Lock is poisoned, err={}",
                    err
                ))
            })?;
        Ok(())
    }
}

fn lock<T>(mutex: &Mutex<T>) -> ANNResult<MutexGuard<T>> {
    let guard = mutex.lock().map_err(|err| {
        ANNError::log_lock_poison_error(format!("ConcurrentQueue lock is poisoned, err={}", err))
    })?;
    Ok(guard)
}

#[derive(Debug)]
pub struct ArcConcurrentBoxedQueue<T> {
    internal_queue: Arc<ConcurrentQueue<Box<T>>>,
}

impl<T> ArcConcurrentBoxedQueue<T> {
    pub fn new() -> Self {
        Self {
            internal_queue: Arc::new(ConcurrentQueue::new()),
        }
    }
}

impl<T> Default for ArcConcurrentBoxedQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for ArcConcurrentBoxedQueue<T> {
    fn clone(&self) -> Self {
        Self {
            internal_queue: Arc::clone(&self.internal_queue),
        }
    }
}

impl<T> Deref for ArcConcurrentBoxedQueue<T> {
    type Target = ConcurrentQueue<Box<T>>;

    fn deref(&self) -> &Self::Target {
        &self.internal_queue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let q = ConcurrentQueue::new();
        q.push(1).unwrap();
        q.push(2).unwrap();
        assert_eq!(q.size().unwrap(), 2);

        let v = q.pop().unwrap();
        assert_eq!(v, Some(1));
        assert_eq!(q.size().unwrap(), 1);
    }

    #[test]
    fn test_pop_empty() {
        let q = ConcurrentQueue::<i32>::new();
        let v = q.pop().unwrap();
        assert_eq!(v, None);
    }

    #[test]
    fn test_empty_queue() {
        let q = ConcurrentQueue::new();
        q.push(1).unwrap();
        q.push(2).unwrap();
        q.empty_queue().unwrap();
        assert!(q.is_empty().unwrap());
    }

    #[test]
    fn test_insert_batch() {
        let q = ConcurrentQueue::new();
        q.insert(vec![1, 2, 3]).unwrap();
        assert_eq!(q.size().unwrap(), 3);
        assert_eq!(q.pop().unwrap(), Some(1));
        assert_eq!(q.pop().unwrap(), Some(2));
        assert_eq!(q.pop().unwrap(), Some(3));
    }

    #[test]
    fn test_arc_concurrent_boxed_queue() {
        let q = ArcConcurrentBoxedQueue::<i32>::new();
        q.push(Box::new(42)).unwrap();
        let val = q.pop().unwrap().unwrap();
        assert_eq!(*val, 42);
    }
}
