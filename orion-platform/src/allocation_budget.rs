/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Allocation-scoped payload accounting. Budgets never own buffers; each buffer
//! owns one lease. Shared views retain that lease through their shared backing.
//! Counters describe managed payload bytes, not RSS or allocator overhead.
use crate::region::Placement;
use std::{
    collections::BTreeMap,
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex,
    },
};

#[derive(Debug)]
struct State {
    limit: Option<u64>,
    resident: AtomicU64,
    mapped: AtomicU64,
    objects: Mutex<BTreeMap<String, Arc<Counters>>>,
}
#[derive(Debug, Default)]
struct Counters {
    resident: AtomicU64,
    mapped: AtomicU64,
}

/// Shared accounting domain. Cloning it does not reserve or release bytes.
#[derive(Debug, Clone)]
pub struct AllocationBudget(Arc<State>);
/// A named object may own several independently allocated regions.
#[derive(Debug, Clone)]
pub struct AllocationAccount {
    budget: AllocationBudget,
    counters: Arc<Counters>,
}
impl PartialEq for AllocationAccount {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.counters, &other.counters)
    }
}
impl Eq for AllocationAccount {}

/// One owned charge; drop only after the backing has been released.
#[derive(Debug)]
pub(crate) struct AllocationLease {
    account: AllocationAccount,
    resident: u64,
    mapped: u64,
}

/// Observational totals. Concurrent operations can change counts between reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocationUsage {
    pub resident_bytes: u64,
    pub mapped_bytes: u64,
    pub limit_bytes: Option<u64>,
}
impl AllocationBudget {
    pub fn new(limit_bytes: Option<u64>) -> Self {
        Self(Arc::new(State {
            limit: limit_bytes,
            resident: AtomicU64::new(0),
            mapped: AtomicU64::new(0),
            objects: Mutex::new(BTreeMap::new()),
        }))
    }
    /// Obtain an allocation account; registering an object allocates no payload.
    pub fn account(&self, name: &str) -> io::Result<AllocationAccount> {
        let mut objects = self
            .0
            .objects
            .lock()
            .map_err(|_| io::Error::other("allocation registry poisoned"))?;
        let counters = objects.entry(name.into()).or_default().clone();
        Ok(AllocationAccount {
            budget: self.clone(),
            counters,
        })
    }
    pub fn usage(&self) -> AllocationUsage {
        AllocationUsage {
            resident_bytes: self.0.resident.load(Ordering::Acquire),
            mapped_bytes: self.0.mapped.load(Ordering::Acquire),
            limit_bytes: self.0.limit,
        }
    }
    /// Include only accounts with live allocations. This never inspects buffers.
    pub fn objects(&self) -> io::Result<Vec<(String, AllocationUsage)>> {
        let objects = self
            .0
            .objects
            .lock()
            .map_err(|_| io::Error::other("allocation registry poisoned"))?;
        Ok(objects
            .iter()
            .filter_map(|(name, counters)| {
                let resident_bytes = counters.resident.load(Ordering::Acquire);
                let mapped_bytes = counters.mapped.load(Ordering::Acquire);
                (resident_bytes != 0 || mapped_bytes != 0).then(|| {
                    (
                        name.clone(),
                        AllocationUsage {
                            resident_bytes,
                            mapped_bytes,
                            limit_bytes: self.0.limit,
                        },
                    )
                })
            })
            .collect())
    }
}
impl AllocationAccount {
    pub(crate) fn reserve(
        &self,
        bytes: usize,
        placement: Placement,
    ) -> io::Result<AllocationLease> {
        let (resident, mapped) = match placement {
            Placement::Resident => (bytes as u64, 0),
            Placement::Mapped => (0, bytes as u64),
        };
        let state = &self.budget.0;
        // Reserve atomically before touching the allocator. Other constructors
        // cannot spend this capacity until the lease or failed attempt releases it.
        let reservation =
            state
                .resident
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                    used.checked_add(resident)
                        .filter(|&total| state.limit.is_none_or(|limit| total <= limit))
                });
        reservation.map_err(|used| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                format!(
                    "allocation budget exhausted: resident={used}, requested={resident}, limit={:?}",
                    state.limit,
                ),
            )
        })?;
        if state
            .mapped
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(mapped)
            })
            .is_err()
        {
            state.resident.fetch_sub(resident, Ordering::AcqRel);
            return Err(io::Error::new(
                io::ErrorKind::OutOfMemory,
                "mapped accounting overflow",
            ));
        }
        self.counters.resident.fetch_add(resident, Ordering::AcqRel);
        self.counters.mapped.fetch_add(mapped, Ordering::AcqRel);
        Ok(AllocationLease {
            account: self.clone(),
            resident,
            mapped,
        })
    }
}
impl AllocationLease {
    pub(crate) fn account(&self) -> &AllocationAccount {
        &self.account
    }
}
impl Drop for AllocationLease {
    fn drop(&mut self) {
        self.account
            .counters
            .resident
            .fetch_sub(self.resident, Ordering::AcqRel);
        self.account
            .counters
            .mapped
            .fetch_sub(self.mapped, Ordering::AcqRel);
        self.account
            .budget
            .0
            .resident
            .fetch_sub(self.resident, Ordering::AcqRel);
        self.account
            .budget
            .0
            .mapped
            .fetch_sub(self.mapped, Ordering::AcqRel);
    }
}
