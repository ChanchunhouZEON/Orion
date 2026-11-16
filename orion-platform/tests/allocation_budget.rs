/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Exercise real backing allocations, not synthetic reservation reconciliation.
use orion_ann_platform::{
    allocation_budget::AllocationBudget,
    region::{ArrayView, MutableRegion, Placement, RegionOptions},
};
use std::{
    io,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Barrier,
    },
};

fn options(budget: &AllocationBudget, name: &str, placement: Placement) -> RegionOptions {
    RegionOptions {
        placement,
        accounting: Some(budget.account(name).unwrap()),
        ..Default::default()
    }
}

#[test]
fn rejection_precedes_allocation_and_fill() {
    let budget = AllocationBudget::new(Some(15));
    let options = options(&budget, "codes", Placement::Resident);
    let mut called = false;
    let result = ArrayView::<u32>::build_rows::<io::Error>(4, 1, 64, &options, |_, _| {
        called = true;
        Ok(())
    });
    assert_eq!(result.unwrap_err().kind(), io::ErrorKind::OutOfMemory);
    assert!(!called);
    assert_eq!(budget.usage().resident_bytes, 0);
    assert!(budget.objects().unwrap().is_empty());
}

#[test]
fn shared_sections_and_freeze_keep_one_full_backing_charge() {
    for placement in [Placement::Resident, Placement::Mapped] {
        let budget = AllocationBudget::new(Some(64));
        let options = options(&budget, "graph", placement);
        let mut buffer = MutableRegion::new(64, 64, &options).unwrap();
        buffer.as_bytes_mut().fill(7);
        let charged = budget.usage();
        assert_eq!(charged.resident_bytes + charged.mapped_bytes, 64);
        let region = buffer.freeze().unwrap();
        assert_eq!(budget.usage(), charged);
        let slice = region.section(8..12).unwrap();
        let copy = slice.clone();
        drop(region);
        drop(slice);
        assert_eq!(budget.usage(), charged);
        assert_eq!(copy.as_bytes(), &[7; 4]);
        let account = options.accounting.unwrap();
        let mut same = copy.clone();
        same.account_storage(&account).unwrap();
        assert_eq!(budget.usage(), charged);
        drop(copy);
        std::thread::spawn(move || drop(same)).join().unwrap();
        assert_eq!(
            budget.usage().resident_bytes + budget.usage().mapped_bytes,
            0
        );
    }
}

#[test]
fn adoption_accounts_backing_not_section_and_rejects_rebinding() {
    let budget = AllocationBudget::new(Some(64));
    let account = budget.account("base").unwrap();
    let region = MutableRegion::new(64, 64, &Default::default())
        .unwrap()
        .freeze()
        .unwrap();
    let mut section = region.section(8..12).unwrap();
    assert!(section.account_storage(&account).is_err());
    drop(region);
    section.account_storage(&account).unwrap();
    assert_eq!(budget.usage().resident_bytes, 64);
    assert!(section
        .account_storage(&budget.account("other").unwrap())
        .is_err());
    assert_eq!(budget.usage().resident_bytes, 64);
    drop(section);
    assert_eq!(budget.usage().resident_bytes, 0);
}

#[test]
fn fill_errors_and_unwind_release_only_discarded_buffers() {
    for placement in [Placement::Resident, Placement::Mapped] {
        let root = tempfile::tempdir().unwrap();
        let budget = AllocationBudget::new(Some(128));
        let mut options = options(&budget, "sidecar", placement);
        options.directory = Some(root.path().into());
        let published = ArrayView::copy_from(&[1u8; 32], &options).unwrap();
        let before = budget.usage();
        let fail = ArrayView::<u8>::build_rows::<io::Error>(64, 1, 64, &options, |_, _| {
            Err(io::Error::other("injected decode failure"))
        });
        assert!(fail.is_err());
        assert_eq!(budget.usage(), before);
        let panic = std::panic::catch_unwind(|| {
            let _ = ArrayView::<u8>::build_rows::<io::Error>(64, 1, 64, &options, |_, _| {
                panic!("injected constructor panic")
            });
        });
        assert!(panic.is_err());
        assert_eq!(budget.usage(), before);
        drop(published);
        assert_eq!(
            budget.usage().resident_bytes + budget.usage().mapped_bytes,
            0
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0);
    }
}

#[test]
fn mapped_constructor_failure_releases_charge_without_resident_budget() {
    let root = tempfile::tempdir().unwrap();
    let budget = AllocationBudget::new(Some(0));
    let mut options = options(&budget, "base", Placement::Mapped);
    options.directory = Some(root.path().join("missing"));
    assert!(MutableRegion::new(64, 64, &options).is_err());
    assert_eq!(budget.usage().mapped_bytes, 0);
    options.directory = Some(root.path().into());
    let mapping = MutableRegion::new(64, 64, &options).unwrap();
    assert_eq!(budget.usage().mapped_bytes, 64);
    assert_eq!(budget.usage().resident_bytes, 0);
    drop(mapping);
    assert_eq!(budget.usage().mapped_bytes, 0);
}

#[test]
fn concurrent_allocators_cannot_overspend_and_last_owner_releases() {
    let budget = AllocationBudget::new(Some(256));
    let barrier = Arc::new(Barrier::new(8));
    let success = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let barrier = barrier.clone();
            let budget = &budget;
            let success = &success;
            scope.spawn(move || {
                barrier.wait();
                let buffer =
                    MutableRegion::new(64, 64, &options(budget, "shared", Placement::Resident));
                if buffer.is_ok() {
                    success.fetch_add(1, Ordering::Relaxed);
                }
                barrier.wait();
                assert_eq!(budget.usage().resident_bytes, 256);
                barrier.wait();
                drop(buffer);
            });
        }
    });
    assert_eq!(success.load(Ordering::Relaxed), 4);
    assert_eq!(budget.usage().resident_bytes, 0);
    assert!(budget.objects().unwrap().is_empty());
}

#[test]
fn replacement_charges_overlap_and_failure_keeps_old_buffer() {
    let budget = AllocationBudget::new(Some(64));
    let options = options(&budget, "pq", Placement::Resident);
    let mut old = ArrayView::copy_from(&[3u8; 40], &options).unwrap();
    assert!(ArrayView::copy_from(&[4u8; 40], &options).is_err());
    assert_eq!(old.as_slice(), &[3; 40]);
    assert_eq!(budget.usage().resident_bytes, 40);
    let replacement = ArrayView::copy_from(&[5u8; 24], &options).unwrap();
    assert_eq!(budget.usage().resident_bytes, 64);
    old = replacement;
    assert_eq!(budget.usage().resident_bytes, 24);
    assert_eq!(old.as_slice(), &[5; 24]);
    drop(old);
    assert_eq!(budget.usage().resident_bytes, 0);
}
