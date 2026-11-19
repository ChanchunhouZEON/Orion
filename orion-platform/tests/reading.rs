/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_platform::{
    allocation_budget::AllocationBudget,
    reading::{ObjectReader, ReadLease, ReadRequest},
    region::{ArrayView, Placement, RegionOptions},
};
use std::{
    future::Future,
    io,
    pin::pin,
    task::{Context, Poll, Waker},
};

fn immediately<F: Future>(future: F) -> F::Output {
    match pin!(future).poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("region read unexpectedly suspended"),
    }
}

#[test]
fn borrowed_reads_share_bytes_and_owned_leases_retain_backing_and_budget() {
    for placement in [Placement::Resident, Placement::Mapped] {
        let budget = AllocationBudget::new(Some(1024));
        let options = RegionOptions {
            placement,
            accounting: Some(budget.account("rows").unwrap()),
            ..Default::default()
        };
        let data = ArrayView::copy_from(&[1u32, 7, 11, 19], &options).unwrap();
        let usage = budget.usage();
        let lease = immediately(data.region().read(ReadRequest::new(4, 8).unwrap())).unwrap();
        assert_eq!(lease.as_slice::<u32>().unwrap(), &[7, 11]);
        assert_eq!(
            lease.as_bytes().as_ptr(),
            data.region().as_bytes()[4..].as_ptr()
        );
        assert_eq!(budget.usage(), usage);
        let owned = lease.into_owned().unwrap();
        assert_eq!(budget.usage(), usage);
        drop(data);
        std::thread::spawn(move || {
            assert_eq!(owned.as_slice::<u32>().unwrap(), &[7, 11]);
        })
        .join()
        .unwrap();
        assert_eq!(budget.usage().resident_bytes, 0);
        assert_eq!(budget.usage().mapped_bytes, 0);
    }
}

#[test]
fn ranges_empty_reads_alignment_and_overflow_are_checked() {
    let data = ArrayView::copy_from(&[1u64, 2], &Default::default()).unwrap();
    assert_eq!(data.region().byte_len(), 16);
    for (offset, bytes) in [(17, 0), (15, 2), (u64::MAX, 0)] {
        let error =
            immediately(data.region().read(ReadRequest::new(offset, bytes).unwrap())).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    }
    for offset in [0, 8, 16] {
        assert!(
            immediately(data.region().read(ReadRequest::new(offset, 0).unwrap()))
                .unwrap()
                .as_bytes()
                .is_empty()
        );
    }
    assert!(ReadRequest::new(u64::MAX, 1).is_err());
    assert!(ReadRequest::new(0, usize::MAX).is_err());
    let unaligned = immediately(data.region().read(ReadRequest::new(1, 8).unwrap())).unwrap();
    assert!(unaligned.as_slice::<u64>().is_err());
    let partial = immediately(data.region().read(ReadRequest::new(0, 7).unwrap())).unwrap();
    assert!(partial.as_slice::<u64>().is_err());
    assert!(partial.as_slice::<()>().is_err());
    let all = ReadLease::from_region(data.region().clone());
    assert_eq!(
        all.into_owned().unwrap().as_slice::<u64>().unwrap(),
        &[1, 2]
    );
}

#[test]
fn nested_region_reads_use_section_relative_offsets() {
    let values =
        ArrayView::copy_from(&(0u32..64).collect::<Vec<_>>(), &Default::default()).unwrap();
    let section = values.region().section(32..128).unwrap();
    let lease = immediately(section.read(ReadRequest::new(4, 12).unwrap())).unwrap();
    assert_eq!(lease.as_slice::<u32>().unwrap(), &[9, 10, 11]);
    assert_eq!(section.byte_len(), 96);
}

#[test]
fn pending_region_future_can_be_dropped_without_changing_accounting() {
    let budget = AllocationBudget::new(Some(8));
    let options = RegionOptions {
        accounting: Some(budget.account("data").unwrap()),
        ..Default::default()
    };
    let data = ArrayView::copy_from(&[0u64], &options).unwrap();
    drop(data.region().read(ReadRequest::new(0, 8).unwrap()));
    assert_eq!(budget.usage().resident_bytes, 8);
    drop(data);
    assert_eq!(budget.usage().resident_bytes, 0);
}
