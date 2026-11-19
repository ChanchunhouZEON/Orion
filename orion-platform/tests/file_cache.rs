/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(any(target_os = "linux", target_os = "macos"))]

use orion_ann_platform::{
    allocation_budget::AllocationBudget,
    file_cache::{sync_and_reclaim_file, RECLAIM_WINDOW_BYTES},
    region::{ArrayView, MutableRegion, Placement, RegionOptions},
};
use std::io::{self, Read, Seek, SeekFrom, Write};

#[test]
fn reclaimed_mapping_preserves_bytes_views_and_budget_until_last_owner() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = RECLAIM_WINDOW_BYTES + 137;
    let budget = AllocationBudget::new(Some(0));
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
        accounting: Some(budget.account("mapped").unwrap()),
        reclaim_after_write: true,
    };
    let mut region = MutableRegion::new(bytes, 64, &options).unwrap();
    for (i, v) in region.as_bytes_mut().iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let charged = budget.usage();
    // Exercise a non-page-aligned start, a short tail and an empty checkpoint.
    region.flush_and_reclaim(3..bytes).unwrap();
    region.flush_and_reclaim(bytes..bytes).unwrap();
    assert_eq!(
        region.flush_and_reclaim(0..bytes + 1).unwrap_err().kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(budget.usage(), charged);
    let published = region.freeze().unwrap();
    let view = published.section(7..bytes).unwrap();
    drop(published);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let view = &view;
            scope.spawn(move || {
                view.reclaim_file_pages(1..view.len_bytes()).unwrap();
                assert!(view
                    .as_bytes()
                    .iter()
                    .enumerate()
                    .all(|(i, &v)| v == ((i + 7) % 251) as u8));
            });
        }
    });
    assert!(view.reclaim_file_pages(0..bytes).is_err());
    assert_eq!(budget.usage(), charged);
    drop(view);
    assert_eq!(budget.usage().mapped_bytes, 0);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn file_checkpoints_preserve_completed_prefix_and_final_tail() {
    let mut file = tempfile::tempfile().unwrap();
    let chunk = vec![29u8; 1024 * 1024];
    for _ in 0..33 {
        file.write_all(&chunk).unwrap();
    }
    sync_and_reclaim_file(&file, 0..RECLAIM_WINDOW_BYTES as u64).unwrap();
    file.write_all(&[7; 137]).unwrap();
    let len = file.metadata().unwrap().len();
    sync_and_reclaim_file(&file, RECLAIM_WINDOW_BYTES as u64..len).unwrap();
    sync_and_reclaim_file(&file, len..len).unwrap();
    assert!(sync_and_reclaim_file(&file, 0..len + 1).is_err());
    assert!(sync_and_reclaim_file(&file, std::ops::Range { start: 2, end: 1 }).is_err());
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut block = vec![0; chunk.len()];
    for _ in 0..33 {
        file.read_exact(&mut block).unwrap();
        assert_eq!(block, chunk);
    }
    let mut tail = [0; 137];
    file.read_exact(&mut tail).unwrap();
    assert_eq!(tail, [7; 137]);
}

#[test]
fn checkpointed_rows_match_resident_and_failure_returns_the_lease() {
    let dir = tempfile::tempdir().unwrap();
    let budget = AllocationBudget::new(Some(0));
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
        accounting: Some(budget.account("codes").unwrap()),
        reclaim_after_write: true,
    };
    // Five blocks cross a checkpoint before the injected producer error.
    let bytes = RECLAIM_WINDOW_BYTES + 2 * 8 * 1024 * 1024;
    let result = ArrayView::<u8>::build_rows::<io::Error>(bytes, 1, 64, &options, |start, out| {
        if start > RECLAIM_WINDOW_BYTES {
            return Err(io::Error::other("injected producer failure"));
        }
        out.fill(11);
        Ok(())
    });
    assert!(result.is_err());
    assert_eq!(budget.usage().mapped_bytes, 0);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    let values = ArrayView::<u32>::build_rows::<io::Error>(4097, 3, 64, &options, |start, out| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (start * 3 + i) as u32;
        }
        Ok(())
    })
    .unwrap();
    values
        .region()
        .reclaim_file_pages(0..values.region().len_bytes())
        .unwrap();
    assert!(values.iter().enumerate().all(|(i, &v)| v == i as u32));
    drop(values);
    assert_eq!(budget.usage().mapped_bytes, 0);
}

#[test]
fn resident_reclamation_is_a_noop_and_reversed_ranges_are_rejected() {
    let mut owned = MutableRegion::new(64, 64, &RegionOptions::default()).unwrap();
    owned.as_bytes_mut().fill(73);
    owned.flush_and_reclaim(1..63).unwrap();
    assert!(owned
        .flush_and_reclaim(std::ops::Range { start: 5, end: 3 })
        .is_err());
    let published = owned.freeze().unwrap();
    published.reclaim_file_pages(0..64).unwrap();
    assert_eq!(published.as_bytes(), &[73; 64]);
}
