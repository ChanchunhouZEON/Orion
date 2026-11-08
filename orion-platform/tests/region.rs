/* Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License. */
use orion_ann_platform::region::{ArrayView, Placement, RegionOptions};
use std::io;

#[test]
fn sections_keep_mapping_alive_and_validate_typed_alignment() {
    for placement in [Placement::Resident, Placement::Mapped] {
        let dir = tempfile::tempdir().unwrap();
        let options = RegionOptions {
            placement,
            directory: Some(dir.path().into()),
        };
        let values =
            ArrayView::<u32>::build_rows::<io::Error>(16, 8, 64, &options, |start, out| {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = (start * 8 + i) as u32;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(values.as_ptr().addr() % 64, 0);
        let section = values.region().section(32..64).unwrap();
        assert!(section.section(1..4).unwrap().array::<u32>().is_err());
        assert!(section.section(0..33).is_err());
        assert!(section.array::<()>().is_err());
        drop(values);
        let view = section.array::<u32>().unwrap();
        drop(section);
        assert_eq!(view.as_slice(), &(8..16).collect::<Vec<u32>>());
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let view = &view;
                scope.spawn(move || assert_eq!(view.as_slice()[7], 15));
            }
        });
        drop(view);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
#[test]
fn mapped_builder_is_bounded_and_errors_remove_partial_files() {
    let dir = tempfile::tempdir().unwrap();
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
    };
    let mut calls = 0;
    let result = ArrayView::<u8>::build_rows::<io::Error>(20_000_000, 1, 64, &options, |_, out| {
        assert!(out.len() <= 8 * 1024 * 1024);
        calls += 1;
        if calls == 2 {
            return Err(io::Error::other("injected failure"));
        }
        out.fill(7);
        Ok(())
    });
    assert!(result.is_err());
    assert_eq!(calls, 2);
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    assert!(
        ArrayView::<u64>::build_rows::<io::Error>(usize::MAX, 2, 64, &options, |_, _| Ok(()))
            .is_err()
    );
    assert!(ArrayView::<u8>::build_rows::<io::Error>(1, 1, 3, &options, |_, _| Ok(())).is_err());
}

#[test]
fn exclusive_mapped_construction_freezes_without_copy_and_views_own_lifetime() {
    use orion_ann_platform::region::MutableRegion;
    let dir = tempfile::tempdir().unwrap();
    let options = RegionOptions {
        placement: Placement::Mapped,
        directory: Some(dir.path().into()),
    };
    let mut region = MutableRegion::new(8192, 64, &options).unwrap();
    assert_eq!(region.placement(), Placement::Mapped);
    assert!(region.as_bytes().iter().all(|&v| v == 0));
    let ptr = region.as_bytes().as_ptr();
    region.as_bytes_mut()[4096..].fill(91);
    let published = region.freeze().unwrap();
    assert_eq!(ptr, published.array::<u8>().unwrap().as_ptr());
    let view = published
        .section(4096..8192)
        .unwrap()
        .array::<u8>()
        .unwrap();
    drop(published);
    std::thread::spawn(move || assert!(view.iter().all(|&v| v == 91)))
        .join()
        .unwrap();
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    assert!(MutableRegion::new(16, 3, &options).is_err());
    assert!(MutableRegion::new(usize::MAX, 64, &options).is_err());
}
