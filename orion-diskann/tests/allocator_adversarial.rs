/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Invalid input contracts, independently runnable under Miri.
use orion_ann_diskann::common::AlignedBoxWithSlice;

#[test]
fn invalid_layouts_are_errors() {
    for alignment in [0, 3, 6] {
        assert!(AlignedBoxWithSlice::<u64>::new(1, alignment).is_err());
        assert!(AlignedBoxWithSlice::<u64>::new(0, alignment).is_err());
    }
    assert!(AlignedBoxWithSlice::<u64>::new(usize::MAX, 8).is_err());
    assert!(AlignedBoxWithSlice::<u8>::new(isize::MAX as usize, 64).is_err());
}

#[test]
fn invalid_ranges_and_oversized_copy_leave_data_unchanged() {
    let mut values = AlignedBoxWithSlice::<u32>::new(4, 64).unwrap();
    values.memcpy(&[1, 2, 3, 4]).unwrap();
    for (range, chunk) in [(0..4, 0), (3..2, 1), (0..5, 1), (0..3, 2)] {
        assert!(
            values
                .split_into_nonoverlapping_mut_slices(range, chunk)
                .is_err()
        );
    }
    assert!(values.memcpy(&[9; 5]).is_err());
    assert_eq!(values.as_slice(), &[1, 2, 3, 4]);
}
