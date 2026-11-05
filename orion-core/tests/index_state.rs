/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Ownership and persistence contracts at the public index boundary.
use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
fn index() -> Orion {
    let data = PaddedDataset::from_packed(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap();
    Orion::from_graph_partitions(
        data,
        &[
            (vec![1, 2], vec![], vec![]),
            (vec![0, 2], vec![], vec![]),
            (vec![0, 1], vec![], vec![]),
        ],
        0,
        2,
        0,
    )
    .unwrap()
}

#[test]
fn metadata_and_base_borrows_do_not_copy_or_expose_mutation() {
    let index = index();
    let ptr = index.dataset().as_slice().as_ptr();
    assert_eq!(index.len(), index.graph().num_nodes());
    assert_eq!(index.len(), index.dataset().len());
    assert!(!index.is_empty());
    assert_eq!(index.entry_point(), 0);
    assert!(index.pq_table().is_none());
    assert!(index.pq_codes().is_none());
    assert!(index.num_pq_chunks().is_none());
    // Fitting a sidecar does not change ownership of the base allocation.
    index.ensure_quantized_dataset();
    let mut base = index.into_dataset();
    assert_eq!(base.as_slice().as_ptr(), ptr);
    base.set_row(0, &[42., 43.]).unwrap();
    assert_eq!(base.row(0).unwrap(), &[42., 43.]);
}
