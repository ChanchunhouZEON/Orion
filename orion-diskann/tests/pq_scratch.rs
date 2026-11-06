/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_diskann::model::scratch::pq_scratch::PQScratch;
#[test]
fn pq_scratch_aligns_buffers_and_resets_normalized_queries() {
    for (degree, dimension) in [(1, 32), (64, 128), (96, 960)] {
        let mut scratch = PQScratch::new(degree, dimension).unwrap();
        assert_eq!(scratch.aligned_pq_coord_scratch.len(), degree * 512);
        assert_eq!(scratch.aligned_dist_scratch.len(), degree);
        assert_eq!(scratch.aligned_pqtable_dist_scratch.len(), 256 * 512);
        for ptr in [
            scratch.aligned_pq_coord_scratch.as_ptr() as usize,
            scratch.aligned_dist_scratch.as_ptr() as usize,
            scratch.aligned_pqtable_dist_scratch.as_ptr() as usize,
        ] {
            assert_eq!(ptr % 256, 0);
        }
        let query: Vec<u8> = (0..dimension).map(|i| (i % 251) as u8).collect();
        for norm in [1., 2., 1.] {
            scratch.set(dimension, &query, norm);
            let expected: Vec<_> = query.iter().map(|v| f32::from(*v) / norm).collect();
            assert_eq!(&scratch.rotated_query[..], expected);
            assert_eq!(&scratch.aligned_query_float[..], expected);
        }
    }
}
