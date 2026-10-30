//! Numerical parity for the kernels used by native SIFT admission.
//! Run on the target architecture to exercise its stream unrolling.

use vector::{
    distance_l2_vector_u8_batch4, DistanceStream, FullPrecisionDistance, L2U8Distance, Metric,
};

fn scalar_l2(a: &[u8], b: &[u8]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let difference = i32::from(x) - i32::from(y);
            (difference * difference) as u32
        })
        .sum::<u32>() as f32
}

fn check_stream<const N: usize>() {
    let mut state = 0x91e1_0da5_c79e_7b1d_u64;
    let mut next_byte = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state as u8
    };

    for case in 0..24 {
        // The first two cases exercise both signs of the largest u8
        // difference. The last case checks exact zero distances.
        let query: [u8; N] = std::array::from_fn(|_| match case {
            0 => 255,
            1 => 0,
            _ => next_byte(),
        });

        // Offset the first row to exercise unaligned loads. Keep a
        // mapped tail for the stream's speculative prefetch addresses.
        let mut storage = vec![0_u8; 1 + 33 * N + 128];
        for row in storage[1..1 + 33 * N].chunks_exact_mut(N) {
            for (coordinate, value) in row.iter_mut().enumerate() {
                *value = match case {
                    0 => 0,
                    1 => 255,
                    23 => query[coordinate],
                    _ => next_byte(),
                };
            }
        }
        let base = &storage[1..];

        for count in [0, 1, 2, 3, 7, 16, 33, 40] {
            // Non-sequential IDs include repeats for count > 33;
            // the distance stream must preserve their order.
            let ids: Vec<u32> = (0..count).map(|i| (i * 17 % 33) as u32).collect();
            for lookahead in [1, 10, 64] {
                let mut actual = Vec::new();
                unsafe {
                    DistanceStream::<L2U8Distance>::new(
                        base.as_ptr(),
                        N,
                        N,
                        query.as_ptr(),
                        &ids,
                        lookahead,
                    )
                    .run(|id, distance| actual.push((id, distance)));
                }

                assert_eq!(actual.len(), ids.len());
                for ((id, streamed), expected_id) in actual.into_iter().zip(&ids) {
                    assert_eq!(id, *expected_id);
                    let start = id as usize * N;
                    let row: &[u8; N] = base[start..start + N].try_into().unwrap();
                    let expected = scalar_l2(row, &query);
                    assert_eq!(streamed, expected, "case={case}, N={N}, id={id}");
                    assert_eq!(
                        <[u8; N]>::distance_compare(row, &query, Metric::L2),
                        expected,
                    );
                }
            }
        }

        let rows: Vec<&[u8; N]> = base[..4 * N]
            .chunks_exact(N)
            .map(|row| row.try_into().unwrap())
            .collect();
        let batched = distance_l2_vector_u8_batch4(rows[0], rows[1], rows[2], rows[3], &query);
        for (distance, row) in batched.into_iter().zip(rows) {
            assert_eq!(distance, scalar_l2(row, &query));
        }
    }
}

#[test]
fn sift128_stream_full_and_batch_match_scalar() {
    check_stream::<128>();
}

#[test]
fn u8_256_stream_full_and_batch_match_scalar() {
    // Also covers the stream's four-chunk body on AVX-512.
    check_stream::<256>();
}

#[test]
fn u8_stream_tail_shapes_match_scalar() {
    // With AVX-512's 64-byte chunks: a single tail, pair plus single
    // tail, and four-way body plus pair. SIFT128 above covers pair-only.
    check_stream::<64>();
    check_stream::<192>();
    check_stream::<384>();
}

#[test]
fn u8_kernel_all_byte_pairs_match_squared_difference() {
    use vector::{DistanceFn, L2U8Distance};

    // Exercise the kernel directly, including every negative difference and
    // both extreme byte values. Run in debug too: release wrapping masked a
    // signed-to-unsigned multiplication bug in the non-AVX-512 fallback.
    #[repr(align(64))]
    struct Chunk([u8; 64]);

    for a in 0..=255u8 {
        let base = Chunk([a; 64]);
        for b in 0..=255u8 {
            let query = Chunk([b; 64]);
            let mut acc = L2U8Distance::init();
            // Both arrays cover the kernel's 32- or 64-byte aligned chunk.
            unsafe {
                L2U8Distance::step(&mut acc, base.0.as_ptr(), query.0.as_ptr());
            }
            let difference = i32::from(a) - i32::from(b);
            let expected = difference * difference * L2U8Distance::CHUNK_BYTES as i32;
            assert_eq!(L2U8Distance::reduce(acc), expected as f32, "a={a}, b={b}");
        }
    }
}
