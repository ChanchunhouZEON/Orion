//! Low-dimensional groups and the high-dimensional signed-byte GIST kernel.
use vector::{DistanceFn, DistanceStream, IpI8Distance};

fn check<const N: usize>() {
    // Stream kernels consume whole chunks: 32 bytes on NEON, 64 on
    // AVX512. Pad each row and the query, rather than truncating N/chunk.
    let stride = N.div_ceil(IpI8Distance::CHUNK_BYTES) * IpI8Distance::CHUNK_BYTES;
    let mut query = vec![0_i8; stride];
    for (i, value) in query[..N].iter_mut().enumerate() {
        *value = (i % 127) as i8 - 63;
    }
    let mut base = vec![0_i8; 35 * stride + 128];
    for row in 0..35 {
        for coordinate in 0..N {
            base[row * stride + coordinate] = (((row * N + coordinate) * 17) % 251) as i8;
        }
    }
    for count in [0, 1, 2, 3, 4, 5, 7, 33] {
        let ids: Vec<u32> = (0..count).map(|i| (i * 11 % 35) as u32).collect();
        for lookahead in [1, 10, 64] {
            let mut result = Vec::new();
            unsafe {
                DistanceStream::<IpI8Distance>::new(
                    base.as_ptr().cast(),
                    stride,
                    stride,
                    query.as_ptr().cast(),
                    &ids,
                    lookahead,
                )
                .run(|id, distance| result.push((id, distance)));
            }
            assert_eq!(result.len(), ids.len());
            for ((id, value), expected_id) in result.iter().zip(&ids) {
                assert_eq!(id, expected_id);
                let start = *id as usize * stride;
                let dot: i32 = base[start..start + N]
                    .iter()
                    .zip(&query)
                    .map(|(&a, &b)| i32::from(a) * i32::from(b))
                    .sum();
                assert_eq!(
                    *value,
                    -(dot as f32),
                    "N={N}, stride={stride}, count={count}, lookahead={lookahead}, id={id}"
                );
            }
        }
    }
}

#[test]
fn grouped_i8_stream_preserves_order_and_distances() {
    check::<31>();
    check::<32>();
    check::<33>();
    check::<64>();
    check::<65>();
    check::<128>();
    check::<960>();
}
