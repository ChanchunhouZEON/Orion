//! Verifies the generic DiskANN dataset uses the new full-precision byte kernel.
//! Orion's separate f32-only dataset path is deliberately not implied by this test.
use diskann::model::InmemDataset;
use vector::Metric;

#[test]
fn byte_dataset_retains_exact_sift_distances_with_quarter_payload() {
    let mut bytes = InmemDataset::<u8, 128>::new(2, 1.0).unwrap();
    let mut floats = InmemDataset::<f32, 128>::new(2, 1.0).unwrap();
    for i in 0..256 {
        bytes.data[i] = (i * 73 + 19) as u8;
        floats.data[i] = f32::from(bytes.data[i]);
    }

    assert_eq!(
        bytes.get_distance(0, 1, Metric::L2).unwrap(),
        floats.get_distance(0, 1, Metric::L2).unwrap()
    );
    assert_eq!(bytes.data.len(), 2 * 128 + 64);
    assert_eq!(floats.data.len(), 2 * 128 + 16);
    // Both buffers have a 64-byte SIMD tail; only the vector payload shrinks 4x.
    assert_eq!(
        std::mem::size_of_val(&bytes.data[..256]) * 4,
        std::mem::size_of_val(&floats.data[..256])
    );
}
