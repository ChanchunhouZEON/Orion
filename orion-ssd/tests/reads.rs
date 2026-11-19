/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg(not(target_arch = "wasm32"))]
use orion_ann_platform::{
    allocation_budget::AllocationBudget,
    reading::{ObjectReader, ReadRequest},
    region::{ArrayView, Placement, RegionOptions},
};
use orion_ann_ssd::{FileReader, FileReaderOptions};
use std::{
    future::Future,
    io::{self, Write},
    pin::pin,
    task::{Context, Poll, Waker},
};

async fn bytes(reader: &impl ObjectReader, offset: u64, count: usize) -> Vec<u8> {
    reader
        .read(ReadRequest::new(offset, count).unwrap())
        .await
        .unwrap()
        .as_bytes()
        .to_vec()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn identical_ranges_across_resident_mapping_and_file() {
    let input: Vec<u8> = (0..65_557)
        .map(|i| ((i * 37 + i / 257) % 251) as u8)
        .collect();
    let resident = ArrayView::copy_from(&input, &Default::default()).unwrap();
    let mapped = ArrayView::copy_from(
        &input,
        &RegionOptions {
            placement: Placement::Mapped,
            ..Default::default()
        },
    )
    .unwrap();
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&input).unwrap();
    let reader = FileReader::open(file.path(), FileReaderOptions::default()).unwrap();
    assert_eq!(reader.byte_len(), input.len() as u64);
    for (offset, count) in [(0, 0), (0, 4096), (4093, 8197), (65550, 7), (65557, 0)] {
        let expected = bytes(resident.region(), offset, count).await;
        assert_eq!(bytes(mapped.region(), offset, count).await, expected);
        assert_eq!(bytes(&reader, offset, count).await, expected);
    }
    // Concurrent disjoint reads must not race through a shared file cursor.
    let mut tasks = Vec::new();
    for i in 0..64 {
        let reader = reader.clone();
        tasks.push(tokio::spawn(async move {
            (i, bytes(&reader, i * 997, 997).await)
        }));
    }
    for task in tasks {
        let (i, actual) = task.await.unwrap();
        assert_eq!(actual, input[i as usize * 997..(i as usize + 1) * 997]);
    }
}

#[test]
fn completed_lease_outlives_file_reader_runtime_and_preserves_alignment() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&vec![23; 4096]).unwrap();
    let budget = AllocationBudget::new(Some(4096));
    let reader = FileReader::open(
        file.path(),
        FileReaderOptions {
            accounting: Some(budget.account("io").unwrap()),
            ..Default::default()
        },
    )
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let lease = runtime
        .block_on(reader.read(ReadRequest::new(0, 4096).unwrap()))
        .unwrap()
        .into_owned()
        .unwrap();
    assert_eq!(lease.as_bytes().as_ptr().addr() % 4096, 0);
    drop(runtime);
    drop(reader);
    drop(file);
    assert_eq!(budget.usage().resident_bytes, 4096);
    assert!(lease.as_bytes().iter().all(|&byte| byte == 23));
    drop(lease);
    assert_eq!(budget.usage().resident_bytes, 0);
}

#[tokio::test]
async fn invalid_ranges_budget_failure_and_truncation_release_resources() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().set_len(64).unwrap();
    let budget = AllocationBudget::new(Some(16));
    let reader = FileReader::open(
        file.path(),
        FileReaderOptions {
            max_in_flight: 1,
            max_read_bytes: 32,
            accounting: Some(budget.account("io").unwrap()),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        reader
            .read(ReadRequest::new(64, 1).unwrap())
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        reader
            .read(ReadRequest::new(0, 33).unwrap())
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(
        reader
            .read(ReadRequest::new(0, 32).unwrap())
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::OutOfMemory
    );
    assert_eq!(budget.usage().resident_bytes, 0);
    let lease = reader.read(ReadRequest::new(0, 16).unwrap()).await.unwrap();
    assert_eq!(
        reader
            .read(ReadRequest::new(0, 1).unwrap())
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::OutOfMemory
    );
    drop(lease);
    // A shortened external file is an I/O error, never an invalid memory view.
    file.as_file().set_len(1).unwrap();
    assert_eq!(
        reader
            .read(ReadRequest::new(0, 16).unwrap())
            .await
            .unwrap_err()
            .kind(),
        io::ErrorKind::UnexpectedEof
    );
    assert_eq!(budget.usage().resident_bytes, 0);
    assert_eq!(bytes(&reader, 0, 1).await, vec![0]);
}

#[test]
fn no_runtime_is_a_typed_error_and_unpolled_reads_do_nothing() {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().set_len(1).unwrap();
    let reader = FileReader::open(file.path(), FileReaderOptions::default()).unwrap();
    drop(reader.read(ReadRequest::new(0, 1).unwrap()));
    let mut future = pin!(reader.read(ReadRequest::new(0, 1).unwrap()));
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(Err(error)) => assert_eq!(error.kind(), io::ErrorKind::Unsupported),
        other => panic!("unexpected poll: {other:?}"),
    }
    let mut empty = pin!(reader.read(ReadRequest::new(1, 0).unwrap()));
    assert!(matches!(
        empty.as_mut().poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(_))
    ));
}

#[test]
fn invalid_configuration_and_non_regular_files_are_rejected() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let variants = [
        FileReaderOptions {
            max_in_flight: 0,
            ..Default::default()
        },
        FileReaderOptions {
            max_in_flight: usize::MAX,
            ..Default::default()
        },
        FileReaderOptions {
            max_read_bytes: 0,
            ..Default::default()
        },
        FileReaderOptions {
            buffer_alignment: 3,
            ..Default::default()
        },
    ];
    for options in variants {
        assert!(FileReader::open(file.path(), options).is_err());
    }
    let dir = tempfile::tempdir().unwrap();
    assert!(FileReader::open(dir.path(), FileReaderOptions::default()).is_err());
}

#[tokio::test]
async fn unchanged_distance_stream_consumes_all_three_reader_leases() {
    use orion_ann_vector::{DistanceStream, L2F32Distance};
    for stride in [32, 128, 960] {
        let rows = 37;
        let values: Vec<f32> = (0..rows * stride + 32).map(|i| (i % 67) as f32).collect();
        let query = vec![7.0f32; stride];
        let resident = ArrayView::copy_from(&values, &Default::default()).unwrap();
        let mapped = ArrayView::copy_from(
            &values,
            &RegionOptions {
                placement: Placement::Mapped,
                ..Default::default()
            },
        )
        .unwrap();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(resident.region().as_bytes()).unwrap();
        let reader = FileReader::open(file.path(), FileReaderOptions::default()).unwrap();
        let request = ReadRequest::new(0, resident.region().len_bytes()).unwrap();
        let ids = [0, 36, 7, 19, 4, 30, 3];
        let leases = [
            resident.region().read(request).await.unwrap(),
            mapped.region().read(request).await.unwrap(),
            reader.read(request).await.unwrap(),
        ];
        let mut outputs = Vec::new();
        for lease in leases {
            let mut output = Vec::new();
            // SAFETY: aligned, fully initialized f32 rows plus SIMD tail. All
            // IDs are bounded; the lease and padded query outlive the stream.
            unsafe {
                DistanceStream::<L2F32Distance>::new(
                    lease.as_slice::<f32>().unwrap().as_ptr().cast(),
                    stride * 4,
                    stride * 4,
                    query.as_ptr().cast(),
                    &ids,
                    32,
                )
                .run(|id, distance| output.push((id, distance)));
            }
            outputs.push(output);
        }
        assert_eq!(outputs[0], outputs[1]);
        assert_eq!(outputs[0], outputs[2]);
        for &(id, distance) in &outputs[0] {
            let expected: f32 = values[id as usize * stride..(id as usize + 1) * stride]
                .iter()
                .map(|&x| (x - 7.0).powi(2))
                .sum();
            assert_eq!(distance, expected);
        }
    }
}
