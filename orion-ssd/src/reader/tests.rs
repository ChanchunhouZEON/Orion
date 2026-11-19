/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
use orion_ann_platform::allocation_budget::AllocationBudget;
use std::{
    task::{Context, Poll, Waker},
    time::Duration,
};

fn setup() -> (tempfile::NamedTempFile, FileReader, AllocationBudget) {
    let file = tempfile::NamedTempFile::new().unwrap();
    file.as_file().set_len(4096).unwrap();
    let budget = AllocationBudget::new(Some(4096));
    let reader = FileReader::open(
        file.path(),
        FileReaderOptions {
            max_in_flight: 1,
            accounting: Some(budget.account("io").unwrap()),
            ..Default::default()
        },
    )
    .unwrap();
    (file, reader, budget)
}

#[tokio::test]
async fn cancellation_keeps_started_io_buffer_and_permit_until_completion() {
    let (_file, reader, budget) = setup();
    let request = ReadRequest::new(0, 4096).unwrap();
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let (finish_tx, finish_rx) = std::sync::mpsc::channel();
    let worker_reader = reader.clone();
    let task = tokio::spawn(async move {
        worker_reader
            .read_with(request, move |file, out, offset| {
                started_tx.send(()).unwrap();
                // A timeout prevents a failed assertion from hanging runtime shutdown.
                finish_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                read_exact_at(file, out, offset)
            })
            .await
    });
    started_rx.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert_eq!(budget.usage().resident_bytes, 4096);
    assert_eq!(reader.permits.available_permits(), 0);

    // Cancelling a queued request must not allocate or lose a semaphore slot.
    let mut queued = reader.read(request);
    assert!(matches!(
        queued
            .as_mut()
            .poll(&mut Context::from_waker(Waker::noop())),
        Poll::Pending
    ));
    drop(queued);
    assert_eq!(budget.usage().resident_bytes, 4096);
    finish_tx.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if reader.permits.available_permits() == 1 && budget.usage().resident_bytes == 0 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let result = reader.read(request).await.unwrap();
    assert_eq!(result.as_bytes().len(), 4096);
    drop(result);
    assert_eq!(budget.usage().resident_bytes, 0);
}

#[tokio::test]
async fn injected_error_and_worker_panic_roll_back_allocation_and_queue_slot() {
    let (_file, reader, budget) = setup();
    let request = ReadRequest::new(0, 4096).unwrap();
    let error = reader
        .read_with(request, |_, _, _| {
            Err(io::Error::other("injected read failure"))
        })
        .await
        .unwrap_err();
    assert!(error.to_string().contains("injected read failure"));
    assert_eq!(budget.usage().resident_bytes, 0);
    assert_eq!(reader.permits.available_permits(), 1);
    let error = reader
        .read_with(request, |_, _, _| panic!("injected worker panic"))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("file read worker failed"));
    assert_eq!(budget.usage().resident_bytes, 0);
    assert_eq!(reader.permits.available_permits(), 1);
    assert!(reader.read(request).await.is_ok());
}
