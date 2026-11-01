use super::*;
use std::cell::{Cell, RefCell};

#[derive(Clone, Debug, PartialEq)]
enum Event {
    Computed(usize),
    Line,
}
thread_local! {
    static BASE: Cell<usize> = const { Cell::new(0) };
    static TRACE: RefCell<Vec<Event>> = const { RefCell::new(Vec::new()) };
}
struct Probe<const BYTES: usize>;
impl<const BYTES: usize> DistanceKernelBackend for Probe<BYTES> {
    type Storage = u8;
    type Acc = usize;
    const CHUNK_BYTES: usize = BYTES;
    fn init() -> usize {
        0
    }
    unsafe fn step(acc: &mut usize, a: *const u8, b: *const u8) {
        for i in 0..BYTES {
            *acc += a.add(i).read().abs_diff(b.add(i).read()) as usize;
        }
        let consumed = BASE.with(|base| a as usize - base.get() + BYTES);
        TRACE.with(|trace| trace.borrow_mut().push(Event::Computed(consumed)));
    }
    fn reduce(acc: usize) -> f32 {
        acc as f32
    }
    fn merge(acc: &mut usize, other: usize) {
        *acc += other;
    }
}

fn record<const BYTES: usize, const LINE: usize>(chunks: usize) -> Vec<Event> {
    let base: Vec<_> = (0..chunks * BYTES).map(|i| (i * 17 % 251) as u8).collect();
    let query: Vec<_> = (0..chunks * BYTES).map(|i| (i * 29 % 253) as u8).collect();
    BASE.with(|ptr| ptr.set(base.as_ptr() as usize));
    TRACE.with(|trace| trace.borrow_mut().clear());
    let distance = unsafe {
        compute_vertex::<Probe<BYTES>, _, LINE>(base.as_ptr(), query.as_ptr(), base.len(), || {
            TRACE.with(|trace| trace.borrow_mut().push(Event::Line));
        })
    };
    let expected: usize = base
        .iter()
        .zip(&query)
        .map(|(a, b)| a.abs_diff(*b) as usize)
        .sum();
    assert_eq!(distance, expected as f32);
    TRACE.with(|trace| trace.borrow().clone())
}

fn check<const BYTES: usize, const LINE: usize>() {
    for chunks in 0..=20 {
        let mut expected = Vec::new();
        let mut boundary = LINE;
        for completed in (1..=chunks).map(|n| n * BYTES) {
            expected.push(Event::Computed(completed));
            while boundary <= completed {
                expected.push(Event::Line);
                boundary += LINE;
            }
        }
        if !(chunks * BYTES).is_multiple_of(LINE) {
            expected.push(Event::Line);
        }
        assert_eq!(
            record::<BYTES, LINE>(chunks),
            expected,
            "chunk={BYTES}, line={LINE}, count={chunks}"
        );
    }
}

#[test]
fn prefetch_follows_cache_line_progress_for_all_builtin_shapes() {
    check::<32, 64>();
    check::<64, 64>();
    check::<32, 128>();
    check::<64, 128>();
}

#[test]
fn custom_kernel_groups_and_multi_line_steps_preserve_cadence() {
    check::<16, 128>();
    check::<192, 128>();
    check::<192, 64>();
}

#[test]
fn two_step_tails_have_no_duplicate_or_missing_requests() {
    use Event::*;
    assert_eq!(record::<32, 64>(2), vec![Computed(32), Computed(64), Line]);
    assert_eq!(
        record::<64, 64>(2),
        vec![Computed(64), Line, Computed(128), Line]
    );
    assert_eq!(
        record::<64, 64>(3),
        vec![Computed(64), Line, Computed(128), Line, Computed(192), Line]
    );
}

#[test]
fn partial_line_is_flushed_after_each_random_vector() {
    use Event::*;
    // Two unrelated short rows each get a line credit, never one shared credit.
    for _ in 0..2 {
        assert_eq!(record::<32, 128>(1), vec![Computed(32), Line]);
    }
    assert_eq!(
        record::<32, 128>(3),
        vec![Computed(32), Computed(64), Computed(96), Line]
    );
}
