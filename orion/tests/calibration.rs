use diskann::model::InmemDataset;
use orion::{CalibrationConfig, CalibrationMetric, Orion};

const DIM: usize = 32;
const POINTS: usize = 128;

#[test]
fn calibration_beam_policy_preserves_base_and_adds_headroom() {
    assert_eq!(orion::calibration_search_list_size!(48, 10), 48);
    assert_eq!(orion::calibration_search_list_size!(48, 100), 200);
    assert_eq!(orion::calibration_search_list_size!(256, 100), 256);
    assert_eq!(
        orion::calibration_search_list_size!(48, usize::MAX),
        usize::MAX
    );
    let mut calls = 0;
    let beam = orion::calibration_search_list_size!(
        {
            calls += 1;
            48
        },
        {
            calls += 1;
            100
        }
    );
    assert_eq!(beam, 200);
    assert_eq!(calls, 2);
}

fn fixture() -> (Orion<DIM>, [[f32; DIM]; 1]) {
    let mut flat = vec![0.0; POINTS * DIM];
    for i in 0..POINTS {
        flat[i * DIM] = (i + 10) as f32;
    }
    flat[0] = 1.0;
    flat[10 * DIM] = 1.1;
    let mut dataset = InmemDataset::<f32, DIM>::new(POINTS, 1.0).unwrap();
    dataset.data.memcpy(&flat).unwrap();
    // A chain fixes discovery order independently of the chosen metric.
    let partitions: Vec<_> = (0..POINTS)
        .map(|i| {
            let local = if i + 1 < POINTS {
                vec![(i + 1) as u32]
            } else {
                vec![]
            };
            (local, vec![], vec![])
        })
        .collect();
    let index = Orion::new(dataset, &partitions, 0, 1, 0, None, None, None, false);
    let mut query = [0.0; DIM];
    query[0] = 1.0;
    (index, [query])
}

#[test]
fn default_preserves_top10_l2() {
    assert_eq!(
        CalibrationConfig::default(),
        CalibrationConfig {
            k: 10,
            metric: CalibrationMetric::L2,
        }
    );
    let (index, queries) = fixture();
    let default = index
        .calibrate(&queries, POINTS, 5, Default::default())
        .unwrap();
    let explicit = index
        .calibrate(
            &queries,
            POINTS,
            5,
            CalibrationConfig {
                k: 10,
                metric: CalibrationMetric::L2,
            },
        )
        .unwrap();
    assert_eq!(default.threshold, explicit.threshold);
    assert_eq!(default.early_exit_limit, explicit.early_exit_limit);
}

#[test]
fn topk_target_changes_gaps_and_coverage() {
    let (index, queries) = fixture();
    let run = |k| {
        index
            .calibrate_with_diagnostics(
                &queries,
                POINTS,
                5,
                CalibrationConfig {
                    k,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let one = run(1);
    let two = run(2);
    let hundred = run(100);
    assert!(one.useful_gaps.is_empty());
    assert_eq!(one.params.early_exit_limit, 3);
    assert_eq!(two.useful_gaps, vec![9]);
    assert_eq!(two.params.early_exit_limit, 9);
    assert_eq!(one.topk_coverage_by_step[0], 1.0);
    assert_eq!(two.topk_coverage_by_step[0], 0.5);
    assert!((hundred.topk_coverage_by_step[0] - 0.02).abs() < 1e-6);
    assert_eq!(hundred.topk_coverage_by_step.last(), Some(&1.0));
}

#[test]
fn inner_product_uses_raw_vector_norms() {
    let (index, queries) = fixture();
    let run = |metric| {
        index
            .calibrate_with_diagnostics(&queries, POINTS, 5, CalibrationConfig { k: 1, metric })
            .unwrap()
    };
    let l2 = run(CalibrationMetric::L2);
    let ip = run(CalibrationMetric::InnerProduct);
    assert_eq!(l2.topk_coverage_by_step[0], 1.0);
    assert_eq!(ip.topk_coverage_by_step[0], 0.0);
    assert_eq!(ip.topk_coverage_by_step[125], 0.0);
    assert_eq!(ip.topk_coverage_by_step[126], 1.0);
    assert_ne!(l2.tail_gaps, ip.tail_gaps);
}

#[test]
fn diagnostics_and_regular_calibration_agree() {
    let (index, queries) = fixture();
    for metric in [
        CalibrationMetric::L2,
        CalibrationMetric::InnerProduct,
        CalibrationMetric::Cosine,
    ] {
        for k in [1, 2, 10, 100] {
            let config = CalibrationConfig { k, metric };
            let params = index.calibrate(&queries, POINTS, 5, config).unwrap();
            let diag = index
                .calibrate_with_diagnostics(&queries, POINTS, 5, config)
                .unwrap();
            assert_eq!(params.threshold, diag.params.threshold);
            assert_eq!(params.early_exit_limit, diag.params.early_exit_limit);
            assert!(diag.topk_coverage_by_step.iter().all(|v| v.is_finite()));
        }
    }
}

#[test]
fn invalid_config_is_rejected_by_both_entry_points() {
    let (index, queries) = fixture();
    for (k, l, window) in [
        (0, 128, 5),
        (100, 48, 5),
        (1, 0, 5),
        (10, 128, 0),
        (10, 128, 65),
    ] {
        let config = CalibrationConfig {
            k,
            ..Default::default()
        };
        assert!(index.calibrate(&queries, l, window, config).is_err());
        assert!(
            index
                .calibrate_with_diagnostics(&queries, l, window, config)
                .is_err()
        );
    }
}
