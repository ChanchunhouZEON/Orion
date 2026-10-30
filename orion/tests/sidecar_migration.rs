//! Immutable pre-migration artifacts: catch changes to quantization and calibration.
#![cfg(target_arch = "aarch64")]
use diskann::model::{InmemDataset, PaddedDataset};
use orion::model::dataset::{
    jl_hadamard_dataset::JlHadamardDataset,
    jl_sparse_dataset::{JLSparseDataset, JLSparseDatasetMips},
    l2_kt_dataset::L2KTDataset,
    quantized_dataset::{L2U8, L2U16, MipsI8, MipsI16, QuantizedDataset},
    rabitq_b4_dataset::RabitQ4Dataset,
    rabitq_dataset::RabitQDataset,
};
use orion::{CalibrationConfig, Orion, StaticOrion, PhasedGraph};
use std::path::Path;

#[test]
#[cfg(target_arch = "aarch64")]
fn frozen_sidecars_and_calibration() {
    let mut base = InmemDataset::<f32, 32>::new(48, 1.0).unwrap();
    for i in 0..48 * 32 {
        base.data[i] = ((i * 31 + i / 32 * 7) % 103) as f32 / 17. - 3.;
    }
    let dir = std::env::temp_dir().join(format!("orion-sidecar-reference-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let check = |name: &str| {
        let actual = std::fs::read(dir.join(name)).unwrap();
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/dynamic-sidecars")
            .join(name);
        assert_eq!(actual, std::fs::read(fixture).unwrap(), "{name}");
    };
    macro_rules! save {
        ($name:literal,$ds:expr) => {{
            $ds.save(dir.join($name)).unwrap();
            check($name);
        }};
    }
    save!("u8.bin", QuantizedDataset::<L2U8>::from_f32_dataset(&base));
    save!(
        "u16.bin",
        QuantizedDataset::<L2U16>::from_f32_dataset(&base)
    );
    save!(
        "i8.bin",
        QuantizedDataset::<MipsI8>::from_f32_dataset(&base)
    );
    save!(
        "i16.bin",
        QuantizedDataset::<MipsI16>::from_f32_dataset(&base)
    );
    save!("kt.bin", L2KTDataset::build_from(&base));
    save!("jl.bin", JLSparseDataset::<1024>::build_from(&base, 42));
    save!(
        "jl-mips.bin",
        JLSparseDatasetMips::<1024>::build_from(&base, 42)
    );
    save!(
        "hadamard.bin",
        JlHadamardDataset::build_with_padding(&base, 42, 64)
    );
    save!("rabitq.bin", RabitQDataset::build_from(&base, 42));
    save!("rabitq4.bin", RabitQ4Dataset::build_from(&base, 42));
    // Same immutable files through padded logical rows, not just the adapter.
    let padded = PaddedDataset::from_packed(
        &base.data[..48 * 32],
        vector::PaddedLayout::new::<f32>(32, 256).unwrap(),
    )
    .unwrap();
    save!(
        "u8.bin",
        QuantizedDataset::<L2U8>::from_f32_dataset(&padded)
    );
    save!(
        "u16.bin",
        QuantizedDataset::<L2U16>::from_f32_dataset(&padded)
    );
    save!(
        "i8.bin",
        QuantizedDataset::<MipsI8>::from_f32_dataset(&padded)
    );
    save!(
        "i16.bin",
        QuantizedDataset::<MipsI16>::from_f32_dataset(&padded)
    );
    save!("kt.bin", L2KTDataset::build_from(&padded));
    save!("jl.bin", JLSparseDataset::<1024>::build_from(&padded, 42));
    save!(
        "jl-mips.bin",
        JLSparseDatasetMips::<1024>::build_from(&padded, 42)
    );
    save!(
        "hadamard.bin",
        JlHadamardDataset::build_with_padding(&padded, 42, 64)
    );
    save!("rabitq.bin", RabitQDataset::build_from(&padded, 42));
    save!("rabitq4.bin", RabitQ4Dataset::build_from(&padded, 42));
    let qs: Vec<[f32; 32]> = (0..8)
        .map(|r| std::array::from_fn(|j| base.data[r * 32 + j] + 0.01))
        .collect();
    let parts: Vec<_> = (0..48)
        .map(|i| ((1..7).map(|j| (i + j * 7) % 48).collect(), vec![], vec![]))
        .collect();
    let graph = PhasedGraph::build_from_partitions(&parts, 8, 0);
    let index = StaticOrion::from_phased_graph(base, graph, 0, None, None);
    let d = index
        .calibrate_with_diagnostics(
            &qs,
            16,
            3,
            CalibrationConfig {
                k: 5,
                ..Default::default()
            },
        )
        .unwrap();
    let dynamic = Orion::from_phased_graph(
        padded,
        PhasedGraph::build_from_partitions(&parts, 8, 0),
        0,
        None,
        None,
    );
    let packed: Vec<_> = qs.iter().flatten().copied().collect();
    let dynamic_d = dynamic
        .calibrate_with_diagnostics(
            &packed,
            16,
            3,
            CalibrationConfig {
                k: 5,
                ..Default::default()
            },
        )
        .unwrap();
    for d in [d, dynamic_d] {
        let mut bytes = Vec::new();
        bytes.extend(d.params.threshold.to_le_bytes());
        bytes.extend((d.params.early_exit_limit as u64).to_le_bytes());
        for list in [&d.admission_rates, &d.topk_coverage_by_step] {
            bytes.extend((list.len() as u64).to_le_bytes());
            for v in list {
                bytes.extend(v.to_le_bytes());
            }
        }
        for list in [&d.useful_gaps, &d.tail_gaps] {
            bytes.extend((list.len() as u64).to_le_bytes());
            for &v in list {
                bytes.extend((v as u64).to_le_bytes());
            }
        }
        std::fs::write(dir.join("calibration.bin"), bytes).unwrap();
        check("calibration.bin");
    }
    std::fs::remove_dir_all(dir).unwrap();
}
