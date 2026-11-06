/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use super::*;
use orion_ann_core::{Orion, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;

#[test]
fn every_official_recipe_agrees_between_single_and_batch_search() {
    let count = 20;
    let dimension = 33;
    let flat: Vec<_> = (0..count * dimension)
        .map(|i| ((i * 31 + i / dimension * 7) % 101) as f32 / 10. - 5.)
        .collect();
    let base = PaddedDataset::from_packed(
        &flat,
        PaddedLayout::for_kernel::<L2F32Distance>(dimension).unwrap(),
    )
    .unwrap();
    let parts: Vec<_> = (0..count as u32)
        .map(|i| {
            (
                (0..count as u32).filter(|j| *j != i).collect(),
                vec![],
                vec![],
            )
        })
        .collect();
    let index = Orion::from_graph_partitions(base, &parts, 0, count as u32, 0).unwrap();
    for (pf, ad, rr) in [
        (
            PrefilterChoice::None,
            AdmissionChoice::PqL2,
            RerankChoice::F32,
        ),
        (
            PrefilterChoice::Jl,
            AdmissionChoice::PqIp,
            RerankChoice::IpF32,
        ),
        (
            PrefilterChoice::None,
            AdmissionChoice::PqL2,
            RerankChoice::None,
        ),
        (
            PrefilterChoice::None,
            AdmissionChoice::L2U8,
            RerankChoice::F32,
        ),
        (
            PrefilterChoice::Jl,
            AdmissionChoice::L2U16,
            RerankChoice::U16,
        ),
        (
            PrefilterChoice::JlHadamard,
            AdmissionChoice::L2Kt,
            RerankChoice::F32,
        ),
        (
            PrefilterChoice::Rabitq,
            AdmissionChoice::L2U8,
            RerankChoice::F32,
        ),
        (
            PrefilterChoice::Jl,
            AdmissionChoice::MipsI8,
            RerankChoice::IpF32,
        ),
        (
            PrefilterChoice::None,
            AdmissionChoice::MipsI16,
            RerankChoice::None,
        ),
        (
            PrefilterChoice::None,
            AdmissionChoice::AdsF32,
            RerankChoice::F32,
        ),
    ] {
        let batch = search_batch_compose(
            &index,
            &flat[..dimension * 3],
            3,
            24,
            2,
            0.,
            usize::MAX,
            pf,
            ad,
            rr,
        )
        .unwrap();
        for (i, q) in flat[..dimension * 3].chunks_exact(dimension).enumerate() {
            let single = search_compose(&index, q, 3, 24, 2, 0., usize::MAX, pf, ad, rr).unwrap();
            assert_eq!(batch.rows().nth(i).unwrap(), single);
            assert_eq!(single.len(), 3);
        }
        // Pinning consumes prepared requirements and cannot rebuild a sidecar.
        let requirements = crate::config::cascade::Cascade {
            prefilter: pf,
            admission: ad,
            rerank: rr,
        }
        .dataset_requirements();
        let residency = crate::residency::pin(
            &index,
            bytemuck::cast_slice(index.dataset().as_slice()),
            orion_ann_core::QuantizedDatasetShape {
                rows: count,
                dimension,
            },
            &requirements,
            &crate::PinOptions {
                budget_bytes: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(residency.pinned_bytes(), 0);
        assert_eq!(
            search_batch_compose(
                &index,
                &flat[..dimension * 3],
                3,
                24,
                2,
                0.,
                usize::MAX,
                pf,
                ad,
                rr
            )
            .unwrap(),
            batch
        );
    }
    assert!(build_admission(&index, AdmissionChoice::NativeL2U8).is_err());
}
