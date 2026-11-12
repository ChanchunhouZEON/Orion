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
            AdmissionChoice::IpI8,
            RerankChoice::IpF32,
        ),
        (
            PrefilterChoice::None,
            AdmissionChoice::IpI16,
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

#[test]
fn cosine_arms_preserve_existing_ip_stream_results_on_normalized_rows() {
    let dimension = 33;
    let count = 32;
    let mut flat: Vec<f32> = (0..count * dimension)
        .map(|i| ((i * 13 + i / dimension * 7) % 101) as f32 - 50.)
        .collect();
    crate::data::validate_and_normalize_vectors(&mut flat, dimension, SearchMetric::Cosine)
        .unwrap();
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
    for (old, new) in [
        (AdmissionChoice::IpI8, AdmissionChoice::CosineI8),
        (AdmissionChoice::IpI16, AdmissionChoice::CosineI16),
        (AdmissionChoice::PqIp, AdmissionChoice::PqCosine),
    ] {
        for pf in [PrefilterChoice::None, PrefilterChoice::Jl] {
            let run = |ad, rr| {
                search_batch_compose(
                    &index,
                    &flat[..dimension * 4],
                    5,
                    32,
                    2,
                    0.,
                    usize::MAX,
                    pf,
                    ad,
                    rr,
                )
                .unwrap()
            };
            assert_eq!(
                run(old, RerankChoice::IpF32),
                run(new, RerankChoice::CosineF32)
            );
            assert_eq!(run(old, RerankChoice::None), run(new, RerankChoice::None));
            let old_session = build_admission(&index, old).unwrap();
            let new_session = build_admission(&index, new).unwrap();
            let a = old_session.open(&flat[..dimension]);
            let b = new_session.open(&flat[..dimension]);
            for id in 0..count as u32 {
                assert_eq!(a.entry_distance(id), b.entry_distance(id));
            }
        }
    }
}
