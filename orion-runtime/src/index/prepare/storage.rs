/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Establish the storage domain independently of calibration. Base and graph
//! constructors consume placement options before allocating. This final step
//! binds lazy sidecars, saved extras and native norms to the same domain.
use crate::{config::ResolvedRunConfig, data::BaseElement, Result};

pub(in crate::index) fn configure_object_storage<T: BaseElement>(
    index: &orion_ann_core::Orion<T>,
    config: &ResolvedRunConfig,
    snapshot_codecs: impl Iterator<Item = String>,
) -> Result<()> {
    crate::placement::configure_storage_policy(index, config, snapshot_codecs)
}

#[cfg(test)]
mod tests {
    use crate::{BaseInput, CascadeConfig, Index, IndexOptions, StorageMode, StorageOptions};

    #[test]
    fn generation_clones_keep_buffers_charged_until_final_drop() {
        let root = tempfile::tempdir().unwrap();
        let values: Vec<f32> = (0..64 * 8).map(|i| (i % 37) as f32).collect();
        let mut index = Index::new(
            BaseInput::from_f32(&values, 8).unwrap(),
            IndexOptions {
                cache_dir: root.path().into(),
                threads: 1,
                auto_pin: false,
                storage_options: StorageOptions {
                    mode: StorageMode::Resident,
                    ..Default::default()
                },
                cascade: CascadeConfig {
                    prefilter: Some("none".into()),
                    admission: Some("l2-u8".into()),
                    rerank: Some("none".into()),
                },
                ..Default::default()
            },
        )
        .unwrap();
        // The domain and lazy allocation policy exist before calibration.
        let budget = index.config.allocation_budget.get().unwrap().clone();
        let before = budget.usage();
        assert!(before.resident_bytes > 0);
        index
            .calibrate(values[..8 * 8].to_vec(), 3, Some(16))
            .unwrap();
        let prepared = budget.usage();
        assert!(prepared.resident_bytes > before.resident_bytes);
        assert_eq!(prepared.limit_bytes, before.limit_bytes);
        let shared = std::sync::Arc::new(index);
        let in_flight = shared.clone();
        drop(shared);
        assert_eq!(budget.usage(), prepared);
        std::thread::spawn(move || drop(in_flight)).join().unwrap();
        assert_eq!(budget.usage().resident_bytes, 0);
        assert!(budget.objects().unwrap().is_empty());
    }
}
