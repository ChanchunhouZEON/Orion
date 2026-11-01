//! Generate a portable trace with the production native-u8 search path.
use orion_ann_runtime::diagnostics::TraceLimits;
use orion_ann_runtime::{BaseInput, GraphConfig, Index, IndexOptions, Result};
fn main() -> Result<()> {
    let destination = std::env::args()
        .nth(1)
        .ok_or("provide an output JSON filename")?;
    let cache = std::env::temp_dir().join(format!("orion-trace-example-{}", std::process::id()));
    let result = (|| {
        let base: Vec<u8> = (0..64 * 13)
            .map(|i| ((i * 37 + i / 13 * 17) % 251) as u8)
            .collect();
        let query: Vec<f32> = base[..13].iter().map(|&v| v as f32).collect();
        let mut index = Index::new(
            BaseInput::from_u8(&base, 13)?,
            IndexOptions {
                threads: 1,
                cache_dir: cache.clone(),
                graph: GraphConfig {
                    degree: 8,
                    build_l: 16,
                    max_extra: 4,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        let expected = index.search(query.clone(), 5, 16, false)?;
        let trace = index.trace_search(query, 5, 16, false, TraceLimits::default())?;
        assert_eq!(
            &trace.result_ids,
            &expected.ids[..expected.counts[0] as usize]
        );
        std::fs::write(&destination, trace.to_json()?)?;
        let prefix = index.trace_search(
            base[..13].iter().map(|&v| v as f32).collect(),
            5,
            16,
            false,
            TraceLimits {
                max_events: 1,
                max_items: 6,
            },
        )?;
        std::fs::write(
            std::path::Path::new(&destination).with_extension("truncated.json"),
            prefix.to_json()?,
        )?;
        let mut unsupported = trace;
        unsupported.schema_version = 999;
        let error = orion_ann_runtime::diagnostics::SearchTrace::from_json(&serde_json::to_string(
            &unsupported,
        )?)
        .unwrap_err();
        std::fs::write(
            std::path::Path::new(&destination).with_extension("error.json"),
            serde_json::to_string(&error.report())?,
        )?;
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(cache);
    result
}
