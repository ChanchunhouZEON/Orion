use orion_ann_runtime::{BaseInput, ErrorCode, Index, IndexOptions};
use std::process::Command;

#[test]
fn rust_and_cli_preserve_the_same_missing_input_report() {
    let path = std::env::temp_dir().join(format!("orion-missing-{}.fbin", std::process::id()));
    assert!(!path.exists());
    let rust = match Index::new(BaseInput::File(path.clone()), IndexOptions::default()) {
        Ok(_) => panic!("missing file accepted"),
        Err(error) => error,
    };
    assert_eq!(rust.code(), ErrorCode::Io);
    let output = Command::new(env!("CARGO_BIN_EXE_orion"))
        .args([
            "anonymous",
            "--metric",
            "l2",
            "--prepare-only",
            "--no-groundtruth",
            "--base",
        ])
        .arg(&path)
        .args(["--error-format", "json"])
        .env_remove("ORION_SWEEP_CONFIG")
        .env_remove("ORION_GRAPH")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let text = String::from_utf8(output.stderr).unwrap();
    let cli: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(cli, serde_json::to_value(rust.report()).unwrap());
}
