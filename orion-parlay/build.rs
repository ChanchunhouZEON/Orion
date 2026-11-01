fn main() {
    println!("cargo:rerun-if-changed=cpp");
    println!("cargo:rerun-if-changed=vendor");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    assert!(matches!(target_os.as_str(), "linux" | "macos"),
        "ParlayANN construction currently supports Linux/macOS; disable the runtime parlay feature on other targets");
    let mut build = cc::Build::new();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // cc otherwise chooses the host SDK version instead of Rust's minimum.
        if std::env::var_os("MACOSX_DEPLOYMENT_TARGET").is_none() {
            std::env::set_var("MACOSX_DEPLOYMENT_TARGET", "11.0");
        }
    }
    if std::env::var_os("CARGO_FEATURE_REFERENCE_TESTS").is_some() {
        build.define("ORION_REFERENCE_TESTS", None);
    }
    build
        .cpp(true)
        .std("c++17")
        .include("vendor/parlaylib/include")
        .include("vendor/algorithms")
        .file("cpp/bridge.cpp")
        // Build portable artifacts; never inherit the benchmark's -march=native.
        .flag_if_supported("-pthread")
        .flag_if_supported("-fvisibility=hidden")
        .flag_if_supported("-Wno-nontrivial-memcall")
        .flag_if_supported("-Wno-c++20-extensions")
        .warnings(false) // Upstream template headers are not warning-clean.
        .compile("orion_parlay");
}
