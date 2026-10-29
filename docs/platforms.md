# Platforms and build variants

[README](../README.md)

The pinned toolchain is [Rust 1.92](../rust-toolchain.toml). Run Cargo from the
repository root. [.cargo/config.toml](../.cargo/config.toml) enables dot-product
instructions on Apple Silicon (and requests LLD) and AVX2/FMA on x86 targets.
Explicit `RUSTFLAGS` replaces those target flags; record it in experiment metadata.

## AArch64

Apple Silicon builds use NEON and `+dotprod`. The local result snapshots use
macOS/AArch64; Linux THP is not active there. Ensure the target CPU supports any
optional instructions enabled at compilation. Other operating systems/targets
need their own runtime verification.

## x86 Linux

For the tested Xeon Platinum 8358P configuration:

```bash
RUSTFLAGS='-C target-cpu=x86-64 -C target-feature=+avx2,+fma,+f16c,+avx512f,+avx512bw,+avx512dq,+avx512vl,+avx512vpopcntdq' \
  cargo build --locked --release --target x86_64-unknown-linux-gnu \
  -p benchmark --bin orion-sweep
```

This is an AVX-512 build for a compatible CPU. The AVX2 variant uses
`-C target-cpu=x86-64 -C target-feature=+avx2,+fma,+f16c`. Do not use
`target-cpu=native` on an unrelated build host as a substitute for the deployment
CPU's feature set. Cross-compilation also needs a compatible linker/sysroot and
runtime libraries; specifying a Rust target alone does not supply them.

CI compiles AVX2, AVX-512 and AVX-512-with-vector-popcount variants. Execution is
conditional on the runner's detected features. It does not guarantee AVX-512
runtime coverage on every GitHub-hosted runner. See [Testing](testing.md).

## Linux transparent huge pages

`AlignedBoxWithSlice` requests at least 2 MiB alignment for allocations of at
least 2 MiB on Linux and advises complete pages with `MADV_HUGEPAGE` before
initialization. The OS hook is in [platform/memory.rs](../platform/src/memory.rs).
This best-effort hint does not require the application to reserve hugetlb pages,
and failure leaves ordinary pages usable. It neither guarantees THP coverage nor
requests 1 GiB pages.

Inspect the running process's `/proc/<pid>/smaps` or `smaps_rollup` to measure
`AnonHugePages` against the relevant resident mappings. Accepted advice alone
is insufficient evidence. Record CPU affinity, NUMA placement, kernel THP
settings and measured coverage alongside QPS. The shared allocator can affect
multiple engines, so baseline binaries must be identified when comparing them.

On non-Linux targets this THP path is not applied. No equivalent macOS large-page
implementation is claimed by the current code.
