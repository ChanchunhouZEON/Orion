<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> orion-ann-vector

SIMD distance kernels and padded vector layouts for Orion ANN.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

This component is versioned with the Orion ANN workspace. Most applications
should depend on `orion-ann` or `orion-ann-core` instead of this lower-level
implementation crate. See rustdoc for contracts of individual APIs.

On baseline x86-64 builds, the f32 and Half L2 entry points detect AVX2/FMA
at runtime (Half also requires F16C) and cache the selected kernel. Unsupported
CPUs use scalar arithmetic. Explicit AVX2 builds use direct calls; explicit
AVX-512 builds retain their existing kernels. These entry points accept
naturally aligned inputs and handle short tails.

The seven streaming distance kernels also select a backend automatically:
scalar, AVX2/FMA, AVX-512, or AVX-512 with vector popcount on x86-64;
scalar, NEON, or NEON with dot product on AArch64. Selection happens before
the stream, with computation, sink and prefetch scheduling inlined inside the
chosen backend. A 32-byte padded row that cannot safely use a 64-byte kernel
uses AVX2 for the whole stream instead; no search-time repacking is required.

Explicit target features retain the direct specialized path where possible.
They are optional for ordinary downstream dependencies. Never distribute a
binary compiled with global CPU features to an unsupported CPU: runtime
dispatch cannot make instructions emitted elsewhere in that binary portable.

Streaming near prefetch advances in `CACHE_LINE_BYTES` units of computation,
independent of SIMD chunk width. The four accumulator chains are retained;
trigger positions specialize at compilation for each backend. A partial last
line is accounted for after each vector, without carrying credits between
random vertex IDs. Group/burst scheduling and near/far overlap remain separate.
The low-level kernel trait is `DistanceKernelBackend`. Higher layers select
cascade stages independently of coordinate storage and ISA dispatch.
