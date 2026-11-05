<h1 align="center"><img src="orion-core/assets/orion-astraeus.png" alt="" width="64" align="center" /> 猎户·阿斯特俄斯：面向候选准入的图搜索</h1>

<p align="center"><a href="README.md" lang="en">English</a> · <strong>简体中文</strong></p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img alt="Safety tests: Miri and adversarial inputs" src="https://img.shields.io/badge/safety-Miri_%2B_adversarial_tests-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="orion-runtime/src/guide/cache.md"><img alt="Verified persistence: SHA-256" src="https://img.shields.io/badge/integrity-SHA--256-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md"><img alt="Tested platforms: Linux x86-64 and macOS ARM64" src="https://img.shields.io/badge/platforms-Linux_x86__64_%7C_macOS_ARM64-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="#使用示例"><img alt="Rust and Python APIs" src="https://img.shields.io/badge/API-Rust_%2B_Python-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md"><img alt="Automatic runtime SIMD dispatch" src="https://img.shields.io/badge/SIMD-auto_dispatch-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="rust-toolchain.toml"><img alt="Rust 1.92" src="https://img.shields.io/badge/Rust-1.92-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Rust tests: 504 passed" src="https://img.shields.io/badge/Rust_tests-504_passed-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Doctests: 42 passed" src="https://img.shields.io/badge/doctests-42_passed-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/testing.md#readme-validation-snapshot"><img alt="Rust line coverage: 90.20%" src="https://img.shields.io/badge/Rust_line_coverage-90.20%25-A77A2E?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml"><img alt="Coverage check" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml"><img alt="Release validation" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml/badge.svg?branch=main" /></a>
</p>

**猎户·阿斯特俄斯（Orion Astraeus，简称 Orion）** 是一种 **Vamana 风格的图算法与内存近似最近邻检索引擎**，提供 Rust 和 Python 接口。其准入感知搜索跟踪候选节点进入搜索队列的过程，在邻域逐渐收敛时调整图遍历，并通过级联距离计算减少在低潜力候选节点上的开销。

只需提供向量：Orion 会使用内置的 ParlayANN 构图，根据距离度量和维度选择搜索方案，并自动选择 CPU 支持的 SIMD 后端。使用有代表性的查询完成校准后，即可通过共享索引执行搜索。使用库接口时，无需数据集预设、预先构建的图、ground truth 或手动设置 SIMD 编译选项。

**面向嵌入式检索：** 支持 f32 和原生 u8 输入、动态维度、并发查询、结构化错误，以及可复用图和 sidecar 缓存对象的校验快照。Sidecar 是索引按需构建的量化等辅助数据。

[性能](#性能) · [使用示例](#使用示例) · [支持范围](#支持范围) · [CLI](#cli) · [基准测试与开发](#基准测试与开发)

## 性能

在论文涵盖更多基线的六数据集对照中，Orion 在 **Recall@10 ≥ 0.95 时均取得最高实测 QPS**。下表同时列出十亿向量规模的 SIFT1B 和高维 Wiki-Ada-1M 结果。所有列均使用 **8 线程、k=10、recall ≥ 0.95**；每个结果取达到目标召回率的最快实测 QPS 中位数，不进行插值。K 表示 1,000 QPS，前六列沿用论文的舍入精度。

| 引擎 · QPS | SIFT1M | GloVe-25 | GloVe-100 | GIST | Deep10M | MS-MARCO | SIFT1B | Wiki-Ada-1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Orion** | **341K** | **383K** | **47K** | **48K** | **170K** | **52K** | **33,897** | **143,954** |
| ParlayANN | 257K | 247K | 37K | 29K | 143K | 38K | 33,086 | 151,258 |
| Microsoft Vamana (DiskANN) | 91K | 151K | 12K | 7K | 39K | — | 10,371 | 39,915 |
| USearch HNSW | 59K | 102K | — | — | 21K | 8K | 未测试 | 未测试 |
| LanceDB HNSW (native) | 1K | 1K | — | — | — | — | 未测试 | 未测试 |
| HNSW (hnswlib) | 51K | 139K | — | — | 21K | 6K | 未测试 | 未测试 |
| FAISS HNSW+SQ | 54K | — | — | — | 15K | 5K | 未测试 | 17,730 |
| FAISS IVF-Flat | 9K | 33K | 3K | — | 2K | — | 未测试 | 未测试 |
| Annoy | 1K | 3K | — | — | — | — | 未测试 | 未测试 |

前六个数据集的实验运行于 Apple M4 Max。Wiki-Ada-1M 包含 100 万条 1,536 维归一化向量，运行于 Apple Silicon；SIFT1B 包含 10 亿条 128 维原生 u8 向量，运行于 Xeon Platinum 8358P、Linux/AVX-512，Orion 启用了透明大页。搜索计时不包含数据加载、构图和校准。

「未测试」表示没有保留对应测量结果；「—」表示实测曲线未达到目标召回率，或结果不可比。Microsoft Vamana 指 DiskANN 的内存实现，其 MS-MARCO 结果因距离度量不一致而未纳入比较。FAISS IVF-PQ 在前六个数据集上的 recall 均低于 0.62，故未列入。**粗体用于突出 Orion 的结果。**
[精确 QPS、实际召回率与运行元数据](docs/results/2026-11-05/readme-performance.json) · [实验方法](docs/benchmarks.md) · [SIFT1B 资源规划](docs/large-datasets.md)。

## 使用示例

源码构建需要 **Rust 1.92 和 C++17 编译器**；后者用于内置的图构建器。

### Rust：一个依赖，完成构图与搜索

假设应用目录与 `Orion` 源码目录相邻：

```toml
[dependencies]
orion-ann = { path = "../Orion/orion-runtime" }
```

```rust
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};

fn main() -> orion_ann::Result<()> {
    // 将示例数据替换为按行连续存储的向量。
    let base: Vec<f32> = (0..1_024 * 128)
        .map(|i| ((i * 73 + i / 128 * 11) % 251) as f32)
        .collect();
    let queries = &base[..32 * 128];
    let mut index = Index::new(
        BaseInput::from_f32(&base, 128)?,
        IndexOptions::default(),
    )?;

    // 生产环境应使用有代表性的应用查询进行校准，无需 ground truth。
    index.calibrate(queries.to_vec(), 10, None)?;
    let result = index.search(
        QueryBatch::from_f32(queries, 128)?,
        &SearchOptions::new(10, 100),
    )?;
    for neighbors in result.rows() {
        println!("{neighbors:?}");
    }
    Ok(())
}
```

文件输入可使用 `BaseInput::from_file("base.fbin")`；原生字节输入可使用 `BaseInput::from_u8(&bytes, dimension)?`。Orion 会在内部处理逐行 padding。Rust 搜索借用查询数据，并通过 `&self` 访问索引，因此可以使用 `Arc<Index>`，通过共享线程池处理并发请求。每个需要使用的 `k` 均需校准一次；缺少对应校准时，搜索返回 `InvalidState`。
[更多 Rust 示例](orion-runtime/README.md) · [配置指南](orion-runtime/src/guide/configuration.md)。

### Python：输入 NumPy 数组，获取邻居 ID

在仓库根目录中创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.24,<3' 'maturin>=1.9,<2'
maturin develop --manifest-path orion-py/Cargo.toml --profile python
```

```python
import numpy as np
from orion_ann import Index

rng = np.random.default_rng(42)
base = rng.random((1_024, 128), dtype=np.float32)
queries = rng.random((32, 128), dtype=np.float32)

index = Index.from_array(base, metric="l2", threads=8)
index.calibrate(queries, k=10)
result = index.search(queries, k=10, search_list_size=100)
print(result.ids[0, :result.counts[0]])
```

需要归一化的嵌入向量可使用 `metric="cosine"`；原始内积可使用 `metric="ip"`。接口接受 C 连续布局的 `float32` 和 `uint8` 数组，原生字节输入使用 L2 度量。Python 接口在释放 GIL 前复制输入；并发搜索共享配置好的 Rust 线程池。

### 读取文件并保存已校准的快照

```python
index = Index.from_file("base.fbin", metric="l2", cache_dir="cache/build")
index.calibrate("queries.fbin", k=10)
result = index.search("queries.fbin", k=10, search_list_size=100)

snapshot = index.save("snapshots/k10")  # 请选择尚不存在的目录。
restored = Index.open(snapshot, base="base.fbin", threads=8)
result = restored.search("queries.fbin", k=10, search_list_size=100)
```

快照是一个小型 manifest：图和 sidecar 保存在共享的、按 SHA-256 寻址的对象库中，打开快照时会恢复校准状态。请同时保留 manifest、外部 base 数据和对象库；快照不内嵌 base 向量。Rust 提供相同的 `Index::save` / `Index::open` 工作流。
[持久化指南](orion-runtime/src/guide/cache.md) · [Python 指南](orion-py/README.md)。

## 支持范围

| 能力 | 当前支持 |
|---|---|
| CI 验证的原生平台 | Linux x86-64、macOS Apple Silicon |
| 接口 | Rust `Index`；Python 3.9+ / NumPy；独立 CLI |
| 距离度量 | f32 L2、余弦和内积；原生 u8 L2 |
| 维度 | 运行时维度，自动逐行 padding |
| SIMD | 根据 CPU 能力与行布局，自动选择 scalar、NEON/dot-product、AVX2/FMA 或 AVX-512 |
| 构图 | 默认内置 ParlayANN 内存构图；f32 Rust DiskANN 构图；STAG 导入 |
| 输入 | 连续 Rust 切片、NumPy 数组、`fvecs`、`bvecs`、`fbin`、`u8bin`；CLI 查询流 |
| 持久化 | SHA-256 校验的图与 sidecar 对象，以及已校准快照 |
| 错误处理 | Rust、Python、CLI 共享错误码与详情；Python 将可展开的 panic 转换为异常 |
| 安全验证 | Miri、畸形输入测试、随机压力测试、跨接口一致性检查 |
| 大块内存分配 | 尽力启用 Linux 透明大页建议 |
| 浏览器工具 | 通过 `orion-wasm` 回放 WebAssembly 搜索轨迹 |

运行时 SIMD 分派无需应用手动设置指令集编译选项。面向固定部署环境时，仍可显式指定架构参数，详见[平台支持](docs/platforms.md)。若只需不含 C++ 构图器的精简 f32 Rust 构建，可设置 `default-features = false`。[图来源选择](docs/graph-sources.md)说明了缓存、导入的优先级与显式后端选择规则。

使用问题或可复现的缺陷报告，请提交至 [GitHub Issues](https://github.com/ChanchunhouZEON/Orion/issues)。请尽量附上平台、包版本或提交号、距离度量与维度，以及相关的结构化错误信息。

## CLI

需要直接对文件执行搜索时，可以构建 CLI：

```bash
cargo build --locked --release -p orion-ann-cli --bin orion
./target/release/orion \
  --base /data/base.fbin --query /data/query.fbin --metric l2 \
  --k 10 --search-l 100 --output results.jsonl
```

CLI 会构建或复用图，并使用查询流进行校准。数据集预设和 ground truth 均为可选项。使用 `--snapshot DIR` 恢复校准，使用 `--groundtruth /data/gt.ivecs` 评估召回率，或使用 `--query - --query-format bvecs` 从标准输入读取查询。`--prepare-only` 仅准备工作图缓存，`--preflight` 估算内存需求。详见 [CLI 指南](docs/cli.md)。

## 基准测试与开发

实验程序使用独立的 Cargo workspace，并启用 `benchmark` feature。在仓库根目录执行：

```bash
cargo build --locked --release --target-dir target \
  --manifest-path benchmark/Cargo.toml --bin orion-sweep
./target/release/orion-sweep sift --graph-source parlayann \
  --k 10 --threads 8 --trials 3
```

在 [sweep.yaml](orion-runtime/configs/sweep.yaml) 中配置数据集路径。Sweep 在单个进程中仅加载一次索引，然后完成各个 L 值及重复轮次。复现实验时，请记录硬件、CPU 绑定、图来源、编译选项和计时设置。
[基准测试脚本](benchmark/scripts/README.md) · [测量方法](docs/benchmarks.md)。

```bash
cargo test --locked -p orion-ann -p orion-ann-cli
cargo test --locked -p orion-ann-vector --release
cargo doc --locked -p orion-ann --no-deps --open
```

[架构](docs/architecture.md) · [优化](docs/optimizations.md) · [测试与 CI](docs/testing.md) · [代码结构](docs/code-structure.md) · [消融实验](docs/ablation-study.md) · [历史图表](docs/historical-benchmarks.md)。

[贡献指南](CONTRIBUTION.md) · [变更记录](CHANGLOG.md)。

## 许可证

[MIT](LICENSE)。Copyright Chanchunhou；继承自 Microsoft DiskANN 的文件保留其原有 Microsoft 版权声明。
