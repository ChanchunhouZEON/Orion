<h1 align="center"><img src="orion-core/assets/orion-astraeus.png" alt="" width="64" align="center" /> 獵戶·阿斯特俄斯：面向候選准入嘅圖搜尋</h1>

<p align="center"><a href="README.md" lang="en">English</a> · <a href="README.zh-CN.md" lang="zh-CN">简体中文</a> · <strong lang="yue">粵語</strong></p>

<p align="center">
  <a href=".github/workflows/ci.yml"><img alt="Safety tests: Miri and adversarial inputs" src="https://img.shields.io/badge/safety-Miri_%2B_adversarial_tests-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="orion-runtime/src/guide/cache.md"><img alt="Verified persistence: SHA-256" src="https://img.shields.io/badge/integrity-SHA--256-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md#native-ci-matrix"><img alt="OS targets: Linux and macOS" src="https://img.shields.io/badge/OS-Linux_%7C_macOS-377EAD?style=flat&amp;labelColor=17263D" /></a>
  <a href="docs/platforms.md#native-ci-matrix"><img alt="CPU targets: x86-64 and ARM64" src="https://img.shields.io/badge/CPU-x86__64_%7C_ARM64-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="#使用例子"><img alt="Rust and Python APIs" src="https://img.shields.io/badge/API-Rust_%2B_Python-377EAD?style=flat&amp;labelColor=17263D" /></a>
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
  <a href="https://chanchunhouzeon.github.io/Orion/"><img alt="Homepage: Orion Astraeus" src="https://img.shields.io/badge/homepage-Orion_Astraeus-377EAD?style=flat&amp;labelColor=17263D" /></a>
</p>

<p align="center">
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/ci.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml"><img alt="Coverage check" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/coverage.yml/badge.svg?branch=main" /></a>
  <a href="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml"><img alt="Release validation" src="https://github.com/ChanchunhouZEON/Orion/actions/workflows/release-check.yml/badge.svg?branch=main" /></a>
</p>

**獵戶·阿斯特俄斯（Orion Astraeus，簡稱 Orion）** 係一種 **Vamana 風格嘅圖算法同記憶體近似最近鄰搜尋引擎**，提供 Rust 同 Python 接口。佢嘅准入感知搜尋會追蹤候選節點點樣進入搜尋佇列，喺鄰域逐漸收斂時調整圖遍歷，再透過級聯距離計算，減少花喺低潛力候選節點上嘅運算。

你只需要提供向量：Orion 就會用內置嘅 ParlayANN 建圖，按距離度量同維度揀搜尋方案，並自動選用 CPU 支援嘅 SIMD 後端。用有代表性嘅查詢完成校準之後，就可以透過共享索引搜尋。用程式庫接口唔使預設數據集、預先建好圖、提供 ground truth，亦唔使手動設定 SIMD 編譯選項。

**方便整合到應用程式嘅搜尋功能：** 支援 f32 同原生 u8 輸入、動態維度、並行查詢、結構化錯誤，以及經校驗、可以重用圖同 sidecar 快取物件嘅快照。Sidecar 係索引按需要建立嘅量化等輔助數據。

[效能](#效能) · [使用例子](#使用例子) · [支援範圍](#支援範圍) · [CLI](#cli) · [基準測試同開發](#基準測試同開發)

## 效能

喺 [Orion Astraeus 項目主頁](https://chanchunhouzeon.github.io/Orion/)了解圖搜尋同剪枝機制、admission 診斷，並互動比較 QPS–recall 曲線、cascade 組合同消融實驗。

呢批數據集涵蓋圖像描述子、學習得到嘅視覺特徵、詞向量同段落向量，用嚟檢驗低維同高維搜尋，以及由百萬到十億條向量規模下嘅效能。所有欄位都用 **8 條執行緒、k=10、Recall@10 ≥ 0.95**；每個結果都係達到目標召回率嘅最快實測 QPS 中位數，唔做插值。Wiki-Ada-1M 欄入面嘅括號會標出實際 Recall@10；揀出嚟嘅測量點並唔係同一個召回率。

| 引擎 · QPS | SIFT1M | GloVe-25 | GloVe-100 | GIST | Deep10M | MS-MARCO | SIFT1B | Wiki-Ada-1M<br>QPS (Recall@10) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Orion** | **349K** | **360K** | **48K** | **52K** | **175K** | **59K** | **34K** | **136K** (0.98433) |
| ParlayANN | 257K | 247K | 37K | 29K | 143K | 38K | 33K | 151K (0.95378) |
| Microsoft Vamana (DiskANN) | 91K | 151K | 12K | 7K | 39K | — | 10K | 40K (0.98470) |
| USearch HNSW | 59K | 102K | — | — | 21K | 8K | 未測試 | 未測試 |
| LanceDB HNSW (native) | 1K | 1K | — | — | — | — | 未測試 | 未測試 |
| HNSW (hnswlib) | 51K | 139K | — | — | 21K | 6K | 未測試 | 未測試 |
| FAISS HNSW+SQ | 54K | — | — | — | 15K | 5K | 未測試 | 18K (0.95508) |
| FAISS IVF-Flat | 9K | 33K | 3K | — | 2K | — | 未測試 | 未測試 |
| Annoy | 1K | 3K | — | — | — | — | 未測試 | 未測試 |

SIFT1B 喺 Intel Xeon Platinum 8358P（Linux/AVX-512）上執行；其餘七個數據集就喺 Apple M4 Max（macOS/NEON）上執行。

「未測試」表示冇保留對應嘅測量結果；「—」表示實測曲線未達到目標召回率，或者結果唔可以直接比較。Microsoft Vamana 指 DiskANN 嘅記憶體實作；佢嘅 MS-MARCO 結果因為距離度量唔一致，所以冇納入比較。FAISS IVF-PQ 喺前六個數據集上嘅 recall 都低過 0.62，所以冇列出。**粗體用嚟突出 Orion 嘅結果。**
[實驗方法](docs/benchmarks.md) · [SIFT1B 資源規劃](docs/large-datasets.md)。

## 使用例子

由原始碼建置需要 **Rust 1.92 同 C++17 編譯器**；C++ 編譯器係畀內置嘅圖構建器用。

### Rust：一個依賴，完成建圖同搜尋

假設你嘅應用程式目錄同 `Orion` 原始碼目錄放喺同一層：

```toml
[dependencies]
orion-ann = { path = "../Orion/orion-runtime" }
```

```rust
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};

fn main() -> orion_ann::Result<()> {
    // 將例子數據換成逐行連續儲存嘅向量。
    let base: Vec<f32> = (0..1_024 * 128)
        .map(|i| ((i * 73 + i / 128 * 11) % 251) as f32)
        .collect();
    let queries = &base[..32 * 128];
    let mut index = Index::new(
        BaseInput::from_f32(&base, 128)?,
        IndexOptions::default(),
    )?;

    // 正式使用時，用有代表性嘅應用查詢做校準，唔使提供 ground truth。
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

要讀檔案，可以用 `BaseInput::from_file("base.fbin")`；要輸入原生位元組，可以用 `BaseInput::from_u8(&bytes, dimension)?`。Orion 會喺內部處理逐行 padding。Rust 搜尋會借用查詢數據，並透過 `&self` 存取索引，所以你可以用 `Arc<Index>` 同共享執行緒池處理並行請求。每個要用嘅 `k` 都要校準一次；如果未有對應校準，搜尋會回傳 `InvalidState`。
[更多 Rust 例子](orion-runtime/README.md) · [設定指南](orion-runtime/src/guide/configuration.md)。

### Python：輸入 NumPy 陣列，取得鄰居 ID

喺儲存庫根目錄建立並啟用虛擬環境：

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

需要正規化嘅嵌入向量可以用 `metric="cosine"`；原始內積就用 `metric="ip"`。接口接受 C 連續佈局嘅 `float32` 同 `uint8` 陣列，原生位元組輸入用 L2 度量。Python 接口會喺釋放 GIL 之前複製輸入；並行搜尋會共用已設定好嘅 Rust 執行緒池。

### 讀取檔案，保存已校準嘅快照

```python
index = Index.from_file("base.fbin", metric="l2", cache_dir="cache/build")
index.calibrate("queries.fbin", k=10)
result = index.search("queries.fbin", k=10, search_list_size=100)

snapshot = index.save("snapshots/k10")  # 請揀一個仲未存在嘅目錄。
restored = Index.open(snapshot, base="base.fbin", threads=8)
result = restored.search("queries.fbin", k=10, search_list_size=100)
```

快照係一份細小嘅 manifest：圖同 sidecar 存喺共享、按 SHA-256 定址嘅物件庫入面，開啟快照時會恢復校準狀態。請一齊保留 manifest、外部 base 數據同物件庫；快照本身唔包含 base 向量。Rust 亦提供相同嘅 `Index::save` / `Index::open` 工作流程。
[持久化指南](orion-runtime/src/guide/cache.md) · [Python 指南](orion-py/README.md)。

## 支援範圍

| 能力 | 目前支援 |
|---|---|
| 原生 CI 矩陣 | Linux、macOS × x86-64、ARM64；[驗證狀態](docs/platforms.md#native-ci-matrix) |
| 接口 | Rust `Index`；Python 3.9+ / NumPy；獨立 CLI |
| 距離度量 | f32 L2、餘弦同內積；原生 u8 L2 |
| 維度 | 執行時維度，自動逐行 padding |
| SIMD | 按 CPU 能力同每行嘅佈局，自動揀 scalar、NEON/dot-product、AVX2/FMA 或 AVX-512 |
| 建圖 | 預設內置 ParlayANN 記憶體建圖；f32 Rust DiskANN 建圖；STAG 匯入 |
| 輸入 | 連續 Rust 切片、NumPy 陣列、`fvecs`、`bvecs`、`fbin`、`u8bin`；CLI 查詢串流 |
| 持久化 | 經 SHA-256 校驗嘅圖同 sidecar 物件，以及已校準快照 |
| 錯誤處理 | Rust、Python、CLI 共用錯誤碼同詳細資料；Python 會將可以展開堆疊嘅 panic 轉成例外 |
| 安全驗證 | Miri、無效輸入測試、隨機壓力測試、跨接口一致性檢查 |
| 大塊記憶體分配 | 盡可能啟用 Linux 透明大頁建議 |
| 瀏覽器工具 | 透過 `orion-wasm` 重播 WebAssembly 搜尋軌跡 |

執行時 SIMD 分派唔使應用程式手動設定指令集編譯選項。如果部署環境固定，你仍然可以明確指定架構參數，詳見[平台支援](docs/platforms.md)。如果只需要唔包含 C++ 圖構建器嘅精簡 f32 Rust 建置，可以設定 `default-features = false`。[圖來源選擇](docs/graph-sources.md)解釋咗快取、匯入嘅優先次序，同埋明確選擇後端嘅規則。

有使用問題，或者想報告可以重現嘅缺陷，請到 [GitHub Issues](https://github.com/ChanchunhouZEON/Orion/issues)。請盡量附上平台、套件版本或者 commit、距離度量同維度，以及相關嘅結構化錯誤資料。

## CLI

想直接對檔案執行搜尋，可以建置 CLI：

```bash
cargo build --locked --release -p orion-ann-cli --bin orion
./target/release/orion \
  --base /data/base.fbin --query /data/query.fbin --metric l2 \
  --k 10 --search-l 100 --output results.jsonl
```

CLI 會建立或者重用圖，再用查詢串流做校準。CLI、Rust 同 Python 共用預設配置，ground truth 係可選嘅。用 `--snapshot DIR` 可以恢復校準；用 `--groundtruth /data/gt.ivecs` 可以評估召回率；用 `--query - --query-format bvecs` 就可以由標準輸入讀取查詢。`--prepare-only` 只準備工作圖快取，`--preflight` 就估算記憶體需求。詳見 [CLI 指南](docs/cli.md)。

## 基準測試同開發

實驗程式有獨立嘅 Cargo workspace，並啟用 `benchmark` feature。喺儲存庫根目錄執行：

```bash
cargo build --locked --release --target-dir target \
  --manifest-path benchmark/Cargo.toml --bin orion-sweep
./target/release/orion-sweep sift --graph-source parlayann \
  --k 10 --threads 8 --trials 3
```

喺 [sweep.yaml](benchmark/configs/sweep.yaml) 設定數據集路徑。Sweep 會喺單一程序入面只載入一次索引，再完成各個 L 值同重複輪次。重現實驗時，請記錄硬件、CPU 綁定、圖來源、編譯選項同計時設定。
[基準測試腳本](benchmark/scripts/README.md) · [測量方法](docs/benchmarks.md)。

```bash
cargo test --locked -p orion-ann -p orion-ann-cli
cargo test --locked -p orion-ann-vector --release
cargo doc --locked -p orion-ann --no-deps --open
```

[架構](docs/architecture.md) · [優化](docs/optimizations.md) · [測試同 CI](docs/testing.md) · [程式碼結構](docs/code-structure.md) · [消融實驗](docs/ablation-study.md) · [歷史圖表](docs/historical-benchmarks.md)。

[貢獻指南](CONTRIBUTION.md) · [變更記錄](CHANGELOG.md)。

## 授權條款

[MIT](LICENSE)。Copyright Chanchunhou；繼承自 Microsoft DiskANN 嘅檔案保留原有嘅 Microsoft 版權聲明。
