# Historical benchmark figures

[README](../README.md) · [Current dated summaries](benchmarks.md)

The figures and visualization datasets previously embedded in the README remain
in place and unchanged. They document earlier benchmark states. They have not
been regenerated to incorporate the October 2026 updates summarized in the new
benchmark guide. Older README text is retained in Git history.

| Dataset | Existing figure | Existing data |
|---|---|---|
| `sift` | [Figure](../visualizations/qps_recall_sift_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_sift.json) |
| `glove25` | [Figure](../visualizations/qps_recall_glove25_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_glove25.json) |
| `glove100` | [Figure](../visualizations/qps_recall_glove100_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_glove100.json) |
| `gist` | [Figure](../visualizations/qps_recall_gist_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_gist.json) |
| `deep10m` | [Figure](../visualizations/qps_recall_deep10m_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_deep10m.json) |
| `msmarco_bert_1M` | [Figure](../visualizations/qps_recall_msmarco_bert_1M_all3.png) | [Data](../visualizations/sweep_orion_vs_parlayann_msmarco_bert_1M.json) |

Additional build, memory, convergence and thread-scaling artifacts remain under
[visualizations](../visualizations/); cascade and adaptive-search studies are
indexed in [Ablation study](ablation-study.md).

Historical claims of identical topology from equal R/L/alpha, identical cold
cache state from eviction buffers, static test counts/coverage, and unqualified
panel-wide acceleration are not carried forward as current guarantees. Use the
protocol and provenance accompanying each study. In particular, do not mix
historical tie-aware PA recall with strict ID recall without stating the metric.
