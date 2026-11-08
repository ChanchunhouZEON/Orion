<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Ablation } from "../../types";
  import { ablationRows, ablationColor } from "../../model/selection.mjs";
  let { study }: { study: Ablation } = $props();
  let metric = $state<"ndc" | "ndcReduction" | "qpsChange" | "recallChange">(
    "ndc",
  );
  let width = $state(800);
  const height = 350,
    pad = { left: 65, right: 20, top: 35, bottom: 48 };
  const titles = {
    ndc: "Stage distance evaluations / query ↓",
    ndcReduction: "NDC reduction vs. Full neighbors (%) ↑",
    qpsChange: "QPS change vs. Full neighbors (%) ↑",
    recallChange: "Recall change vs. Full neighbors (pp) ↑",
  };
  const short = {
    ndc: "NDC / query",
    ndcReduction: "NDC reduction",
    qpsChange: "QPS change",
    recallChange: "Recall change",
  };
  type Row = {
    id: string;
    label: string;
    L: number;
    ndc: number;
    qps: number;
    recall: number;
    ndcReduction: number;
    qpsChange: number;
    recallChange: number;
  };
  let rows: Row[] = $derived(ablationRows(study));
  let widths = $derived(study.barWidths || []);
  let references = $derived(
    rows.filter((r) => r.id === study.referenceVariant),
  );
  let plottedVariants = $derived(
    metric === "ndc"
      ? study.variants
      : study.variants.filter((v) => v.id !== study.referenceVariant),
  );
  let low = $derived(Math.min(0, ...rows.map((r) => r[metric])) * 1.18);
  let high = $derived(Math.max(0, ...rows.map((r) => r[metric])) * 1.18);
  let span = $derived(Math.max(high - low, 0.01));
  let cell = $derived(
    (width - pad.left - pad.right) / Math.max(widths.length, 1),
  );
  const y = (v: number) =>
    height - pad.bottom - ((v - low) / span) * (height - pad.top - pad.bottom);
  const fmt = (n: number, d = 1) =>
    n.toLocaleString("en-US", { maximumFractionDigits: d });
  const signed = (n: number) => `${n > 0 ? "+" : ""}${fmt(n, 2)}`;
</script>

<div class="metric-controls">
  {#each Object.entries(short) as [key, label]}<button
      class:active={metric === key}
      aria-pressed={metric === key}
      onclick={() => (metric = key as typeof metric)}>{label}</button
    >{/each}
</div>
<p class="chart-caption">
  Same L, not matched recall. Changes are relative to Full neighbors (adaptive
  and early stop off).
</p>
<section
  class="reference-panel"
  aria-label="Full neighbors reference measurements"
>
  <h4>Full neighbors · absolute reference</h4>
  <p>
    Adaptive off · early stop off. Each percentage uses the reference at the
    same L.
  </p>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>Beam width L</th>{#each references as r}<th>{r.L}</th>{/each}</tr
        ></thead
      >
      <tbody>
        <tr
          ><td>NDC / query</td>{#each references as r}<td>{fmt(r.ndc, 2)}</td
            >{/each}</tr
        >
        <tr
          ><td>QPS</td>{#each references as r}<td>{fmt(r.qps, 0)}</td
            >{/each}</tr
        >
        <tr
          ><td>Recall@10</td>{#each references as r}<td
              >{r.recall.toFixed(5)}</td
            >{/each}</tr
        >
      </tbody>
    </table>
  </div>
  {#if metric !== "ndc"}<p class="reference-formula">
      {metric === "ndcReduction"
        ? "Reduction = 100 × (1 − NDC / reference NDC)"
        : metric === "qpsChange"
          ? "Change = 100 × (QPS / reference QPS − 1)"
          : "Change (pp) = 100 × (recall − reference recall)"}. Full neighbors
      is the zero line.
    </p>{/if}
</section>
<div class="bar-scroll">
  <div class="grouped-bars plot" bind:clientWidth={width}>
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={titles[metric]}
    >
      <title>{titles[metric]} at the same beam width</title>
      <text x={pad.left} y="15" class="axis-title">{titles[metric]}</text>
      {#each [0, 1, 2, 3, 4] as i}{@const value = low + (span * i) / 4}
        <line
          class="gridline"
          x1={pad.left}
          x2={width - pad.right}
          y1={y(value)}
          y2={y(value)}
        />
        <text x={pad.left - 9} y={y(value) + 4} text-anchor="end"
          >{fmt(value)}</text
        >
      {/each}
      <line
        x1={pad.left}
        x2={width - pad.right}
        y1={y(0)}
        y2={y(0)}
        stroke="#94a3b8"
        stroke-width="1"
      />
      {#each widths as L, li}
        <text
          x={pad.left + (li + 0.5) * cell}
          y={height - 24}
          text-anchor="middle">{L}</text
        >
        {#each plottedVariants as variant, vi}
          {@const row = rows.find((r) => r.id === variant.id && r.L === L)}
          {#if row}
            {@const value = row[metric]}
            {@const bw = (cell * 0.78) / plottedVariants.length}
            <rect
              x={pad.left + li * cell + cell * 0.11 + vi * bw}
              y={Math.min(y(value), y(0))}
              width={Math.max(1, bw - 2)}
              height={Math.max(0.8, Math.abs(y(value) - y(0)))}
              fill={ablationColor(study, variant.id)}
              rx="2"
            >
              <title
                >{variant.label} · L={L} · {short[metric]}: {fmt(value, 3)} · recall
                {row.recall.toFixed(5)}</title
              >
            </rect>
          {/if}
        {/each}
      {/each}
      <text
        x={(width + pad.left - pad.right) / 2}
        y={height - 3}
        text-anchor="middle"
        class="axis-title">Beam width L</text
      >
    </svg>
  </div>
</div>
<div class="legend">
  {#each plottedVariants as variant, i}<span
      ><i style={`background:${ablationColor(study, variant.id)}`}
      ></i>{variant.label}</span
    >{/each}
</div>
<p class="chart-caption">
  Default = adaptive + early stop. NDC sums evaluations across stages; each
  stage has a different cost.
</p>
<details class="metric-table" open>
  <summary>Exact values & changes</summary>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>L</th><th>Variant</th><th>NDC / query</th><th>NDC reduction</th
          ><th>QPS</th><th>QPS Δ</th><th>Recall</th><th>Recall Δ (pp)</th></tr
        ></thead
      >
      <tbody
        >{#each rows as row}<tr
            class:orion-row={row.id === study.defaultVariant}
            ><td>{row.L}</td><td>{row.label}</td><td>{fmt(row.ndc, 2)}</td><td
              >{signed(row.ndcReduction)}%</td
            ><td>{fmt(row.qps, 0)}</td><td>{signed(row.qpsChange)}%</td><td
              >{row.recall.toFixed(5)}</td
            ><td>{signed(row.recallChange)}</td></tr
          >{/each}</tbody
      >
    </table>
  </div>
</details>
