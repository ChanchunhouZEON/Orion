<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { ticks, type Point, type Quantile } from "./model";
  let {
    series,
    quantile,
    minRecall = 0.9,
    targetRecall = 0.95,
  }: {
    series: { rows: Point[]; name: string; color: string; selected?: Point }[];
    quantile: Quantile;
    minRecall?: number;
    targetRecall?: number;
  } = $props();
  // Each percentile has its own log scale. Hidden/out-of-window points must
  // not influence the scale of any panel, and a control point stays unjoined.
  let visible = $derived(
    series.map((s) => ({
      ...s,
      rows: s.rows.filter((p) => p.recall >= minRecall),
    })),
  );
  let rows = $derived(visible.flatMap((s) => s.rows));
  let axis = $derived(ticks(rows, quantile));
  let lo = $derived(Math.log10(axis[0]));
  let hi = $derived(Math.log10(axis.at(-1)!));
  const left = 58,
    right = 344,
    top = 14,
    bottom = 220;
  const x = (recall: number) =>
    left + ((recall - minRecall) / (1 - minRecall)) * (right - left);
  const y = (us: number) =>
    bottom - ((Math.log10(us) - lo) / (hi - lo)) * (bottom - top);
  const path = (points: Point[]) =>
    points.map((p) => `${x(p.recall)},${y(p[quantile])}`).join(" ");
  const format = (us: number) =>
    us < 1000
      ? `${us.toLocaleString("en-US")} µs`
      : `${(us / 1000).toLocaleString("en-US")} ms`;
  let xTicks = $derived(
    minRecall === 0.95
      ? [0.95, 0.96, 0.97, 0.98, 0.99, 1]
      : [0.9, 0.92, 0.94, 0.96, 0.98, 1],
  );
</script>

<svg
  viewBox="0 0 360 265"
  role="img"
  aria-label={`${quantile.slice(0, 3).toUpperCase()} request latency versus recall`}
>
  {#if rows.length}
    {#each axis as tick}
      <line x1={left} x2={right} y1={y(tick)} y2={y(tick)} stroke="#e2e8f0" />
      <text x={left - 8} y={y(tick) + 3.5} text-anchor="end"
        >{format(tick)}</text
      >
    {/each}
    {#each xTicks as tick}<text x={x(tick)} y={bottom + 19} text-anchor="middle"
        >{tick.toFixed(2)}</text
      >{/each}
    <text x={(left + right) / 2} y="260" text-anchor="middle">Recall@10</text>
    {#each visible as s}
      {#if s.rows.length > 1 && s.rows[0].source !== "native-control"}
        <polyline
          points={path(s.rows)}
          fill="none"
          stroke={s.color}
          stroke-width={s.name === "Orion" ? 2.5 : 1.8}
        />
      {/if}
      {#each s.rows as p}
        <circle
          cx={x(p.recall)}
          cy={y(p[quantile])}
          r={p.source === "native-control" ? 4.5 : 2.5}
          fill={s.color}
          stroke="white"
          stroke-width=".7"
        >
          <title
            >{s.name} · recall {p.recall.toFixed(5)} · effort {p.effort} · {quantile
              .slice(0, 3)
              .toUpperCase()}
            {p[quantile].toFixed(2)} µs · {p.requests.toLocaleString("en-US")} requests</title
          >
        </circle>
      {/each}
    {/each}
    <!-- Draw selections last so another engine's curve cannot hide a ring.
         Coordinates use the same measured point and transform as the curve. -->
    {#each visible as s}
      {#if s.selected && s.rows.includes(s.selected)}
        {@const p = s.selected}
        <g
          data-selected-engine={s.name}
          data-selected-recall={p.recall}
          data-selected-value={p[quantile]}
        >
          <title
            >{s.name} · table selection for recall ≥ {targetRecall.toFixed(2)} · actual
            recall {p.recall.toFixed(5)} · effort {p.effort} · {quantile
              .slice(0, 3)
              .toUpperCase()}
            {p[quantile].toFixed(2)} µs</title
          >
          <circle cx={x(p.recall)} cy={y(p[quantile])} r="7.5" fill="white" />
          <circle
            cx={x(p.recall)}
            cy={y(p[quantile])}
            r="6"
            fill="white"
            stroke={s.color}
            stroke-width="2"
          />
          <circle cx={x(p.recall)} cy={y(p[quantile])} r="2.5" fill={s.color} />
        </g>
      {/if}
    {/each}
  {:else}
    <text x="180" y="125" text-anchor="middle"
      >No qualified measurements in this range</text
    >
  {/if}
</svg>

<style>
  svg {
    display: block;
    width: 100%;
    font: 10px var(--font-sans);
    fill: #52647c;
    font-variant-numeric: tabular-nums;
    overflow: visible;
  }
</style>
