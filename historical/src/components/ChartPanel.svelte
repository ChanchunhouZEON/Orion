<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import type { AtlasState } from "../app/state.svelte";
  import type { Series, PlotOptions } from "../types";
  import Chart from "./Chart.svelte";
  import { fmt } from "../model/presentation";
  let {
    atlas,
    title,
    subtitle,
    series,
    scale,
    plot = {},
    controls,
    children,
  }: {
    atlas: AtlasState;
    title: string;
    subtitle: string;
    series: Series[];
    scale?: "log" | "speedup";
    plot?: PlotOptions;
    controls?: Snippet;
    children?: Snippet;
  } = $props();
</script>

<section class="chart-card" aria-label="Performance chart">
  <div class="chart-header">
    <div>
      <h3>{title}</h3>
      <p>{subtitle}</p>
    </div>
    {#if scale}<label class="compact-toggle"
        ><input
          type="checkbox"
          checked={scale === "speedup" ? atlas.speedup : atlas.logarithmic}
          onchange={(e) => {
            if (scale === "speedup") atlas.speedup = e.currentTarget.checked;
            else atlas.logarithmic = e.currentTarget.checked;
          }}
        />{scale === "speedup" ? "Relative speedup" : "Log scale"}</label
      >{/if}
  </div>
  {#if controls}{@render controls()}{/if}
  {#if children}{@render children()}
  {:else}
    <Chart
      {series}
      {...plot}
      onpoint={(s, p) =>
        (atlas.detail = `${s.label} · ${p.detail || fmt(p.y)}`)}
    />
    <div class="legend">
      {#each series as s}<span
          ><i style={`background:${s.color}`}></i>{s.label}</span
        >{/each}
    </div>
    {#if atlas.detail}<p class="point-detail" aria-live="polite">
        {atlas.detail}
      </p>{/if}
  {/if}
</section>
