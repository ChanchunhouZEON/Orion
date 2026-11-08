<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../../app/state.svelte";
  import type { SearchCurve } from "../../types";
  import EnginePanel from "../../components/EnginePanel.svelte";
  import ChartPanel from "../../components/ChartPanel.svelte";
  import SearchResults from "../../components/SearchResults.svelte";
  import ProtocolNotes from "../../components/ProtocolNotes.svelte";
  import CascadeBuilder from "./CascadeBuilder.svelte";
  import { searchCurves } from "../../model/selection.mjs";
  import {
    engineName,
    engineColor,
    cascadeColor,
    searchSeries,
    comparisonStudies,
  } from "../../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let dataset = $derived(atlas.dataset);
  let curves: SearchCurve[] = $derived(
    searchCurves(dataset, "cascades", atlas.cascadeIds),
  );
  let shown = $derived(curves.filter((c) => !atlas.hidden.includes(c.engine)));
  const color = (id: string) => cascadeColor(atlas.catalog, atlas.dataset, id);
  let series = $derived(
    searchSeries(
      shown,
      (c) =>
        c.engine === "orion" ? c.label : engineName(atlas.catalog, c.engine),
      (c) =>
        c.engine === "orion"
          ? color(c.id)
          : engineColor(atlas.catalog, c.engine),
    ),
  );
  let outsideZoom = $derived(
    shown.filter(
      (c) => !c.points.some((p) => p.recall >= atlas.cascadeMinimum),
    ),
  );
</script>

<div class="workspace cascade-workspace">
  <EnginePanel
    {atlas}
    engines={[...new Set(["orion", ...curves.map((c) => c.engine)])]}
    count={series.length}
  >
    {#snippet orionControls()}{#key dataset.id}<CascadeBuilder
          {dataset}
          selectedIds={atlas.cascadeIds}
          {color}
          onchange={(ids) => atlas.updateCascades(ids)}
        />{/key}{/snippet}
  </EnginePanel>
  <ChartPanel
    {atlas}
    {series}
    title="Throughput & recall"
    subtitle="8 threads · k = 10"
    scale="log"
    plot={{
      logarithmic: atlas.logarithmic,
      target: atlas.target,
      xMinimum: atlas.cascadeMinimum,
      frontier: !atlas.allCascadePoints,
      scatterOnly: atlas.allCascadePoints,
      yLabel: "Queries / second ↑",
    }}
  >
    {#snippet controls()}
      <div class="chart-range-controls">
        <div class="segmented" aria-label="Cascade recall range">
          {#each [0.9, 0.95, 0.98, 0] as r}<button
              class:active={atlas.cascadeMinimum === r}
              aria-pressed={atlas.cascadeMinimum === r}
              onclick={() => (atlas.cascadeMinimum = r)}
              >{r === 0 ? "Full range" : `${r.toFixed(2)}–1.00`}</button
            >{/each}
        </div>
        <label class="compact-toggle"
          ><input type="checkbox" bind:checked={atlas.allCascadePoints} /> All measured
          points</label
        >
      </div>
      <p class="chart-caption">
        {atlas.allCascadePoints
          ? "Every measured point; no connecting lines."
          : "Measured Pareto frontier: points that are not beaten in both recall and QPS."}
        The QPS axis fits the visible recall range.
      </p>
      {#if shown.some((c) => c.engine === "orion" && c.id.includes("__pq-"))}<p
          class="chart-caption"
        >
          PQ + exact reranks at most 20 candidates (k=10); increasing L does not
          increase this budget. Exact refers to rerank precision, not exhaustive
          search.
        </p>{/if}
      {#if outsideZoom.length}<details class="range-note">
          <summary
            >{outsideZoom.length} selected {outsideZoom.length === 1
              ? "curve does"
              : "curves do"} not reach this range</summary
          >{#each outsideZoom as c}<p>
              {c.engine === "orion"
                ? c.label
                : engineName(atlas.catalog, c.engine)} · max recall {Math.max(
                ...c.points.map((p) => p.recall),
              ).toFixed(5)}
            </p>{/each}
        </details>{/if}
    {/snippet}
  </ChartPanel>
</div>
<SearchResults {atlas} curves={shown} configurationLabels />
<ProtocolNotes catalog={atlas.catalog} studies={comparisonStudies(dataset)} />
