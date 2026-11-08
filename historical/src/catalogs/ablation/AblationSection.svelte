<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../../app/state.svelte";
  import type { SearchCurve } from "../../types";
  import EnginePanel from "../../components/EnginePanel.svelte";
  import ChartPanel from "../../components/ChartPanel.svelte";
  import SearchResults from "../../components/SearchResults.svelte";
  import ProtocolNotes from "../../components/ProtocolNotes.svelte";
  import AblationMetrics from "./AblationMetrics.svelte";
  import { searchCurves, ablationColor } from "../../model/selection.mjs";
  import { searchSeries } from "../../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let study = $derived(atlas.study);
  let barView = $derived(
    !!study.referenceVariant && atlas.ablationView === "bars",
  );
  let curves: SearchCurve[] = $derived(
    searchCurves(
      atlas.dataset,
      "ablation",
      [],
      atlas.studyId,
      atlas.armId,
      atlas.compare,
    ),
  );
  let shown = $derived(curves.filter((c) => !atlas.hidden.includes(c.engine)));
  let series = $derived(
    searchSeries(
      shown,
      (c) => c.label,
      (c) => ablationColor(study, c.id),
    ),
  );
</script>

{#snippet viewControls()}
  {#if study.referenceVariant}
    <div class="segmented ablation-view" aria-label="Ablation display">
      <button
        class:active={barView}
        onclick={() => (atlas.ablationView = "bars")}>NDC & changes</button
      >
      <button
        class:active={!barView}
        onclick={() => (atlas.ablationView = "curves")}>QPS vs. recall</button
      >
    </div>
  {/if}
{/snippet}
<div class="workspace">
  <EnginePanel
    {atlas}
    engines={["orion"]}
    count={barView ? study.variants.length : series.length}
    variants={barView}
    hideVisibility={barView}
  >
    {#snippet orionControls()}
      <label class="field-label" for="ablation-study">Study</label>
      <select
        id="ablation-study"
        value={study.id}
        onchange={(e) => atlas.selectStudy(e.currentTarget.value)}
      >
        {#each atlas.dataset.ablations as a}<option value={a.id}
            >{a.label}</option
          >{/each}
      </select>
      {#if !barView}
        <label class="field-label" for="ablation-arm">Variant</label>
        <select
          id="ablation-arm"
          bind:value={atlas.armId}
          disabled={atlas.compare}
        >
          {#each study.variants as a}<option value={a.id}>{a.label}</option
            >{/each}
        </select>
        <label class="visible-toggle compare-toggle"
          ><input
            id="compare-ablations"
            type="checkbox"
            bind:checked={atlas.compare}
          /> Compare all variants</label
        >
      {/if}
    {/snippet}
  </EnginePanel>
  {#if barView}
    <ChartPanel
      {atlas}
      {series}
      title="Adaptive & early stop"
      subtitle="8 threads · k = 10 · same-L comparison"
      controls={viewControls}
    >
      <AblationMetrics {study} />
    </ChartPanel>
  {:else}
    <ChartPanel
      {atlas}
      {series}
      title="Throughput & recall"
      subtitle="8 threads · k = 10"
      scale="log"
      controls={viewControls}
      plot={{
        logarithmic: atlas.logarithmic,
        target: atlas.target,
        xMinimum: 0.5,
        inlineLabel:
          shown.length === 1 && shown[0].id === study.defaultVariant
            ? "Default"
            : null,
      }}
    />
  {/if}
</div>
{#if !barView}<SearchResults {atlas} curves={shown} configurationLabels />{/if}
<ProtocolNotes catalog={atlas.catalog} studies={[study.study]} />
