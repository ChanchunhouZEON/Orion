<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../../app/state.svelte";
  import type { SearchCurve } from "../../types";
  import EnginePanel from "../../components/EnginePanel.svelte";
  import ChartPanel from "../../components/ChartPanel.svelte";
  import SearchResults from "../../components/SearchResults.svelte";
  import ProtocolNotes from "../../components/ProtocolNotes.svelte";
  import { searchCurves, recipeLabel } from "../../model/selection.mjs";
  import {
    engineName,
    engineColor,
    searchSeries,
    comparisonStudies,
  } from "../../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let dataset = $derived(atlas.dataset);
  let curves: SearchCurve[] = $derived(searchCurves(dataset, "default"));
  let shown = $derived(curves.filter((c) => !atlas.hidden.includes(c.engine)));
  let series = $derived(
    searchSeries(
      shown,
      (c) => engineName(atlas.catalog, c.engine),
      (c) => engineColor(atlas.catalog, c.engine),
    ),
  );
</script>

<div class="workspace">
  <EnginePanel
    {atlas}
    engines={curves.map((c) => c.engine)}
    count={series.length}
  >
    {#snippet orionControls()}<p class="small-note">
        {dataset.defaultRecipe
          ? recipeLabel(
              dataset.cascades.find((c) => c.id === dataset.defaultRecipe),
            )
          : "Native u8 · exact admission"}
      </p>{/snippet}
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
      xMinimum: 0.5,
      yLabel: "Queries / second ↑",
    }}
  />
</div>
<SearchResults {atlas} curves={shown} />
<ProtocolNotes catalog={atlas.catalog} studies={comparisonStudies(dataset)} />
