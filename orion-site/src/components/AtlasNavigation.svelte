<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { sections, type AtlasState } from "../app/state.svelte";
  import { supports } from "../model/selection.mjs";
  import { fmt } from "../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let dataset = $derived(atlas.dataset);
</script>

<nav class="sections" aria-label="Experiment section">
  {#each sections as s}<button
      class:active={atlas.section === s.id}
      aria-pressed={atlas.section === s.id}
      onclick={() => atlas.selectSection(s.id)}>{s.label}</button
    >{/each}
</nav>
<div class="dataset-strip" aria-label="Dataset">
  {#each atlas.datasets as d}<button
      class:active={d.id === atlas.datasetId}
      disabled={!supports(d, atlas.section)}
      title={!supports(d, atlas.section)
        ? "Not measured in this study"
        : `${fmt(d.count)} vectors · ${d.dimension} dimensions`}
      aria-pressed={d.id === atlas.datasetId}
      onclick={() => atlas.selectDataset(d.id)}>{d.label}</button
    >{/each}
</div>
<div class="dataset-heading">
  <div>
    <h2>{dataset.label}</h2>
    <p>
      {fmt(dataset.count)} vectors <span>·</span>
      {dataset.dimension} dimensions <span>·</span>
      {dataset.hardware}
    </p>
  </div>
  <a
    class="download"
    href={`${import.meta.env.BASE_URL}data/${dataset.id}.json`}
    download>Dataset JSON ↓</a
  >
</div>
