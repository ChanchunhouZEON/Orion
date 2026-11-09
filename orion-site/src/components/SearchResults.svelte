<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../app/state.svelte";
  import type { SearchCurve } from "../types";
  import { bestMeasured } from "../model/selection.mjs";
  import { engineName, fmt } from "../model/presentation";
  let {
    atlas,
    curves,
    configurationLabels = false,
  }: {
    atlas: AtlasState;
    curves: SearchCurve[];
    configurationLabels?: boolean;
  } = $props();
</script>

<section class="results-card">
  <div class="results-heading">
    <div>
      <h3>At your recall target</h3>
      <p>Fastest measured point meeting the target. No interpolation.</p>
    </div>
    <div class="segmented" aria-label="Recall target">
      {#each [0.95, 0.98, 0.99] as t}<button
          class:active={atlas.target === t}
          aria-pressed={atlas.target === t}
          onclick={() => (atlas.target = t)}>≥ {t.toFixed(2)}</button
        >{/each}
    </div>
  </div>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>Engine / variant</th><th>QPS ↑</th><th>Recall@10</th><th
            >Search setting</th
          ></tr
        ></thead
      >
      <tbody
        >{#each curves as c}{@const p = bestMeasured(c.points, atlas.target)}<tr
            class:orion-row={c.engine === "orion"}
            ><td
              >{configurationLabels && c.engine === "orion"
                ? `Orion · ${c.label}`
                : engineName(atlas.catalog, c.engine)}</td
            ><td>{p ? fmt(p.qps) : "Not reached"}</td><td
              >{p ? p.recall.toFixed(5) : "—"}</td
            ><td class="search-setting"
              >{p?.L !== undefined
                ? `L = ${p.L}`
                : p?.Q !== undefined
                  ? `Q = ${p.Q}`
                  : p?.setting || "—"}</td
            ></tr
          >{/each}</tbody
      >
    </table>
  </div>
</section>
