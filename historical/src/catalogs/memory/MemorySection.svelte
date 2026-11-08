<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../../app/state.svelte";
  import type { Series } from "../../types";
  import EnginePanel from "../../components/EnginePanel.svelte";
  import ChartPanel from "../../components/ChartPanel.svelte";
  import ProtocolNotes from "../../components/ProtocolNotes.svelte";
  import { fmt, engineName, engineColor } from "../../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let profile = $derived(atlas.dataset.memory!);
  let series: Series[] = $derived(
    (["orion", "diskann"] as const)
      .filter((e) => !atlas.hidden.includes(e))
      .map((e) => ({
        id: e,
        engine: e,
        label: engineName(atlas.catalog, e),
        color: engineColor(atlas.catalog, e),
        points: [{ x: 1, y: profile[e].peakBytes / 2 ** 30 }],
      })),
  );
</script>

<div class="workspace">
  <EnginePanel
    {atlas}
    mark="bar"
    engines={["orion", "diskann"]}
    count={series.length}
  />
  <ChartPanel
    {atlas}
    {series}
    title="Index allocation"
    subtitle="Peak index bytes above loaded inputs"
    plot={{
      mode: "bars",
      logarithmic: false,
      xMinimum: 0,
      target: null,
      yLabel: "Peak allocation (GiB) ↓",
      inlineLabel: null,
    }}
  />
</div>
<section class="results-card">
  <div class="results-heading">
    <div>
      <h3>Measurements</h3>
      <p>Retained measurements from this study.</p>
    </div>
  </div>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>Engine</th><th>Peak index allocation (GiB) ↓</th><th
            >Retained index allocation (GiB)</th
          ></tr
        ></thead
      >
      <tbody
        >{#each series as s}<tr class:orion-row={s.engine === "orion"}
            ><td>{s.label}</td><td>{fmt(s.points[0].y, 3)}</td><td
              >{s.engine === "orion"
                ? fmt(profile.orion.retainedBytes / 2 ** 30, 3)
                : "Not measured"}</td
            ></tr
          >{/each}</tbody
      >
    </table>
  </div>
</section>
<ProtocolNotes catalog={atlas.catalog} studies={[profile.study]} />
