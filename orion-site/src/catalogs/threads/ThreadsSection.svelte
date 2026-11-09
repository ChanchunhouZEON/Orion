<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { AtlasState } from "../../app/state.svelte";
  import type { Series } from "../../types";
  import EnginePanel from "../../components/EnginePanel.svelte";
  import ChartPanel from "../../components/ChartPanel.svelte";
  import ProtocolNotes from "../../components/ProtocolNotes.svelte";
  import { fmt, engineName, engineColor } from "../../model/presentation";
  let { atlas }: { atlas: AtlasState } = $props();
  let profile = $derived(atlas.dataset.threadScaling!);
  let series: Series[] = $derived(
    profile.series
      .filter((s) => !atlas.hidden.includes(s.engine))
      .map((s) => ({
        id: s.engine,
        engine: s.engine,
        label: engineName(atlas.catalog, s.engine),
        color: engineColor(atlas.catalog, s.engine),
        points: s.points.map((p) => ({
          x: p.threads,
          y: atlas.speedup ? p.qps / s.points[0].qps : p.qps,
          detail: `${p.threads} threads · ${fmt(p.qps)} QPS · ${fmt(p.qps / s.points[0].qps, 2)}×`,
        })),
      })),
  );
</script>

<div class="workspace">
  <EnginePanel
    {atlas}
    engines={profile.series.map((s) => s.engine)}
    count={series.length}
  />
  <ChartPanel
    {atlas}
    {series}
    title="Thread scalability"
    subtitle={`k = ${profile.k} · L = ${profile.L} · fixed search list`}
    scale="speedup"
    plot={{
      logarithmic: false,
      xMinimum: 1,
      target: null,
      xLabel: "Threads",
      yLabel: atlas.speedup ? "Speedup vs. one thread" : "Queries / second ↑",
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
          ><th>Threads</th>{#each series as s}<th
              >{s.label} {atlas.speedup ? "speedup" : "QPS"}</th
            >{/each}</tr
        ></thead
      >
      <tbody
        >{#each profile.series[0].points as p, i}<tr
            ><td>{p.threads}</td>{#each series as s}<td
                class:orion-row={s.engine === "orion"}
                >{fmt(s.points[i].y, atlas.speedup ? 2 : 0)}{atlas.speedup
                  ? "×"
                  : ""}</td
              >{/each}</tr
          >{/each}</tbody
      >
    </table>
  </div>
</section>
<ProtocolNotes catalog={atlas.catalog} studies={[profile.study]} />
