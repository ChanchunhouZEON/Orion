<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import data from "../../public/data/request-latency.json";
  import LatencyChart from "./LatencyChart.svelte";
  import LatencySummary from "./LatencySummary.svelte";
  import LatencySelectionNote from "./LatencySelectionNote.svelte";
  import {
    curve,
    select,
    DEFAULT_ENGINES,
    QUANTILES,
    toggleEngine,
    type Point,
  } from "./model";
  const points = data.points as Point[];
  const names: Record<string, string> = data.engines;
  const datasets: Record<string, string> = data.datasets;
  const colors: Record<string, string> = data.colors;
  let dataset = $state("sift");
  let concurrency = $state(8);
  let target = $state(0.95);
  let zoom = $state(false);
  let enabled = $state([...DEFAULT_ENGINES]);
  let minRecall = $derived(zoom ? 0.95 : 0.9);
  // Stable catalog order keeps legend, panels and table aligned after toggling.
  let engines = $derived(Object.keys(names).filter((e) => enabled.includes(e)));
  let legend = $derived(
    Object.keys(names).filter(
      (e) => DEFAULT_ENGINES.includes(e) || enabled.includes(e),
    ),
  );
  let series = $derived(
    engines.map((e) => ({
      name: names[e],
      color: colors[e],
      rows: curve(points, dataset, e, concurrency, minRecall),
      selected: select(points, dataset, e, concurrency, target),
    })),
  );
  let outsideZoom = $derived(
    series.filter((s) => s.selected && s.selected.recall < minRecall),
  );
  let reference = $derived(
    select(points, dataset, "orion", concurrency, target),
  );
  let missing = $derived(
    engines.filter(
      (e) => !curve(points, dataset, e, concurrency, minRecall).length,
    ),
  );
  let excluded = $derived(
    data.excluded.filter(
      (p) =>
        p.dataset === dataset &&
        p.concurrency === concurrency &&
        engines.includes(p.engine),
    ),
  );
  const toggle = (engine: string) => {
    enabled = toggleEngine(enabled, engine);
  };
  function absence(engine: string): string {
    const job = data.coverage.find(
      (j) => j.dataset === dataset && j.engine === engine,
    );
    if (job?.status === "failed") return "result-ID consistency check failed";
    if (job?.status === "unsupported")
      return "base does not satisfy the unit-norm requirement";
    if (engine === "usearch-native")
      return "no control point in this dataset / recall range";
    return "no sampling-qualified point in this recall range";
  }
  const triples = (p: { p50_us: number; p95_us: number; p99_us: number }) =>
    [p.p50_us, p.p95_us, p.p99_us]
      .map((v) => v.toLocaleString("en-US", { maximumFractionDigits: 2 }))
      .join(" / ");
</script>

<section id="latency" class="home-section" aria-labelledby="latency-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">REQUEST LATENCY</p>
      <h2 id="latency-title">From throughput to response time.</h2>
    </div>
    <p>
      Compare the median and the tail together. One query per request, at 1 or 8
      concurrent clients.
    </p>
  </div>
  <div class="latency-card">
    <div class="controls">
      <label
        >Dataset<select bind:value={dataset}
          >{#each Object.entries(datasets) as [id, name]}<option value={id}
              >{name}</option
            >{/each}</select
        ></label
      >
      <label
        >Concurrent clients<select bind:value={concurrency}
          ><option value={1}>1 client</option><option value={8}
            >8 clients</option
          ></select
        ></label
      >
      <label class="zoom"
        ><input type="checkbox" bind:checked={zoom} /> High recall · 0.95–1.00</label
      >
      <details class="more-engines">
        <summary>More engines</summary>
        <div class="engine-options">
          {#each Object.entries(names).filter(([id]) => !DEFAULT_ENGINES.includes(id)) as [id, name]}
            <label
              ><input
                type="checkbox"
                checked={enabled.includes(id)}
                onchange={() => toggle(id)}
              />{name}</label
            >
          {/each}
        </div>
      </details>
    </div>
    <div class="plot-title">
      <h3>{datasets[dataset]}</h3>
      <span>Apple M4 Max · macOS · k=10</span>
    </div>
    <div class="shared-legend" role="group" aria-label="Latency engines">
      {#each legend as engine}<button
          type="button"
          aria-pressed={enabled.includes(engine)}
          onclick={() => toggle(engine)}
          class:inactive={!enabled.includes(engine)}
          style:--engine-color={colors[engine]}
          ><i aria-hidden="true"></i>{names[engine]}</button
        >{/each}
    </div>
    <div class="triptych">
      {#each QUANTILES as quantile}<div class="percentile-panel">
          <h4>
            {quantile.slice(0, 3).toUpperCase()}
            <span
              >{quantile === "p50_us"
                ? "Median"
                : quantile === "p95_us"
                  ? "95th percentile"
                  : "99th percentile"}</span
            >
          </h4>
          <LatencyChart {series} {quantile} {minRecall} targetRecall={target} />
        </div>{/each}
    </div>
    <p class="caption">
      Lower is better · independent log latency axes · shared recall range.
      Click the legend to hide or show engines. Lines join measured points; no
      interpolation in comparisons. <span class="selection-key"
        ><i aria-hidden="true"></i>
        Rings mark the table's selected settings.</span
      >
      Native USearch remains a single control point.
    </p>
    {#if outsideZoom.length}<p class="notice">
        Selected settings outside this zoom: {outsideZoom
          .map((s) => s.name)
          .join(", ")}. Turn off high-recall zoom to see their markers; table
        selections are unchanged.
      </p>{/if}
    {#if missing.length}<p class="notice">
        {#each missing as engine, i}{i ? " · " : ""}<strong
            >{names[engine]}:</strong
          >
          {absence(engine)}{/each}
      </p>{/if}
    <LatencySelectionNote
      {target}
      point={enabled.includes("orion") ? reference : undefined}
    />
    <LatencySummary {dataset} {concurrency} {engines} bind:target />
  </div>
  <div class="latency-notes">
    <details>
      <summary>Timing protocol &amp; excluded measurements</summary>
      <p>
        Three rounds per setting, with at least 10,000 requests and one second
        per included round; 30-second cap. P50/P95/P99 pool individual request
        samples across rounds. Closed-loop clients issue their next query after
        the previous one returns. This is not an open-loop service-load test.
      </p>
      <p>
        Timing includes each Python-facing adapter, binding/GIL overhead and
        result assembly. Loading, building, calibration, warmup and the
        one-second cooldown are excluded. The figures are not HTTP or
        pure-kernel latency. Processes ran at different times; these comparisons
        are descriptive, not significance-tested.
      </p>
      <p>
        The main sweep contains {data.point_counts.qualified.toLocaleString(
          "en-US",
        )} qualified settings and {data.point_counts.undersampled} undersampled settings.
        Qualified points from an otherwise incomplete job are retained. Failed/unsupported
        jobs are excluded.
      </p>
      <p>
        {excluded.length} undersampled settings for this dataset, client count and
        selected engines; excluded from the chart and comparisons.
      </p>
      {#if excluded.length}<div class="table-scroll">
          <table>
            <thead
              ><tr
                ><th>Engine</th><th>Effort</th><th>Recall</th><th
                  >Requests per round</th
                ></tr
              ></thead
            ><tbody
              >{#each excluded as p}<tr
                  ><td>{names[p.engine]}</td><td>{p.effort}</td><td
                    >{p.recall.toFixed(5)}</td
                  ><td>{p.round_requests.join(" / ")}</td></tr
                >{/each}</tbody
            >
          </table>
        </div>{/if}
    </details>
    <details>
      <summary>USearch: separate the engine from the Python GIL</summary>
      <p>
        A separate control uses the same graph, effort and installed serial
        distance kernel within each dataset. IDs and distances were checked,
        including concurrent parity. The native-GIL and native-no-GIL variants
        share an adapter; their difference isolates GIL release more closely
        than comparison with the stock Python wrapper.
      </p>
      <p>
        The native-no-GIL points cover three datasets and one effort each. They
        retain a Python/ctypes request boundary and are not a complete recall
        curve or a SIMD-optimized USearch upper bound. They are measured
        separately from Orion.
      </p>
      <div class="table-scroll">
        <table>
          <caption
            >8 concurrent clients · all three rounds meet sampling requirements</caption
          ><thead
            ><tr
              ><th>Dataset</th><th>Variant</th><th>Recall</th><th
                >P50 / P95 / P99 (µs)</th
              ></tr
            ></thead
          ><tbody
            >{#each data.control.filter((p) => p.concurrency === 8) as p}<tr
                ><td>{datasets[p.dataset]}</td><td>{p.variant}</td><td
                  >{p.recall.toFixed(5)}</td
                ><td>{triples(p)}</td></tr
              >{/each}</tbody
          >
        </table>
      </div>
    </details>
    <div class="downloads">
      <a href={`${import.meta.env.BASE_URL}data/request-latency.json`} download
        >Measurements &amp; coverage JSON ↓</a
      ><a href={`${import.meta.env.BASE_URL}latency/latency-p99-c8.pdf`}
        >P99 panel · PDF ↗</a
      >
    </div>
  </div>
</section>

<style>
  .selection-key {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-weight: 550;
  }
  .selection-key i {
    width: 10px;
    height: 10px;
    border: 2px solid var(--blue);
    border-radius: 50%;
    background: white;
  }
  .latency-card {
    background: white;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 24px;
  }
  .controls {
    display: flex;
    align-items: end;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 18px;
  }
  label {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 12px;
    color: var(--muted);
  }
  select {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 8px;
    color: #25364b;
    min-width: 140px;
  }
  .zoom {
    flex-direction: row;
    align-items: center;
    align-self: center;
    padding-top: 18px;
    gap: 8px;
  }
  .more-engines {
    position: relative;
    margin-left: auto;
    align-self: center;
    padding-top: 18px;
    font-size: 13px;
  }
  .engine-options {
    position: absolute;
    z-index: 5;
    right: 0;
    top: 100%;
    min-width: 260px;
    max-width: calc(100vw - 64px);
    background: white;
    padding: 12px;
    border: 1px solid var(--line);
    border-radius: 9px;
    box-shadow: 0 6px 22px #17263d14;
  }
  .engine-options label {
    flex-direction: row;
    align-items: center;
    padding: 7px;
    gap: 9px;
    font-size: 13px;
    color: #25364b;
  }
  .plot-title {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin: 12px 0;
  }
  .plot-title span,
  .caption {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.65;
  }
  .shared-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 14px 0 20px;
  }
  .shared-legend button {
    display: flex;
    align-items: center;
    gap: 7px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: white;
    padding: 6px 10px;
    font-size: 12px;
    color: #25364b;
  }
  .shared-legend i {
    display: inline-block;
    width: 17px;
    height: 3px;
    background: var(--engine-color);
    border-radius: 2px;
  }
  .shared-legend .inactive {
    color: var(--muted);
    background: #f8fafc;
    text-decoration: line-through;
  }
  .shared-legend .inactive i {
    opacity: 0.25;
  }
  .triptych {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 20px;
    margin: 10px 0;
  }
  .percentile-panel {
    min-width: 0;
  }
  .percentile-panel h4 {
    font-size: 14px;
    margin: 0 0 8px 14px;
  }
  .percentile-panel h4 span {
    font-size: 11px;
    color: var(--muted);
    font-weight: 400;
    margin-left: 7px;
  }
  .notice {
    font-size: 12px;
    line-height: 1.65;
    background: #fff8e8;
    padding: 9px 12px;
    border-radius: 7px;
    margin-top: 10px;
  }
  .latency-notes {
    display: grid;
    gap: 10px;
    margin-top: 14px;
  }
  .latency-notes details {
    padding: 14px 18px;
    background: white;
    border: 1px solid var(--line);
    border-radius: 10px;
  }
  .latency-notes p {
    margin: 12px 0;
    line-height: 1.7;
    color: var(--muted);
    font-size: 14px;
  }
  summary {
    font-weight: 550;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    margin: 15px 0;
  }
  th,
  td {
    text-align: right;
    padding: 9px 10px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }
  th:first-child,
  td:first-child {
    text-align: left;
  }
  .downloads {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    color: var(--blue);
    font-size: 13px;
    padding: 8px 0;
  }
  @media (max-width: 850px) {
    .triptych {
      grid-template-columns: 1fr;
      gap: 24px;
    }
    .percentile-panel {
      max-width: 600px;
      width: 100%;
      margin: auto;
    }
    .latency-card {
      padding: 16px;
    }
  }
  @media (max-width: 500px) {
    .controls {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .controls > label:not(.zoom) {
      flex: 1;
      min-width: 0;
    }
    select {
      min-width: 0;
      width: 100%;
    }
    .zoom,
    .more-engines {
      grid-column: 1 / -1;
      padding-top: 0;
    }
    .more-engines {
      margin-left: 0;
    }
    .engine-options {
      left: 0;
      right: auto;
    }
  }
</style>
