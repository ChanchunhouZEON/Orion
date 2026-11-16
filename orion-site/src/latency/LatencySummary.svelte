<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import data from "../../public/data/request-latency.json";
  import { select, QUANTILES, type Point } from "./model";
  let {
    dataset,
    concurrency,
    engines,
    target = $bindable(0.95),
  }: {
    dataset: string;
    concurrency: number;
    engines: string[];
    target?: number;
  } = $props();
  const names: Record<string, string> = data.engines;
  const datasets: Record<string, string> = data.datasets;
  let selected = $derived(
    engines.map((engine) => ({
      engine,
      point: select(
        data.points as Point[],
        dataset,
        engine,
        concurrency,
        target,
      ),
    })),
  );
  // Orion remains the ratio reference even when its curve is hidden.
  let reference = $derived(
    select(data.points as Point[], dataset, "orion", concurrency, target),
  );
  let aggregates = $derived(
    data.aggregates.filter(
      (a) =>
        engines.includes(a.engine) &&
        a.concurrency === concurrency &&
        a.target === target,
    ),
  );
  const number = (n: number) =>
    n.toLocaleString("en-US", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
</script>

<div class="threshold-heading">
  <h3>Selected settings above your recall floor</h3>
  <label
    >Recall@10 ≥ <select bind:value={target}
      >{#each data.targets as value}<option {value}>{value.toFixed(2)}</option
        >{/each}</select
    ></label
  >
</div>
<p class="caption">
  Lowest qualified pooled P50 among settings meeting the recall floor; P95/P99
  use the same setting. Rings on the charts mark these actual recalls, not the
  floor itself. Ratios = baseline / Orion; above 1 favors Orion.
</p>
{#if engines.length}
  <p class="mobile-table-hint">
    Scroll the table horizontally for all percentiles →
  </p>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>Engine</th><th>Effort</th><th>Actual recall</th
          >{#each QUANTILES as q}<th>{q.slice(0, 3).toUpperCase()} (µs)</th
            >{/each}</tr
        ></thead
      >
      <tbody
        >{#each selected as row}<tr>
            <th scope="row">{names[row.engine]}</th>
            {#if row.point}
              <td
                title={`${row.point.requests.toLocaleString("en-US")} pooled requests`}
                >{row.point.effort}</td
              ><td>{row.point.recall.toFixed(5)}</td>
              {#each QUANTILES as q}<td
                  ><span class:ours={row.engine === "orion"}
                    >{number(row.point[q])}</span
                  >
                  {#if reference && row.engine !== "orion"}{@const ratio =
                      row.point[q] / reference[q]}
                    <small
                      class:orion-lower={ratio >= 1}
                      class:baseline-lower={ratio < 1}
                      title={ratio >= 1
                        ? "Orion has lower latency"
                        : "Baseline has lower latency"}
                      >{ratio.toFixed(2)}×</small
                    >
                  {/if}
                </td>{/each}
            {:else}<td colspan="5">No qualified point at this threshold</td
              >{/if}
          </tr>{/each}</tbody
      >
    </table>
  </div>
{:else}<p class="caption empty">
    Select an engine in the legend to show its measurements.
  </p>{/if}
<details class="aggregate-details">
  <summary>Across common datasets · geometric means</summary>
  <p class="caption">
    Separate from {datasets[dataset]}. Coverage varies by engine and threshold;
    this is not a ranking across baselines. Orion remains the reference when
    hidden.
  </p>
  <div class="table-scroll">
    <table>
      <thead
        ><tr
          ><th>Baseline</th><th>Datasets</th><th>P50 ratio</th><th>P95 ratio</th
          ><th>P99 ratio</th></tr
        ></thead
      >
      <tbody
        >{#each aggregates as a}<tr
            ><th scope="row">{names[a.engine]}</th><td
              title={a.datasets.map((d) => datasets[d]).join(", ")}
              >{a.count}</td
            >
            {#if a.ratios}{#each QUANTILES as q}<td
                  >{a.ratios[q].toFixed(2)}×</td
                >{/each}{:else}<td colspan="3">No common qualified datasets</td
              >{/if}
          </tr>{/each}</tbody
      >
    </table>
  </div>
  {#each aggregates.filter((a) => a.count > 0) as a}<p class="caption">
      <strong>{names[a.engine]}:</strong>
      {a.datasets.map((d) => datasets[d]).join(" · ")}
    </p>{/each}
</details>

<style>
  thead th {
    text-transform: none;
  }
  .mobile-table-hint {
    display: none;
  }
  @media (max-width: 500px) {
    .mobile-table-hint {
      display: block;
      font-size: 11px;
      color: var(--muted);
      margin: 8px 0;
    }
  }

  .threshold-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    border-top: 1px solid var(--line);
    padding-top: 16px;
    margin: 16px 0 8px;
  }
  label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--muted);
  }
  select {
    background: #f8fafc;
    border: 1px solid var(--line);
    border-radius: 7px;
    padding: 7px;
    color: #25364b;
  }
  .caption {
    font-size: 12px;
    color: var(--muted);
    line-height: 1.65;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    margin: 10px 0;
  }
  th,
  td {
    text-align: right;
    padding: 9px 10px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
  }
  th:first-child {
    text-align: left;
  }
  .ours {
    color: var(--blue);
    font-weight: 650;
  }
  small {
    font-size: 11px;
    display: inline-block;
    min-width: 44px;
    margin-left: 8px;
    color: var(--muted);
  }
  .orion-lower {
    color: #176d58;
  }
  .baseline-lower {
    color: #9b5c19;
  }
  summary {
    font-size: 13px;
    font-weight: 550;
  }
  .aggregate-details {
    margin-top: 12px;
    padding: 12px 0 0;
  }
  .aggregate-details p {
    margin: 10px 0;
  }
  .empty {
    padding: 12px 0;
  }
</style>
