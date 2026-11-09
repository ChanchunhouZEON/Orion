<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  // The curated evidence is independent of the throughput catalog: these are
  // instrumented diagnostic studies, not extra repetitions of its QPS runs.
  import evidence from "../../public/data/mechanism-evidence.json";
</script>

<section id="evidence" class="home-section" aria-labelledby="evidence-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">BEHIND THE SPEEDUP</p>
      <h2 id="evidence-title">Measure the work. Explain the gain.</h2>
    </div>
    <p>Controlled diagnostics isolate what changes inside a query.</p>
  </div>
  <div class="evidence-grid">
    <article class="story-card">
      <span class="story-kicker">EVIDENCE 01 / MATCHED SEARCH STATE</span>
      <h3>Same starting state. Different continuation.</h3>
      <p>
        Replay an identical search prefix, then change only neighbor selection.
        Early exit stays off.
      </p>
      <div
        class="checkpoint-flow"
        aria-label="A shared, verified checkpoint branches into full-neighbor and local-plus-extra continuations"
      >
        <div>Shared prefix <span>Verified checkpoint</span></div>
        <span aria-hidden="true">↗<br />↘</span>
        <div class="branches">
          <span>Full neighbors</span><span>Local + extra</span>
        </div>
      </div>
      <div class="evidence-stats">
        {#each evidence.checkpoint.rows as row}
          <div>
            <span>{row.dataset}</span><strong
              >−{row.tailSavingsPercent.toFixed(1)}<small>%</small></strong
            >
            <p>post-checkpoint NDC</p>
            <span class="recall-tradeoff"
              >−{row.recallLossPp.toFixed(2)} pp recall</span
            >
          </div>
        {/each}
      </div>
      <p class="evidence-caption">
        Relative to full-neighbor continuation · deployed cascade · L=256 · k=10
        · 200 queries per dataset
      </p>
      <details class="evidence-details">
        <summary>Read the comparison &amp; exact values</summary>
        <p>
          Beam contents, expansion flags, pending buffer, visited IDs,
          convergence history, prefilter state, and flush counters match before
          branching. Means include all 200 queries; tail NDC excludes final
          reranking.
        </p>
        <div class="table-scroll">
          <table>
            <caption>Local + extra relative to full neighbors</caption><thead
              ><tr
                ><th>Dataset</th><th>Total NDC saved</th><th>Tail NDC saved</th
                ><th>Recall loss (pp)</th></tr
              ></thead
            ><tbody
              >{#each evidence.checkpoint.rows as row}<tr
                  ><td>{row.dataset}</td><td
                    >{row.totalSavingsPercent.toFixed(2)}%</td
                  ><td>{row.tailSavingsPercent.toFixed(2)}%</td><td
                    >{row.recallLossPp.toFixed(2)}</td
                  ></tr
                >{/each}</tbody
            >
          </table>
        </div>
        <p>
          NDC counts executed scoring-kernel calls, whose costs differ. These
          fixed-L measurements demonstrate a work–recall tradeoff, not
          equal-recall throughput or a guarantee of lossless switching.
        </p>
      </details>
    </article>
    <article class="story-card">
      <span class="story-kicker">EVIDENCE 02 / FILTERING LATE IN SEARCH</span>
      <h3>Close to the query. Still worth filtering.</h3>
      <p>
        The query-relative cutoff continues to reject candidates before
        admission scoring, even in the final quarter of expansions.
      </p>
      <div class="rejection-legend">
        <span><i class="whole-swatch"></i>Whole search</span><span
          ><i class="late-swatch"></i>Final quarter</span
        >
      </div>
      <div
        class="rejection-chart"
        aria-label="Median per-query prefilter rejection rates, on a shared zero to 100 percent scale"
      >
        {#each evidence.prefilter.rows as row}
          <div class="rejection-group">
            <h4>{row.dataset}</h4>
            {#each ["whole", "late"] as window}
              {@const value = row[window as "whole" | "late"].median}
              <div class="rejection-row">
                <span class="sr-only"
                  >{window === "whole"
                    ? "Whole search"
                    : "Final quarter"}:</span
                >
                <div class="bar-track" aria-hidden="true">
                  <div
                    class:late={window === "late"}
                    style:width={`${value}%`}
                  ></div>
                </div>
                <strong>{value.toFixed(1)}%</strong>
              </div>
            {/each}
          </div>
        {/each}
        <div class="rejection-scale" aria-hidden="true">
          <span>0%</span><span>50%</span><span>100%</span>
        </div>
      </div>
      <p class="evidence-caption">
        Median across queries · L=256 · k=10 · JL-enabled cascades · adaptive
        search and early exit enabled
      </p>
      <details class="evidence-details">
        <summary>Read the protocol &amp; distributions</summary>
        <p>
          Each query’s rejection rate is rejected / tested candidates after
          visited-set deduplication. Late means its final quarter of expansions.
          Bypassed steps do not count as filter tests. These are per-query
          medians, not pooled candidate percentages.
        </p>
        <div class="table-scroll">
          <table>
            <caption>Median rejection % [25th, 75th percentile]</caption><thead
              ><tr
                ><th>Dataset</th><th>Whole search</th><th>Final quarter</th><th
                  >Recall@10</th
                ></tr
              ></thead
            ><tbody
              >{#each evidence.prefilter.rows as row}<tr
                  ><td
                    >{row.dataset}<br /><small
                      >{row.queries.toLocaleString("en-US")} queries</small
                    ></td
                  ><td
                    >{row.whole.median.toFixed(1)} [{row.whole.q25.toFixed(1)}, {row.whole.q75.toFixed(
                      1,
                    )}]</td
                  ><td
                    >{row.late.median.toFixed(1)} [{row.late.q25.toFixed(1)}, {row.late.q75.toFixed(
                      1,
                    )}]</td
                  ><td>{row.recall.toFixed(5)}</td></tr
                >{/each}</tbody
            >
          </table>
        </div>
        <p>
          Sequential diagnostic runs produce the same result lists as the
          uninstrumented observer path. Rejection rates measure avoided
          admission evaluations; throughput benefits are measured separately.
        </p>
      </details>
    </article>
  </div>
  <div class="evidence-footer">
    <p>
      Mechanism diagnostics and throughput sweeps answer different questions.
      Explore recall–QPS tradeoffs below.
    </p>
    <a href={`${import.meta.env.BASE_URL}data/mechanism-evidence.json`} download
      >Evidence JSON ↓</a
    >
  </div>
</section>
