<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import diagnostics from "../../public/data/search-diagnostics.json";
  const zones = ["local", "extra", "remote"] as const;
  const yieldPercent = (zone: { admitted: number; unseen: number }) =>
    (100 * zone.admitted) / zone.unseen;
</script>

<article class="story-card">
  <span class="story-kicker">WHY KEEP PRUNING RESIDUALS?</span>
  <h3>Discarded edges can still offer useful candidates.</h3>
  <p>
    Orion retains selected nearby pruning candidates for refinement. In an
    all-zone diagnostic walk, extras have comparable or higher candidate
    admission yield than local neighbors.
  </p>
  <div class="zone-legend">
    {#each zones as zone}<span><i class={zone}></i>{zone}</span>{/each}
  </div>
  <div
    class="zone-chart"
    aria-label="Post-convergence candidate admission yield; all bars share a zero to two percent scale"
  >
    {#each diagnostics.datasets as d}<div class="zone-row">
        <h4>{d.label}</h4>
        <div>
          {#each zones as zone}<div class="zone-bar">
              <span class="sr-only">{zone}</span>
              <div class="bar-track" aria-hidden="true">
                <div
                  class={zone}
                  style:width={`${(yieldPercent(d.zones[zone]) / 2) * 100}%`}
                ></div>
              </div>
              <span>{yieldPercent(d.zones[zone]).toFixed(3)}%</span>
            </div>{/each}
        </div>
      </div>{/each}
    <p class="zone-scale">Common bar scale: 0–2% of examined candidates</p>
  </div>
  <p class="evidence-caption">
    Post-convergence candidate eligibility · L=100 · admitted / unseen
    candidates · separate from step-level admission rate
  </p>
  <details class="evidence-details">
    <summary>What this evidence establishes</summary>
    <p>
      Eligibility means passing the admission cutoff, not surviving the beam
      merge or appearing in the final top-k. The all-zone diagnostic examines
      local, extra, and remote candidates after convergence.
    </p>
    <p>
      Remote yield remains meaningful, especially on GIST, and remote paths may
      discover useful downstream vertices. These measurements support selective
      reuse of extras, not a guarantee that remote edges are dispensable. Base
      counts are not recorded in the zone files.
    </p>
  </details>
</article>
