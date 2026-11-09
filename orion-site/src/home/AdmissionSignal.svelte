<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import diagnostics from "../../public/data/search-diagnostics.json";
  let selected = $state("sift");
  let sample = $derived(diagnostics.datasets.find((d) => d.id === selected)!);
  let calibration = $derived(sample.calibration);
  let rates = $derived(calibration.admission_rates);
  const x = (i: number, count: number) => 52 + (570 * i) / (count - 1);
  const y = (rate: number) => 232 - 198 * rate;
  let path = $derived(
    rates
      .map((rate, i) => `${i ? "L" : "M"}${x(i, rates.length)},${y(rate)}`)
      .join(" "),
  );
</script>

<section id="admission" class="home-section" aria-labelledby="admission-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">OBSERVE THE SEARCH</p>
      <h2 id="admission-title">When does exploration stop paying off?</h2>
    </div>
    <p>Admission history supplies a signal for changing the search phase.</p>
  </div>
  <div class="signal-grid">
    <div class="signal-definition">
      <h3>Count productive steps.</h3>
      <p>
        A step is admission-positive if at least one candidate passes the
        admission cutoff. Average these binary events over a recent window.
      </p>
      <div
        class="binary-window"
        aria-label="Example window: 1, 0, 0, 1, 0. Two admission-positive steps out of five."
      >
        {#each [1, 0, 0, 1, 0] as value}<span class:positive={value === 1}
            >{value}</span
          >{/each}<strong>2/5</strong>
      </div>
      <p>
        With W=5, this example gives an admission rate of 0.4. It counts <strong
          >steps</strong
        >, not the fraction of accepted neighbors.
      </p>
      <p>
        Low activity suggests refinement. Renewed activity restores navigation.
        A separate zero-admission streak controls early exit.
      </p>
      <a class="text-action" href="#mechanism">See the phase switch ↓</a>
    </div>
    <article class="story-card signal-chart">
      <div
        class="phase-control"
        role="group"
        aria-label="Admission diagnostic dataset"
      >
        {#each diagnostics.datasets as d}<button
            aria-pressed={selected === d.id}
            onclick={() => (selected = d.id)}>{d.label}</button
          >{/each}
      </div>
      <svg
        viewBox="0 0 650 285"
        role="img"
        aria-label={`${sample.label} historical aggregate admission-rate curve over ${rates.length} recorded steps. Calibration threshold ${calibration.threshold.toFixed(3)}. 100,000 base vectors.`}
      >
        {#each [0, 0.25, 0.5, 0.75, 1] as tick}<line
            x1="52"
            x2="622"
            y1={y(tick)}
            y2={y(tick)}
            stroke="#e5ebf1"
          /><text x="42" y={y(tick) + 4} text-anchor="end"
            >{tick.toFixed(2)}</text
          >{/each}
        {#each [0, Math.floor((rates.length - 1) / 2), rates.length - 1] as tick}<text
            x={x(tick, rates.length)}
            y="252"
            text-anchor="middle">{tick}</text
          >{/each}
        <text x="52" y="18">Admission rate (aggregated)</text><text
          x="335"
          y="277"
          text-anchor="middle">Search step</text
        >
        <!-- Clip the area at τ so every crossing follows the measured polyline,
             including any later rise above the threshold. -->
        <defs>
          <clipPath id="admission-above-threshold">
            <rect
              x="52"
              y="34"
              width="570"
              height={y(calibration.threshold) - 34}
            />
          </clipPath>
        </defs>
        <path
          d={`${path} L622,232 L52,232 Z`}
          fill="#0284c7"
          fill-opacity="0.16"
          clip-path="url(#admission-above-threshold)"
        />
        <line
          x1="52"
          x2="622"
          y1={y(calibration.threshold)}
          y2={y(calibration.threshold)}
          stroke="#ad8236"
          stroke-dasharray="5 5"
        />
        <path d={path} stroke="#0284c7" stroke-width="2.5" fill="none" />
        <!-- The early curve stays near 1 here, leaving room above τ even
             when a dataset's late-search curve hugs the threshold. -->
        <rect
          x="59"
          y={y(calibration.threshold) - 26}
          width="78"
          height="20"
          rx="3"
          fill="white"
          fill-opacity="0.94"
        />
        <text x="65" y={y(calibration.threshold) - 12} style="fill: #92702e"
          >τ = {calibration.threshold.toFixed(3)}</text
        >
      </svg>
      <p class="evidence-caption">
        <strong>{sample.label} · 100K base subset</strong> · historical calibration
        diagnostic · shaded area: above τ
      </p>
      <details class="evidence-details">
        <summary>How to read this signal</summary>
        <p>
          The historical diagnostic first measures admission-positive fractions
          across queries still active at each step, then smooths those fractions
          across neighboring steps. This recorded calibration curve is not an
          individual query’s sliding-window trace. Crossing the threshold here
          does not mean all individual queries switch at that step. These
          historical 100K runs are separate from the full-dataset throughput
          sweeps below.
        </p>
        <p>
          Calibration uses unlabeled queries. GloVe-25’s historical file records
          its padded 32-dimensional layout. Query counts and beam width are not
          recorded in these calibration JSON files and are not inferred here.
        </p>
        <a
          href={`${import.meta.env.BASE_URL}data/search-diagnostics.json`}
          download>Download signal and zone data ↓</a
        >
      </details>
    </article>
  </div>
</section>
