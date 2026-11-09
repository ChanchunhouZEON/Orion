<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import ZoneEvidence from "./ZoneEvidence.svelte";
  let phase = $state<"navigation" | "refinement">("navigation");
  const nodes = [
    { x: 90, y: 55, kind: "remote" },
    { x: 300, y: 45, kind: "remote" },
    { x: 335, y: 170, kind: "remote" },
    { x: 155, y: 80, kind: "local" },
    { x: 250, y: 105, kind: "local" },
    { x: 205, y: 190, kind: "local" },
    { x: 110, y: 165, kind: "extra" },
    { x: 285, y: 210, kind: "extra" },
  ];
</script>

<section id="mechanism" class="home-section" aria-labelledby="mechanism-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">THE SEARCH MECHANISM</p>
      <h2 id="mechanism-title">Change the neighborhood as search converges.</h2>
    </div>
    <p>
      Retain the navigation graph. Reuse selected pruning residuals for
      refinement.
    </p>
  </div>
  <div class="mechanism-grid">
    <article class="story-card">
      <span class="story-kicker">01 / WHERE TO EXPLORE</span>
      <h3>Follow admission activity.</h3>
      <p>
        Navigation uses local and remote neighbors. When admission activity
        falls, refinement uses local neighbors and extras retained from pruning.
      </p>
      <div
        class="phase-control"
        role="group"
        aria-label="Illustrated search phase"
      >
        <button
          aria-pressed={phase === "navigation"}
          onclick={() => (phase = "navigation")}>Navigation</button
        >
        <button
          aria-pressed={phase === "refinement"}
          onclick={() => (phase = "refinement")}>Refinement</button
        >
      </div>
      <svg
        class="neighbor-diagram"
        viewBox="0 0 420 250"
        role="img"
        aria-label={`${phase === "navigation" ? "Navigation selects local and remote" : "Refinement selects local and extra"} neighbors around the expanded vertex. Schematic, not a measured query.`}
      >
        <ellipse
          cx="205"
          cy="135"
          rx="105"
          ry="85"
          fill="#f0f7fc"
          stroke="#dbeaf4"
          stroke-dasharray="4 5"
        />
        {#each nodes as node}
          {@const active =
            node.kind === "local" ||
            node.kind === (phase === "navigation" ? "remote" : "extra")}
          <line
            x1="200"
            y1="130"
            x2={node.x}
            y2={node.y}
            stroke={node.kind === "extra" ? "#b28a42" : "#0284c7"}
            stroke-width={active ? 2 : 1}
            opacity={active ? 0.7 : 0.14}
            stroke-dasharray={node.kind === "extra" ? "4 4" : undefined}
          />
          <circle
            cx={node.x}
            cy={node.y}
            r={active ? 7 : 5}
            fill={active
              ? node.kind === "extra"
                ? "#b28a42"
                : "#0284c7"
              : "#e2e8f0"}
          />
        {/each}
        <circle
          cx="200"
          cy="130"
          r="13"
          fill="#25364b"
          stroke="white"
          stroke-width="4"
        />
        <text x="200" y="157" text-anchor="middle">expanded vertex</text>
        <text x="90" y="36" text-anchor="middle">remote</text>
        <text x="250" y="88" text-anchor="middle">local</text>
        <text x="108" y="190" text-anchor="middle" fill="#916e31">extra</text>
      </svg>
      <p class="phase-explanation" aria-live="polite">
        {phase === "navigation"
          ? "Local + remote: preserve broad navigation through the base graph."
          : "Local + extra: focus expansion on a different neighborhood after convergence."}
      </p>
      <div class="mechanism-notes">
        <p>
          <strong>The switch is reversible.</strong> Renewed admissions restore navigation.
          A sustained zero-admission streak can trigger early exit during refinement.
        </p>
        <p>
          Admission rate counts steps with at least one accepted candidate over
          a recent window. Calibration uses unlabeled warmup queries, without
          ground truth.
        </p>
      </div>
      <span class="schematic-label"
        >Schematic · select a phase to inspect its neighbors</span
      >
    </article>
    <ZoneEvidence />
  </div>
</section>
