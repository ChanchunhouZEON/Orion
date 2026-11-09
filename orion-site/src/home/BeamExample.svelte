<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { beamNodes, type BeamFrame } from "./beamWalkthrough";
  let { frame }: { frame: BeamFrame } = $props();
</script>

<div class="beam-example" aria-live="polite">
  <div class="beam-example-heading">
    <h4>Inside the beam</h4>
    <span>L = 3 · k = 2</span>
  </div>
  <p>Same nodes as the graph above · lower distance is better.</p>
  <ol class="beam-slots" aria-label="Beam candidates ordered by query distance">
    {#each frame.beam as id}
      {@const candidate = beamNodes.find((node) => node.id === id)!}
      <li
        class:beam-new={id === frame.added}
        class:beam-result={frame.result?.includes(id)}
      >
        <strong>{id}<span>d = {candidate.distance.toFixed(2)}</span></strong>
        <span>{frame.expanded.includes(id) ? "Expanded" : "Unexpanded"}</span>
        {#if frame.result?.includes(id)}<span class="result-label"
            >Returned</span
          >{/if}
      </li>
    {/each}
    {#each Array(3 - frame.beam.length) as _}<li class="beam-empty">
        Empty slot
      </li>{/each}
  </ol>
  {#if frame.pending}<p class="beam-pending">
      D · d = 0.40 · pending admission, not yet in the beam
    </p>{/if}
  <p>
    Expanded means its neighbors have been examined, not that it is removed from
    the result candidates.
  </p>
</div>
