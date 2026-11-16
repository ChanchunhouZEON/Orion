<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Point } from "./model";
  let { target, point }: { target: number; point?: Point } = $props();
</script>

<aside aria-label="Understanding the selected settings">
  <div>
    <h4>A recall floor, not a fixed recall</h4>
    <p>
      At recall ≥ {target.toFixed(2)}, the table chooses each engine's measured
      setting with the lowest P50 among all qualified settings above that floor.
      The same setting supplies P95 and P99. Its ring can sit farther right on
      the curve; the table is not a comparison at exactly {target.toFixed(2)}.
    </p>
    {#if point}<p class="example">
        Orion here: <strong>recall {point.recall.toFixed(5)}</strong> · effort {point.effort}.
        All three panels use this one setting; its rings appear when it is
        within the view.
      </p>{/if}
  </div>
  <div>
    <h4>Why higher recall can have lower latency</h4>
    <p>
      These timings include Python input and result handling, not just vector
      search. With concurrent callers, short searches can compete to reacquire
      the GIL. More native work may spread completions and reduce that
      contention, even while search itself does more work. Separate GIL
      diagnostics observed repeated waits, but do not establish the cause of
      every dip in these curves. Scheduling and run-to-run variation also
      matter.
    </p>
  </div>
</aside>

<style>
  aside {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    padding: 18px 20px;
    margin: 16px 0;
    background: #f4f8fc;
    border-left: 3px solid var(--blue);
    border-radius: 0 8px 8px 0;
    color: var(--muted);
  }
  h4 {
    margin: 0 0 8px;
    font-size: 13px;
    color: #25364b;
  }
  p {
    margin: 0;
    font-size: 12px;
    line-height: 1.7;
  }
  .example {
    margin-top: 8px;
    color: #315673;
  }
  @media (max-width: 700px) {
    aside {
      grid-template-columns: 1fr;
      gap: 16px;
    }
  }
</style>
