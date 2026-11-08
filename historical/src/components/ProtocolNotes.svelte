<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Catalog } from "../types";
  let { catalog, studies }: { catalog: Catalog; studies: string[] } = $props();
</script>

<details class="method">
  <summary>Measurement notes & protocol</summary>
  <p>
    Connecting lines guide the eye; they do not add observations. Missing
    measurements remain unavailable. SIFT1B runs on Xeon; the other datasets run
    on Apple Silicon.
  </p>
  {#each studies as key}<div>
      <h4>{catalog.studies[key].label}</h4>
      <p>{catalog.studies[key].protocol}</p>
    </div>{/each}
</details>
