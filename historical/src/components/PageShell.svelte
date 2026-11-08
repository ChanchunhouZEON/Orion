<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  let { children }: { children: Snippet } = $props();
</script>

<header class="masthead">
  <a class="brand" href="https://github.com/ChanchunhouZEON/Orion"
    ><img
      src={`${import.meta.env.BASE_URL}orion-astraeus.svg`}
      alt="Orion dragon emblem"
    /><span>ORION <span class="brand-sub">ASTRAEUS</span></span></a
  >
  <nav aria-label="Project">
    <a href="https://github.com/ChanchunhouZEON/Orion">Source ↗</a><a
      href="https://github.com/ChanchunhouZEON/Orion-Explorer">Explorer ↗</a
    >
  </nav>
</header>
<main>
  <div class="intro">
    <p class="eyebrow">PERFORMANCE ATLAS</p>
    <h1>Find the frontier.</h1>
    <p>
      Explore the speed–recall tradeoff, from compact embeddings to a billion
      vectors.
    </p>
  </div>
  {@render children()}
  <footer>
    <div class="footer-signature">
      <svg class="footer-stars" viewBox="0 0 104 40" aria-hidden="true">
        <path
          d="M12 29 52 20 92 11"
          fill="none"
          stroke="currentColor"
          opacity=".45"
        />
        <path
          d="m12 21 2.5 5.5L20 29l-5.5 2.5L12 37l-2.5-5.5L4 29l5.5-2.5ZM52 10l3 7 7 3-7 3-3 7-3-7-7-3 7-3ZM92 3l2.5 5.5L100 11l-5.5 2.5L92 19l-2.5-5.5L84 11l5.5-2.5Z"
          fill="currentColor"
        />
      </svg>
      <span>ORION <span class="footer-astraeus">ASTRAEUS</span></span>
    </div>
    <p class="footer-motto">Navigate the vector universe.</p>
  </footer>
</main>
