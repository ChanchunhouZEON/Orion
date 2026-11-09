<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { onMount } from "svelte";
  import { loadAtlas } from "./app/load";
  import { AtlasState } from "./app/state.svelte";
  import PageShell from "./components/PageShell.svelte";
  import AtlasNavigation from "./components/AtlasNavigation.svelte";
  import DefaultSection from "./catalogs/default/DefaultSection.svelte";
  import CascadesSection from "./catalogs/cascades/CascadesSection.svelte";
  import AblationSection from "./catalogs/ablation/AblationSection.svelte";
  import BuildSection from "./catalogs/build/BuildSection.svelte";
  import MemorySection from "./catalogs/memory/MemorySection.svelte";
  import ThreadsSection from "./catalogs/threads/ThreadsSection.svelte";
  import GraphPrimer from "./home/GraphPrimer.svelte";
  import AdmissionSignal from "./home/AdmissionSignal.svelte";
  import CascadeJourney from "./home/CascadeJourney.svelte";
  import Overview from "./home/Overview.svelte";
  import SearchMechanism from "./home/SearchMechanism.svelte";
  import MechanismEvidence from "./home/MechanismEvidence.svelte";
  import "./home/home.css";
  let atlas = $state<AtlasState | null>(null);
  let error = $state("");
  onMount(async () => {
    try {
      const { catalog, datasets } = await loadAtlas(import.meta.env.BASE_URL);
      atlas = new AtlasState(catalog, datasets);
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    }
  });
</script>

<svelte:head>
  <title>Orion Astraeus · Admission-aware graph search</title>
  <meta
    name="description"
    content="Explore how Orion combines adaptive Vamana-style graph traversal and scoring cascades, with controlled diagnostics and interactive recall–throughput benchmarks."
  />
</svelte:head>
<PageShell>
  <Overview />
  <GraphPrimer />
  <AdmissionSignal />
  <SearchMechanism />
  <CascadeJourney />
  <MechanismEvidence />
  <section id="atlas" aria-labelledby="atlas-title">
    <div class="home-heading atlas-intro">
      <div>
        <p class="eyebrow">PERFORMANCE ATLAS</p>
        <h2 id="atlas-title">Explore the measured frontier.</h2>
      </div>
      <p>
        Compare engines, compose cascades, and inspect the tradeoffs at your
        recall target.
      </p>
    </div>
    {#if error}<p role="alert" class="error">{error}. Reload to try again.</p>
    {:else if !atlas}<p role="status">Loading measurements…</p>
    {:else}
      <AtlasNavigation {atlas} />
      {#if atlas.section === "default"}<DefaultSection {atlas} />
      {:else if atlas.section === "cascades"}<CascadesSection {atlas} />
      {:else if atlas.section === "ablation"}<AblationSection {atlas} />
      {:else if atlas.section === "build"}<BuildSection {atlas} />
      {:else if atlas.section === "memory"}<MemorySection {atlas} />
      {:else}<ThreadsSection {atlas} />{/if}
    {/if}
  </section>
</PageShell>
