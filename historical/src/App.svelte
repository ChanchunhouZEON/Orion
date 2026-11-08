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

<svelte:head><title>Orion Astraeus · Performance</title></svelte:head>
<PageShell>
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
</PageShell>
