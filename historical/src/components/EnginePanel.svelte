<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Snippet } from "svelte";
  import { sections, type AtlasState } from "../app/state.svelte";
  import type { Section } from "../types";
  import { engineColor, engineName } from "../model/presentation";
  let {
    atlas,
    engines,
    count,
    mark = "curve",
    variants = false,
    hideVisibility = false,
    orionControls,
  }: {
    atlas: AtlasState;
    engines: string[];
    count: number;
    mark?: "curve" | "bar";
    variants?: boolean;
    hideVisibility?: boolean;
    orionControls?: Snippet;
  } = $props();
  let search = $derived(
    ["default", "cascades", "ablation"].includes(atlas.section),
  );
</script>

<aside class="engine-panel" aria-label="Engine configurations">
  <div class="panel-label">
    ENGINES <span
      >{count}
      {variants ? "variants" : `${mark}${count === 1 ? "" : "s"}`}</span
    >
  </div>
  {#each engines as engine}
    <details class="engine" open={engine === "orion"}>
      <summary
        ><span
          class="swatch"
          style={`--color:${engineColor(atlas.catalog, engine)}`}
        ></span><span>{engineName(atlas.catalog, engine)}</span><span
          class="chevron">⌄</span
        ></summary
      >
      <div class="engine-body">
        {#if !hideVisibility}<label class="visible-toggle"
            ><input
              type="checkbox"
              checked={!atlas.hidden.includes(engine)}
              onchange={() => atlas.toggleEngine(engine)}
            />
            Show {mark}{engine === "orion" &&
            (atlas.compare || atlas.section === "cascades")
              ? "s"
              : ""}</label
          >{/if}
        {#if engine === "orion" && search}
          <label class="field-label" for="orion-mode">Configuration</label>
          <select
            id="orion-mode"
            value={atlas.section}
            onchange={(e) =>
              atlas.selectSection(e.currentTarget.value as Section)}
          >
            <option value="default">Default</option><option
              value="cascades"
              disabled={!atlas.dataset.cascades.length}>Cascade</option
            ><option value="ablation" disabled={!atlas.dataset.ablations.length}
              >Ablation study</option
            >
          </select>
          {#if orionControls}{@render orionControls()}{/if}
        {:else}<p class="small-note">
            {search
              ? "One baseline curve"
              : sections.find((s) => s.id === atlas.section)?.label}
          </p>{/if}
      </div>
    </details>
  {/each}
  <p class="sidebar-note">
    {mark === "bar"
      ? "Each bar represents one engine; toggle visibility to compare measurements."
      : atlas.section === "cascades"
        ? "Compare Orion recipes here; baseline engines can be enabled individually."
        : "Default shows one curve per engine. Use Cascades and Ablation to compare Orion variants."}
  </p>
</aside>
