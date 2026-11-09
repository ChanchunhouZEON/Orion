<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import type { Dataset } from "../../types";
  import {
    findCascade,
    addCascadeSelection,
    formatStage,
    recipeLabel,
  } from "../../model/selection.mjs";
  type Stage = "prefilter" | "admission" | "rerank";
  let {
    dataset,
    selectedIds,
    color,
    onchange,
  }: {
    dataset: Dataset;
    selectedIds: string[];
    color: (id: string) => string;
    onchange: (ids: string[]) => void;
  } = $props();
  const stages: Stage[] = ["prefilter", "admission", "rerank"];
  let changes = $state<Partial<Record<Stage, string>>>({});
  let message = $state("");
  let initial = $derived(
    dataset.cascades.find((c) => c.id === dataset.defaultRecipe) ||
      dataset.cascades[0],
  );
  let draft = $derived({
    prefilter: changes.prefilter ?? initial?.prefilter ?? "",
    admission: changes.admission ?? initial?.admission ?? "",
    rerank: changes.rerank ?? initial?.rerank ?? "",
  });
  let candidate = $derived(findCascade(dataset.cascades, draft));
  let alreadyAdded = $derived(
    !!candidate && selectedIds.includes(candidate.id),
  );
  function submit(event: SubmitEvent) {
    event.preventDefault();
    const result = addCascadeSelection(dataset.cascades, selectedIds, draft);
    if (result.status === "added") {
      onchange(result.ids);
      message = `Added ${recipeLabel(candidate)}.`;
    } else
      message =
        result.status === "duplicate"
          ? "This combination is already in the chart."
          : "No measurements are available for this combination.";
  }
</script>

<div class="cascade-builder">
  <form onsubmit={submit} aria-label="Add cascade combination">
    <h4>Add a combination</h4>
    {#each stages as stage}
      <label class="field-label" for={`cascade-${stage}`}>{stage}</label>
      <select
        id={`cascade-${stage}`}
        value={draft[stage]}
        onchange={(e) => {
          changes = { ...changes, [stage]: e.currentTarget.value };
          message = "";
        }}
      >
        {#each [...new Set(dataset.cascades.map((c) => c[stage]))] as value}<option
            {value}>{formatStage(value)}</option
          >{/each}
      </select>
    {/each}
    <p id="cascade-availability" class="form-note">
      {!candidate
        ? "No measured data for this combination. Select different stages."
        : alreadyAdded
          ? "Already added. Change a stage to compare another combination."
          : "Measured combination available. Add it to overlay its curve."}
    </p>
    <button
      class="add-combination"
      type="submit"
      disabled={!candidate || alreadyAdded}
      aria-describedby="cascade-availability"
      >{alreadyAdded ? "Already added" : "+ Add to chart"}</button
    >
  </form>
  <p class="form-status" role="status">{message}</p>
  <div class="combination-heading">
    <h4>Added combinations <span>({selectedIds.length})</span></h4>
    <button
      type="button"
      disabled={!selectedIds.length}
      onclick={() => {
        onchange([]);
        message = "All combinations removed.";
      }}>Clear all</button
    >
  </div>
  <div class="combination-list" aria-label="Added cascade combinations">
    {#each selectedIds as id (id)}
      {@const c = dataset.cascades.find((c) => c.id === id)}
      {#if c}<article
          class="combination-card"
          style={`--recipe-color:${color(c.id)}`}
        >
          <div class="combination-title">
            <span
              >{c.id === dataset.defaultRecipe
                ? "Default"
                : "Orion cascade"}</span
            ><button
              type="button"
              aria-label={`Remove ${recipeLabel(c)}`}
              onclick={() => {
                onchange(selectedIds.filter((selected) => selected !== id));
                message = `Removed ${recipeLabel(c)}.`;
              }}>Remove</button
            >
          </div>
          <dl>
            {#each stages as stage}<div>
                <dt>{stage}</dt>
                <dd>{formatStage(c[stage])}</dd>
              </div>{/each}
          </dl>
        </article>{/if}
    {:else}<p class="small-note">
        No combinations added. Choose the three stages above and add a curve.
      </p>{/each}
  </div>
</div>
