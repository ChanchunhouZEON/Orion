<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  let selected = $state(0);
  const recipes = [
    {
      label: "Quantized + exact",
      note: "Use compact scores throughout traversal, then refine the retained finalists with full-precision distances.",
      stages: [null, "Scalar-quantized vectors", "Full-precision vectors"],
    },
    {
      label: "High-dimensional + filter",
      note: "Read a compact sketch first. Rejected candidates avoid the more expensive admission representation; exact reranking resolves the finalists.",
      stages: [
        "Compact JL sketch",
        "Quantized vectors",
        "Full-precision vectors",
      ],
    },
    {
      label: "Native u8",
      note: "When admission already computes exact distances on native byte vectors, a second representation and exact rerank are unnecessary.",
      stages: [null, "Native u8 vectors · exact L2", null],
    },
  ];
  const stages = [
    {
      name: "Prefilter",
      question: "Is this candidate worth scoring?",
      decision:
        "Reject noncompetitive candidates using a compact representation.",
      saving:
        "Avoid admission computation and vector reads for rejected candidates.",
    },
    {
      name: "Admission",
      question: "Should it enter the beam?",
      decision:
        "Supply distance scores for beam maintenance and continued exploration.",
      saving:
        "Use the selected representation for the repeated scoring work of traversal.",
    },
    {
      name: "Rerank",
      question: "How should finalists be ordered?",
      decision: "Recompute exact distances for the retained finalist set.",
      saving: "Reserve full-precision evaluation for a small retained set.",
    },
  ];
  let recipe = $derived(recipes[selected]);
</script>

<section id="cascade" class="home-section" aria-labelledby="cascade-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">MATCH PRECISION TO PURPOSE</p>
      <h2 id="cascade-title">Three roles. A workload-aware cascade.</h2>
    </div>
    <p>
      Different representations can serve different decisions within the same
      search.
    </p>
  </div>
  <div
    class="phase-control recipe-control"
    role="group"
    aria-label="Illustrative cascade recipe"
  >
    {#each recipes as r, i}<button
        aria-pressed={i === selected}
        onclick={() => (selected = i)}>{r.label}</button
      >{/each}
  </div>
  <p class="recipe-description" aria-live="polite">{recipe.note}</p>
  <div class="cascade-cards">
    {#each stages as stage, i}<article
        class="story-card cascade-stage"
        class:bypassed={!recipe.stages[i]}
      >
        <span class="story-kicker">0{i + 1} / {stage.name.toUpperCase()}</span>
        <h3>{stage.question}</h3>
        <span class="representation"
          >{recipe.stages[i] ?? "Bypassed in this recipe"}</span
        >
        {#if recipe.stages[i]}<dl class="term-list">
            <div>
              <dt>Decision</dt>
              <dd>{stage.decision}</dd>
            </div>
            <div>
              <dt>Work saved</dt>
              <dd>{stage.saving}</dd>
            </div>
          </dl>{:else}<p>
            {i === 0
              ? "Candidates proceed directly to admission scoring."
              : "Admission scores already provide exact ordering of the retained candidates."}
          </p>{/if}
        <p class="stage-output">
          {i === 0
            ? "Survivors → admission"
            : i === 1
              ? "Scores → beam → continued search"
              : "Finalists → top-k results"}
        </p>
      </article>{/each}
  </div>
  <div class="cascade-takeaway">
    <p>
      <strong>Exact reranking acts on retained candidates.</strong> It cannot recover
      a vector missed during traversal or discarded before the finalist set. Exact
      native-u8 scoring likewise does not make graph traversal exhaustive.
    </p>
    <a href="#atlas">Explore measured cascade combinations ↓</a>
  </div>
  <p class="schematic-label">
    Illustrative recipes · stage roles and representations, not a measured
    candidate funnel
  </p>
</section>
