<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { beamNodes, beamEdges, beamFrames } from "./beamWalkthrough";
  import BeamExample from "./BeamExample.svelte";
  let step = $state(0);
  let pruned = $state(false);
  const steps = beamFrames;
  let frame = $derived(steps[step]);
</script>

<section id="graph-search" class="home-section" aria-labelledby="graph-title">
  <div class="home-heading">
    <div>
      <p class="eyebrow">START WITH THE GRAPH</p>
      <h2 id="graph-title">A sparse graph. A guided search.</h2>
    </div>
    <p>
      Construction chooses the connections. Search follows them toward a query.
    </p>
  </div>
  <div class="mechanism-grid">
    <article class="story-card">
      <span class="story-kicker">QUERY TIME / BEAM SEARCH</span>
      <h3>Keep a frontier of promising candidates.</h3>
      <p>
        <strong>The beam is a bounded candidate pool.</strong> It keeps up to L promising
        vectors found so far, ranked by their distance to the query. L controls the
        search breadth; k is the number of results returned.
      </p>
      <div class="search-steps" role="group" aria-label="Beam search steps">
        {#each steps as item, i}<button
            aria-label={item.title}
            aria-pressed={step === i}
            onclick={() => (step = i)}>{i + 1}</button
          >{/each}
      </div>
      <svg
        class="primer-graph"
        viewBox="0 0 440 190"
        role="img"
        aria-label={`Beam search schematic: ${steps[step].title}`}
      >
        <defs
          ><marker
            id="beam-edge-arrow"
            viewBox="0 0 10 10"
            refX="10"
            refY="5"
            markerWidth="8"
            markerHeight="8"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
            ><path d="M0 0 10 5 0 10Z" fill="context-stroke" /></marker
          ></defs
        >
        {#each beamEdges as [from, to]}
          {@const a = beamNodes.find((node) => node.id === from)!}
          {@const b = beamNodes.find((node) => node.id === to)!}
          {@const length = Math.hypot(b.x - a.x, b.y - a.y)}
          {@const dx = (b.x - a.x) / length}
          {@const dy = (b.y - a.y) / length}
          <!-- Radius 9 + half the outline 1 + clearance 4. The marker's
               reference point is its tip, so it ends at this trimmed endpoint. -->
          <line
            x1={a.x + dx * 14}
            y1={a.y + dy * 14}
            x2={b.x - dx * 14}
            y2={b.y - dy * 14}
            stroke={frame.expanded.includes(from) ? "#0284c7" : "#dbe4ed"}
            stroke-width="2"
            marker-end="url(#beam-edge-arrow)"
          />
        {/each}
        {#each beamNodes as node}
          <circle
            cx={node.x}
            cy={node.y}
            r="9"
            fill={frame.pending === node.id
              ? "#b28a42"
              : frame.beam.includes(node.id)
                ? "#0284c7"
                : "#cbd7e2"}
            stroke={frame.expanded.includes(node.id) ? "#25364b" : "white"}
            stroke-width="2"
          />
          <text
            class="node-label"
            x={node.x}
            y={node.labelY}
            text-anchor="middle"
            >{node.id}{node.id === "A" ? " · entry" : ""}</text
          >
        {/each}
        <path d="m370 70 4 9 9 4-9 4-4 9-4-9-9-4 9-4Z" fill="#ad8236" />
        <text class="node-label" x="370" y="112" text-anchor="middle">query q</text>
      </svg>
      <p class="graph-key">
        Blue: in beam · gold: pending · dark outline: expanded
      </p>
      <div class="walkthrough-copy" aria-live="polite">
        <h4>{steps[step].title}</h4>
        <p>{steps[step].text}</p>
      </div>
      <button
        class="text-action"
        onclick={() => (step = (step + 1) % steps.length)}
        >{step === steps.length - 1 ? "Restart" : "Next step"} →</button
      >
      <BeamExample {frame} />
      <dl class="term-list">
        <div>
          <dt>Beam width L</dt>
          <dd>
            Candidate capacity during search; distinct from the k results
            returned.
          </dd>
        </div>
        <div>
          <dt>Admission</dt>
          <dd>A candidate passes the current beam’s acceptance cutoff.</dd>
        </div>
        <div>
          <dt>Visited set</dt>
          <dd>Tracks encountered IDs to avoid repeated exploration.</dd>
        </div>
      </dl>
      <span class="schematic-label"
        >Schematic · positions and steps are illustrative</span
      >
    </article>
    <article class="story-card">
      <span class="story-kicker">BUILD TIME / VAMANA PRUNING</span>
      <h3>Retain useful directions with fewer edges.</h3>
      <p>
        Vamana prunes candidate edges using distances between graph vertices. A
        retained neighbor w can occlude a candidate v, making the direct edge u
        → v redundant under the pruning rule.
      </p>
      <svg
        class="primer-graph"
        viewBox="0 0 440 190"
        role="img"
        aria-label={pruned
          ? "Direct edge u to v is pruned; u to w remains, with w to v shown as a distance comparison."
          : "Source u has candidate edges to w and v."}
      >
        <path d="M70 145 230 125" stroke="#0284c7" stroke-width="3" />
        <path
          d="M70 145 275 45"
          stroke={pruned ? "#cbd7e2" : "#0284c7"}
          stroke-width="2"
          stroke-dasharray={pruned ? "5 5" : undefined}
        />
        <path
          d="M230 125 275 45"
          stroke="#ad8236"
          stroke-width="2"
          stroke-dasharray="3 4"
        />
        {#each [[70, 145, "u"], [230, 125, "w"], [275, 45, "v"]] as node}<circle
            cx={node[0]}
            cy={node[1]}
            r="9"
            fill={node[2] === "v" && pruned ? "#ad8236" : "#0284c7"}
          /><text
            class="node-label"
            x={node[0]}
            y={Number(node[1]) + (node[2] === "v" ? -21 : 28)}
            text-anchor="middle">{node[2]}</text
          >{/each}
        <text x="293" y="94">distance check</text>
      </svg>
      <button
        class="prune-toggle"
        aria-pressed={pruned}
        onclick={() => (pruned = !pruned)}
        >{pruned ? "Restore candidate edge" : "Apply pruning rule"}</button
      >
      <p class="prune-explanation" aria-live="polite">
        {pruned
          ? "u → v is removed from the navigation graph. A selected nearby discarded candidate can still be retained by Orion as an extra."
          : "The solid edges are candidates from u. The dashed w–v segment is a distance comparison, not a guaranteed graph edge."}
      </p>
      <details class="evidence-details">
        <summary>The α-occlusion rule</summary>
        <p>
          In the rule used here, a retained w occludes v when <code
            >α · d(w, v) &lt; d(u, v)</code
          >. Degree bounds also constrain the retained graph. A larger α makes
          this occlusion condition harder to satisfy.
        </p>
      </details>
      <div class="mechanism-notes">
        <p>
          <strong>Pruned does not mean useless for every query.</strong> Build-time
          redundancy is assessed relative to u and w; query-time usefulness depends
          on q and the current beam.
        </p>
      </div>
    </article>
  </div>
</section>
