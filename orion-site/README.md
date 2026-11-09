<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->

# Orion Astraeus Website

A light project homepage with interactive method explanations and performance charts built with Svelte, Tailwind CSS and Vite.
The curve colors and white chart background follow the Python plotting style.
A single visible Orion curve is labeled **Ours** at its screen-space midpoint,
without an arrow. Ablations call the complete configuration **Default**;
multiple Orion curves use their recipe or variant names in the legend.

The homepage introduces admission-aware Vamana-style search before the Atlas.
`src/home` separates the overview, phase-selection and cascade schematics, and
controlled evidence panels. The schematics are illustrative, not query traces.
The explanatory sections remain readable while Atlas measurements load or fail.

`public/data/mechanism-evidence.json` records two independent diagnostic studies:
the deployed-cascade, L=256 rows from **Matched-State Continuations / Work,
Recall, and Violations**, and the complete table from **Prefilter Activity
Throughout a Query**. Matched-state results include recall loss beside NDC
savings; prefilter results retain per-query medians, quartiles, query counts,
and recall. Both use k=10. They are not extra throughput trials and do not enter
the engine catalog, QPS curves, or speedup selection. Expandable protocols state
the denominators and distinguish fixed-L work savings from matched-recall QPS.

The method narrative starts with an interactive beam-search walkthrough and
an α-occlusion pruning schematic. It then introduces the admission signal,
phase-specific neighbor selection, and three illustrative cascade recipes.
The dashed w–v segment in the pruning diagram denotes a distance comparison,
not a guaranteed graph edge. Recipe cards are explanatory, not new benchmarks.

`public/data/search-diagnostics.json` curates the historical calibration and
zone-admission records for SIFT, GIST, GloVe-25, and GloVe-100. Calibration files
record 100K base subsets; their admission curves are already smoothed by the
diagnostic producer, and the website adds no smoothing. The producer aggregates
admission-positive fractions over queries still active at each step. These are
not individual sliding-window controller traces. The zone files retain mean
admitted/unseen counts and L=100, but do not record base counts. Their candidate
eligibility denominator is distinct from the controller's step denominator.

```sh
cd orion-site
npm ci
npm test
npm run build
npm run dev
```

Open http://127.0.0.1:8079/ for local development. The public homepage is
https://chanchunhouzeon.github.io/Orion/.

This directory and its curated JSON live only on the independent `pages`
branch. Run `npm run build -- --base /Orion/` to reproduce the project-site
asset paths locally. The production output is `dist/` and is not committed.

The Orion Website Pages workflow runs `npm ci`, `npm test` and the production
build on Ubuntu with Node 24. A push to `pages` deploys only after these checks
pass; pull requests targeting `pages` build without publishing. Manual runs
must select `pages`. Website data updates do not enter `main`.

## Reader controls

Choose a section, dataset, then an engine's configuration. The Default section
shows one curve per engine. Cascades starts with other engines hidden and one
Orion recipe per admission method, preserving the default's other stages where
possible. This is not a fastest-run selection. The combination form has separate
prefilter, admission and rerank selectors. **Add to chart** appends a measured
recipe without changing the existing curves. Each submitted combination has a
**Remove** button; **Clear all** removes the current set. Duplicate submissions
and unmeasured stage combinations are disabled. Editing the draft alone does
not change the chart. Baselines can be enabled separately.
Cascades defaults to recall 0.90–1.00, with narrower 0.95 and 0.98 views and a
full-range option. The QPS axis uses only visible points. The default line joins
measured Pareto points; it never introduces interpolated observations. **All
measured points** switches to a scatter plot, preserving nonmonotonic recall
without misleading connectors. Curves below the zoom range are listed explicitly.

PQ + exact means exact reranking of at most 20 candidates at k=10, not a full
search in exact distance. The PQ training/code size and rerank budget were not
tuned in this screen; poor recall here is not a tuned PQ baseline.
Ablation curves offer an explicit **Compare all variants** overlay.

SIFT and GIST also include adaptive × early-stop diagnostics. Grouped bars show
NDC per query, NDC reduction, QPS change or recall change at L=64/128/256/512/1024.
The reference is Full neighbors (both switches off); Default enables both.
A permanently visible Full neighbors reference table provides absolute NDC,
QPS and recall at every displayed L, including when the bars show percentages.
The reference itself is the zero line in change views, not a zero-cost result.
These are same-L comparisons, not matched recall. The table retains actual
recall and signed changes. The QPS–recall curves remain available in that study.
No arrows appear on bars or curves.

The sections are Default, Cascades, Ablation, Build, Memory and Thread scaling.
Unavailable dataset/section combinations are disabled. The table selects the
fastest observed QPS meeting the recall target, without interpolation. A curve
that does not reach the target is reported as such, rather than as zero.

## Curated data

`public/data/catalog.json` defines engines, colors, datasets and study protocols.
One JSON file per dataset stores:

- One baseline curve per engine.
- Orion's default cascade ID and every measured cascade (168 configurations,
  6,896 points across seven datasets); SIFT1B has a standalone Orion curve.
- Within-study ablation variants, without repeated raw runs.
- Build, memory and thread-scaling aggregates where measured.

Each point stores recall and queries/second; search settings are retained when
recorded. Study IDs resolve within the catalog. The public archive contains no
references to external workspace JSON files, local run directories or input
datasets. Original experiment files and figures are unchanged.

The default selections match the README measurements. Baselines retain their
selected historical measurements. No experiments were rerun or repooled.
Connecting lines guide the eye; they do not invent observations. Native
ParlayANN recall can include boundary ties; historical values are not silently
recalibrated. Protocol details are available in the page's measurement notes.

Build profiles describe the historical Rust/DiskANN construction path, not the
current default ParlayANN builder. Memory profiles measure allocator-accounted
index bytes above loaded inputs, **not process RSS**. Thread scaling uses a
fixed search list and each engine's own one-thread normalization, not matched
recall. Missing metrics are not inferred.

To refresh data, explicitly review the selected study and its protocol, then
replace the corresponding aggregate JSON. Do not auto-select the latest or
fastest run. `npm test` checks the selection rules, curve coverage, public-data
references and README threshold values. `npm run build` includes Svelte and
TypeScript checks.

## Source layout

`App.svelte` loads the catalog and composes the selected section. Each catalog
owns its measurement transforms, controls and result tables:

```text
src/
  App.svelte
  app/
    load.ts                 # Public JSON loading and load errors
    state.svelte.ts         # Shared selections and navigation transitions
  catalogs/
    default/DefaultSection.svelte
    cascades/               # Cascade view and combination form
    ablation/               # Ablation view and same-L metrics
    build/BuildSection.svelte
    memory/MemorySection.svelte
    threads/ThreadsSection.svelte
  components/               # Shell, navigation, engine panel, chart and tables
  model/
    selection.mjs           # Pure measurement-selection and plotting rules
    presentation.ts         # Shared labels, colors and curve conversion
  types.ts
  style.css
```

The shared state survives section switches; catalog components derive views
without changing the archived measurements. Chart and panel components accept
series, plot options and snippets rather than inspecting every catalog type.
There are no re-export-only wrapper modules. Component-rendering tests cover all
six catalogs, selection transitions, the cascade form's selection state,
ablation display modes and the subdirectory-aware loader.

`public/orion-astraeus.svg` is a vector redrawing of the dragon and three-star
emblem. It uses a transparent background and blue/gold colors for the light
page, and is shared by the header and favicon.

## Typography

Geist Sans is used for the interface, prose and charts; table measurements and
axis labels use lining, tabular numerals. Geist Mono is reserved for code and
configuration values. The normal variable faces and the Sans italic face use
Latin WOFF2 subsets from pinned Fontsource packages. Vite emits content-hashed
font files alongside the site, so readers make no third-party font requests.
`font-display: swap` keeps text visible while fonts load; unsupported characters
use the system fallback. Font licenses are included under `public/licenses/`.
