// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

// Compile the actual Svelte modules: the tests exercise composition and rune
// state without needing a network service or a browser installation.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  appType: "custom",
});
after(() => server.close());
const { render } = await server.ssrLoadModule("svelte/server");
const { AtlasState } = await server.ssrLoadModule("/src/app/state.svelte.ts");
const { loadAtlas } = await server.ssrLoadModule("/src/app/load.ts");
const readJson = async (name) =>
  JSON.parse(
    await readFile(
      new URL(`../public/data/${name}.json`, import.meta.url),
      "utf8",
    ),
  );
const catalog = await readJson("catalog");
const evidence = await readJson("mechanism-evidence");
const datasets = await Promise.all(catalog.datasets.map((d) => readJson(d.id)));
const modules = {};
for (const [id, name] of Object.entries({
  default: "Default",
  cascades: "Cascades",
  ablation: "Ablation",
  build: "Build",
  memory: "Memory",
  threads: "Threads",
})) {
  modules[id] = (
    await server.ssrLoadModule(`/src/catalogs/${id}/${name}Section.svelte`)
  ).default;
}
const html = (atlas) =>
  render(modules[atlas.section], { props: { atlas } }).body;

test("beam walkthrough retains sorted candidates and follows graph adjacency", async () => {
  const { beamNodes, beamEdges, beamFrames } = await server.ssrLoadModule(
    "/src/home/beamWalkthrough.ts",
  );
  const nodes = new Map(beamNodes.map((n) => [n.id, n]));
  for (const frame of beamFrames) {
    assert.ok(frame.beam.length <= 3);
    assert.equal(new Set(frame.beam).size, frame.beam.length);
    const distances = frame.beam.map((id) => nodes.get(id).distance);
    assert.deepEqual(
      distances,
      [...distances].sort((a, b) => a - b),
    );
    if (frame.pending) {
      assert.ok(!frame.beam.includes(frame.pending));
      assert.ok(
        beamEdges.some(
          ([a, b]) => frame.expanded.includes(a) && b === frame.pending,
        ),
      );
    }
    if (frame.result) {
      assert.deepEqual(frame.result, frame.beam.slice(0, 2));
      assert.ok(frame.beam.every((id) => frame.expanded.includes(id)));
    }
  }
  const admission = beamFrames.find((f) => f.added);
  assert.ok(admission.beam.includes("C") && admission.expanded.includes("C"));
  assert.ok(!admission.beam.includes("A"));
});

test("the explanatory homepage remains available before Atlas data loads", async () => {
  const { default: App } = await server.ssrLoadModule("/src/App.svelte");
  const output = render(App).body.replace(/\s+/g, " ");
  assert.match(output, /Vamana-style/);
  assert.match(output, /Loading measurements/);
  for (const anchor of ["graph-search", "mechanism", "evidence", "atlas"]) {
    assert.match(output, new RegExp(`id="${anchor}"`));
    assert.match(output, new RegExp(`href="#${anchor}"`));
  }
  assert.match(output, /Schematic/);
  assert.match(output, /Early exit stays off/);
  assert.match(output, /32\.1/);
  assert.match(output, /22\.1/);
  assert.match(output, /0\.05 pp recall/);
  assert.match(output, /0\.35 pp recall/);
  assert.match(output, /87\.0%/);
  assert.match(output, /88\.4%/);
  assert.match(output, /not equal-recall throughput/);
  assert.match(output, /data\/mechanism-evidence\.json/);
  assert.doesNotMatch(output, /\bNaN\b|\bundefined\b|\/Users\/|sigmod_v2/);
});

test("diagnostic data preserves distinct denominators and measured distributions", () => {
  const { checkpoint, prefilter } = evidence;
  assert.equal(checkpoint.earlyExit, false);
  assert.equal(checkpoint.queriesPerDataset, 200);
  assert.equal(checkpoint.beam, prefilter.beam);
  for (const row of checkpoint.rows) {
    assert.ok(row.totalSavingsPercent < row.tailSavingsPercent);
    assert.ok(row.tailSavingsPercent < 100);
    assert.ok(row.recallLossPp > 0, "do not omit measured recall losses");
  }
  for (const row of prefilter.rows) {
    assert.ok(row.queries >= 1000);
    assert.ok(row.recall > 0 && row.recall <= 1);
    for (const window of [row.whole, row.late]) {
      assert.ok(0 <= window.q25 && window.q25 <= window.median);
      assert.ok(window.median <= window.q75 && window.q75 <= 100);
    }
  }
  assert.ok(!catalog.datasets.some((d) => d.id === "mechanism-evidence"));
});

test("historical signal and zone data retain their measurement scope", async () => {
  const data = await readJson("search-diagnostics");
  assert.equal(data.datasets.length, 4);
  for (const d of data.datasets) {
    assert.equal(d.calibration.num_points, 100000);
    assert.equal(d.zoneSearchListSize, 100);
    assert.ok(d.calibration.admission_rates.length > 100);
    assert.ok(d.calibration.admission_rates.every((v) => v >= 0 && v <= 1));
    for (const zone of Object.values(d.zones)) {
      assert.ok(zone.unseen > 0 && zone.admitted >= 0);
      assert.ok(zone.admitted <= zone.unseen);
      assert.ok(
        (100 * zone.admitted) / zone.unseen <= 2,
        "shared chart domain must contain every yield",
      );
    }
  }
  assert.match(data.calibrationSource, /smoothed/);
  assert.match(data.zoneSource, /do not record base count/);
});

test("the method story distinguishes search, pruning, controller rate and candidate yield", async () => {
  const { default: App } = await server.ssrLoadModule("/src/App.svelte");
  const output = render(App).body.replace(/\s+/g, " ");
  assert.match(output, /100K base subset/);
  assert.match(output, /not an individual query/);
  assert.match(output, /candidate eligibility/);
  assert.match(output, /not a guaranteed graph edge/);
  assert.match(output, /cannot recover a vector missed/);
  const ids = [
    "graph-search",
    "admission",
    "mechanism",
    "cascade",
    "evidence",
    "atlas",
  ];
  const positions = ids.map((id) => output.indexOf(`id="${id}"`));
  assert.ok(
    positions.every((p, i) => p >= 0 && (i === 0 || p > positions[i - 1])),
  );
});

test("all six catalogs render their measurements and protocol independently", () => {
  const atlas = new AtlasState(catalog, datasets);
  for (const [section, expected] of Object.entries({
    default: "At your recall target",
    cascades: "Add a combination",
    ablation: "Full neighbors",
    build: "Index construction",
    memory: "Retained index allocation",
    threads: "Thread scalability",
  })) {
    atlas.selectSection(section);
    const output = html(atlas);
    assert.ok(output.includes(expected), section);
    assert.match(output, /Measurement notes &amp; protocol/);
    assert.doesNotMatch(output, /\bNaN\b|\bundefined\b/);
    if (section === "build" || section === "memory") {
      assert.match(output, /Show bar/);
      assert.match(output, /2\s+bars/);
      assert.doesNotMatch(output, /Show curve/);
      atlas.toggleEngine("diskann");
      assert.match(html(atlas), /1\s+bar/);
    }
  }
});

test("dataset fallback and view switches retain shared display preferences", () => {
  const atlas = new AtlasState(catalog, datasets);
  atlas.logarithmic = false;
  atlas.target = 0.99;
  atlas.selectDataset("sift1b");
  atlas.selectSection("build");
  assert.notEqual(atlas.datasetId, "sift1b");
  assert.ok(atlas.dataset.build);
  assert.equal(atlas.logarithmic, false);
  assert.equal(atlas.target, 0.99);
  atlas.selectSection("default");
  assert.match(html(atlas), /At your recall target/);
});

test("cascade add/remove preserves order and can display an empty selection", () => {
  const atlas = new AtlasState(catalog, datasets);
  atlas.selectSection("cascades");
  assert.ok(atlas.hidden.includes("diskann"));
  const ids = atlas.dataset.cascades.slice(0, 2).map((c) => c.id);
  atlas.toggleEngine("orion");
  atlas.updateCascades(ids);
  assert.deepEqual(atlas.cascadeIds, ids);
  assert.ok(!atlas.hidden.includes("orion"));
  atlas.updateCascades(ids.slice(1));
  assert.deepEqual(atlas.cascadeIds, ids.slice(1));
  atlas.updateCascades([]);
  assert.match(html(atlas), /Add to chart/);
  assert.doesNotMatch(html(atlas), /\bNaN\b/);
});

test("ablation switches between absolute-reference bars and variant curves", () => {
  const atlas = new AtlasState(catalog, datasets);
  atlas.selectSection("ablation");
  const referenceStudy = atlas.dataset.ablations.find(
    (s) => s.referenceVariant,
  );
  atlas.selectStudy(referenceStudy.id);
  assert.match(html(atlas), /Full neighbors/);
  atlas.ablationView = "curves";
  atlas.compare = true;
  assert.match(html(atlas), /Compare all variants/);
  assert.match(html(atlas), /At your recall target/);
  atlas.selectStudy(
    atlas.dataset.ablations.find((s) => !s.referenceVariant).id,
  );
  assert.equal(atlas.armId, atlas.study.defaultVariant);
  assert.equal(atlas.compare, false);
  assert.match(html(atlas), /At your recall target/);
});

test("thread scaling uses each engine’s own one-thread reference", () => {
  const atlas = new AtlasState(catalog, datasets);
  atlas.selectSection("threads");
  atlas.speedup = true;
  assert.match(html(atlas), /Speedup vs. one thread/);
  assert.match(html(atlas), /1×/);
  atlas.toggleEngine("diskann");
  assert.equal(atlas.hidden.includes("diskann"), true);
  assert.match(html(atlas), /Thread scalability/);
});

test("loader supports a hosting subdirectory and reports missing datasets", async () => {
  const requested = [];
  const data = await loadAtlas("/atlas/", async (url) => {
    requested.push(url);
    const name = url.split("/").at(-1).replace(".json", "");
    return new Response(JSON.stringify(await readJson(name)));
  });
  assert.equal(data.datasets.length, datasets.length);
  assert.ok(requested.every((url) => url.startsWith("/atlas/data/")));
  await assert.rejects(
    loadAtlas("/", async (url) =>
      url.endsWith("/catalog.json")
        ? new Response(JSON.stringify(catalog))
        : new Response("", { status: 404 }),
    ),
    /Cannot load sift/,
  );
});
