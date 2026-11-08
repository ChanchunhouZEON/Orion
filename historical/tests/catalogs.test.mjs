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
