// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false, ws: false, watch: null },
  appType: "custom",
});
after(() => server.close());
const { select, curve, ticks } = await server.ssrLoadModule(
  "/src/latency/model.ts",
);
const data = JSON.parse(
  await readFile(
    new URL("../public/data/request-latency.json", import.meta.url),
    "utf8",
  ),
);
const { render } = await server.ssrLoadModule("svelte/server");

test("portable measurements contain no local paths and preserve coverage", () => {
  assert.doesNotMatch(
    JSON.stringify(data),
    /\/Users\/|\/GLOBALFS\/|\.npy|file:\/\//,
  );
  assert.equal(data.point_counts.qualified, 1473);
  assert.equal(data.point_counts.undersampled, 147);
  assert.equal(data.points.length, 1479);
  assert.equal(data.control.length, 18);
  for (const p of data.points) {
    assert.ok(p.p50_us > 0 && p.p50_us <= p.p95_us && p.p95_us <= p.p99_us);
    assert.equal(p.round_requests.length, 3);
    assert.ok(p.round_requests.every((n) => n >= 10000));
    assert.equal(
      p.requests,
      p.round_requests.reduce((a, b) => a + b, 0),
    );
  }
});

test("native control stays separate and cannot be extrapolated", () => {
  assert.equal(curve(data.points, "sift", "usearch-native", 8).length, 1);
  assert.equal(curve(data.points, "gist", "usearch-native", 8).length, 0);
  assert.equal(
    select(data.points, "sift", "usearch-native", 8, 0.98),
    undefined,
  );
  assert.equal(select(data.points, "gist", "parlayann", 8, 0.95), undefined);
});

test("one P50-selected configuration supplies P95 and P99", () => {
  const rows = [
    {
      dataset: "sift",
      engine: "orion",
      concurrency: 8,
      recall: 0.96,
      effort: 10,
      p50_us: 10,
      p95_us: 30,
      p99_us: 100,
    },
    {
      dataset: "sift",
      engine: "orion",
      concurrency: 8,
      recall: 0.99,
      effort: 20,
      p50_us: 20,
      p95_us: 25,
      p99_us: 30,
    },
  ];
  assert.equal(select(rows, "sift", "orion", 8, 0.95).p99_us, 100);
  assert.equal(select(rows, "sift", "orion", 8, 0.98).p99_us, 30);
  assert.equal(select(rows, "sift", "orion", 8, 1), undefined);
});

test("SIFT table selections are highlighted at their actual recall in every panel", async () => {
  const { default: Chart } = await server.ssrLoadModule(
    "/src/latency/LatencyChart.svelte",
  );
  const { default: Summary } = await server.ssrLoadModule(
    "/src/latency/LatencySummary.svelte",
  );
  for (const target of [0.95, 0.98]) {
    for (const concurrency of [1, 8]) {
      const selected = select(
        data.points,
        "sift",
        "orion",
        concurrency,
        target,
      );
      const rows = curve(data.points, "sift", "orion", concurrency);
      const table = render(Summary, {
        props: {
          dataset: "sift",
          concurrency,
          engines: ["orion"],
          target,
        },
      }).body;
      assert.ok(table.includes(selected.recall.toFixed(5)));
      for (const quantile of ["p50_us", "p95_us", "p99_us"]) {
        const chart = render(Chart, {
          props: {
            series: [{ name: "Orion", color: "#0EA5E9", rows, selected }],
            quantile,
            targetRecall: target,
          },
        }).body;
        assert.equal((chart.match(/data-selected-engine=/g) || []).length, 1);
        assert.ok(chart.includes(`data-selected-recall="${selected.recall}"`));
        assert.ok(
          chart.includes(`data-selected-value="${selected[quantile]}"`),
        );
        assert.ok(
          table.includes(
            selected[quantile].toLocaleString("en-US", {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            }),
          ),
        );
        assert.ok(
          chart
            .replace(/\s+/g, " ")
            .includes(`actual recall ${selected.recall.toFixed(5)}`),
        );
      }
    }
  }
  const chosen = select(data.points, "sift", "orion", 8, 0.95);
  assert.equal(chosen.effort, 76);
  assert.equal(chosen.recall, 0.9908);
  assert.equal(chosen.p95_us, 147.5);
  assert.equal(
    curve(data.points, "sift", "orion", 8).find((p) => p.effort === 25).p95_us,
    208.292,
  );
});

test("a selected point outside the zoom is not moved to its boundary", async () => {
  const { default: Chart } = await server.ssrLoadModule(
    "/src/latency/LatencyChart.svelte",
  );
  const rows = curve(data.points, "sift", "hnswlib", 1);
  const selected = select(data.points, "sift", "hnswlib", 1, 0.9);
  assert.ok(selected.recall < 0.95);
  const html = render(Chart, {
    props: {
      series: [{ name: "hnswlib", color: "#123456", rows, selected }],
      quantile: "p95_us",
      minRecall: 0.95,
      targetRecall: 0.9,
    },
  }).body;
  assert.doesNotMatch(html, /data-selected-engine/);
  assert.match(html, /polyline/);
});

test("website selection reproduces all exported geometric means", () => {
  for (const a of data.aggregates) {
    const pairs = Object.keys(data.datasets)
      .map((ds) => [
        ds,
        select(data.points, ds, "orion", a.concurrency, a.target),
        select(data.points, ds, a.engine, a.concurrency, a.target),
      ])
      .filter(([, o, b]) => o && b);
    assert.deepEqual(
      pairs.map(([ds]) => ds),
      a.datasets,
    );
    if (!pairs.length) {
      assert.equal(a.ratios, null);
      continue;
    }
    for (const q of ["p50_us", "p95_us", "p99_us"]) {
      const ratio = Math.exp(
        pairs.reduce((sum, [, o, b]) => sum + Math.log(b[q] / o[q]), 0) /
          pairs.length,
      );
      assert.ok(Math.abs(ratio - a.ratios[q]) < 1e-10);
    }
  }
});

test("log axes cover long tails, including degenerate or empty selections", () => {
  for (const rows of [
    [],
    [{ p99_us: 100 }],
    curve(data.points, "sift", "usearch", 8),
  ]) {
    const values = ticks(rows, "p99_us");
    assert.ok(values[0] > 0 && values.at(-1) > values[0]);
    assert.ok(
      rows.every((p) => p.p99_us >= values[0] && p.p99_us <= values.at(-1)),
    );
  }
});

test("latency section renders controls, exclusions and protocol", async () => {
  const { default: Section } = await server.ssrLoadModule(
    "/src/latency/LatencySection.svelte",
  );
  const html = render(Section).body;
  for (const text of [
    'id="latency"',
    "Actual recall",
    "P50 / P95 / P99",
    "GIL",
    "not HTTP",
    "147",
  ])
    assert.ok(html.includes(text), text);
  assert.match(html, /polyline/);
  assert.match(html, /A recall floor, not a fixed recall/);
  assert.match(html, /Why higher recall can have lower latency/);
  assert.match(html.replace(/\s+/g, " "), /do not establish the cause of every dip/);
  assert.equal((html.match(/data-selected-engine=/g) || []).length, 12);
  assert.doesNotMatch(html, /NaN|Infinity/);
});

test("triptych defaults to four engines with one shared accessible legend", async () => {
  const { default: Section } = await server.ssrLoadModule(
    "/src/latency/LatencySection.svelte",
  );
  const html = render(Section).body;
  assert.equal((html.match(/request latency versus recall/g) || []).length, 3);
  assert.equal((html.match(/aria-pressed="true"/g) || []).length, 4);
  assert.equal((html.match(/<polyline/g) || []).length, 12);
  assert.match(html, /High recall/);
  assert.match(html, /More engines/);
  assert.doesNotMatch(html, /Compare with|<strong>1\.72×/);
  assert.match(html, /<details class="aggregate-details[^>]*>/);
  assert.doesNotMatch(html, /<details[^>]*open/);
});

test("legend toggling can hide all engines and restores without duplicates", async () => {
  const { DEFAULT_ENGINES, toggleEngine } = await server.ssrLoadModule(
    "/src/latency/model.ts",
  );
  assert.deepEqual(DEFAULT_ENGINES, [
    "orion",
    "parlayann",
    "diskann",
    "hnswlib",
  ]);
  let selected = [...DEFAULT_ENGINES];
  for (const e of DEFAULT_ENGINES) selected = toggleEngine(selected, e);
  assert.deepEqual(selected, []);
  selected = toggleEngine(selected, "usearch-native");
  selected = toggleEngine(selected, "hnswlib");
  selected = toggleEngine(selected, "usearch-native");
  assert.deepEqual(selected, ["hnswlib"]);
  assert.equal(DEFAULT_ENGINES.length, 4);
});

test("zoom filters plotted points and y scale without changing threshold selection", async () => {
  const { default: Chart } = await server.ssrLoadModule(
    "/src/latency/LatencyChart.svelte",
  );
  const rows = [
    {
      recall: 0.91,
      p50_us: 100000,
      effort: 10,
      source: "main",
      requests: 30000,
    },
    { recall: 0.96, p50_us: 100, effort: 20, source: "main", requests: 30000 },
  ];
  const html = render(Chart, {
    props: {
      series: [{ name: "test", color: "#0EA5E9", rows }],
      quantile: "p50_us",
      minRecall: 0.95,
    },
  }).body;
  assert.doesNotMatch(html, /100 ms|0\.91000|polyline/);
  assert.match(html, /0\.96000/);
  assert.doesNotMatch(html, /NaN|Infinity/);
  const before = select(data.points, "glove25", "orion", 8, 0.95);
  const zoomed = curve(data.points, "glove25", "orion", 8, 0.95);
  assert.ok(zoomed.every((p) => p.recall >= 0.95));
  assert.equal(select(data.points, "glove25", "orion", 8, 0.95), before);
});

test("empty charts and native controls never fabricate connecting lines", async () => {
  const { default: Chart } = await server.ssrLoadModule(
    "/src/latency/LatencyChart.svelte",
  );
  const empty = render(Chart, {
    props: { series: [], quantile: "p99_us" },
  }).body;
  assert.match(empty, /No qualified measurements/);
  assert.doesNotMatch(empty, /polyline|NaN|Infinity/);
  const rows = curve(data.points, "sift", "usearch-native", 8);
  const native = render(Chart, {
    props: {
      series: [{ name: "native", color: "#06B6D4", rows }],
      quantile: "p99_us",
    },
  }).body;
  assert.equal((native.match(/<circle/g) || []).length, 1);
  assert.doesNotMatch(native, /polyline/);
});

test("compact table preserves GloVe-25's disadvantage and missing results", async () => {
  const { default: Summary } = await server.ssrLoadModule(
    "/src/latency/LatencySummary.svelte",
  );
  const glove = render(Summary, {
    props: {
      dataset: "glove25",
      concurrency: 8,
      engines: ["orion", "hnswlib"],
    },
  }).body;
  assert.match(glove, /0\.77×/);
  assert.match(glove, /Baseline has lower latency/);
  assert.match(glove, /0\.99016/);
  assert.match(glove, /0\.95644/);
  const gist = render(Summary, {
    props: { dataset: "gist", concurrency: 8, engines: ["orion", "parlayann"] },
  }).body;
  assert.match(gist, /No qualified point/);
  const hidden = render(Summary, {
    props: { dataset: "sift", concurrency: 8, engines: [] },
  }).body;
  assert.match(hidden, /Select an engine/);
});
