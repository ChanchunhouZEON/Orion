// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import { readFileSync, readdirSync } from "node:fs";
import assert from "node:assert/strict";
import test from "node:test";
import {
  bestMeasured,
  chooseCascade,
  defaultOrion,
  defaultCascadeIds,
  defaultHiddenEngines,
  ablationRows,
  lineMidpoint,
  paretoPoints,
  chartSeries,
  findCascade,
  addCascadeSelection,
  searchCurves,
  supports,
} from "../src/model/selection.mjs";
const folder = new URL("../public/data/", import.meta.url);
const read = (name) =>
  JSON.parse(readFileSync(new URL(name + ".json", folder), "utf8"));
const catalog = read("catalog");
const datasets = catalog.datasets.map((d) => read(d.id));

test("every dataset has one default curve per engine and no baseline duplication", () => {
  for (const d of datasets) {
    const curves = searchCurves(d, "default");
    assert.equal(
      new Set(curves.map((c) => c.engine)).size,
      curves.length,
      d.id,
    );
    assert.ok(curves.every((c) => c.points.length));
    assert.equal(curves.filter((c) => c.engine === "orion").length, 1);
    assert.ok(defaultOrion(d).length);
  }
});
test("changing stages replaces Orion and preserves valid measured cascades", () => {
  for (const d of datasets) {
    for (const stage of ["prefilter", "admission", "rerank"]) {
      for (const value of new Set(d.cascades.map((c) => c[stage]))) {
        const id = chooseCascade(d.cascades, d.defaultRecipe, stage, value);
        assert.equal(d.cascades.find((c) => c.id === id)[stage], value);
        const curves = searchCurves(d, "cascades", id);
        assert.equal(curves.filter((c) => c.engine === "orion").length, 1);
        assert.deepEqual(
          curves.find((c) => c.engine === "orion").points,
          d.cascades.find((c) => c.id === id).points,
        );
      }
    }
  }
});
test("ablation overlay retains distinct variant names", () => {
  for (const d of datasets)
    for (const study of d.ablations) {
      const single = searchCurves(
        d,
        "ablation",
        "",
        study.id,
        study.variants[1].id,
      );
      assert.equal(single.length, 1);
      assert.equal(single[0].id, study.variants[1].id);

      const overlay = searchCurves(d, "ablation", "", study.id, "", true);
      assert.equal(overlay.length, study.variants.length);

      assert.equal(new Set(overlay.map((c) => c.label)).size, overlay.length);
    }
});
test("measured threshold selection neither interpolates nor fills a missing result", () => {
  assert.equal(
    bestMeasured(
      [
        { recall: 0.94, qps: 100 },
        { recall: 0.96, qps: 50 },
      ],
      0.95,
    ).qps,
    50,
  );
  assert.equal(bestMeasured([{ recall: 0.94, qps: 100 }], 0.95), null);
  const expected = [348599, 359635, 47584, 51879, 174975, 58988, 135765, 33897];
  datasets.forEach((d, i) =>
    assert.equal(
      Math.round(bestMeasured(defaultOrion(d), 0.95).qps),
      expected[i],
      d.id,
    ),
  );
});
test("archive retains the entire measured cascade screen without duplicate baseline runs", () => {
  assert.equal(
    datasets.reduce((n, d) => n + d.cascades.length, 0),
    168,
  );
  assert.equal(
    datasets.reduce(
      (n, d) => n + d.cascades.reduce((m, c) => m + c.points.length, 0),
      0,
    ),
    6896,
  );
  for (const d of datasets) {
    assert.equal(
      new Set(d.baselines.map((b) => b.engine)).size,
      d.baselines.length,
    );
    for (const c of [...d.cascades, ...d.baselines])
      for (const p of c.points) {
        assert.ok(Number.isFinite(p.qps) && p.qps > 0);
        assert.ok(p.recall >= 0 && p.recall <= 1 + 1e-12);
      }
  }
});
test("public data is self-contained and does not leak local run paths", () => {
  for (const filename of readdirSync(folder)) {
    const text = readFileSync(new URL(filename, folder), "utf8");
    assert.doesNotMatch(
      text,
      /\/Users\/|\/GLOBALFS\/|experiment\/|runs\/|\.\.\/|file:\/\//,
    );
    assert.doesNotMatch(
      text,
      /"(?:source_path|json_path|source_file|raw_runs)"/,
    );
  }
  for (const d of datasets) {
    for (const item of [
      ...d.baselines,
      ...d.ablations,
      d.build,
      d.memory,
      d.threadScaling,
    ].filter(Boolean))
      assert.ok(catalog.studies[item.study]);
    if (d.build)
      assert.ok(
        Math.abs(
          d.build.orion.graphSeconds +
            d.build.orion.overheadSeconds -
            d.build.orion.totalSeconds,
        ) < 1e-6,
      );
    if (d.memory)
      assert.ok(d.memory.orion.peakBytes >= d.memory.orion.retainedBytes);
  }
  assert.equal(
    supports(
      datasets.find((d) => d.id === "sift1b"),
      "build",
    ),
    false,
  );
});

test("Cascades starts with multiple Orion recipes and hides other engines", () => {
  for (const d of datasets.filter((d) => d.cascades.length)) {
    const ids = defaultCascadeIds(d);
    assert.ok(ids.length > 1);
    assert.ok(ids.includes(d.defaultRecipe));
    const curves = searchCurves(d, "cascades", ids);
    const hidden = defaultHiddenEngines(d, "cascades");
    const visible = curves.filter((c) => !hidden.includes(c.engine));
    assert.equal(visible.length, ids.length);
    assert.ok(visible.every((c) => c.engine === "orion"));
    assert.equal(new Set(visible.map((c) => c.id)).size, ids.length);
    assert.equal(new Set(visible.map((c) => c.label)).size, ids.length);
    assert.deepEqual(defaultHiddenEngines(d, "default"), []);
    assert.equal(
      searchCurves(d, "cascades", []).filter((c) => c.engine === "orion")
        .length,
      0,
    );
    assert.equal(
      searchCurves(
        d,
        "cascades",
        d.cascades.map((c) => c.id),
      ).filter((c) => c.engine === "orion").length,
      d.cascades.length,
    );
  }
});
test("same-L diagnostics preserve reference semantics and signed changes", () => {
  for (const d of datasets)
    for (const study of d.ablations) {
      assert.equal(
        study.variants.find((v) => v.id === study.defaultVariant).label,
        "Default",
      );
      if (!study.referenceVariant) continue;
      assert.equal(study.defaultVariant, "adaptive_ee");
      assert.equal(study.referenceVariant, "full");
      const rows = ablationRows(study);
      assert.equal(rows.length, 20);
      for (const row of rows) {
        const ref = rows.find((r) => r.L === row.L && r.id === "full");
        assert.ok(
          Math.abs(row.ndcReduction - 100 * (1 - row.ndc / ref.ndc)) < 1e-10,
        );
        assert.ok(
          Math.abs(row.qpsChange - 100 * (row.qps / ref.qps - 1)) < 1e-10,
        );
        assert.ok(
          Math.abs(row.recallChange - 100 * (row.recall - ref.recall)) < 1e-10,
        );
        if (row.id === "full") assert.equal(row.ndcReduction, 0);
      }
      assert.ok(
        rows.some((r) => r.recallChange < 0),
        "recall regressions must remain visible",
      );
      assert.equal(study.queryCount, study.diagnosticCount);
    }
});
test("curve labels use the visible screen-space midpoint on either scale", () => {
  assert.equal(lineMidpoint([]), null);
  assert.deepEqual(lineMidpoint([{ x: 12, y: 8 }]), { x: 12, y: 8 });
  assert.deepEqual(
    lineMidpoint([
      { x: 0, y: 0 },
      { x: 0, y: 20 },
      { x: 80, y: 20 },
    ]),
    { x: 30, y: 20 },
  );
  for (const log of [false, true]) {
    const points = [
      { x: 10, y: 1 },
      { x: 40, y: 10 },
      { x: 80, y: 100 },
    ].map((p) => ({ x: p.x, y: log ? Math.log10(p.y) : p.y }));
    const m = lineMidpoint(points);
    assert.ok(Number.isFinite(m.x) && Number.isFinite(m.y));
    assert.ok(
      m.x >= 10 && m.x <= 80 && m.y >= points[0].y && m.y <= points.at(-1).y,
    );
  }
});

test("Pareto view keeps actual observations and preserves recall-target maxima", () => {
  for (const d of datasets)
    for (const c of d.cascades) {
      const points = c.points.map((p) => ({ x: p.recall, y: p.qps }));
      const front = paretoPoints(points);
      assert.ok(front.every((p) => points.includes(p)));
      for (const target of [0.5, 0.9, 0.95, 0.98, 0.99]) {
        const max = (ps) =>
          Math.max(0, ...ps.filter((p) => p.x >= target).map((p) => p.y));
        assert.equal(max(front), max(points));
      }
      for (let i = 1; i < front.length; i++) {
        assert.ok(front[i].x > front[i - 1].x);
        assert.ok(front[i].y < front[i - 1].y);
      }
    }
});
test("zoom excludes low-recall throughput from the plotted QPS scale", () => {
  const input = [
    { id: "pq", points: [{ x: 0.7, y: 1e6 }] },
    {
      id: "exact-rerank",
      points: [
        { x: 0.91, y: 100 },
        { x: 0.96, y: 90 },
        { x: 0.99, y: 40 },
      ],
    },
  ];
  const zoomed = chartSeries(input, 0.95, 1, true);
  assert.equal(zoomed[0].points.length, 0);
  assert.deepEqual(zoomed[1].points, [input[1].points[1], input[1].points[2]]);
  assert.equal(
    Math.max(...zoomed.flatMap((s) => s.points.map((p) => p.y))),
    90,
  );
  assert.equal(input[0].points.length, 1, "raw measurements remain intact");
  assert.deepEqual(chartSeries(input, 0, 1, false), input);
});

test("combination forms add independently, deduplicate and remove without replacing peers", () => {
  for (const d of datasets.filter((d) => d.cascades.length)) {
    let ids = [];
    const first = d.cascades[0],
      second = d.cascades[1];
    const draft = (c) => ({
      prefilter: c.prefilter,
      admission: c.admission,
      rerank: c.rerank,
    });
    const result = addCascadeSelection(d.cascades, ids, draft(first));
    assert.equal(result.status, "added");
    assert.deepEqual(ids, [], "submitting does not mutate the old selection");
    ids = result.ids;
    assert.equal(
      addCascadeSelection(d.cascades, ids, draft(first)).status,
      "duplicate",
    );
    assert.deepEqual(
      addCascadeSelection(d.cascades, ids, draft(first)).ids,
      ids,
    );
    ids = addCascadeSelection(d.cascades, ids, draft(second)).ids;
    assert.deepEqual(
      searchCurves(d, "cascades", ids)
        .filter((c) => c.engine === "orion")
        .map((c) => c.id),
      [first.id, second.id],
    );
    ids = ids.filter((id) => id !== first.id);
    assert.deepEqual(
      searchCurves(d, "cascades", ids)
        .filter((c) => c.engine === "orion")
        .map((c) => c.id),
      [second.id],
    );
    ids = addCascadeSelection(d.cascades, ids, draft(first)).ids;
    assert.deepEqual(
      searchCurves(d, "cascades", ids)
        .filter((c) => c.engine === "orion")
        .map((c) => c.id),
      [second.id, first.id],
      "re-added recipes preserve submission order",
    );
    assert.equal(
      searchCurves(d, "cascades", []).filter((c) => c.engine === "orion")
        .length,
      0,
    );
  }
});
test("form accepts every measured combination and rejects unmeasured cross-stage choices", () => {
  for (const d of datasets.filter((d) => d.cascades.length)) {
    const initial = [d.defaultRecipe];
    for (const c of d.cascades)
      assert.equal(findCascade(d.cascades, c).id, c.id);
    const pf = [...new Set(d.cascades.map((c) => c.prefilter))];
    const ad = [...new Set(d.cascades.map((c) => c.admission))];
    const rr = [...new Set(d.cascades.map((c) => c.rerank))];
    for (const prefilter of pf)
      for (const admission of ad)
        for (const rerank of rr) {
          const draft = { prefilter, admission, rerank };
          if (findCascade(d.cascades, draft)) continue;
          const result = addCascadeSelection(d.cascades, initial, draft);
          assert.equal(result.status, "unmeasured");
          assert.deepEqual(result.ids, initial);
        }
    assert.equal(
      addCascadeSelection(d.cascades, initial, {
        prefilter: "unknown",
        admission: "unknown",
        rerank: "unknown",
      }).status,
      "unmeasured",
    );
  }
});
