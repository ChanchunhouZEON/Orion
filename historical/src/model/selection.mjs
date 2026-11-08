// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
export function bestMeasured(points, target) {
  return points
    .filter((p) => p.recall + 1e-12 >= target)
    .reduce((best, p) => (!best || p.qps > best.qps ? p : best), null);
}
export function defaultOrion(dataset) {
  return (
    dataset.orionCurve ||
    dataset.cascades.find((c) => c.id === dataset.defaultRecipe)?.points ||
    []
  );
}
// Retain the other stages when possible; never create an invalid combination.
export function chooseCascade(cascades, currentId, stage, value) {
  const current = cascades.find((c) => c.id === currentId);
  const stages = ["prefilter", "admission", "rerank"];
  return (
    cascades
      .filter((c) => c[stage] === value)
      .sort((a, b) => {
        const score = (c) =>
          stages.filter((s) => s !== stage && c[s] === current?.[s]).length;
        return score(b) - score(a);
      })[0]?.id || currentId
  );
}
export function supports(dataset, section) {
  return (
    section === "default" ||
    (section === "cascades" && dataset.cascades.length > 0) ||
    (section === "ablation" && dataset.ablations.length > 0) ||
    (section === "build" && !!dataset.build) ||
    (section === "memory" && !!dataset.memory) ||
    (section === "threads" && !!dataset.threadScaling)
  );
}
export function formatStage(value) {
  return (
    {
      none: "None",
      jl: "JL",
      "jl-hadamard": "JL–Hadamard",
      rabitq: "RaBitQ",
      "l2-u8": "L2 · u8",
      "l2-u16": "L2 · u16",
      "l2-kt": "L2 · KT",
      "pq-l2": "PQ · L2",
      "mips-i8": "IP · i8",
      "mips-i16": "IP · i16",
      "pq-ip": "PQ · IP",
      f32: "Exact f32",
      "ip-f32": "Exact f32",
      u16: "u16",
    }[value] || value
  );
}
export function recipeLabel(cascade) {
  return [cascade.prefilter, cascade.admission, cascade.rerank]
    .map(formatStage)
    .join(" → ");
}
// Select one representative per admission method, preserving the default's
// other stages where possible. This is a recipe selection, not a QPS ranking.
export function defaultCascadeIds(dataset) {
  return [...new Set(dataset.cascades.map((c) => c.admission))].map(
    (admission) =>
      chooseCascade(
        dataset.cascades,
        dataset.defaultRecipe,
        "admission",
        admission,
      ),
  );
}
export function defaultHiddenEngines(dataset, section) {
  return section === "cascades" ? dataset.baselines.map((b) => b.engine) : [];
}
export function ablationRows(study, widths = study.barWidths || []) {
  const reference = study.variants.find((v) => v.id === study.referenceVariant);
  if (!reference) return [];
  return widths.flatMap((L) => {
    const base = reference.points.find((p) => p.L === L);
    if (!base || !Number.isFinite(base.ndc) || base.ndc <= 0 || base.qps <= 0)
      return [];
    return study.variants.flatMap((variant) => {
      const point = variant.points.find((p) => p.L === L);
      if (!point || !Number.isFinite(point.ndc)) return [];
      return [
        {
          id: variant.id,
          label: variant.label,
          ...point,
          ndcReduction: 100 * (1 - point.ndc / base.ndc),
          qpsChange: 100 * (point.qps / base.qps - 1),
          recallChange: 100 * (point.recall - base.recall),
        },
      ];
    });
  });
}
// Cascades may overlay selected recipes; baseline engines remain single curves.
export function searchCurves(
  dataset,
  section,
  cascadeIds = [],
  studyId,
  armId,
  compare = false,
) {
  if (section === "ablation") {
    const study =
      dataset.ablations.find((s) => s.id === studyId) || dataset.ablations[0];
    if (!study) return [];
    const arm = study.variants.find((v) => v.id === armId) || study.variants[0];
    return (compare ? study.variants : [arm]).map((v) => ({
      ...v,
      engine: "orion",
      study: study.study,
    }));
  }
  const selected = Array.isArray(cascadeIds) ? cascadeIds : [cascadeIds];
  const orion =
    section === "cascades"
      ? [...new Set(selected)]
          .map((id) => dataset.cascades.find((c) => c.id === id))
          .filter(Boolean)
          .map((c) => ({
            id: c.id,
            engine: "orion",
            label: recipeLabel(c),
            points: c.points,
          }))
      : [
          {
            id: "orion",
            engine: "orion",
            label: "Default",
            points: defaultOrion(dataset),
          },
        ];
  return [
    ...orion,
    ...dataset.baselines.map((b) => ({
      ...b,
      id: b.engine,
      label: "Baseline",
    })),
  ];
}

// Arc-length midpoint in screen coordinates, recomputed for either axis scale.
export function lineMidpoint(points) {
  if (!points.length) return null;
  const lengths = points
    .slice(1)
    .map((p, i) => Math.hypot(p.x - points[i].x, p.y - points[i].y));
  let remaining = lengths.reduce((a, b) => a + b, 0) / 2;
  for (let i = 0; i < lengths.length; i++) {
    if (remaining <= lengths[i] && lengths[i] > 0) {
      const t = remaining / lengths[i];
      return {
        x: points[i].x + t * (points[i + 1].x - points[i].x),
        y: points[i].y + t * (points[i + 1].y - points[i].y),
      };
    }
    remaining -= lengths[i];
  }
  return points[0];
}

export function ablationColor(study, id) {
  if (id === study.defaultVariant) return "#0EA5E9";
  if (study.referenceVariant)
    return (
      { full: "#64748B", adaptive: "#10B981", ee_only: "#8B5CF6" }[id] ||
      "#64748B"
    );
  const colors = ["#0EA5E9", "#F43F5E", "#F59F00", "#7262FD"];
  return colors[
    Math.max(
      0,
      study.variants.findIndex((v) => v.id === id),
    ) % colors.length
  ];
}

// Keep only measured, nondominated points; no interpolated throughput is added.
export function paretoPoints(points) {
  const ordered = points
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.y > 0)
    .slice()
    .sort((a, b) => b.x - a.x || b.y - a.y);
  let best = -Infinity;
  return ordered
    .filter((p) => {
      if (p.y <= best) return false;
      best = p.y;
      return true;
    })
    .reverse();
}
export function chartSeries(series, minimum, maximum, frontier = false) {
  return series.map((s) => {
    const points = s.points.filter(
      (p) =>
        Number.isFinite(p.x) &&
        Number.isFinite(p.y) &&
        p.y > 0 &&
        p.x >= minimum &&
        p.x <= maximum,
    );
    return { ...s, points: frontier ? paretoPoints(points) : points };
  });
}

// Match all stages exactly: the form never silently changes a user's recipe.
export function findCascade(cascades, draft) {
  return cascades.find((c) =>
    ["prefilter", "admission", "rerank"].every(
      (stage) => c[stage] === draft[stage],
    ),
  );
}
export function addCascadeSelection(cascades, selectedIds, draft) {
  const match = findCascade(cascades, draft);
  if (!match) return { status: "unmeasured", ids: selectedIds };
  if (selectedIds.includes(match.id))
    return { status: "duplicate", ids: selectedIds };
  return { status: "added", ids: [...selectedIds, match.id] };
}
