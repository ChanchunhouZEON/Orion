// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
import type { Catalog, Dataset, SearchCurve, Series } from "../types";
export const fmt = (n: number, decimals = 0) =>
  n.toLocaleString("en-US", { maximumFractionDigits: decimals });
export const engineName = (catalog: Catalog, id: string) =>
  catalog.engines[id] || id;
export const engineColor = (catalog: Catalog, id: string) =>
  catalog.colors[id] || "#64748b";
export const cascadeColor = (catalog: Catalog, dataset: Dataset, id: string) =>
  id === dataset.defaultRecipe
    ? engineColor(catalog, "orion")
    : `hsl(${(dataset.cascades.findIndex((c) => c.id === id) * 137.5 + 20) % 360} 62% 43%)`;

/** Preserve measured points; filtering/frontier selection belongs to the chart. */
export function searchSeries(
  curves: SearchCurve[],
  label: (curve: SearchCurve) => string,
  color: (curve: SearchCurve) => string,
): Series[] {
  return curves.map((c) => ({
    id: c.id,
    engine: c.engine,
    label: label(c),
    color: color(c),
    points: c.points.map((p) => ({
      x: p.recall,
      y: p.qps,
      detail: `Recall ${p.recall.toFixed(5)} · ${fmt(p.qps)} QPS${p.L !== undefined ? ` · L=${p.L}` : p.Q !== undefined ? ` · Q=${p.Q}` : ""}`,
    })),
  }));
}
export function comparisonStudies(dataset: Dataset): string[] {
  return [
    ...new Set([
      dataset.id === "sift1b" ? "sift1b-engines" : "orion-cascades",
      ...dataset.baselines.map((b) => b.study),
    ]),
  ];
}
