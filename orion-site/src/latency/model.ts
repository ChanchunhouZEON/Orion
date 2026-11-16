// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
export type Quantile = "p50_us" | "p95_us" | "p99_us";
export const QUANTILES: Quantile[] = ["p50_us", "p95_us", "p99_us"];
export const DEFAULT_ENGINES = ["orion", "parlayann", "diskann", "hnswlib"];

export function toggleEngine(engines: string[], engine: string): string[] {
  return engines.includes(engine)
    ? engines.filter((e) => e !== engine)
    : [...engines, engine];
}

export interface Point {
  dataset: string;
  engine: string;
  source: string;
  concurrency: number;
  effort: number;
  recall: number;
  p50_us: number;
  p95_us: number;
  p99_us: number;
  requests: number;
  round_requests: number[];
}

// Threshold selection uses P50 once, regardless of the displayed percentile.
export function select(
  points: Point[],
  dataset: string,
  engine: string,
  concurrency: number,
  target: number,
): Point | undefined {
  return points
    .filter(
      (p) =>
        p.dataset === dataset &&
        p.engine === engine &&
        p.concurrency === concurrency &&
        p.recall >= target,
    )
    .sort((a, b) => a.p50_us - b.p50_us || a.effort - b.effort)[0];
}

export function curve(
  points: Point[],
  dataset: string,
  engine: string,
  concurrency: number,
  minRecall = 0.9,
): Point[] {
  return points
    .filter(
      (p) =>
        p.dataset === dataset &&
        p.engine === engine &&
        p.concurrency === concurrency &&
        p.recall >= minRecall,
    )
    .sort((a, b) => a.recall - b.recall || a.effort - b.effort);
}

export function ticks(rows: Point[], quantile: Quantile): number[] {
  if (!rows.length) return [10, 100, 1000];
  const low = Math.floor(Math.log10(Math.min(...rows.map((p) => p[quantile]))));
  const high = Math.max(
    low + 1,
    Math.ceil(Math.log10(Math.max(...rows.map((p) => p[quantile])))),
  );
  const values: number[] = [];
  for (let power = low; power <= high; power++) {
    for (const multiplier of [1, 2, 5]) {
      const value = multiplier * 10 ** power;
      if (value <= 10 ** high) values.push(value);
    }
  }
  const minimum = Math.min(...rows.map((p) => p[quantile]));
  const maximum = Math.max(...rows.map((p) => p[quantile]));
  const first = values.reduce((last, v, i) => (v <= minimum ? i : last), 0);
  const last = Math.max(
    first + 1,
    values.findIndex((v) => v >= maximum),
  );
  return values.slice(first, last + 1);
}
