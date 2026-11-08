// Copyright (c) Chanchunhou. All rights reserved.
// Licensed under the MIT License.
export type Section =
  "default" | "cascades" | "ablation" | "build" | "memory" | "threads";
export interface Point {
  recall: number;
  qps: number;
  L?: number;
  Q?: number;
  setting?: string;
  visits?: number;
  ndc?: number;
}
export interface Cascade {
  id: string;
  prefilter: string;
  admission: string;
  rerank: string;
  points: Point[];
}
export interface Variant {
  id: string;
  label: string;
  points: Point[];
}
export interface Ablation {
  id: string;
  label: string;
  study: string;
  variants: Variant[];
  defaultVariant: string;
  referenceVariant?: string;
  barWidths?: number[];
  queryCount?: number;
  diagnosticCount?: number;
}
export interface Dataset {
  id: string;
  label: string;
  count: number;
  dimension: number;
  hardware: string;
  k: number;
  threads: number;
  defaultRecipe: string | null;
  orionCurve: Point[] | null;
  baselines: { engine: string; study: string; points: Point[] }[];
  cascades: Cascade[];
  ablations: Ablation[];
  build: null | {
    study: string;
    trials: number;
    alpha: number;
    orion: {
      graphSeconds: number;
      overheadSeconds: number;
      totalSeconds: number;
    };
    diskann: { totalSeconds: number };
  };
  memory: null | {
    study: string;
    orion: { peakBytes: number; retainedBytes: number };
    diskann: { peakBytes: number };
  };
  threadScaling: null | {
    study: string;
    k: number;
    L: number;
    trials: number;
    series: { engine: string; points: { threads: number; qps: number }[] }[];
  };
}
export interface Catalog {
  version: number;
  datasets: { id: string; label: string }[];
  engines: Record<string, string>;
  colors: Record<string, string>;
  studies: Record<
    string,
    { label: string; date?: string; commit?: string; protocol: string }
  >;
}
export interface Series {
  id: string;
  engine: string;
  label: string;
  color: string;
  points: { x: number; y: number; detail?: string }[];
  dashed?: boolean;
}
export interface SearchCurve {
  id: string;
  engine: string;
  label: string;
  points: Point[];
}
export interface PlotOptions {
  logarithmic?: boolean;
  mode?: "line" | "bars";
  xLabel?: string;
  yLabel?: string;
  target?: number | null;
  xMinimum?: number | null;
  inlineLabel?: string | null;
  frontier?: boolean;
  scatterOnly?: boolean;
}
