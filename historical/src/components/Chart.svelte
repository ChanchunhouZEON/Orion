<!-- Copyright (c) Chanchunhou. All rights reserved. Licensed under the MIT License. -->
<script lang="ts">
  import { lineMidpoint, chartSeries } from "../model/selection.mjs";
  import type { Series } from "../types";
  let {
    series,
    xLabel = "Recall@10",
    yLabel = "Queries / second",
    logarithmic = true,
    target = null,
    xMinimum = null,
    mode = "line",
    inlineLabel = "Ours",
    frontier = false,
    scatterOnly = false,
    onpoint = () => {},
  }: {
    series: Series[];
    xLabel?: string;
    yLabel?: string;
    logarithmic?: boolean;
    target?: number | null;
    xMinimum?: number | null;
    mode?: "line" | "bars";
    inlineLabel?: string | null;
    frontier?: boolean;
    scatterOnly?: boolean;
    onpoint?: (s: Series, p: Series["points"][number]) => void;
  } = $props();
  let width = $state(800);
  const height = 390;
  const pad = { left: 76, right: 28, top: 45, bottom: 55 };
  let valid = $derived(
    series
      .flatMap((s) => s.points)
      .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && p.y > 0),
  );
  let minX = $derived(xMinimum ?? Math.min(...valid.map((p) => p.x), 0.5));
  let maxX = $derived(
    xLabel.startsWith("Recall") ? 1 : Math.max(...valid.map((p) => p.x), 1),
  );
  let plotted: Series[] = $derived(chartSeries(series, minX, maxX, frontier));
  let all = $derived(plotted.flatMap((s) => s.points));
  let values = $derived(all.map((p) => (logarithmic ? Math.log10(p.y) : p.y)));
  let lo = $derived(
    logarithmic && values.length ? Math.floor(Math.min(...values)) : 0,
  );
  let hi = $derived(
    !values.length
      ? 1
      : logarithmic
        ? Math.max(...values) + 0.3
        : Math.max(...values) * 1.25,
  );
  let yTicks = $derived(
    logarithmic
      ? Array.from(
          { length: Math.max(0, Math.floor(hi) - Math.ceil(lo) + 1) },
          (_, i) => 10 ** (Math.ceil(lo) + i),
        )
      : [0, 1, 2, 3, 4].map((i) => (hi * i) / 4),
  );
  let xTicks = $derived(
    xLabel === "Threads"
      ? [
          minX,
          ...Array.from(
            { length: Math.floor(maxX / 4) },
            (_, i) => (i + 1) * 4,
          ).filter((v) => v > minX),
        ]
      : Array.from({ length: 6 }, (_, i) => minX + ((maxX - minX) * i) / 5),
  );
  let hover = $state<{
    label: string;
    detail: string;
    x: number;
    y: number;
  } | null>(null);
  const x = (v: number) =>
    pad.left +
    ((v - minX) / Math.max(maxX - minX, 0.001)) *
      (width - pad.left - pad.right);
  const y = (v: number) =>
    height -
    pad.bottom -
    (((logarithmic ? Math.log10(v) : v) - lo) / Math.max(hi - lo, 0.001)) *
      (height - pad.top - pad.bottom);
  let orionLines = $derived(
    plotted.filter((s) => s.engine === "orion" && s.points.length > 0),
  );
  let midpoint = $derived(
    mode === "line" && !scatterOnly && inlineLabel && orionLines.length === 1
      ? lineMidpoint(
          orionLines[0].points
            .filter((p) => p.x >= minX && p.x <= maxX && p.y > 0)
            .sort((a, b) => a.x - b.x)
            .map((p) => ({ x: x(p.x), y: y(p.y) })),
        )
      : null,
  );
  const number = (v: number) =>
    v >= 1000
      ? `${(v / 1000).toLocaleString("en-US", { maximumFractionDigits: 1 })}K`
      : v.toLocaleString("en-US", { maximumFractionDigits: 2 });
</script>

<div class="plot" bind:clientWidth={width}>
  {#if !all.length}<div class="empty-state">
      {valid.length
        ? "No measurements in this recall range. Choose a wider range."
        : "Select a configuration to display its measurements."}
    </div>
  {:else}
    <svg
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`${yLabel} versus ${xLabel}. Exact values appear in the table below.`}
    >
      <title>{yLabel} versus {xLabel}</title>
      {#each yTicks as val}
        <line
          x1={pad.left}
          x2={width - pad.right}
          y1={y(val)}
          y2={y(val)}
          class="gridline"
        />
        <text x={pad.left - 12} y={y(val) + 4} text-anchor="end"
          >{number(val)}</text
        >
      {/each}
      {#if mode === "line"}
        {#each xTicks as tick}
          <text x={x(tick)} y={height - pad.bottom + 25} text-anchor="middle"
            >{xLabel.startsWith("Recall")
              ? tick.toFixed(2)
              : number(tick)}</text
          >
        {/each}
      {/if}
      <text x={pad.left} y="16" class="axis-title">{yLabel}</text>
      {#if mode === "line"}<text
          x={(width + pad.left - pad.right) / 2}
          y={height - 5}
          text-anchor="middle"
          class="axis-title">{xLabel}</text
        >{/if}
      {#if target !== null && target >= minX && target <= maxX}
        <line
          x1={x(target)}
          x2={x(target)}
          y1={pad.top}
          y2={height - pad.bottom}
          class="targetline"
        />
      {/if}
      {#each plotted as s, si}
        {@const pts = s.points
          .filter((p) => p.y > 0 && p.x >= minX)
          .sort((a, b) => a.x - b.x)}
        {#if mode === "line"}
          {#if !scatterOnly}<polyline
              points={pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" ")}
              fill="none"
              stroke={s.color}
              stroke-width={s.engine === "orion" ? 2.7 : 1.8}
              stroke-dasharray={s.dashed ? "5 4" : undefined}
              opacity="0.9"
            />{/if}
          {#each pts as p}
            <circle
              cx={x(p.x)}
              cy={y(p.y)}
              r="3.1"
              fill={s.color}
              tabindex="0"
              role="button"
              aria-label={`${s.label}, ${xLabel} ${p.x}, ${yLabel} ${p.y}`}
              onpointerenter={() =>
                (hover = {
                  label: s.label,
                  detail: p.detail || `${p.x.toFixed(4)} · ${number(p.y)}`,
                  x: Math.min(x(p.x) + 8, width - 230),
                  y: y(p.y) - 16,
                })}
              onpointerleave={() => (hover = null)}
              onfocus={() =>
                (hover = {
                  label: s.label,
                  detail: p.detail || number(p.y),
                  x: Math.min(x(p.x) + 8, width - 230),
                  y: y(p.y) - 16,
                })}
              onblur={() => (hover = null)}
              onclick={() => onpoint(s, p)}
              onkeydown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onpoint(s, p);
                }
              }}
            />
          {/each}
        {:else}
          {@const p = s.points[0]}
          {@const cell = (width - pad.left - pad.right) / series.length}
          {@const bx = pad.left + si * cell + cell * 0.2}
          <rect
            x={bx}
            y={y(p.y)}
            width={cell * 0.6}
            height={height - pad.bottom - y(p.y)}
            rx="6"
            fill={s.color}
            opacity="0.9"
          />
          <text
            x={bx + cell * 0.3}
            y={y(p.y) - 10}
            text-anchor="middle"
            class="bar-value">{number(p.y)}</text
          >
          <text
            x={bx + cell * 0.3}
            y={height - pad.bottom + 24}
            text-anchor="middle">{s.label}</text
          >
        {/if}
      {/each}
      {#if midpoint && inlineLabel}
        <text
          class="curve-label"
          x={Math.max(
            pad.left + 28,
            Math.min(width - pad.right - 28, midpoint.x),
          )}
          y={Math.max(pad.top + 13, midpoint.y - 10)}
          text-anchor="middle">{inlineLabel}</text
        >
      {/if}
    </svg>
    {#if hover}<div
        class="point-tooltip"
        style={`left:${Math.max(0, hover.x)}px;top:${Math.max(0, hover.y)}px`}
      >
        <strong>{hover.label}</strong><span>{hover.detail}</span>
      </div>{/if}
  {/if}
</div>
