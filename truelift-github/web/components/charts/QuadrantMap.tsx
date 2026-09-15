"use client";

import { colorKind, fmt, pct } from "@/lib/utils";

export type MapPoint = {
  i: number;
  x: number;
  y: number;
  kind: string;
  ours: boolean;
  theirs: boolean;
};

export function QuadrantMap({
  points,
  budget,
  onPick,
}: {
  points: MapPoint[];
  budget: number;
  onPick: (index: number) => void;
}) {
  const w = 640;
  const h = 320;
  const pad = 36;
  const xs = points.map((d) => d.x);
  const ys = points.map((d) => d.y);
  const x0 = 0;
  const x1 = Math.max(0.35, ...xs);
  const y0 = Math.min(-0.08, ...ys);
  const y1 = Math.max(0.12, ...ys);
  const X = (v: number) => pad + ((v - x0) / (x1 - x0)) * (w - 2 * pad);
  const Y = (v: number) => h - pad - ((v - y0) / (y1 - y0)) * (h - 2 * pad);
  const yZero = Y(0);

  return (
    <div className="map-wrap">
      <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Uplift quadrant map">
        <line x1={pad} y1={yZero} x2={w - pad} y2={yZero} stroke="#243044" strokeDasharray="4 4" />
        <text x={w - pad - 4} y={yZero - 6} fill="#7d8ba3" fontSize={10} textAnchor="end">
          τ = 0
        </text>
        {points.map((d, idx) => {
          const r = d.ours ? 4.2 : 2.8;
          const stroke = d.theirs && !d.ours ? "var(--blue)" : d.ours && !d.theirs ? "var(--gold)" : "transparent";
          const op = d.ours || d.theirs ? 0.85 : 0.35;
          return (
            <circle
              key={d.i}
              className="map-dot"
              style={{ animationDelay: `${(idx % 20) * 0.015}s` }}
              cx={X(d.x)}
              cy={Y(d.y)}
              r={r}
              fill={colorKind(d.kind)}
              fillOpacity={op}
              stroke={stroke}
              strokeWidth={1.5}
              onClick={() => onPick(d.i)}
            >
              <title>{`${d.kind} · μ₀ ${fmt(d.x)} · τ ${fmt(d.y)}`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="note">Click a dot for a person dossier. Budget {pct(budget)}.</p>
    </div>
  );
}
