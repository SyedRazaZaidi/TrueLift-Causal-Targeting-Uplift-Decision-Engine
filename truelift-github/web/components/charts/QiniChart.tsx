"use client";

import { fmt } from "@/lib/utils";

type QiniPayload = {
  qini: number[];
  random: number[];
  qini_lo?: number[];
  qini_hi?: number[];
  fracs?: number[];
  auuc?: number;
};

export function QiniChart({
  q,
  resp,
  animate = true,
}: {
  q: QiniPayload;
  resp?: { qini?: number[]; auuc?: number } | null;
  animate?: boolean;
}) {
  const w = 640;
  const h = 260;
  const pad = 28;
  const respQ = resp?.qini ?? q.random;
  const series = [
    { y: q.qini, c: "#e8c468", n: "champion", d: "" },
    { y: q.random, c: "#64748b", n: "random", d: "d1" },
    { y: respQ, c: "#60a5fa", n: "response", d: "d2" },
  ];
  const all = [...series.flatMap((s) => s.y), ...(q.qini_lo ?? []), ...(q.qini_hi ?? [])];
  const min = Math.min(0, ...all);
  const max = Math.max(...all, 1e-6);
  const fracs = q.fracs ?? series[0].y.map((_, i) => i / (series[0].y.length - 1));
  const x = (i: number) => pad + (i / (fracs.length - 1)) * (w - 2 * pad);
  const yv = (v: number) => h - pad - ((v - min) / (max - min)) * (h - 2 * pad);
  const path = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${x(i)},${yv(v)}`).join(" ");
  let band = null;
  if (q.qini_lo && q.qini_hi) {
    const up = q.qini_hi.map((v, i) => `${i ? "L" : "M"}${x(i)},${yv(v)}`).join(" ");
    const dn = [...q.qini_lo].reverse().map((v, i) => `L${x(q.qini_lo!.length - 1 - i)},${yv(v)}`).join(" ");
    band = <path className="band-fill" d={`${up} ${dn} Z`} fill="rgba(232,196,104,0.12)" stroke="none" />;
  }
  const lineCls = animate ? "line-draw" : "";
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${w} ${h}`}>
        {band}
        {series.map((s) => (
          <path key={s.n} className={`${lineCls} ${s.d}`.trim()} d={path(s.y)} stroke={s.c} />
        ))}
      </svg>
      <div className="map-legend">
        {series.map((s) => (
          <span key={s.n}>
            <i className="sw" style={{ background: s.c }} />
            {s.n}
          </span>
        ))}
        <span>
          AUUC champion {fmt(q.auuc, 2)}
          {resp?.auuc != null ? ` · response ${fmt(resp.auuc, 2)}` : ""}
        </span>
      </div>
    </div>
  );
}
