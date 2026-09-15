"use client";

import { useEffect, useState } from "react";
import { QiniChart } from "@/components/charts/QiniChart";
import { QuadrantMap } from "@/components/charts/QuadrantMap";
import { api } from "@/lib/api";
import { colorKind, fmt, pct } from "@/lib/utils";

async function fetchStage(budget: number) {
  return api<any>(`/api/stage?budget=${budget}`);
}

function PolicyBars({ policies }: { policies: any[] }) {
  const maxInc = Math.max(...policies.map((p) => p.incremental_rate), 1e-6);
  const [w, setW] = useState<Record<string, number>>({});

  useEffect(() => {
    const next: Record<string, number> = {};
    policies.forEach((p) => {
      next[p.name] = Math.max(4, (100 * p.incremental_rate) / maxInc);
    });
    const t = requestAnimationFrame(() => setW(next));
    return () => cancelAnimationFrame(t);
  }, [policies, maxInc]);

  return (
    <>
      {policies
        .slice()
        .sort((a, b) => a.incremental_rate - b.incremental_rate)
        .map((p) => {
          const cls = p.role === "us" ? "us" : p.role === "naive" ? "naive" : "waste";
          return (
            <div key={p.name} className={`policy-row ${p.role === "us" ? "us" : ""}`}>
              <div className="policy-name">
                {p.name}
                {p.role === "us" ? " ★" : ""}
                <small>
                  {p.n_treated.toLocaleString()} contacts · spend {fmt(p.spend, 0)}
                </small>
              </div>
              <div className="bar-track">
                <div
                  className={`bar-fill animated ${cls}`}
                  style={{ width: w[p.name] != null ? `${w[p.name]}%` : "0%" }}
                />
              </div>
              <div className="policy-num">
                {fmt(p.incremental_rate, 4)}
                <br />
                <span style={{ color: "var(--muted)", fontSize: 10 }}>Δ rate</span>
              </div>
            </div>
          );
        })}
    </>
  );
}

export function StageView({
  budget,
  onBudgetChange,
  onPickPerson,
  onGotoAllocate,
  onStageLoaded,
}: {
  budget: number;
  onBudgetChange: (b: number) => void;
  onPickPerson: (i: number) => void;
  onGotoAllocate: () => void;
  onStageLoaded?: (s: any) => void;
}) {
  const [stage, setStage] = useState<any>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancel = false;
    setErr("");
    fetchStage(budget)
      .then((s) => {
        if (!cancel) {
          setStage(s);
          onStageLoaded?.(s);
        }
      })
      .catch((e) => {
        if (!cancel) setErr(String(e));
      });
    return () => {
      cancel = true;
    };
  }, [budget]);

  if (err) return <p className="bad">{err}</p>;
  if (!stage) return <p className="note">Loading campaign holdout…</p>;

  const gates = (stage.gates || []) as { ok: boolean; id: string; detail: string }[];

  return (
    <>
      <div className="grid g2">
        <section className="card">
          <h2>{stage.campaign}</h2>
          <p className="note" style={{ fontSize: 15, color: "var(--text)", marginBottom: 16 }}>
            {stage.thesis}
          </p>
          <div className="budget-bar">
            <label className="note">Mail budget — top fraction of holdout</label>
            <input
              type="range"
              min={0.05}
              max={0.65}
              step={0.05}
              value={budget}
              onChange={(e) => onBudgetChange(Number(e.target.value))}
            />
            <div className="budget-val">
              Targeting <strong>{pct(budget)}</strong> of {stage.n_test.toLocaleString()} people
            </div>
          </div>
          <h3 style={{ marginTop: 20 }}>Policy value at this budget</h3>
          <PolicyBars policies={stage.policies} />
        </section>
        <section className="card">
          <h2>Business impact</h2>
          <div className="stat-grid">
            <div className="stat">
              <div className="label">Wasted contacts</div>
              <div className="val bad">{stage.wasted.n.toLocaleString()}</div>
              <div className="note">~{fmt(stage.wasted.spend, 0)} spend on ~zero lift</div>
            </div>
            <div className="stat">
              <div className="label">Rescued persuadables</div>
              <div className="val ok">{stage.rescued.n.toLocaleString()}</div>
              <div className="note">{stage.rescued.label}</div>
            </div>
            <div className="stat">
              <div className="label">Sleeping dogs mailed</div>
              <div className="val bad">{stage.dogs_mailed.toLocaleString()}</div>
              <div className="note">by conversion model alone</div>
            </div>
          </div>
          <div className={`kpi ${stage.ship ? "ok" : "bad"}`} style={{ fontSize: 22, marginTop: 18 }}>
            {stage.ship ? "Ship list" : "Hold"}
            <small>
              AUUC {fmt(stage.auuc, 2)} vs response {fmt(stage.response_auuc, 2)}
            </small>
          </div>
          <div style={{ marginTop: 16, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <a className="btn" href={`/api/export.csv?budget=${budget}&mode=knapsack`}>
              Download treat/skip CSV
            </a>
            <button type="button" className="btn secondary" onClick={onGotoAllocate}>
              Fine-tune allocator
            </button>
          </div>
        </section>
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <section className="card">
          <h2>Quadrant map — baseline propensity vs τ</h2>
          <QuadrantMap points={stage.map} budget={budget} onPick={onPickPerson} />
          <div className="map-legend">
            {["persuadable", "sure_thing", "lost_cause", "sleeping_dog"].map((k) => (
              <span key={k}>
                <i className="sw" style={{ background: colorKind(k) }} />
                {k.replace("_", " ")}
              </span>
            ))}
          </div>
        </section>
        <section className="card">
          <h2>Offer mix under TrueLift policy</h2>
          <div className="offer-chips">
            {Object.entries(stage.offers || {}).map(([k, v]) => (
              <span key={k}>
                {k}
                <strong>{Number(v).toLocaleString()}</strong>
              </span>
            ))}
          </div>
        </section>
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <section className="card">
          <h2>Qini — champion vs random vs response</h2>
          <QiniChart q={stage.qini} resp={{ qini: stage.qini_response?.qini, auuc: stage.response_auuc }} />
        </section>
        <section className="card">
          <h2>Ship gates</h2>
          {gates.map((x) => (
            <div key={x.id} className="gate">
              <span className={x.ok ? "ok" : "bad"}>{x.ok ? "✓" : "✗"}</span>
              <span>{x.id}</span>
              <span className="note">{x.detail}</span>
            </div>
          ))}
        </section>
      </div>
    </>
  );
}

export type StageMeta = { campaign: string; thesis: string; ship: boolean; champion: string; auuc: number; response_auuc: number; n_test: number; budget: number };

export function stageHero(stage: StageMeta | null, run: any, budget: number) {
  if (stage) {
    return {
      title: stage.campaign,
      tagline: `${stage.thesis} Champion ${stage.champion} · budget ${pct(budget)}.`,
      meta: [
        stage.ship ? "SHIP" : "HOLD",
        `AUUC ${fmt(stage.auuc, 2)}`,
        `Response AUUC ${fmt(stage.response_auuc, 2)}`,
        `${stage.n_test.toLocaleString()} people`,
      ],
    };
  }
  const top = run.leaderboard.find((r: any) => r.model === run.champion) || run.leaderboard[0];
  return {
    title: "Causal targeting — not another conversion ranker",
    tagline: "Rank people by incremental lift τ, allocate budget to persuadables, and prove uplift beats mailing likely buyers.",
    meta: [run.champion, `AUUC ${fmt(top.auuc, 2)}`, `${run.n_test.toLocaleString()} holdout`, (run.arm_names || []).join(" · ")],
  };
}
