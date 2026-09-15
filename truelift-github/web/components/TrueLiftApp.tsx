"use client";

import { useCallback, useEffect, useState } from "react";
import { PersonDrawer, Person } from "@/components/PersonDrawer";
import { QiniChart } from "@/components/charts/QiniChart";
import { stageHero, StageView } from "@/components/views/StageView";
import { api } from "@/lib/api";
import { ROOMS, RoomId, fmt } from "@/lib/utils";

export function TrueLiftApp() {
  const [run, setRun] = useState<any>(null);
  const [err, setErr] = useState("");
  const [room, setRoom] = useState<RoomId>("stage");
  const [budget, setBudget] = useState(0.3);
  const [stageSnap, setStageSnap] = useState<any>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [person, setPerson] = useState<Person | null>(null);

  const reloadRun = useCallback(() => api<any>("/api/run").then(setRun), []);

  useEffect(() => {
    reloadRun().catch((e) => setErr(String(e.message || e)));
  }, [reloadRun]);

  const openPerson = async (index: number) => {
    setDrawerOpen(true);
    setDrawerLoading(true);
    setPerson(null);
    try {
      const p = await api<Person>(`/api/person/${index}`);
      setPerson(p);
    } finally {
      setDrawerLoading(false);
    }
  };

  if (err) {
    return (
      <div id="boot">
        <div className="boot-inner">
          <p style={{ color: "var(--bad)", maxWidth: 360, lineHeight: 1.6 }}>
            {err}
            <br />
            Run: <span className="mono">truelift all --dataset synthetic</span> then{" "}
            <span className="mono">truelift serve</span>
          </p>
        </div>
      </div>
    );
  }

  if (!run) {
    return (
      <div id="boot">
        <div className="boot-inner">
          <div className="boot-ring" />
          <p>Loading campaign holdout</p>
        </div>
      </div>
    );
  }

  const hero = stageHero(room === "stage" ? stageSnap : null, run, budget);
  const shipOk = run.card.ship;

  return (
    <>
      <div className="shell">
        <aside className="rail">
          <div className="rail-brand">
            <span className="mark" />
            <div>
              <strong>TrueLift</strong>
              <span>Causal targeting</span>
            </div>
          </div>
          <nav className="rail-nav">
            {ROOMS.map((r) => (
              <button key={r.id} type="button" className={room === r.id ? "on" : ""} onClick={() => setRoom(r.id)}>
                {r.label}
              </button>
            ))}
          </nav>
          <div className="rail-foot">
            <div className={`ship-pill ${shipOk ? "ok" : "bad"}`}>{shipOk ? "READY TO SHIP" : "HOLD — REVIEW GATES"}</div>
            <a className="rail-dl" href={`/api/export.csv?budget=${budget}&mode=knapsack`}>
              Export list
            </a>
          </div>
        </aside>
        <div className="main">
          <header className="hero">
            <h1>{hero.title}</h1>
            <p className="tagline">{hero.tagline}</p>
            <div className="hero-meta">
              {hero.meta.map((m, i) => (
                <span key={m} className={i === 0 ? "hi" : ""}>
                  {m}
                </span>
              ))}
            </div>
          </header>
          <section className="content">
            {room === "stage" && (
              <StageView
                budget={budget}
                onBudgetChange={setBudget}
                onPickPerson={openPerson}
                onGotoAllocate={() => setRoom("allocate")}
                onStageLoaded={setStageSnap}
              />
            )}
            {room === "lab" && <LabRoom run={run} />}
            {room === "identify" && <IdentifyRoom run={run} />}
            {room === "allocate" && <AllocateRoom run={run} budget={budget} setBudget={setBudget} />}
            {room === "desk" && <DeskRoom run={run} onPick={openPerson} />}
            {room === "data" && <DataRoom onRetrain={reloadRun} setRoom={setRoom} />}
            {room === "risk" && <RiskRoom run={run} />}
          </section>
        </div>
      </div>
      <PersonDrawer
        open={drawerOpen}
        loading={drawerLoading}
        person={person}
        onClose={() => setDrawerOpen(false)}
        onOpenDesk={() => {
          setDrawerOpen(false);
          setRoom("desk");
        }}
      />
    </>
  );
}

function LabRoom({ run }: { run: any }) {
  const models = run.leaderboard.filter((r: any) => !r.baseline);
  const [sel, setSel] = useState(run.champion);
  return (
    <div className="grid g2">
      <section className="card">
        <h2>Estimator zoo</h2>
        <table>
          <thead>
            <tr>
              {["Model", "AUUC", "Norm", "u@10", "u@30", "PEHE"].map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((r: any) => (
              <tr key={r.model}>
                <td>{r.model === run.champion ? "★ " : ""}{r.model}</td>
                <td className="gold">{fmt(r.auuc, 2)}</td>
                <td>{fmt(r.normalized_auuc)}</td>
                <td>{fmt(r["uplift@10"])}</td>
                <td>{fmt(r["uplift@30"])}</td>
                <td>{r.pehe != null ? fmt(r.pehe) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <section className="card">
        <h2>Qini curves</h2>
        <select value={sel} onChange={(e) => setSel(e.target.value)} style={{ marginBottom: 8 }}>
          {models.map((r: any) => (
            <option key={r.model} value={r.model}>
              {r.model}
            </option>
          ))}
        </select>
        <QiniChart q={run.qini[sel]} resp={run.qini_response} />
      </section>
    </div>
  );
}

function IdentifyRoom({ run }: { run: any }) {
  const id = run.identification;
  const ov = id.overlap;
  const hist = ov.histogram || [];
  return (
    <div className="grid g2">
      <section className="card">
        <h2>Estimand</h2>
        <div className="kpi">
          {id.estimand}
          <small>{id.randomized ? "Randomized experiment" : "Observational"}</small>
        </div>
        <p className="note">{(id.notes || []).join(" ")}</p>
      </section>
      <section className="card">
        <h2>Overlap</h2>
        <div className={`kpi ${ov.positivity_ok ? "ok" : "bad"}`}>{fmt(ov.common_support_frac, 3)}</div>
        <HistMini hist={hist} />
      </section>
    </div>
  );
}

function HistMini({ hist }: { hist: number[] }) {
  const m = Math.max(...hist, 1);
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 120, marginTop: 8 }}>
      {hist.map((v, i) => (
        <div
          key={i}
          className="hist-bar"
          style={{
            flex: 1,
            height: `${(v / m) * 100}%`,
            background: "linear-gradient(180deg,#5eead4,#134e4a)",
            borderRadius: "3px 3px 0 0",
            transitionDelay: `${i * 0.012}s`,
          }}
        />
      ))}
    </div>
  );
}

function AllocateRoom({ run, budget, setBudget }: { run: any; budget: number; setBudget: (b: number) => void }) {
  const [state, setState] = useState({ budget, mode: "knapsack", conformal: false, fairness: false });
  const [pol, setPol] = useState<any>(null);

  useEffect(() => {
    setState((s) => ({ ...s, budget }));
  }, [budget]);

  useEffect(() => {
    const q = new URLSearchParams(state as any);
    api<any>(`/api/policy?${q}`).then(setPol);
  }, [state]);

  return (
    <div>
      <section className="card">
        <h2>Allocator</h2>
        <input
          type="range"
          min={0.05}
          max={0.8}
          step={0.05}
          value={state.budget}
          onChange={(e) => {
            const b = Number(e.target.value);
            setState((s) => ({ ...s, budget: b }));
            setBudget(b);
          }}
        />
        <div className="budget-val">{state.budget.toFixed(2)}</div>
        {pol && (
          <p className="mono">
            treated {pol.n_treated} · incremental {fmt(pol.incremental_rate, 4)} · spent {fmt(pol.spent, 1)}
          </p>
        )}
        <a className="gold" href={`/api/export.csv?${new URLSearchParams(state as any)}`}>
          Download assignment CSV
        </a>
      </section>
      <section className="card" style={{ marginTop: 16 }}>
        <h2>Policy tree</h2>
        <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>
          {JSON.stringify(run.policy_tree, null, 2)}
        </pre>
      </section>
    </div>
  );
}

function DeskRoom({ run, onPick }: { run: any; onPick: (i: number) => void }) {
  const [items, setItems] = useState(run.people);
  const [kind, setKind] = useState("all");
  useEffect(() => {
    api<any>(`/api/people?kind=${kind}`).then((d) => setItems(d.items || d.featured || []));
  }, [kind]);
  return (
    <section className="card">
      <h2>People</h2>
      <div className="toggle">
        {["all", "persuadable", "sure_thing", "lost_cause", "sleeping_dog"].map((k) => (
          <button key={k} type="button" className={kind === k ? "on" : ""} onClick={() => setKind(k)}>
            {k.replace("_", " ")}
          </button>
        ))}
      </div>
      <div style={{ marginTop: 12 }}>
        {items.map((p: any) => (
          <button key={p.index} type="button" className="chip" onClick={() => onPick(p.index)}>
            {p.kind.replace("_", " ")} · τ {fmt(p.tau)} · id {p.id}
          </button>
        ))}
      </div>
    </section>
  );
}

function DataRoom({ onRetrain, setRoom }: { onRetrain: () => Promise<void>; setRoom: (r: RoomId) => void }) {
  const [msg, setMsg] = useState("");
  const [dataset, setDataset] = useState("synthetic");
  return (
    <div className="grid g2">
      <section className="card">
        <h2>Train</h2>
        <div className="toggle">
          {["synthetic", "hillstrom", "ihdp", "jobs"].map((d) => (
            <button key={d} type="button" className={dataset === d ? "on" : ""} onClick={() => setDataset(d)}>
              {d}
            </button>
          ))}
        </div>
        <a className="gold" href="/api/template.csv">
          Template CSV
        </a>
        <button
          type="button"
          className="btn"
          style={{ marginTop: 12, display: "block" }}
          onClick={async () => {
            setMsg("Training…");
            const fd = new FormData();
            fd.append("dataset", dataset);
            fd.append("treatment", "treatment");
            fd.append("outcome", "visit");
            const r = await fetch("/api/train", { method: "POST", body: fd });
            const body = await r.json();
            setMsg(r.ok ? `Done — ${body.champion}` : JSON.stringify(body));
            if (r.ok) {
              await onRetrain();
              setRoom("stage");
            }
          }}
        >
          Train engine
        </button>
        <p className="note">{msg}</p>
      </section>
    </div>
  );
}

function RiskRoom({ run }: { run: any }) {
  return (
    <div className="grid g2">
      <section className="card">
        <h2>Model card</h2>
        <pre className="mono" style={{ whiteSpace: "pre-wrap", fontSize: 11 }}>
          {JSON.stringify(run.card, null, 2)}
        </pre>
      </section>
      <section className="card">
        <h2>Gates</h2>
        {(run.gates || []).map((x: any) => (
          <div key={x.id} className="gate">
            <span className={x.ok ? "ok" : "bad"}>{x.ok ? "✓" : "✗"}</span>
            <span>{x.id}</span>
            <span className="note">{x.detail}</span>
          </div>
        ))}
      </section>
    </div>
  );
}
