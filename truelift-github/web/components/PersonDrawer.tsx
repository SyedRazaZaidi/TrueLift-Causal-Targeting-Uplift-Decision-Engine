"use client";

import { useEffect } from "react";
import { fmt } from "@/lib/utils";

export type Person = {
  id: number;
  index?: number;
  kind?: string;
  tau: number;
  tau_lo?: number;
  tau_hi?: number;
  mu0: number;
  mu1: number;
  assign?: number;
  response_score?: number;
  true_tau?: number | null;
  features?: Record<string, number>;
  contributions?: { feature: string; delta_tau: number }[];
};

export function PersonDrawer({
  open,
  loading,
  person,
  onClose,
  onOpenDesk,
}: {
  open: boolean;
  loading: boolean;
  person: Person | null;
  onClose: () => void;
  onOpenDesk: () => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!open) return null;

  const feats = Object.entries(person?.features ?? {}).slice(0, 12);
  const max = Math.max(...feats.map(([, v]) => Math.abs(v)), 1e-6);

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} role="presentation" />
      <aside className="drawer" aria-label="Person dossier">
        <div className="drawer-head">
          <strong>
            {loading
              ? "Loading…"
              : person
                ? `Person ${person.id} · ${(person.kind ?? "").replace("_", " ")}`
                : "Person"}
          </strong>
          <button type="button" className="drawer-close" onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>
        <div className="drawer-body">
          {loading && <p className="note">Fetching dossier…</p>}
          {!loading && person && (
            <>
              <div className="kpi">
                {fmt(person.tau, 3)}
                <small>
                  τ [{fmt(person.tau_lo)} , {fmt(person.tau_hi)}]
                </small>
              </div>
              <p className="mono" style={{ margin: "12px 0" }}>
                Y(0) {fmt(person.mu0)} · Y(1) {fmt(person.mu1)}
                <br />
                Policy: {(person.assign ?? 0) > 0 ? "TREAT" : "SKIP"} · response {fmt(person.response_score, 3)}
                {person.true_tau != null && (
                  <>
                    <br />
                    Oracle τ {fmt(person.true_tau)}
                  </>
                )}
              </p>
              <h3>Features</h3>
              {feats.map(([k, v]) => (
                <div key={k} className="mono">
                  {k}
                  <div className="bar">
                    <i style={{ width: `${Math.min(100, 50 + (50 * v) / max)}%` }} />
                  </div>
                </div>
              ))}
              <h3 style={{ marginTop: 12 }}>τ drivers</h3>
              {(person.contributions ?? []).slice(0, 8).map((c) => (
                <div key={c.feature} className="mono">
                  {c.feature} {fmt(c.delta_tau, 4)}
                </div>
              ))}
              {!(person.contributions ?? []).length && <p className="note">No contribution vector for this run.</p>}
              <button type="button" className="btn secondary" style={{ marginTop: 16 }} onClick={onOpenDesk}>
                Open in People room
              </button>
            </>
          )}
        </div>
      </aside>
    </>
  );
}
