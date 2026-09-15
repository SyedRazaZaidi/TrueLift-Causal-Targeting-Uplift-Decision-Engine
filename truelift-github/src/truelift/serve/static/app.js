const $ = (s, el = document) => el.querySelector(s);
const api = (p) => fetch(p).then((r) => { if (!r.ok) throw new Error(r.status + " " + p); return r.json(); });

const ROOMS = [
  { id: "stage", label: "Campaign" },
  { id: "lab", label: "Models" },
  { id: "identify", label: "Identify" },
  { id: "allocate", label: "Allocate" },
  { id: "desk", label: "People" },
  { id: "data", label: "Data" },
  { id: "risk", label: "Governance" },
];

let RUN = null;
let STAGE = null;
let ROOM = "stage";
let budget = 0.3;

function fmt(n, d = 3) {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: d });
}

function pct(x) {
  return `${(100 * x).toFixed(0)}%`;
}

async function boot() {
  try {
    RUN = await api("/api/run");
    $("#shell").hidden = false;
    $("#boot").setAttribute("hidden", "");
    renderNav();
    paintShip();
    paintHero();
    await show(ROOM);
  } catch (e) {
    const boot = $("#boot");
    boot.innerHTML = `<div class="boot-inner"><p style="color:var(--bad);max-width:360px;line-height:1.6">No trained run found.<br/>Run: <span class="mono">truelift all --dataset synthetic</span><br/>Then refresh.</p></div>`;
  }
}

function renderNav() {
  $("#nav").innerHTML = ROOMS.map(
    (r) => `<button data-r="${r.id}" class="${r.id === ROOM ? "on" : ""}">${r.label}</button>`
  ).join("");
  $("#nav").onclick = (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    ROOM = b.dataset.r;
    renderNav();
    show(ROOM);
  };
}

function paintShip() {
  const ok = RUN.card.ship;
  const el = $("#ship");
  el.textContent = ok ? "READY TO SHIP" : "HOLD — REVIEW GATES";
  el.className = "ship-pill " + (ok ? "ok" : "bad");
}

function paintHero() {
  const top = RUN.leaderboard.find((r) => r.model === RUN.champion) || RUN.leaderboard[0];
  $("#hero").innerHTML = `
    <h1>Causal targeting — not another conversion ranker</h1>
    <p class="tagline">Rank people by incremental lift τ, allocate budget to persuadables, and prove uplift beats mailing likely buyers.</p>
    <div class="hero-meta">
      <span class="hi">${RUN.champion}</span>
      <span>AUUC ${fmt(top.auuc, 2)}</span>
      <span>${RUN.n_test.toLocaleString()} holdout people</span>
      <span>${(RUN.arm_names || []).join(" · ")}</span>
    </div>`;
}

function card(title, body, delay = 0) {
  return `<section class="card" style="animation-delay:${delay}s"><h2>${title}</h2>${body}</section>`;
}

async function show(room) {
  const fn = { stage, lab, identify, allocate, desk, data, risk }[room];
  $("#app").innerHTML = `<div class="note" style="padding:24px">Loading…</div>`;
  await fn();
}

async function stage() {
  STAGE = await api("/api/stage?budget=" + budget);
  paintHeroStage();
  const maxInc = Math.max(...STAGE.policies.map((p) => p.incremental_rate), 1e-6);
  const policyRows = STAGE.policies
    .slice()
    .sort((a, b) => a.incremental_rate - b.incremental_rate)
    .map((p) => {
      const w = Math.max(4, (100 * p.incremental_rate) / maxInc);
      const cls = p.role === "us" ? "us" : p.role === "naive" ? "naive" : "waste";
      return `<div class="policy-row ${p.role === "us" ? "us" : ""}">
        <div class="policy-name">${p.name}${p.role === "us" ? " ★" : ""}<small>${p.n_treated.toLocaleString()} contacts · spend ${fmt(p.spend, 0)}</small></div>
        <div class="bar-track"><div class="bar-fill ${cls}" data-target="${w}"></div></div>
        <div class="policy-num">${fmt(p.incremental_rate, 4)}<br/><span style="color:var(--muted);font-size:10px">Δ rate</span></div>
      </div>`;
    })
    .join("");

  const offerHtml = Object.entries(STAGE.offers || {})
    .map(([k, v]) => `<span>${k}<strong>${v.toLocaleString()}</strong></span>`)
    .join("");

  const gates = (STAGE.gates || [])
    .map((x) => `<div class="gate"><span class="${x.ok ? "ok" : "bad"}">${x.ok ? "✓" : "✗"}</span><span>${x.id}</span><span class="note">${x.detail}</span></div>`)
    .join("");

  $("#app").innerHTML = `
    <div class="grid g2">
      ${card(
        STAGE.campaign,
        `
        <p class="note" style="font-size:15px;color:var(--text);margin-bottom:16px">${STAGE.thesis}</p>
        <div class="budget-bar">
          <label class="note">Mail budget — top fraction of holdout</label>
          <input type="range" id="stage-bud" min="0.05" max="0.65" step="0.05" value="${budget}" />
          <div class="budget-val">Targeting <strong>${pct(budget)}</strong> of ${STAGE.n_test.toLocaleString()} people</div>
        </div>
        <h3 style="margin-top:20px">Policy value at this budget</h3>
        ${policyRows}
        <p class="note" style="margin-top:14px">Incremental rate = treated-group lift observable in the RCT holdout (not a forecast).</p>
      `,
        0
      )}
      ${card(
        "Business impact",
        `
        <div class="stat-grid">
          <div class="stat"><div class="label">Wasted contacts</div><div class="val bad">${STAGE.wasted.n.toLocaleString()}</div><div class="note">~${fmt(STAGE.wasted.spend, 0)} spend on ~zero lift</div></div>
          <div class="stat"><div class="label">Rescued persuadables</div><div class="val ok">${STAGE.rescued.n.toLocaleString()}</div><div class="note">${STAGE.rescued.label}</div></div>
          <div class="stat"><div class="label">Sleeping dogs mailed</div><div class="val bad">${STAGE.dogs_mailed.toLocaleString()}</div><div class="note">by conversion model alone</div></div>
        </div>
        <div style="margin-top:18px">
          <div class="kpi ${STAGE.ship ? "ok" : "bad"}" style="font-size:22px">${STAGE.ship ? "Ship list" : "Hold"}<small>AUUC ${fmt(STAGE.auuc, 2)} vs response ${fmt(STAGE.response_auuc, 2)}</small></div>
        </div>
        <div style="margin-top:16px;display:flex;gap:10px;flex-wrap:wrap">
          <a class="btn" id="stage-export" href="/api/export.csv?budget=${budget}&mode=knapsack">Download treat/skip CSV</a>
          <button class="btn secondary" type="button" id="goto-alloc">Fine-tune allocator</button>
        </div>
      `,
        0.05
      )}
    </div>
    <div class="grid g2" style="margin-top:16px">
      ${card("Quadrant map — baseline propensity vs τ", `<div class="map-wrap"><div id="map"></div><div class="map-legend">${legendKinds()}</div></div>`, 0.1)}
      ${card("Offer mix under TrueLift policy", `<div class="offer-chips">${offerHtml}</div><p class="note" style="margin-top:12px">Arms: ${(STAGE.arms || []).join(", ")} · cost/contact ${fmt(STAGE.cost, 2)}</p>`, 0.12)}
    </div>
    <div class="grid g2" style="margin-top:16px">
      ${card("Qini — champion vs random vs response", `<div id="qini-stage"></div>`)}
      ${card("Ship gates", gates)}
    </div>`;

  drawMap($("#map"), STAGE.map, { onPick: openPersonDrawer });
  drawQini($("#qini-stage"), STAGE.qini, { qini: STAGE.qini_response, auuc: STAGE.response_auuc }, true);
  animatePolicyBars();

  $("#stage-bud").oninput = (e) => {
    budget = Number(e.target.value);
    $("#stage-bud").nextElementSibling.innerHTML = `Targeting <strong>${pct(budget)}</strong> of ${STAGE.n_test.toLocaleString()} people`;
  };
  $("#stage-bud").onchange = () => stage();
  $("#stage-export").href = `/api/export.csv?budget=${budget}&mode=knapsack`;
  $("#rail-export").href = `/api/export.csv?budget=${budget}&mode=knapsack`;
  $("#goto-alloc").onclick = () => {
    ROOM = "allocate";
    renderNav();
    show("allocate");
  };
}

function paintHeroStage() {
  if (!STAGE) return;
  $("#hero").innerHTML = `
    <h1>${STAGE.campaign}</h1>
    <p class="tagline">${STAGE.thesis} Champion <span class="gold">${STAGE.champion}</span> · budget ${pct(STAGE.budget)}.</p>
    <div class="hero-meta">
      <span class="hi">${STAGE.ship ? "SHIP" : "HOLD"}</span>
      <span>AUUC ${fmt(STAGE.auuc, 2)}</span>
      <span>Response AUUC ${fmt(STAGE.response_auuc, 2)}</span>
      <span>${STAGE.n_test.toLocaleString()} people</span>
    </div>`;
}

function legendKinds() {
  return ["persuadable", "sure_thing", "lost_cause", "sleeping_dog"]
    .map((k) => `<span><i class="sw" style="background:${colorKind(k)}"></i>${k.replace("_", " ")}</span>`)
    .join("");
}

function animatePolicyBars() {
  requestAnimationFrame(() => {
    document.querySelectorAll(".bar-fill[data-target]").forEach((el) => {
      const w = el.dataset.target;
      el.classList.add("animated");
      el.style.width = `${w}%`;
    });
  });
}

function initDrawer() {
  const close = () => {
    $("#drawer").setAttribute("hidden", "");
    $("#drawer-scrim").setAttribute("hidden", "");
  };
  $("#drawer-close").onclick = close;
  $("#drawer-scrim").onclick = close;
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") close();
  });
}

async function openPersonDrawer(index) {
  initDrawerOnce();
  $("#drawer-title").textContent = "Loading…";
  $("#drawer-body").innerHTML = `<p class="note">Fetching dossier…</p>`;
  $("#drawer").removeAttribute("hidden");
  $("#drawer-scrim").removeAttribute("hidden");
  try {
    const person = await api("/api/person/" + index);
    $("#drawer-title").textContent = `Person ${person.id} · ${(person.kind || "").replace("_", " ")}`;
    $("#drawer-body").innerHTML = renderPersonBody(person);
    animateBars();
    const go = $("#drawer-body").querySelector("#drawer-goto-desk");
    if (go) {
      go.onclick = () => {
        $("#drawer").setAttribute("hidden", "");
        $("#drawer-scrim").setAttribute("hidden", "");
        ROOM = "desk";
        renderNav();
        show("desk");
      };
    }
  } catch (e) {
    $("#drawer-body").innerHTML = `<p class="bad">${e}</p>`;
  }
}

let drawerReady = false;
function initDrawerOnce() {
  if (drawerReady) return;
  drawerReady = true;
  initDrawer();
}

function renderPersonBody(p) {
  const feats = Object.entries(p.features || {}).slice(0, 12);
  const max = Math.max(...feats.map(([, v]) => Math.abs(v)), 1e-6);
  return `
    <div class="kpi">${fmt(p.tau, 3)}<small>τ [${fmt(p.tau_lo)} , ${fmt(p.tau_hi)}]</small></div>
    <p class="mono" style="margin:12px 0">Y(0) ${fmt(p.mu0)} · Y(1) ${fmt(p.mu1)}<br/>
    Policy: ${p.assign > 0 ? "TREAT" : "SKIP"} · response ${fmt(p.response_score, 3)}
    ${p.true_tau != null ? `<br/>Oracle τ ${fmt(p.true_tau)}` : ""}</p>
    <p class="note">Click any dot on the map to inspect someone in the holdout.</p>
    <h3 style="margin-top:16px">Features</h3>
    ${feats.map(([k, v]) => `<div class="mono">${k}<div class="bar"><i data-w="${Math.min(100, 50 + (50 * v) / max)}"></i></div></div>`).join("")}
    <h3 style="margin-top:12px">τ drivers</h3>
    ${(p.contributions || []).slice(0, 8).map((c) => `<div class="mono">${c.feature} ${fmt(c.delta_tau, 4)}</div>`).join("") || '<p class="note">No contribution vector for this run.</p>'}
    <button type="button" class="btn secondary" style="margin-top:16px" id="drawer-goto-desk">Open in People room</button>
  `;
}

function drawMap(el, points, opts = {}) {
  if (!el || !points?.length) return;
  const w = 640,
    h = 320,
    p = 36;
  const xs = points.map((d) => d.x),
    ys = points.map((d) => d.y);
  const x0 = 0,
    x1 = Math.max(0.35, ...xs);
  const y0 = Math.min(-0.08, ...ys),
    y1 = Math.max(0.12, ...ys);
  const X = (v) => p + ((v - x0) / (x1 - x0)) * (w - 2 * p);
  const Y = (v) => h - p - ((v - y0) / (y1 - y0)) * (h - 2 * p);
  const dots = points
    .map((d, idx) => {
      const r = d.ours ? 4.2 : 2.8;
      const stroke = d.theirs && !d.ours ? "var(--blue)" : d.ours && !d.theirs ? "var(--gold)" : "transparent";
      const op = d.ours || d.theirs ? 0.85 : 0.35;
      const delay = (idx % 20) * 0.015;
      return `<circle class="map-dot" style="animation-delay:${delay}s" cx="${X(d.x)}" cy="${Y(d.y)}" r="${r}" fill="${colorKind(d.kind)}" fill-opacity="${op}" stroke="${stroke}" stroke-width="1.5" data-i="${d.i}"><title>${d.kind} · μ₀ ${fmt(d.x)} · τ ${fmt(d.y)} · click for dossier</title></circle>`;
    })
    .join("");
  el.innerHTML = `<svg class="chart" viewBox="0 0 ${w} ${h}">
    <line x1="${p}" y1="${Y(0)}" x2="${w - p}" y2="${Y(0)}" stroke="#243044" stroke-dasharray="4 4"/>
    <text x="${w - p - 4}" y="${Y(0) - 6}" fill="#7d8ba3" font-size="10" text-anchor="end">τ = 0</text>
    <text x="${p}" y="${h - 8}" fill="#7d8ba3" font-size="10">P(Y|no offer) →</text>
    <text x="8" y="${p}" fill="#7d8ba3" font-size="10" transform="rotate(-90 12 ${p})">uplift τ →</text>
    ${dots}
  </svg>
  <p class="note">Click a dot for a person dossier. Gold ring = TrueLift only · blue = conversion model only.</p>`;
  el.querySelector("svg").onclick = (ev) => {
    const c = ev.target.closest("[data-i]");
    if (!c) return;
    const i = Number(c.dataset.i);
    if (opts.onPick) opts.onPick(i);
  };
}

async function data() {
  $("#app").innerHTML = `
    <div class="grid g2">
      ${card(
        "Train on experiment data",
        `
        <p class="note">Upload a labeled CSV (treatment + outcome + features) or retrain a public benchmark. Engine fits propensity, CATE zoo, conformal intervals, and writes artifacts for this dashboard.</p>
        <div class="toggle" id="src">
          <button class="on" data-s="synthetic">synthetic</button>
          <button data-s="hillstrom">hillstrom</button>
          <button data-s="ihdp">ihdp</button>
          <button data-s="jobs">jobs</button>
        </div>
        <p class="mono"><a class="gold" href="/api/template.csv">Download campaign template CSV</a></p>
        <label class="note">Upload labeled CSV (optional)</label>
        <input type="file" id="csv" accept=".csv" />
        <div class="grid g2" style="margin-top:10px">
          <label class="note">Treatment column<input type="text" id="tcol" value="treatment" /></label>
          <label class="note">Outcome column<input type="text" id="ocol" value="visit" /></label>
        </div>
        <button class="btn" type="button" id="go" style="margin-top:14px">Train engine</button>
        <p class="note" id="trainmsg">Synthetic ~30s · Hillstrom longer.</p>
      `
      )}
      ${card(
        "Score cold audience (no outcomes yet)",
        `
        <p class="note">Features-only CSV scored with the saved champion. Returns treat/skip with τ and conformal bounds.</p>
        <input type="file" id="scorefile" accept=".csv" />
        <label class="note">Budget fraction <span class="mono" id="sbudv">0.30</span></label>
        <input type="range" id="sbud" min="0.05" max="0.8" step="0.05" value="0.3" />
        <button class="btn secondary" type="button" id="scorego" style="margin-top:12px">Score & download</button>
        <p class="mono" id="scoremsg"></p>
      `
      )}
    </div>
    <div style="margin-top:16px">${card(
      "Sample files",
      `<p class="note">Bundled under <span class="mono">data/samples/</span>: <strong>campaign_labeled.csv</strong> (train), <strong>new_customers.csv</strong> (score), <strong>tiny_walkthrough.csv</strong>.</p>`
    )}</div>`;

  let dataset = "synthetic";
  $("#src").onclick = (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    dataset = b.dataset.s;
    [...$("#src").children].forEach((x) => x.classList.toggle("on", x === b));
  };
  $("#sbud").oninput = (e) => {
    $("#sbudv").textContent = Number(e.target.value).toFixed(2);
  };
  $("#go").onclick = async () => {
    $("#trainmsg").textContent = "Training… keep this tab open.";
    const fd = new FormData();
    fd.append("dataset", dataset);
    fd.append("treatment", $("#tcol").value);
    fd.append("outcome", $("#ocol").value);
    const f = $("#csv").files[0];
    if (f) fd.append("file", f);
    try {
      const r = await fetch("/api/train", { method: "POST", body: fd });
      const body = await r.json();
      if (!r.ok) throw new Error(JSON.stringify(body));
      $("#trainmsg").textContent = `Done — ${body.champion} · AUUC ${fmt(body.auuc, 2)} · ${body.ship ? "SHIP" : "HOLD"}`;
      RUN = await api("/api/run");
      paintShip();
      paintHero();
      ROOM = "stage";
      renderNav();
      await stage();
    } catch (e) {
      $("#trainmsg").textContent = String(e);
    }
  };
  $("#scorego").onclick = async () => {
    const f = $("#scorefile").files[0];
    if (!f) {
      $("#scoremsg").textContent = "Choose a CSV.";
      return;
    }
    const fd = new FormData();
    fd.append("file", f);
    fd.append("budget", $("#sbud").value);
    fd.append("mode", "knapsack");
    const r = await fetch("/api/score", { method: "POST", body: fd });
    if (!r.ok) {
      $("#scoremsg").textContent = await r.text();
      return;
    }
    const blob = await r.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "truelift_scored.csv";
    a.click();
    $("#scoremsg").textContent = "Downloaded scored list.";
  };
}

function lab() {
  const rows = RUN.leaderboard
    .filter((r) => !r.baseline)
    .map(
      (r, i) => `<tr>
        <td>${r.model === RUN.champion ? "★ " : ""}${r.model}</td>
        <td class="gold">${fmt(r.auuc, 2)}</td>
        <td>${fmt(r.normalized_auuc)}</td>
        <td>${fmt(r["uplift@10"])}</td>
        <td>${fmt(r["uplift@30"])}</td>
        <td>${r.pehe != null ? fmt(r.pehe) : "—"}</td>
      </tr>`
    )
    .join("");
  $("#app").innerHTML = `
    <div class="grid g2">
      ${card(
        "Estimator zoo — same split, same people",
        `
        <table><thead><tr><th>Model</th><th>AUUC</th><th>Norm</th><th>u@10</th><th>u@30</th><th>PEHE</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <p class="note">Dominance vs response ranker: Δ AUUC ${fmt(RUN.dominance_vs_response.delta_auuc_mean, 2)}
        [${fmt(RUN.dominance_vs_response.delta_lo, 2)}, ${fmt(RUN.dominance_vs_response.delta_hi, 2)}]
        ${RUN.dominance_vs_response.beats ? "— champion above response interval." : "— compare policy value on Allocate."}</p>`
      )}
      ${card(
        "Qini curves",
        `<label class="note">Model</label>
        <select id="mselect">${RUN.leaderboard.filter((r) => !r.baseline).map((r) => `<option ${r.model === RUN.champion ? "selected" : ""}>${r.model}</option>`).join("")}</select>
        <div id="qini2"></div>`
      )}
    </div>`;
  const sel = $("#mselect");
  const paint = () => drawQini($("#qini2"), RUN.qini[sel.value], RUN.qini_response, true);
  sel.onchange = paint;
  paint();
}

function identify() {
  const id = RUN.identification;
  const ov = id.overlap;
  const hist = ov.histogram || [];
  $("#app").innerHTML = `
    <div class="grid g2">
      ${card(
        "Estimand",
        `
        <div class="kpi">${id.estimand}<small>${id.randomized ? "Randomized experiment" : "Observational"}</small></div>
        <p class="note">${(id.notes || []).join(" ")}</p>
        <p class="mono">Unconfoundedness: ${id.unconfoundedness}<br/>IV / LATE: ${id.iv_ready}<br/>ATE (IPW): ${fmt(RUN.ate_ipw)}
        ${RUN.late ? `<br/>ITT ${fmt(RUN.late.itt)} · first-stage ${fmt(RUN.late.first_stage)} · Wald LATE ${fmt(RUN.late.wald_late)}` : ""}</p>`
      )}
      ${card(
        "Overlap / positivity",
        `
        <div class="kpi ${ov.positivity_ok ? "ok" : "bad"}">${fmt(ov.common_support_frac, 3)}<small>Common support</small></div>
        <div id="hist"></div>
        <p class="mono">ē ${fmt(ov.e_mean)} · min ${fmt(ov.e_min)} · max ${fmt(ov.e_max)} · ESS_t ${fmt(ov.ess_treated, 0)} · ESS_c ${fmt(ov.ess_control, 0)}</p>`
      )}
    </div>
    <div class="grid g2" style="margin-top:16px">
      ${card("Rosenbaum Γ", `<div class="kpi">${fmt(RUN.rosenbaum.gamma_critical)}<small>${RUN.rosenbaum.note}</small></div>`)}
      ${card(
        "Leakage scan",
        RUN.leakage.ok
          ? `<p class="ok">No high-severity post-treatment flags in features.</p>`
          : `<p class="bad">Review flags</p>` + (RUN.leakage.flags || []).map((f) => `<div class="mono">${f.feature} AUC ${f.auc_vs_treatment} (${f.severity})</div>`).join("")
      )}
    </div>`;
  drawHist($("#hist"), hist);
}

async function allocate() {
  $("#app").innerHTML = `
    <div class="card">
      <h2>Allocator — knapsack, conformal, fairness</h2>
      <div class="grid g3">
        <div><label class="note">Budget fraction</label><input type="range" id="bud" min="0.05" max="0.8" step="0.05" value="${budget}" /><div class="budget-val" id="budv">${budget.toFixed(2)}</div></div>
        <div class="toggle" id="mode">
          <button class="on" data-m="knapsack">Knapsack</button>
          <button data-m="topk">Top-k</button>
        </div>
        <div class="toggle">
          <button id="cf">Conformal τ_lo&gt;0</button>
          <button id="ff">Fairness slack</button>
        </div>
      </div>
      <div id="pol" class="grid g4" style="margin-top:12px"></div>
      <div id="kinds" class="map-legend" style="margin-top:12px"></div>
      <p class="mono" style="margin-top:12px"><a id="dl" class="gold" href="/api/export.csv?budget=${budget}&mode=knapsack">Download assignment CSV</a></p>
    </div>
    <div class="grid g2" style="margin-top:16px">
      ${card("Pareto — value vs treat-rate gap", `<div id="pareto"></div>`)}
      ${card("Distilled policy tree", `<pre class="mono" style="white-space:pre-wrap;font-size:11px">${JSON.stringify(RUN.policy_tree, null, 2)}</pre>`)}
    </div>`;
  const state = { budget, mode: "knapsack", conformal: false, fairness: false };
  const load = async () => {
    const q = new URLSearchParams(state);
    const p = await api("/api/policy?" + q.toString());
    $("#pol").innerHTML = [
      ["Treated", p.n_treated],
      ["Incremental rate", fmt(p.incremental_rate, 4)],
      ["Spent", fmt(p.spent, 1)],
      ["Value", fmt(p.expected_value ?? p.policy_value_dr, 2)],
    ]
      .map(([k, v], i) => `<div class="card" style="animation-delay:${i * 0.05}s;padding:14px"><h3>${k}</h3><div class="kpi" style="font-size:22px">${v}</div></div>`)
      .join("");
    const ks = p.kinds || {};
    $("#kinds").innerHTML = Object.entries(ks)
      .map(([k, v]) => `<span><i class="sw" style="background:${colorKind(k)}"></i>${k} ${v}</span>`)
      .join("");
    $("#dl").href = "/api/export.csv?" + new URLSearchParams({ ...state, conformal: state.conformal, fairness: state.fairness }).toString();
  };
  $("#bud").oninput = (e) => {
    state.budget = Number(e.target.value);
    budget = state.budget;
    $("#budv").textContent = state.budget.toFixed(2);
  };
  $("#bud").onchange = load;
  $("#mode").onclick = (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    state.mode = b.dataset.m;
    [...$("#mode").children].forEach((x) => x.classList.toggle("on", x === b));
    load();
  };
  $("#cf").onclick = () => {
    state.conformal = !state.conformal;
    $("#cf").classList.toggle("on", state.conformal);
    load();
  };
  $("#ff").onclick = () => {
    state.fairness = !state.fairness;
    $("#ff").classList.toggle("on", state.fairness);
    load();
  };
  drawPareto($("#pareto"), RUN.pareto);
  await load();
}

async function desk() {
  const feat = RUN.people;
  $("#app").innerHTML = `
    <div class="grid g2">
      ${card(
        "Four archetypes",
        `<div class="toggle" id="kinds">
        ${["all", "persuadable", "sure_thing", "lost_cause", "sleeping_dog"].map((k) => `<button data-k="${k}" class="${k === "all" ? "on" : ""}">${k.replace("_", " ")}</button>`).join("")}
      </div><div id="list"></div>`
      )}
      <div id="detail">${renderPerson(feat[0])}</div>
    </div>`;
  const list = $("#list");
  const renderList = (items) => {
    list.innerHTML = items
      .map((p) => `<div class="chip" data-i="${p.index}">${p.kind.replace("_", " ")} · τ ${fmt(p.tau)} · id ${p.id}</div>`)
      .join("");
    list.onclick = async (ev) => {
      const c = ev.target.closest("[data-i]");
      if (!c) return;
      const person = await api("/api/person/" + c.dataset.i);
      $("#detail").innerHTML = renderPerson(person);
      animateBars();
    };
  };
  renderList(feat);
  $("#kinds").onclick = async (ev) => {
    const b = ev.target.closest("button");
    if (!b) return;
    [...$("#kinds").children].forEach((x) => x.classList.toggle("on", x === b));
    const data = await api("/api/people?kind=" + b.dataset.k);
    renderList(data.items || data.featured || []);
  };
  animateBars();
}

function renderPerson(p) {
  if (!p) return card("Person", "—");
  const feats = Object.entries(p.features || {}).slice(0, 10);
  const max = Math.max(...feats.map(([, v]) => Math.abs(v)), 1e-6);
  return card(
    `Person ${p.id} · ${(p.kind || "").replace("_", " ")}`,
    `
    <div class="person-panel">
      <div><div class="kpi">${fmt(p.tau, 3)}<small>τ [${fmt(p.tau_lo)} , ${fmt(p.tau_hi)}]</small></div></div>
      <div class="mono">Y(0) ${fmt(p.mu0)} · Y(1) ${fmt(p.mu1)}<br/>
      Policy: ${p.assign > 0 || p.we_treat ? "TREAT" : "SKIP"} · response score ${fmt(p.response_score ?? p.response, 3)}
      ${p.true_tau != null ? `<br/>Oracle τ ${fmt(p.true_tau)}` : ""}</div>
    </div>
    <h3>Features</h3>
    ${feats.map(([k, v]) => `<div class="mono">${k}<div class="bar"><i data-w="${Math.min(100, 50 + (50 * v) / max)}"></i></div></div>`).join("")}
    <h3>τ drivers</h3>
    ${(p.contributions || []).slice(0, 6).map((c) => `<div class="mono">${c.feature} ${fmt(c.delta_tau, 4)}</div>`).join("") || '<p class="note">Train with SHAP-enabled run for contributions.</p>'}
  `
  );
}

function risk() {
  const r = RUN;
  $("#app").innerHTML = `
    <div class="grid g2">
      ${card("Model card", `<pre class="mono" style="white-space:pre-wrap;font-size:11px">${JSON.stringify(r.card, null, 2)}</pre>`)}
      ${card("Fairness", `<pre class="mono" style="font-size:11px">${JSON.stringify(r.fairness, null, 2)}</pre>`)}
    </div>
    <div class="grid g2" style="margin-top:16px">
      ${card(
        "Robustness gym",
        (r.robustness.scenarios || [])
          .map((s) => `<div class="mono">${s.scenario} · AUUC ${fmt(s.auuc, 2)} · norm ${fmt(s.normalized_auuc)}</div>`)
          .join("")
      )}
      ${card("Bootstrap AUUC", `<div class="kpi">${fmt(r.bootstrap.auuc_mean, 2)}<small>[${fmt(r.bootstrap.auuc_lo, 2)}, ${fmt(r.bootstrap.auuc_hi, 2)}]</small></div>`)}
    </div>
    <div style="margin-top:16px">${card(
      "Gates",
      (r.gates || []).map((x) => `<div class="gate"><span class="${x.ok ? "ok" : "bad"}">${x.ok ? "✓" : "✗"}</span><span>${x.id}</span><span class="note">${x.detail}</span></div>`).join("")
    )}</div>`;
}

function colorKind(k) {
  return { persuadable: "#5eead4", sure_thing: "#e8c468", lost_cause: "#64748b", sleeping_dog: "#f472b6" }[k] || "#60a5fa";
}

function drawQini(el, q, resp, animate = false) {
  if (!el || !q) return;
  const w = 640,
    h = 260,
    p = 28;
  const respQ = resp && resp.qini ? resp.qini : q.random;
  const series = [
    { y: q.qini, c: "#e8c468", n: "champion", d: "" },
    { y: q.random, c: "#64748b", n: "random", d: "d1" },
    { y: respQ, c: "#60a5fa", n: "response", d: "d2" },
  ];
  const all = series.flatMap((s) => s.y).concat(q.qini_lo || []).concat(q.qini_hi || []);
  const min = Math.min(0, ...all),
    max = Math.max(...all, 1e-6);
  const fracs = q.fracs || series[0].y.map((_, i) => i / (series[0].y.length - 1));
  const x = (i) => p + (i / (fracs.length - 1)) * (w - 2 * p);
  const yv = (v) => h - p - ((v - min) / (max - min)) * (h - 2 * p);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i)},${yv(v)}`).join(" ");
  let band = "";
  if (q.qini_lo && q.qini_hi) {
    const up = q.qini_hi.map((v, i) => `${i ? "L" : "M"}${x(i)},${yv(v)}`).join(" ");
    const dn = [...q.qini_lo].reverse().map((v, i) => `L${x(q.qini_lo.length - 1 - i)},${yv(v)}`).join(" ");
    band = `<path class="band-fill" d="${up} ${dn} Z" fill="rgba(232,196,104,0.12)" stroke="none" />`;
  }
  const lineCls = animate ? "line-draw" : "";
  el.innerHTML = `<svg class="chart" viewBox="0 0 ${w} ${h}">
    ${band}
    ${series.map((s) => `<path class="${lineCls} ${s.d}" d="${path(s.y)}" stroke="${s.c}" />`).join("")}
  </svg>
  <div class="map-legend">${series.map((s) => `<span><i class="sw" style="background:${s.c}"></i>${s.n}</span>`).join("")}<span>AUUC champion ${fmt(q.auuc, 2)}${resp && resp.auuc != null ? ` · response ${fmt(resp.auuc, 2)}` : ""}</span></div>`;
}

function drawHist(el, hist) {
  if (!el) return;
  const m = Math.max(...hist, 1);
  el.innerHTML = `<div style="display:flex;align-items:flex-end;gap:3px;height:120px">${hist
    .map((v, i) => `<div class="hist-bar" data-h="${(v / m) * 100}" style="flex:1;background:linear-gradient(180deg,#5eead4,#134e4a);border-radius:3px 3px 0 0;transition-delay:${i * 0.012}s"></div>`)
    .join("")}</div>`;
  requestAnimationFrame(() => {
    el.querySelectorAll(".hist-bar").forEach((b) => {
      b.style.height = `${b.dataset.h}%`;
    });
  });
}

function drawPareto(el, pts) {
  if (!el || !pts) return;
  const w = 520,
    h = 220,
    p = 28;
  const xs = pts.map((d) => d.fairness_gap),
    ys = pts.map((d) => d.value);
  const x0 = Math.min(...xs),
    x1 = Math.max(...xs, x0 + 1e-6);
  const y0 = Math.min(...ys),
    y1 = Math.max(...ys, y0 + 1e-6);
  const X = (v) => p + ((v - x0) / (x1 - x0)) * (w - 2 * p);
  const Y = (v) => h - p - ((v - y0) / (y1 - y0)) * (h - 2 * p);
  const line = pts.map((d, i) => `${i ? "L" : "M"}${X(d.fairness_gap)},${Y(d.value)}`).join(" ");
  el.innerHTML = `<svg class="chart" viewBox="0 0 ${w} ${h}">
    <path class="line-draw" d="${line}" stroke="#5eead4" />
    ${pts.map((d, i) => `<circle class="map-dot" style="animation-delay:${i * 0.06}s" cx="${X(d.fairness_gap)}" cy="${Y(d.value)}" r="5" fill="#e8c468"><title>budget ${d.budget_frac}</title></circle>`).join("")}
  </svg><div class="note">x = treat-rate gap · y = expected incremental value</div>`;
}

function animateBars() {
  requestAnimationFrame(() => {
    document.querySelectorAll(".bar > i").forEach((el) => {
      el.style.width = (el.dataset.w || 0) + "%";
    });
  });
}

boot();
