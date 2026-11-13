const state = {
  activeTab: "tab-eda",
  selectedDepartment: "All Departments",
  activeModel: "xgb",
  overview: null,
  departments: [],
  triage: null,
  cashflow: null,
  selectedInvoice: null,
  docsList: [],
  mainChart: null,
  shapChart: null,
};

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

const fmtInt = (n) => Number(n || 0).toLocaleString("en-US");
const fmtCurrency = (n) =>
  "$" + Number(n || 0).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  return response.json();
}

function updateWorkspaceHeaderForTab() {
  const h1 = document.getElementById("workspace-headline");
  const sub = document.getElementById("workspace-subheadline");
  if (!h1 || !sub) return;

  const copy = {
    "tab-eda": [
      "Metro Nashville Vendor Payment Grain Audit & Statistical Distribution",
      "1,307,617 raw ArcGIS lines consolidated via DuckDB into 811,914 clean invoices with a 61-day right-censoring buffer.",
    ],
    "tab-risk": [
      "Track 1 — Late Payment Risk Scoring & Calibrated Probability Triage",
      "Calibrated XGBoost achieves 0.4092 PR-AUC and 2.83x Lift@5% (58.81% precision) on 195,889 out-of-time 2025 invoices.",
    ],
    "tab-cashflow": [
      "Track 2 — Settlement Duration Regression & Treasury Cash Outflow Forecast",
      "log1p-trained XGBoost reduces Test MAE to 8.73 days and 14-day cumulative cash outflow MAPE to 0.65% ($1,769.6M vs $1,781.1M).",
    ],
  }[state.activeTab];

  if (copy) {
    h1.textContent = copy[0];
    sub.textContent = copy[1];
  }
}

function renderHeaderKpis() {
  const strip = document.getElementById("header-kpi-strip");
  if (!strip || !state.overview) return;

  const testXgbCls =
    state.overview.classification_benchmark.find(
      (r) => r.model === "XGBoost (Isotonic)" && r.split === "Test"
    ) || state.overview.classification_benchmark.find((r) => r.split === "Test");

  const testXgbReg =
    state.overview.regression_benchmark.find(
      (r) => r.model === "XGBoost Regressor" && r.split === "Test"
    ) || state.overview.regression_benchmark.find((r) => r.split === "Test");

  const kpis = [
    {
      label: "Test PR-AUC",
      value: testXgbCls ? testXgbCls.pr_auc.toFixed(4) : "0.4092",
      sub: testXgbCls ? `Lift@5% ${testXgbCls.lift_5.toFixed(2)}x` : "Calibrated",
    },
    {
      label: "Test MAE",
      value: testXgbReg ? `${testXgbReg.mae.toFixed(2)}d` : "8.73d",
      sub: testXgbReg ? `RMSE ${testXgbReg.rmse.toFixed(2)}d` : "log1p GBDT",
    },
    {
      label: "14d Cashflow MAPE",
      value: testXgbReg ? `${testXgbReg.mape_14d.toFixed(2)}%` : "0.65%",
      sub: "$1,769.6M vs $1,781.1M",
    },
  ];

  strip.innerHTML = kpis
    .map(
      (k) => `
      <div class="px-3 py-1.5 rounded-md bg-slate-50 border border-slate-200 text-right">
        <div class="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">${escapeHtml(k.label)}</div>
        <div class="text-sm font-bold font-mono-num text-slate-900">${escapeHtml(k.value)}</div>
        <div class="text-[10px] font-mono-num text-emerald-600">${escapeHtml(k.sub)}</div>
      </div>`
    )
    .join("");
}

function renderSidebarDepartments(filterText = "") {
  const listEl = document.getElementById("department-list");
  const countEl = document.getElementById("sidebar-dept-count");
  if (!listEl) return;

  const query = filterText.trim().toLowerCase();
  const filtered = state.departments.filter((d) =>
    d.department.toLowerCase().includes(query)
  );
  if (countEl) {
    countEl.textContent = `${filtered.length} depts`;
  }

  const testSplit = (state.overview?.splits || []).find((s) =>
    s.split.startsWith("Test")
  );
  const allItem = {
    department: "All Departments",
    invoices: testSplit ? testSplit.invoices : 195889,
    actual_late_rate_pct: testSplit ? testSplit.late_rate_pct : 20.76,
  };

  const rows = [allItem, ...filtered];
  listEl.innerHTML = rows
    .map((item) => {
      const isActive = state.selectedDepartment === item.department;
      const dotColor =
        item.actual_late_rate_pct >= 30
          ? "bg-rose-500"
          : item.actual_late_rate_pct >= 20
          ? "bg-amber-500"
          : "bg-emerald-500";
      return `
        <div class="dept-item ${isActive ? "active" : ""}" data-dept-name="${escapeHtml(item.department)}">
          <span class="w-3 h-3 rounded-full ${dotColor} mt-1 shrink-0 ring-2 ring-slate-100"></span>
          <div class="min-w-0 flex-1">
            <div class="text-xs font-semibold text-slate-800 truncate">${escapeHtml(item.department)}</div>
            <div class="text-[11px] font-mono-num text-slate-500 flex items-center justify-between mt-0.5">
              <span>${fmtInt(item.invoices)} inv</span>
              <span>${item.actual_late_rate_pct.toFixed(1)}% late</span>
            </div>
          </div>
        </div>`;
    })
    .join("");

  listEl.querySelectorAll(".dept-item").forEach((el) => {
    el.addEventListener("click", async () => {
      state.selectedDepartment = el.getAttribute("data-dept-name") || "All Departments";
      state.selectedInvoice = null;
      const pill = document.getElementById("active-scope-pill");
      if (pill) {
        pill.textContent =
          state.selectedDepartment === "All Departments"
            ? "All Departments (Citywide)"
            : state.selectedDepartment;
      }
      renderSidebarDepartments(document.getElementById("dept-search-input")?.value || "");
      await Promise.all([loadTriageData(), loadCashflowData()]);
      renderMainInteractiveChart();
      renderRightInspector();
    });
  });
}

function getVisualCardsForTab() {
  if (!state.overview) return [];
  const eda = state.overview.eda_plots || [];
  const cls = state.overview.cls_plots || [];
  const reg = state.overview.reg_plots || [];

  if (state.activeTab === "tab-eda") {
    const titles = [
      "1. Audit Funnel (1.31M -> 811.9k)",
      "2. Net-30 Spike & Right-Skew",
      "3. 61-Day Censoring Verification",
    ];
    return eda.slice(0, 3).map((p, i) => ({ url: p.url, title: titles[i] || p.filename, caption: p.caption }));
  }
  if (state.activeTab === "tab-risk") {
    const titles = [
      "1. Precision-Recall (PR-AUC 0.4092)",
      "2. Isotonic Reliability Curve",
      "3. SHAP Global Feature Drivers",
    ];
    return cls.slice(0, 3).map((p, i) => ({ url: p.url, title: titles[i] || p.filename, caption: p.caption }));
  }
  const titles = [
    "1. Horizon Outflow (14d MAPE 0.65%)",
    "2. Fiscal Year-End Seasonality",
    "3. Department & Operating Unit Tests",
  ];
  return [...reg, ...eda.slice(3, 5)]
    .slice(0, 3)
    .map((p, i) => ({ url: p.url, title: titles[i] || p.filename, caption: p.caption }));
}

function renderVisualCards() {
  const grid = document.getElementById("visual-cards-grid");
  if (!grid) return;
  const cards = getVisualCardsForTab();
  if (!cards.length) {
    grid.innerHTML = `<div class="col-span-3 text-xs text-slate-500 p-4 border border-slate-200 rounded">Diagnostic plots running in baseline fallback mode.</div>`;
    return;
  }

  grid.innerHTML = cards
    .map(
      (c) => `
      <article class="visual-card" data-zoom-url="${escapeHtml(c.url)}" data-zoom-caption="${escapeHtml(c.caption)}">
        <div class="visual-card-media">
          <img src="${escapeHtml(c.url)}" alt="${escapeHtml(c.title)}" loading="lazy" />
        </div>
        <div>
          <div class="text-xs font-bold text-slate-800 truncate">${escapeHtml(c.title)}</div>
          <div class="text-[11px] text-slate-500 line-clamp-2 mt-0.5">${escapeHtml(c.caption)}</div>
        </div>
      </article>`
    )
    .join("");

  grid.querySelectorAll(".visual-card").forEach((card) => {
    card.addEventListener("click", () => {
      openLightbox(card.getAttribute("data-zoom-url"), card.getAttribute("data-zoom-caption"));
    });
  });
}

function buildSplitsTableHtml(rows) {
  return `
    <table class="w-full text-left border-collapse text-xs">
      <thead class="bg-slate-50 text-[11px] uppercase text-slate-500 border-b border-slate-200">
        <tr>
          <th class="py-2 px-2.5">Segment</th>
          <th class="py-2 px-2.5">Window</th>
          <th class="py-2 px-2.5 text-right">Invoices</th>
          <th class="py-2 px-2.5 text-right">Late Rate</th>
          <th class="py-2 px-2.5">Guardrail Role</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-100 font-mono-num text-[11px]">
        ${rows
          .map(
            (r) => `
          <tr>
            <td class="py-2 px-2.5 font-sans font-semibold text-slate-800">${escapeHtml(r.split)}</td>
            <td class="py-2 px-2.5 text-slate-600">${escapeHtml(r.window)}</td>
            <td class="py-2 px-2.5 text-right font-semibold">${fmtInt(r.invoices)}</td>
            <td class="py-2 px-2.5 text-right">${r.late_rate_pct.toFixed(2)}%</td>
            <td class="py-2 px-2.5 font-sans text-slate-500">${escapeHtml(r.role_and_guardrail)}</td>
          </tr>`
          )
          .join("")}
      </tbody>
    </table>`;
}

function buildClsTableHtml(rows) {
  return `
    <table class="w-full text-left border-collapse text-xs">
      <thead class="bg-slate-50 text-[11px] uppercase text-slate-500 border-b border-slate-200">
        <tr>
          <th class="py-2 px-2">Split</th>
          <th class="py-2 px-2">Model</th>
          <th class="py-2 px-2 text-right">PR-AUC</th>
          <th class="py-2 px-2 text-right">ROC-AUC</th>
          <th class="py-2 px-2 text-right">Brier</th>
          <th class="py-2 px-2 text-right">Lift@5%</th>
          <th class="py-2 px-2 text-right">Lift@10%</th>
          <th class="py-2 px-2 text-right">ValCap@20%</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-100 font-mono-num text-[11px]">
        ${rows
          .map((r) => {
            const hi = r.model.includes("XGBoost") && r.split === "Test";
            return `
            <tr class="${hi ? "bg-blue-50/70 font-semibold text-slate-900" : ""}">
              <td class="py-1.5 px-2">${escapeHtml(r.split)}</td>
              <td class="py-1.5 px-2 font-sans">${escapeHtml(r.model)}</td>
              <td class="py-1.5 px-2 text-right">${r.pr_auc.toFixed(4)}</td>
              <td class="py-1.5 px-2 text-right">${r.roc_auc.toFixed(4)}</td>
              <td class="py-1.5 px-2 text-right">${r.brier_score.toFixed(4)}</td>
              <td class="py-1.5 px-2 text-right">${r.lift_5.toFixed(2)}x</td>
              <td class="py-1.5 px-2 text-right">${r.lift_10.toFixed(2)}x</td>
              <td class="py-1.5 px-2 text-right">${r.value_captured_20_pct.toFixed(2)}%</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;
}

function buildRegTableHtml(rows) {
  return `
    <table class="w-full text-left border-collapse text-xs">
      <thead class="bg-slate-50 text-[11px] uppercase text-slate-500 border-b border-slate-200">
        <tr>
          <th class="py-2 px-2">Split</th>
          <th class="py-2 px-2">Model</th>
          <th class="py-2 px-2 text-right">MAE (d)</th>
          <th class="py-2 px-2 text-right">RMSE (d)</th>
          <th class="py-2 px-2 text-right">MedianAE</th>
          <th class="py-2 px-2 text-right">7d MAPE</th>
          <th class="py-2 px-2 text-right">14d MAPE</th>
          <th class="py-2 px-2 text-right">30d MAPE</th>
        </tr>
      </thead>
      <tbody class="divide-y divide-slate-100 font-mono-num text-[11px]">
        ${rows
          .map((r) => {
            const hi = r.model.includes("XGBoost") && r.split === "Test";
            return `
            <tr class="${hi ? "bg-emerald-50/70 font-semibold text-slate-900" : ""}">
              <td class="py-1.5 px-2">${escapeHtml(r.split)}</td>
              <td class="py-1.5 px-2 font-sans">${escapeHtml(r.model)}</td>
              <td class="py-1.5 px-2 text-right">${r.mae.toFixed(2)}</td>
              <td class="py-1.5 px-2 text-right">${r.rmse.toFixed(2)}</td>
              <td class="py-1.5 px-2 text-right">${r.median_ae.toFixed(2)}</td>
              <td class="py-1.5 px-2 text-right">${r.mape_7d.toFixed(2)}%</td>
              <td class="py-1.5 px-2 text-right text-emerald-700 font-bold">${r.mape_14d.toFixed(2)}%</td>
              <td class="py-1.5 px-2 text-right">${r.mape_30d.toFixed(2)}%</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;
}

function renderBenchmarkTable() {
  const container = document.getElementById("benchmark-table-container");
  const titleEl = document.getElementById("benchmark-table-title");
  const subEl = document.getElementById("benchmark-table-subtitle");
  if (!container || !state.overview) return;

  if (state.activeTab === "tab-eda") {
    titleEl.textContent = "DuckDB Grain Audit & Out-of-Time Split Guardrails";
    subEl.textContent = "Strict temporal boundaries preventing payment ledger truncation and right-censoring leakage";
    container.innerHTML = buildSplitsTableHtml(state.overview.splits || []);
  } else if (state.activeTab === "tab-risk") {
    titleEl.textContent = "Track 1 — Late Payment Classification Benchmark (is_late: days > 30)";
    subEl.textContent = "Out-of-Time Validation & Test splits with post-hoc Isotonic Calibration";
    container.innerHTML = buildClsTableHtml(state.overview.classification_benchmark || []);
  } else {
    titleEl.textContent = "Track 2 — Payment Duration Regression & Cash Outflow MAPE";
    subEl.textContent = "Trained on log1p(days_to_pay) and inverted via clip(expm1(y), 0)";
    container.innerHTML = buildRegTableHtml(state.overview.regression_benchmark || []);
  }
}

function buildMainChartConfig() {
  if (state.activeTab === "tab-eda") {
    const splits = state.overview.splits || [];
    return {
      title: "Out-of-Time Split Volume & Late Rate (%)",
      sub: "Train vs Val vs Hold-out Test (with 61d right-censoring buffer)",
      config: {
        type: "bar",
        data: {
          labels: splits.map((s) => s.split.replace(" (Hold-out)", "")),
          datasets: [{ label: "Late Rate (%)", data: splits.map((s) => s.late_rate_pct), backgroundColor: ["#94a3b8", "#2563eb", "#0ea5e9", "#10b981", "#f59e0b"], borderRadius: 4 }],
        },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } },
      },
    };
  }
  if (state.activeTab === "tab-risk") {
    const testModels = (state.overview.classification_benchmark || []).filter((r) => r.split === "Test");
    return {
      title: "Test Hold-out Ranking Power (PR-AUC & ROC-AUC)",
      sub: "Dummy vs Logistic vs LightGBM vs XGBoost on 195,889 unseen 2025 invoices",
      config: {
        type: "bar",
        data: {
          labels: testModels.map((m) => m.model.replace(" (Train Prior 27.53%)", "")),
          datasets: [
            { label: "PR-AUC", data: testModels.map((m) => m.pr_auc), backgroundColor: "#2563eb", borderRadius: 4 },
            { label: "ROC-AUC", data: testModels.map((m) => m.roc_auc), backgroundColor: "#93c5fd", borderRadius: 4 },
          ],
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, max: 0.85 } } },
      },
    };
  }
  const series = state.cashflow?.weekly_series || [];
  const weeks = [...new Set(series.map((p) => p.payment_week))];
  const mapSeries = (name) => {
    const lookup = new Map(series.filter((s) => s.series === name).map((s) => [s.payment_week, s.outflow_m]));
    return weeks.map((w) => lookup.get(w) || 0);
  };
  return {
    title: `Weekly Cash Outflow ($M) — ${state.selectedDepartment}`,
    sub: "Actual payment outflow vs XGBoost predicted timing",
    config: {
      type: "line",
      data: {
        labels: weeks,
        datasets: [
          { label: "Actual ($M)", data: mapSeries("Actual Outflow"), borderColor: "#0f172a", borderWidth: 2, pointRadius: 0, tension: 0.25 },
          { label: "XGBoost ($M)", data: mapSeries("XGBoost Predicted"), borderColor: "#10b981", borderWidth: 1.8, pointRadius: 0, tension: 0.25 },
        ],
      },
      options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 6 } }, y: { beginAtZero: true } } },
    },
  };
}

function renderMainInteractiveChart() {
  const canvas = document.getElementById("main-interactive-chart");
  const titleEl = document.getElementById("primary-chart-title");
  const subEl = document.getElementById("primary-chart-subtitle");
  if (!canvas || !window.Chart || !state.overview) return;

  if (state.mainChart) {
    state.mainChart.destroy();
    state.mainChart = null;
  }
  const spec = buildMainChartConfig();
  titleEl.textContent = spec.title;
  subEl.textContent = spec.sub;
  state.mainChart = new window.Chart(canvas, spec.config);
}

function renderRightShapChart() {
  const canvas = document.getElementById("right-shap-chart");
  if (!canvas || !window.Chart || !state.overview) return;

  if (state.shapChart) {
    state.shapChart.destroy();
    state.shapChart = null;
  }
  const top5 = (state.overview.shap_top_features || []).slice(0, 5);
  state.shapChart = new window.Chart(canvas, {
    type: "bar",
    data: {
      labels: top5.map((f) => f.feature.replace("vendor_prior_", "v_").replace("dept_prior_", "d_").replace("_smoothed", "")),
      datasets: [{ label: "mean |SHAP|", data: top5.map((f) => f.mean_abs_shap), backgroundColor: "#2563eb", borderRadius: 3 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { beginAtZero: true }, y: { ticks: { font: { size: 10 } } } },
    },
  });
}

function renderInspectorCards(items) {
  const container = document.getElementById("right-inspector-items");
  if (!container) return;
  container.innerHTML = items
    .map(
      (it) => `
      <div class="flex items-start gap-2.5 p-2.5 rounded-lg border border-slate-200 bg-slate-50">
        <span class="w-3.5 h-3.5 rounded-full ${escapeHtml(it.color)} mt-0.5 shrink-0"></span>
        <div class="min-w-0">
          <div class="text-xs font-bold text-slate-800">${escapeHtml(it.title)}</div>
          <div class="text-[11px] text-slate-500 mt-0.5">${escapeHtml(it.sub)}</div>
        </div>
      </div>`
    )
    .join("");
}

function renderRightInspector() {
  const badge = document.getElementById("inspector-mode-badge");
  if (state.selectedInvoice) {
    const inv = state.selectedInvoice;
    if (badge) {
      badge.textContent = `Invoice #${inv.invoice_number}`;
      badge.className = "text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 font-semibold";
    }
    renderInspectorCards([
      {
        color: inv.pred_prob_late >= 0.6 ? "bg-rose-500" : "bg-amber-500",
        title: `Predicted Late Risk: ${(inv.pred_prob_late * 100).toFixed(1)}%`,
        sub: `Dept: ${inv.primary_department} • Supplier #${inv.supplier_number}`,
      },
      {
        color: "bg-blue-600",
        title: `Invoice Amount: ${fmtCurrency(inv.invoice_amount)}`,
        sub: `Expected Late Exposure: ${fmtCurrency(inv.expected_late_exposure)}`,
      },
      {
        color: inv.is_late ? "bg-rose-500" : "bg-emerald-500",
        title: `XGB Pred: ${inv.xgb_pred_days ?? "--"}d vs Actual: ${inv.days_to_pay}d`,
        sub: `Observed Status: ${inv.is_late ? "LATE (>30 days)" : "ON-TIME (<=30 days)"}`,
      },
    ]);
    return;
  }

  if (badge) {
    badge.textContent = "Department Scope";
    badge.className = "text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 font-semibold";
  }
  const deptObj = state.departments.find((d) => d.department === state.selectedDepartment);
  const s = state.triage?.summary || { matched_count: 0, total_exposure: 0, actual_late_rate: 0 };
  renderInspectorCards([
    {
      color: "bg-blue-600",
      title: state.selectedDepartment,
      sub: deptObj
        ? `${fmtInt(deptObj.invoices)} test invoices • $${deptObj.total_spend_m.toFixed(2)}M total spend`
        : "195,889 test invoices • $3,424.90M citywide holdout spend",
    },
    {
      color: "bg-amber-500",
      title: `Flagged Triage Queue: ${fmtInt(s.matched_count)} invoices`,
      sub: `Queue Exposure: ${fmtCurrency(s.total_exposure)} (${(s.actual_late_rate * 100).toFixed(1)}% true late)`,
    },
    {
      color: "bg-emerald-500",
      title: deptObj
        ? `XGB MAE: ${deptObj.xgb_mae_days.toFixed(2)}d (RMSE ${deptObj.xgb_rmse_days.toFixed(2)}d)`
        : "Citywide XGB MAE: 8.73d (14d Cashflow MAPE: 0.65%)",
      sub: deptObj
        ? `Actual avg: ${deptObj.actual_avg_days.toFixed(1)}d vs Pred: ${deptObj.xgb_pred_avg_days.toFixed(1)}d`
        : "Click any invoice row in the table to inspect individual predictions",
    },
  ]);
}

async function loadTriageData() {
  const minAmount = Number(document.getElementById("triage-min-amount")?.value || 5000);
  const probThreshold = Number(document.getElementById("triage-prob-threshold")?.value || 0.55);
  const params = new URLSearchParams({
    min_amount: String(Math.max(0, minAmount)),
    prob_threshold: String(Math.min(1, Math.max(0, probThreshold))),
    model: state.activeModel,
    limit: "50",
  });
  if (state.selectedDepartment && state.selectedDepartment !== "All Departments") {
    params.set("department", state.selectedDepartment);
  }
  state.triage = await fetchJson(`/api/v1/invoices/triage?${params.toString()}`);
  renderTriageTable();
}

function renderTriageTable() {
  const tbody = document.getElementById("triage-table-body");
  const summaryEl = document.getElementById("triage-summary-text");
  if (!tbody || !state.triage) return;

  const s = state.triage.summary;
  if (summaryEl) {
    summaryEl.textContent = `Matched ${fmtInt(s.matched_count)} high-risk invoices • Exposure: ${fmtCurrency(s.total_exposure)} • Empirical Precision: ${(s.actual_late_rate * 100).toFixed(1)}%`;
  }

  const items = state.triage.items || [];
  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="9" class="py-4 px-3 text-center text-slate-400">No invoices matched the current filter criteria. Lower the probability or dollar threshold.</td></tr>`;
    return;
  }

  tbody.innerHTML = items
    .map((inv, idx) => {
      const isSelected = state.selectedInvoice && state.selectedInvoice.invoice_number === inv.invoice_number;
      const badge = inv.is_late
        ? `<span class="px-1.5 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-200 font-semibold">LATE</span>`
        : `<span class="px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200">ON-TIME</span>`;
      return `
        <tr class="triage-row ${isSelected ? "selected" : ""}" data-inv-idx="${idx}">
          <td class="py-1.5 px-3 font-semibold text-slate-800">${escapeHtml(inv.invoice_number)}</td>
          <td class="py-1.5 px-3 text-slate-600">${escapeHtml(inv.supplier_number)}</td>
          <td class="py-1.5 px-3 font-sans text-slate-700 truncate max-w-[180px]">${escapeHtml(inv.primary_department)}</td>
          <td class="py-1.5 px-3 text-slate-500">${escapeHtml(inv.invoice_date)}</td>
          <td class="py-1.5 px-3 text-right font-semibold">${fmtCurrency(inv.invoice_amount)}</td>
          <td class="py-1.5 px-3 text-right font-bold text-amber-700">${(inv.pred_prob_late * 100).toFixed(1)}%</td>
          <td class="py-1.5 px-3 text-right text-blue-700">${escapeHtml(inv.xgb_pred_days ?? "--")}d</td>
          <td class="py-1.5 px-3 text-right">${inv.days_to_pay}d</td>
          <td class="py-1.5 px-3 text-center">${badge}</td>
        </tr>`;
    })
    .join("");

  tbody.querySelectorAll(".triage-row").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      const idx = Number(rowEl.getAttribute("data-inv-idx"));
      state.selectedInvoice = items[idx] || null;
      renderTriageTable();
      renderRightInspector();
    });
  });
}

function exportTriageCsv() {
  const items = state.triage?.items || [];
  if (!items.length) return;
  const headers = [
    "invoice_number",
    "supplier_number",
    "primary_department",
    "invoice_date",
    "invoice_amount",
    "pred_prob_late",
    "xgb_pred_days",
    "days_to_pay",
    "is_late",
  ];
  const csvRows = [
    headers.join(","),
    ...items.map((row) =>
      headers.map((h) => `"${String(row[h] ?? "").replace(/"/g, '""')}"`).join(",")
    ),
  ];
  const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "nashville_ap_high_risk_triage.csv";
  link.click();
}

async function loadCashflowData() {
  const params = new URLSearchParams();
  if (state.selectedDepartment && state.selectedDepartment !== "All Departments") {
    params.set("department", state.selectedDepartment);
  }
  state.cashflow = await fetchJson(`/api/v1/cashflow/series?${params.toString()}`);
}

function openLightbox(url, caption) {
  const modal = document.getElementById("image-lightbox-modal");
  const img = document.getElementById("lightbox-img");
  const cap = document.getElementById("lightbox-caption");
  if (!modal || !img || !cap || !url) return;
  img.src = url;
  cap.textContent = caption || "Diagnostic Plot";
  modal.classList.remove("hidden");
  modal.classList.add("flex");
}

async function openLearningDocsDrawer() {
  const modal = document.getElementById("learning-docs-modal");
  const nav = document.getElementById("docs-nav-list");
  const viewer = document.getElementById("docs-content-viewer");
  if (!modal || !nav || !viewer) return;

  modal.classList.remove("hidden");
  modal.classList.add("flex");

  if (!state.docsList.length) {
    const res = await fetchJson("/api/v1/docs");
    state.docsList = res.docs || [];
  }

  nav.innerHTML = state.docsList
    .map(
      (d, idx) => `
      <button
        type="button"
        data-doc-slug="${escapeHtml(d.slug)}"
        class="w-full text-left px-3 py-2 rounded text-xs font-medium ${
          idx === 0 ? "bg-blue-50 text-blue-700 font-semibold" : "text-slate-700 hover:bg-slate-100"
        }"
      >
        <div class="truncate">${escapeHtml(d.title)}</div>
        <div class="text-[10px] font-mono-num text-slate-400">${escapeHtml(d.filename)}</div>
      </button>`
    )
    .join("");

  const loadSlug = async (slug) => {
    const detail = await fetchJson(`/api/v1/docs/${encodeURIComponent(slug)}`);
    viewer.textContent = detail.content;
  };

  if (state.docsList.length) {
    await loadSlug(state.docsList[0].slug);
  }

  nav.querySelectorAll("[data-doc-slug]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      nav.querySelectorAll("[data-doc-slug]").forEach((b) => {
        b.className = "w-full text-left px-3 py-2 rounded text-xs font-medium text-slate-700 hover:bg-slate-100";
      });
      btn.className = "w-full text-left px-3 py-2 rounded text-xs font-semibold bg-blue-50 text-blue-700";
      await loadSlug(btn.getAttribute("data-doc-slug") || "");
    });
  });
}

function bindEvents() {
  document.querySelectorAll(".workbench-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".workbench-tab").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.activeTab = btn.getAttribute("data-tab") || "tab-eda";
      updateWorkspaceHeaderForTab();
      renderVisualCards();
      renderBenchmarkTable();
      renderMainInteractiveChart();
    });
  });

  document.getElementById("dept-search-input")?.addEventListener("input", (e) => {
    renderSidebarDepartments(e.target.value);
  });

  document.getElementById("apply-triage-btn")?.addEventListener("click", async () => {
    await loadTriageData();
    renderRightInspector();
  });

  document.getElementById("export-triage-csv-btn")?.addEventListener("click", exportTriageCsv);

  const toggleModel = async () => {
    state.activeModel = state.activeModel === "xgb" ? "lgb" : "xgb";
    const label = document.getElementById("status-bar-model");
    if (label) {
      label.textContent = state.activeModel === "xgb" ? "XGBoost (Isotonic)" : "LightGBM (Isotonic)";
    }
    await loadTriageData();
    renderRightInspector();
  };

  document.getElementById("footer-model-toggle")?.addEventListener("click", toggleModel);

  document.querySelectorAll("[data-rail]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const action = btn.getAttribute("data-rail");
      document.querySelectorAll("[data-rail]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (action === "search") document.getElementById("dept-search-input")?.focus();
      else if (action === "alerts") document.querySelector('[data-tab="tab-risk"]')?.click();
      else if (action === "docs") await openLearningDocsDrawer();
      else if (action === "settings") await toggleModel();
    });
  });

  document.getElementById("open-docs-inline-btn")?.addEventListener("click", openLearningDocsDrawer);
  document.getElementById("close-lightbox-btn")?.addEventListener("click", () => {
    document.getElementById("image-lightbox-modal")?.classList.add("hidden");
  });
  document.getElementById("close-docs-btn")?.addEventListener("click", () => {
    document.getElementById("learning-docs-modal")?.classList.add("hidden");
  });
}

async function initWorkbench() {
  bindEvents();
  const [overview, deptRes] = await Promise.all([
    fetchJson("/api/v1/overview"),
    fetchJson("/api/v1/departments?min_invoices=200"),
  ]);
  state.overview = overview;
  state.departments = deptRes.departments || [];

  const modeEl = document.getElementById("status-bar-mode");
  if (modeEl) {
    modeEl.textContent = overview.kaggle_ready ? "KAGGLE_GBDT_ACTIVE" : "BASELINE_FALLBACK";
  }

  renderHeaderKpis();
  renderSidebarDepartments("");
  renderVisualCards();
  renderBenchmarkTable();
  renderRightShapChart();

  await Promise.all([loadTriageData(), loadCashflowData()]);
  renderMainInteractiveChart();
  renderRightInspector();
}

document.addEventListener("DOMContentLoaded", initWorkbench);
