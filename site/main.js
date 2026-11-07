import {
  SOURCE, STALE_AFTER_H, ageHours, ageLabel, axisMax, categorySlug, localLabel, notForecast, pct,
  rowsFor, stateOf, validateDoc,
} from "./lib.js";

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 5 * 60 * 1000;

const state = { doc: null, horizon: null, region: "" };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.setAttribute("style", v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
}

const ppb = (v) => (Number.isFinite(v) ? v.toFixed(1) : "–");

function showNotice(kind, text, retry) {
  const n = $("notice");
  n.className = `notice${kind === "error" ? " error" : ""}`;
  n.replaceChildren(text);
  if (retry) {
    const b = el("button", { class: "btn", type: "button" }, "Try again");
    b.addEventListener("click", () => load(true));
    n.append(b);
  }
  n.hidden = false;
}

function renderStatus(doc) {
  const age = ageHours(doc.issued_at_utc);
  $("s-issued").replaceChildren(`${doc.issued_at_utc} UTC`, el("small", {}, ageLabel(age)));
  $("s-data").replaceChildren(`${doc.data_as_of_utc ?? "–"} UTC`, el("small", {}, "latest reading used"));
  const ok = doc.stations.length;
  const total = ok + notForecast(doc).length;
  $("s-stations").replaceChildren(`${ok} of ${total}`, el("small", {}, "passed the data contract"));
  $("s-interval").replaceChildren("90%", el("small", {}, "split-conformal"));
  if (age != null && age > STALE_AFTER_H) {
    showNotice("warn", `This forecast was issued ${ageLabel(age)}. The hourly job may be delayed; the numbers below are from that run.`);
  } else {
    $("notice").hidden = true;
  }
}

function renderHorizons(doc) {
  const box = $("horizons");
  if (!doc.horizons_h.includes(state.horizon)) state.horizon = doc.horizons_h.includes(12) ? 12 : doc.horizons_h[0];
  box.replaceChildren(
    ...doc.horizons_h.map((h) => {
      const b = el("button", { type: "button", "aria-pressed": String(h === state.horizon) }, `${h} h`);
      b.addEventListener("click", () => {
        state.horizon = h;
        renderHorizons(doc);
        renderTable(doc);
      });
      return b;
    }),
  );
}

function renderStates(doc) {
  const sel = $("state");
  const states = [...new Set(doc.stations.map((s) => stateOf(s.place)).filter(Boolean))].sort();
  if (state.region && !states.includes(state.region)) state.region = "";
  sel.replaceChildren(el("option", { value: "" }, "All states"), ...states.map((s) => el("option", { value: s }, s)));
  sel.value = state.region;
}

function renderTable(doc) {
  const table = $("table");
  const head = table.firstElementChild;
  const rows = rowsFor(doc, state.horizon).filter((r) => !state.region || stateOf(r.place) === state.region);
  if (!rows.length) {
    table.replaceChildren(head, el("div", { class: "empty" }, "No station has a forecast for this selection."));
    return;
  }
  const max = axisMax(rows);
  const out = rows.map((r) => {
    const bar = el("div", { class: "bar", title: `90% interval ${ppb(r.low)} to ${ppb(r.high)} ppb` },
      el("span", { class: "track" }),
      Number.isFinite(r.low) && Number.isFinite(r.high)
        ? el("span", { class: "ci", style: `left:${pct(r.low, max)}%;width:${Math.max(0.5, pct(r.high, max) - pct(r.low, max))}%` })
        : null,
      r.last != null ? el("span", { class: "ls", style: `left:${pct(r.last, max)}%` }) : null,
      el("span", { class: "pt", style: `left:${pct(r.value, max)}%` }),
    );
    return el("div", { class: "row", role: "row" },
      el("div", { class: "c-place", role: "cell" }, el("span", { class: "place" }, r.place), el("span", { class: "sid" }, r.station)),
      el("div", { class: "c-last num last", role: "cell" }, r.last != null ? ppb(r.last) : "–"),
      el("div", { class: "c-val num", role: "cell" },
        el("span", { class: "val" }, ppb(r.value), el("small", {}, " ppb")),
        el("span", { class: "rng" }, `${ppb(r.low)} to ${ppb(r.high)}`)),
      el("div", { class: "c-bar", role: "cell" }, bar,
        el("div", { class: "axis", "aria-hidden": "true" }, el("span", {}, "0"), el("span", {}, String(max)))),
      el("div", { class: "c-cat", role: "cell" },
        el("span", { class: `cat ${categorySlug(r.category)}` }, r.category || "n/a"),
        el("span", { class: "when" }, localLabel(r.local) ? `${localLabel(r.local)} local` : "")),
    );
  });
  table.replaceChildren(head, ...out);
}

function renderSide(doc) {
  const ex = notForecast(doc);
  $("excluded").replaceChildren(
    ...(ex.length ? ex.map(([s, why]) => el("li", {}, el("code", {}, s), `: ${why}`)) : [el("li", {}, "Every monitor passed.")]),
  );
  const kv = [["Model", doc.model], ["Source", doc.source], ["Units", doc.units],
    ...Object.entries(doc.model_versions ?? {}).map(([k, v]) => [k, v])];
  $("model").replaceChildren(...kv.filter(([, v]) => v).flatMap(([k, v]) => [el("dt", {}, k), el("dd", {}, String(v))]));
}

function render() {
  const doc = state.doc;
  renderStatus(doc);
  renderHorizons(doc);
  renderStates(doc);
  renderTable(doc);
  renderSide(doc);
}

async function load(manual = false) {
  try {
    const r = await fetch(`${SOURCE}?t=${Math.floor(Date.now() / 60000)}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const doc = await r.json();
    const errors = validateDoc(doc);
    if (errors.length) throw new Error(`unexpected data: ${errors.join(", ")}`);
    state.doc = doc;
    render();
  } catch (e) {
    if (state.doc) {
      if (manual) showNotice("error", `Could not refresh (${e.message}). Showing the last forecast loaded.`, true);
      return;
    }
    $("loading")?.replaceChildren("No forecast could be loaded.");
    showNotice("error", `The forecast file could not be loaded (${e.message}).`, true);
  }
}

$("state").addEventListener("change", (e) => {
  state.region = e.target.value;
  if (state.doc) renderTable(state.doc);
});

load();
setInterval(() => { if (document.visibilityState === "visible") load(); }, REFRESH_MS);
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible" && state.doc) load(); });
