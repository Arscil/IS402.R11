/* Web client for the fraud-detection API (Azure Container Apps).
 * Calls GET /health, GET /model-info and POST /predict, then parses and renders the JSON.
 * All values are rendered with textContent (no innerHTML of user/API data). */
(function () {
  "use strict";
  const CFG = window.FRAUD_API;
  const COLS = ["Time"].concat(Array.from({ length: 28 }, (_, i) => "V" + (i + 1)), ["Amount"]);
  const $ = (id) => document.getElementById(id);
  let samples = [];
  let batchRows = null;
  let modelInfo = null;

  const api = () => CFG.regions.find((r) => r.id === $("region").value);
  const fmtMs = (v) => (v >= 1000 ? (v / 1000).toFixed(2) + " s" : Math.round(v) + " ms");
  const pct = (v) => (v * 100).toFixed(1) + "%";

  async function call(path, body) {
    const t0 = performance.now();
    const res = await fetch(api().url + path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : {});
    const rtt = performance.now() - t0;
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : (data.detail || res.statusText);
      throw new Error("HTTP " + res.status + ": " + detail);
    }
    return { data, rtt };
  }

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  // ---------- status ----------
  async function refreshStatus() {
    ["st-health", "st-model", "st-threshold", "st-metrics", "st-latency"].forEach((id) => ($(id).textContent = "…"));
    $("st-note").textContent = "Đang gọi API… lần đầu có thể mất vài giây nếu app đang tắt (scale to zero).";
    $("st-note").classList.remove("error");
    try {
      const h = await call("/health");
      $("st-health").textContent = "Hoạt động (" + h.data.region + ")";
      $("st-latency").textContent = fmtMs(h.rtt);
      const info = await call("/model-info");
      modelInfo = info.data;
      const m = modelInfo.metrics_test;
      $("st-model").textContent = modelInfo.model;
      $("st-threshold").textContent = modelInfo.threshold.toFixed(4);
      $("st-metrics").textContent = m.f1_tuned.toFixed(3) + " / " + m.precision_tuned.toFixed(3) + " / " + m.recall_tuned.toFixed(3);
      $("st-note").textContent = "Instance khởi động lúc " + h.data.started_at + " (UTC).";
    } catch (err) {
      $("st-health").textContent = "Lỗi";
      $("st-note").textContent = err.message;
      $("st-note").classList.add("error");
    }
  }

  // ---------- single prediction ----------
  function buildPcaInputs() {
    const grid = $("pca-grid");
    COLS.slice(1, 29).forEach((c) => {
      const lab = el("label");
      lab.appendChild(el("span", c));
      const inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.id = "f-" + c;
      lab.appendChild(inp);
      grid.appendChild(lab);
    });
  }

  function fillForm(tx) {
    COLS.forEach((c) => ($("f-" + c).value = tx[c]));
  }

  function readForm() {
    const tx = {};
    for (const c of COLS) {
      const v = parseFloat($("f-" + c).value);
      if (!Number.isFinite(v)) throw new Error("Giá trị " + c + " không hợp lệ");
      tx[c] = v;
    }
    return tx;
  }

  function gaussian() {
    const u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  async function predictOne() {
    const btn = $("btn-predict");
    btn.disabled = true;
    const box = $("single-result");
    box.hidden = false;
    $("verdict").textContent = "Đang dự đoán…";
    $("verdict").className = "verdict";
    $("single-meta").replaceChildren();
    try {
      const { data, rtt } = await call("/predict", { transactions: [readForm()] });
      const r = data.results[0];
      $("verdict").textContent = r.is_fraud ? "GIAN LẬN" : "HỢP LỆ";
      $("verdict").className = "verdict " + (r.is_fraud ? "fraud" : "legit");
      $("prob-fill").style.width = Math.max(r.fraud_probability * 100, 0.5) + "%";
      $("prob-thr").style.left = data.threshold * 100 + "%";
      $("prob-text").textContent = "xác suất " + r.fraud_probability.toFixed(4) + " · ngưỡng " + data.threshold.toFixed(4);
      const s = samples[$("sample").value];
      const meta = [
        ["Xác suất gian lận", r.fraud_probability.toFixed(6)],
        ["Nhãn thực tế (mẫu)", s && $("sample").dataset.edited !== "1" ? s.actual : "—"],
        ["Mô hình", data.model],
        ["Region xử lý", data.region],
        ["Thời gian khứ hồi", fmtMs(rtt)],
        ["Xử lý trên server", data.server_time_ms + " ms"],
      ];
      meta.forEach(([k, v]) => { $("single-meta").appendChild(el("dt", k)); $("single-meta").appendChild(el("dd", v)); });
    } catch (err) {
      $("verdict").textContent = err.message;
      $("verdict").className = "verdict error";
    } finally {
      btn.disabled = false;
    }
  }

  // ---------- CSV batch ----------
  function parseCsv(text) {
    const lines = text.split(/\r?\n/).filter((l) => l.trim() !== "");
    if (lines.length < 2) throw new Error("File CSV không có dữ liệu");
    const header = lines[0].split(",").map((h) => h.trim().replace(/^"|"$/g, ""));
    const idx = {};
    COLS.concat(["Class"]).forEach((c) => (idx[c] = header.indexOf(c)));
    const missing = COLS.filter((c) => idx[c] < 0);
    if (missing.length) throw new Error("Thiếu cột: " + missing.join(", "));
    return lines.slice(1).map((line, i) => {
      const parts = line.split(",");
      const tx = {};
      COLS.forEach((c) => {
        const v = parseFloat(parts[idx[c]]);
        if (!Number.isFinite(v)) throw new Error("Dòng " + (i + 2) + ": " + c + " không hợp lệ");
        tx[c] = v;
      });
      const cls = idx.Class >= 0 ? parseFloat(parts[idx.Class]) : NaN;
      return { tx, actual: Number.isFinite(cls) ? (cls >= 0.5 ? 1 : 0) : null };
    });
  }

  async function runBatch() {
    const btn = $("btn-batch");
    btn.disabled = true;
    const sum = $("batch-summary");
    sum.hidden = false;
    sum.replaceChildren(el("span", "Đang gửi " + batchRows.length + " giao dịch…"));
    const tbody = document.querySelector("#batch-table tbody");
    tbody.replaceChildren();
    try {
      const t0 = performance.now();
      const results = [];
      let serverMs = 0;
      for (let i = 0; i < batchRows.length; i += CFG.maxBatch) {
        const chunk = batchRows.slice(i, i + CFG.maxBatch);
        const { data } = await call("/predict", { transactions: chunk.map((r) => r.tx) });
        serverMs += data.server_time_ms;
        results.push(...data.results);
      }
      const total = performance.now() - t0;
      const fraud = results.filter((r) => r.is_fraud).length;
      const labelled = batchRows.filter((r) => r.actual !== null);
      const stats = [["Số giao dịch", batchRows.length], ["Dự đoán gian lận", fraud],
        ["Tổng thời gian", fmtMs(total)], ["Server xử lý", serverMs.toFixed(1) + " ms"]];
      if (labelled.length) {
        let tp = 0, fp = 0, fn = 0, tn = 0;
        batchRows.forEach((r, i) => {
          if (r.actual === null) return;
          const p = results[i].is_fraud ? 1 : 0;
          if (p && r.actual) tp++; else if (p && !r.actual) fp++; else if (!p && r.actual) fn++; else tn++;
        });
        stats.push(["Accuracy", pct((tp + tn) / labelled.length)],
          ["Precision", tp + fp ? pct(tp / (tp + fp)) : "—"], ["Recall", tp + fn ? pct(tp / (tp + fn)) : "—"]);
      }
      sum.replaceChildren(...stats.map(([k, v]) => {
        const s = el("div", undefined, "stat");
        s.append(el("span", k, "stat-label"), el("span", String(v), "stat-value"));
        return s;
      }));
      // show frauds and mistakes first, then the rest (max 500 rows)
      const order = results.map((r, i) => i).sort((a, b) => results[b].fraud_probability - results[a].fraud_probability);
      order.slice(0, 500).forEach((i) => {
        const r = results[i], a = batchRows[i].actual;
        const tr = document.createElement("tr");
        tr.append(el("td", String(i + 1)), el("td", batchRows[i].tx.Amount.toFixed(2)),
          el("td", r.fraud_probability.toFixed(4)), el("td", r.label, r.is_fraud ? "fraud" : "legit"),
          el("td", a === null ? "—" : (a ? "Gian lận" : "Hợp lệ"), a !== null && a !== (r.is_fraud ? 1 : 0) ? "miss" : ""));
        tbody.appendChild(tr);
      });
      $("batch-table-wrap").hidden = false;
    } catch (err) {
      sum.replaceChildren(el("span", err.message, "error"));
    } finally {
      btn.disabled = false;
    }
  }

  function downloadSampleCsv(e) {
    e.preventDefault();
    const rows = [COLS.concat(["Class"]).join(",")];
    samples.forEach((s) => rows.push(COLS.map((c) => s.transaction[c]).concat([s.actual === "Gian lận" ? 1 : 0]).join(",")));
    const url = URL.createObjectURL(new Blob([rows.join("\n")], { type: "text/csv" }));
    const a = el("a");
    a.href = url; a.download = "giao_dich_mau.csv";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }

  // ---------- latency ----------
  async function measureLatency() {
    const btn = $("btn-latency");
    btn.disabled = true;
    const tbody = document.querySelector("#lat-table tbody");
    tbody.replaceChildren();
    $("lat-wrap").hidden = false;
    const body = JSON.stringify({ transactions: [samples[0].transaction] });
    for (const reg of CFG.regions) {
      const tr = document.createElement("tr");
      tr.append(el("td", reg.name), el("td", "…"), el("td", "…"), el("td", "…"), el("td", "…"));
      tbody.appendChild(tr);
      try {
        const rtts = [], srv = [];
        for (let i = 0; i < 20; i++) {
          const t0 = performance.now();
          const res = await fetch(reg.url + "/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body });
          const data = await res.json();
          rtts.push(performance.now() - t0);
          srv.push(data.server_time_ms);
        }
        const sorted = rtts.slice(1).sort((a, b) => a - b);
        const q = (arr, p) => arr[Math.min(arr.length - 1, Math.floor(p * arr.length))];
        tr.children[1].textContent = fmtMs(rtts[0]);
        tr.children[2].textContent = fmtMs(q(sorted, 0.5));
        tr.children[3].textContent = fmtMs(q(sorted, 0.95));
        tr.children[4].textContent = q(srv.slice().sort((a, b) => a - b), 0.5).toFixed(2) + " ms";
      } catch (err) {
        tr.children[1].textContent = "Lỗi: " + err.message;
      }
    }
    btn.disabled = false;
  }

  // ---------- init ----------
  async function init() {
    CFG.regions.forEach((r) => $("region").appendChild(new Option(r.name, r.id)));
    buildPcaInputs();
    try {
      samples = await (await fetch(CFG.samplesUrl)).json();
    } catch (_) {
      samples = [];
    }
    samples.forEach((s, i) => $("sample").appendChild(
      new Option("#" + (i + 1) + " · thực tế: " + s.actual + " · Amount " + s.transaction.Amount, String(i))));
    if (samples.length) fillForm(samples[0].transaction);

    $("sample").addEventListener("change", () => { fillForm(samples[$("sample").value].transaction); $("sample").dataset.edited = "0"; });
    document.querySelectorAll("input[type=number]").forEach((i) => i.addEventListener("input", () => ($("sample").dataset.edited = "1")));
    $("btn-random").addEventListener("click", () => {
      const tx = { Time: Math.round(Math.random() * 172800), Amount: +(Math.random() * 300).toFixed(2) };
      COLS.slice(1, 29).forEach((c) => (tx[c] = +gaussian().toFixed(4)));
      fillForm(tx);
      $("sample").dataset.edited = "1";
    });
    $("btn-predict").addEventListener("click", predictOne);
    $("btn-refresh").addEventListener("click", refreshStatus);
    $("region").addEventListener("change", refreshStatus);
    $("dl-sample").addEventListener("click", downloadSampleCsv);
    $("csv").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      $("btn-batch").disabled = true;
      $("batch-table-wrap").hidden = true;
      if (!f) return;
      try {
        batchRows = parseCsv(await f.text());
        $("batch-summary").hidden = false;
        $("batch-summary").replaceChildren(el("span", "Đã đọc " + batchRows.length + " giao dịch từ " + f.name));
        $("btn-batch").disabled = false;
      } catch (err) {
        $("batch-summary").hidden = false;
        $("batch-summary").replaceChildren(el("span", err.message, "error"));
      }
    });
    $("btn-batch").addEventListener("click", runBatch);
    $("btn-latency").addEventListener("click", measureLatency);
    refreshStatus();
  }

  init();
})();
