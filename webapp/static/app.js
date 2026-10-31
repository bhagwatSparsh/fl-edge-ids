
(function () {
  "use strict";
  const fmtPct1 = d3.format(".1%");
  const fmtPct0 = d3.format(".0%");
  const fmtInt = d3.format(",");
  function fmtBytes(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + " KB";
    return n + " B";
  }

  // ---------- resolve [data-source][data-field] marks ----------
  // The Dashboard artifact type does this automatically; this standalone
  // page has no such runtime, so it's done by hand here, once per element.
  const FIELD_FORMAT = {
    zero_day_catch_rate: fmtPct0,
    macro_f1_float32: fmtPct1,
    macro_f1_float16_roundtrip: fmtPct1,
    spearman_rho: (v) => v.toFixed(2),
  };
  function parseWhere(whereStr) {
    const cond = {};
    whereStr.split("&").forEach(part => {
      const [k, v] = part.split("=").map(decodeURIComponent);
      cond[k] = v;
    });
    return cond;
  }
  function resolveMarks() {
    document.querySelectorAll("[data-source][data-field]").forEach(el => {
      const field = el.getAttribute("data-field");
      if (field === "seq") return; // the live row-count tally fills this one directly
      const r = dash.data(el.getAttribute("data-source"));
      if (r.status !== "ok" || !Array.isArray(r.data)) return;
      let row = r.data[0];
      const whereAttr = el.getAttribute("data-where");
      if (whereAttr) {
        const cond = parseWhere(whereAttr);
        row = r.data.find(rw => Object.entries(cond).every(([k, v]) => String(rw[k]) === v));
      }
      if (!row || !(field in row)) return;
      const fmt = FIELD_FORMAT[field];
      el.textContent = fmt ? fmt(row[field]) : row[field];
    });
  }
  dash.onData(resolveMarks);

  // ---------- generic grouped bar chart ----------
  // Measures text the same way the legend will render it, so items can be
  // wrapped onto as many rows as the real label lengths need instead of
  // assuming a fixed width per item (long labels like "Local-only (no
  // sharing, avg of 9 clients)" overlapped the next item under the old
  // fixed-110px-per-item layout).
  const _measureCtx = document.createElement("canvas").getContext("2d");
  function textWidth(text, font) {
    _measureCtx.font = font;
    return _measureCtx.measureText(text).width;
  }

  function drawGroupedBar(sel, rows, opts) {
    const el = document.querySelector(sel);
    if (!el) return;
    el.innerHTML = "";
    const width = el.clientWidth || 400;
    const groups = Array.from(new Set(rows.map(d => d[opts.groupField])));
    const series = Array.from(new Set(rows.map(d => d[opts.seriesField])));
    const margin = { right: 12, bottom: 34, left: 44 };

    // Lay out the legend first (swatch + label per series, wrapping rows
    // to fit the chart's actual width) so its row count can set the top
    // margin before the bars are scaled.
    const legendFont = "12px " + "IBM Plex Sans, -apple-system, Segoe UI, sans-serif";
    const swatch = 9, gap = 6, itemGapX = 18, rowH = 18;
    const legendItems = [];
    if (opts.legend !== false) {
      let rowX = 0, row = 0;
      series.forEach((s) => {
        const w = swatch + gap + textWidth(s, legendFont);
        if (rowX > 0 && rowX + w > width - margin.left - margin.right) { row++; rowX = 0; }
        legendItems.push({ s, x: rowX, row });
        rowX += w + itemGapX;
      });
    }
    const legendRows = legendItems.length ? legendItems[legendItems.length - 1].row + 1 : 0;
    margin.top = 10 + legendRows * rowH + (legendRows ? 8 : 0);
    const height = 220 + legendRows * rowH;

    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`);
    const x0 = d3.scaleBand().domain(groups).range([margin.left, width - margin.right]).paddingInner(0.3);
    const x1 = d3.scaleBand().domain(series).range([0, x0.bandwidth()]).padding(0.08);
    const yMax = d3.max(rows, d => d[opts.valueField]) || 1;
    const y = d3.scaleLinear().domain([0, Math.max(yMax * 1.15, 0.05)]).range([height - margin.bottom, margin.top]);
    const color = d3.scaleOrdinal().domain(series).range(dash.colors);

    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x0).tickSize(0)).call(g => g.select(".domain").attr("stroke", "var(--cds-chart-axis)"))
      .selectAll("text").attr("font-family", null).attr("font-size", null);
    svg.append("g").attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(opts.yFormat).tickSize(-(width - margin.left - margin.right)))
      .call(g => g.select(".domain").remove())
      .call(g => g.selectAll(".tick line").attr("stroke", "var(--cds-chart-grid)"))
      .selectAll("text").attr("font-family", null).attr("font-size", null);

    const groupG = svg.selectAll(".lx-g").data(groups).join("g")
      .attr("class", "lx-g").attr("transform", d => `translate(${x0(d)},0)`);

    groupG.selectAll("rect").data(g => series.map(s => {
      const row = rows.find(r => r[opts.groupField] === g && r[opts.seriesField] === s);
      return { series: s, value: row ? row[opts.valueField] : 0 };
    })).join("rect")
      .attr("x", d => x1(d.series)).attr("width", x1.bandwidth())
      .attr("y", d => y(d.value)).attr("height", d => y(0) - y(d.value))
      .attr("fill", d => color(d.series)).attr("rx", 3)
      .on("mousemove", function (ev, d) { showTip(ev, `${d.series}: ${opts.yFormat(d.value)}`); })
      .on("mouseleave", hideTip);

    if (legendItems.length) {
      const leg = svg.append("g").attr("transform", `translate(${margin.left},4)`);
      legendItems.forEach(({ s, x, row }) => {
        const gx = leg.append("g").attr("transform", `translate(${x},${row * rowH})`);
        gx.append("rect").attr("width", swatch).attr("height", swatch).attr("y", 1).attr("rx", 2).attr("fill", color(s));
        gx.append("text").attr("x", swatch + gap).attr("y", 9).attr("fill", "var(--color-fg-muted)").text(s);
      });
    }
  }

  let tipEl = null;
  function showTip(ev, text) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "lx-tooltip"; document.body.appendChild(tipEl); }
    tipEl.textContent = text; tipEl.style.left = (ev.clientX + 12) + "px"; tipEl.style.top = (ev.clientY + 12) + "px"; tipEl.style.display = "block";
  }
  function hideTip() { if (tipEl) tipEl.style.display = "none"; }

  // ---------- model comparison ----------
  dash.onData(function () {
    const r = dash.data("model_comparison");
    const box1 = document.querySelector("#chart-compare-nbaiot"), box2 = document.querySelector("#chart-compare-tonIoT");
    if (r.status !== "ok") { if (box1) box1.innerHTML = ""; if (box2) box2.innerHTML = ""; return; }
    const rows = r.data;
    drawGroupedBar("#chart-compare-nbaiot", rows.filter(d => d.dataset === "N-BaIoT"),
      { groupField: "dataset", seriesField: "approach", valueField: "macro_f1", yFormat: fmtPct0 });
    drawGroupedBar("#chart-compare-tonIoT", rows.filter(d => d.dataset === "TON_IoT"),
      { groupField: "dataset", seriesField: "approach", valueField: "macro_f1", yFormat: fmtPct0 });

    const fedprox = rows.find(d => d.dataset === "N-BaIoT" && d.approach === "Federated FedProx");
    const local = rows.find(d => d.dataset === "N-BaIoT" && d.approach === "Local-only (no sharing, avg of 9 clients)");
    const takeaway = document.querySelector("#lx-takeaway");
    if (takeaway && fedprox && local) {
      takeaway.innerHTML = `Nine real IoT devices train one shared attack-detector without ever pooling their traffic. On N-BaIoT, federated training reaches <b>${fmtPct0(fedprox.macro_f1)}</b> macro-F1, well above the <b>${fmtPct0(local.macro_f1)}</b> each device manages completely alone.`;
    }
  });

  // ---------- zero-day ----------
  dash.onData(function () {
    const r = dash.data("zero_day");
    if (r.status !== "ok") return;
    const rows = r.data;
    drawGroupedBar("#chart-zeroday-nbaiot", rows.filter(d => d.dataset === "N-BaIoT"),
      { groupField: "dataset", seriesField: "strategy", valueField: "zero_day_catch_rate", yFormat: fmtPct0, legend: true });
    drawGroupedBar("#chart-zeroday-tonIoT", rows.filter(d => d.dataset === "TON_IoT"),
      { groupField: "dataset", seriesField: "strategy", valueField: "zero_day_catch_rate", yFormat: fmtPct0, legend: true });
  });

  // ---------- explainability ----------
  dash.onData(function () {
    const r = dash.data("top_features");
    const tbody = document.querySelector("#table-features-nbaiot tbody");
    if (!tbody) return;
    if (r.status !== "ok") { tbody.innerHTML = ""; return; }
    const rows = r.data.filter(d => d.dataset === "N-BaIoT").sort((a, b) => a.rank - b.rank);
    tbody.innerHTML = "";
    rows.forEach(d => {
      const tr = document.createElement("tr");
      const tdRank = document.createElement("td"); tdRank.textContent = d.rank;
      const tdFeat = document.createElement("td"); tdFeat.textContent = d.feature;
      tr.appendChild(tdRank); tr.appendChild(tdFeat); tbody.appendChild(tr);
    });
  });

  // ---------- bandwidth ----------
  dash.onData(function () {
    const r = dash.data("bandwidth_totals");
    if (r.status !== "ok") return;
    const rows = r.data;
    const el = document.querySelector("#chart-bandwidth");
    if (!el) return;
    el.innerHTML = "";
    const width = el.clientWidth || 400, height = 420;
    const margin = { top: 12, right: 16, bottom: 56, left: 54 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "none");
    const x = d3.scaleBand().domain(rows.map(d => d.item)).range([margin.left, width - margin.right]).padding(0.25);
    const y = d3.scaleLog().domain([1e5, d3.max(rows, d => d.bytes) * 2]).range([height - margin.bottom, margin.top]);
    svg.append("g").attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4, "~s").tickSize(-(width - margin.left - margin.right)))
      .call(g => g.select(".domain").remove())
      .call(g => g.selectAll(".tick line").attr("stroke", "var(--cds-chart-grid)"))
      .selectAll("text").attr("font-family", null).attr("font-size", null);
    svg.selectAll("rect").data(rows).join("rect")
      .attr("x", d => x(d.item)).attr("width", x.bandwidth())
      .attr("y", d => y(d.bytes)).attr("height", d => y(1e5) - y(d.bytes))
      .attr("fill", (d, i) => d.method === "measured" ? "var(--color-ok)" : dash.colors[i % dash.colors.length])
      .attr("rx", 3)
      .on("mousemove", function (ev, d) { showTip(ev, `${d.item}: ${fmtBytes(d.bytes)} (${d.method})`); })
      .on("mouseleave", hideTip);
    // Short two-line labels for display -- the full strings ("Raw
    // packet-capture equivalent, estimated") are too wide for a ~60px bar
    // slot and were overlapping their neighbors; the full text is still
    // available on hover via the tooltip above.
    const SHORT_LABEL = {
      "Model updates (float32), measured": ["Model update", "(float32)"],
      "Model updates (float16), estimated": ["Model update", "(float16)"],
      "Raw feature-CSV equivalent, estimated": ["Feature CSV", "(estimated)"],
      "Raw packet-capture equivalent, estimated": ["Packet capture", "(estimated)"],
    };
    const labels = svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`);
    rows.forEach(d => {
      const [l1, l2] = SHORT_LABEL[d.item] || [d.item, ""];
      const cx = x(d.item) + x.bandwidth() / 2;
      labels.append("text").attr("x", cx).attr("y", 14).attr("text-anchor", "middle").attr("font-size", "10.5px").text(l1);
      labels.append("text").attr("x", cx).attr("y", 27).attr("text-anchor", "middle").attr("font-size", "10.5px").text(l2);
    });
  });

  // ---------- round time vs delay ----------
  dash.onData(function () {
    const r = dash.data("round_time");
    if (r.status !== "ok") return;
    const rows = r.data.slice().sort((a, b) => a.delay_ms - b.delay_ms);
    const el = document.querySelector("#chart-roundtime");
    if (!el) return;
    el.innerHTML = "";
    const width = el.clientWidth || 400, height = 240;
    const margin = { top: 16, right: 16, bottom: 30, left: 44 };
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`).attr("preserveAspectRatio", "none");
    const x = d3.scaleLinear().domain([0, 55]).range([margin.left, width - margin.right]);
    const y = d3.scaleLinear().domain([0, d3.max(rows, d => d.round_seconds) * 1.2]).range([height - margin.bottom, margin.top]);
    svg.append("g").attr("transform", `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d => d + "ms")).call(g => g.select(".domain").attr("stroke", "var(--cds-chart-axis)"))
      .selectAll("text").attr("font-family", null).attr("font-size", null);
    svg.append("g").attr("transform", `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(4).tickFormat(d => d + "s").tickSize(-(width - margin.left - margin.right)))
      .call(g => g.select(".domain").remove())
      .call(g => g.selectAll(".tick line").attr("stroke", "var(--cds-chart-grid)"))
      .selectAll("text").attr("font-family", null).attr("font-size", null);
    const line = d3.line().x(d => x(d.delay_ms)).y(d => y(d.round_seconds));
    svg.append("path").datum(rows).attr("d", line).attr("fill", "none").attr("stroke", dash.colors[0]).attr("stroke-width", 2);
    svg.selectAll("circle").data(rows).join("circle")
      .attr("cx", d => x(d.delay_ms)).attr("cy", d => y(d.round_seconds)).attr("r", 4.5).attr("fill", dash.colors[0])
      .on("mousemove", function (ev, d) { showTip(ev, `${d.delay_ms}ms delay: ${d.round_seconds.toFixed(1)}s/round`); })
      .on("mouseleave", hideTip);
  });

  // ---------- float16 derived stat ----------
  dash.onData(function () {
    const r = dash.data("float16_check");
    const el = document.querySelector("#stat-sizecut");
    if (r.status !== "ok" || !el) return;
    const row = r.data[0];
    el.textContent = fmtPct0(1 - row.size_ratio_f16_over_f32);
  });

  // ---------- the animated flow demo ----------
  // Built with SVG + CSS transitions rather than <canvas>: it renders
  // through the exact same DOM/paint path as the rest of the page (the
  // charts, the stats -- all of which work reliably everywhere), instead
  // of a separate Canvas2D compositing path that turned out to silently
  // never update on at least one real machine (healthy requestAnimationFrame,
  // zero console errors, correct canvas size -- yet pixels never changed).
  // SVG presentation attributes like fill/stroke DO resolve CSS var()
  // tokens directly (unlike ctx.fillStyle), so no manual color-resolution
  // helper is needed here either.
  const svgNS = "http://www.w3.org/2000/svg";
  const stage = document.querySelector("#demo-canvas");
  const nowEl = document.querySelector("#demo-now");
  const playBtn = document.querySelector("#demo-play");
  const restartBtn = document.querySelector("#demo-restart");
  let trafficRows = null, running = true, spawnIdx = 0;
  let stats = { total: 0, correct: 0, attacks: 0, blindCaught: 0, blindTotal: 0 };

  const VBW = 1000, VBH = 400;
  const GUARD = { x: VBW * 0.83, y: VBH * 0.52 };

  function houseForDevice(id, n) {
    const angle = Math.PI * 0.85 + (id - 1) / (n - 1) * Math.PI * 1.3;
    const cx = VBW * 0.24, cy = VBH * 0.52, r = Math.min(VBW * 0.21, VBH * 0.4);
    return { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
  }

  function svgEl(tag, attrs) {
    const el = document.createElementNS(svgNS, tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // A device "sensor node" glyph: rounded body + lens, read as a small
  // security camera/sensor rather than an abstract dot, plus a static
  // link line to the guard so the whole stage reads as a real network
  // topology diagram even before anything starts moving.
  function deviceGlyph(p, blind) {
    const g = svgEl("g", {});
    g.appendChild(svgEl("line", {
      x1: p.x, y1: p.y, x2: GUARD.x, y2: GUARD.y,
      stroke: "var(--color-border-line)", "stroke-width": 1, "stroke-dasharray": "1 5",
    }));
    if (blind) {
      g.appendChild(svgEl("rect", {
        x: p.x - 16, y: p.y - 16, width: 32, height: 32, rx: 9,
        fill: "none", stroke: "var(--color-accent)", "stroke-width": 1.2, "stroke-dasharray": "3 3", opacity: .55,
      }));
    }
    g.appendChild(svgEl("rect", {
      x: p.x - 11, y: p.y - 11, width: 22, height: 22, rx: 6,
      fill: "var(--color-panel-raised)", stroke: "var(--color-border-line)", "stroke-width": 1.3,
    }));
    g.appendChild(svgEl("circle", {
      cx: p.x, cy: p.y, r: 4.5,
      fill: "var(--color-bg)", stroke: "var(--color-fg-muted)", "stroke-width": 1.1,
    }));
    g.appendChild(svgEl("circle", { cx: p.x, cy: p.y, r: 1.4, fill: "var(--color-fg-muted)" }));
    return g;
  }

  let particleLayer = null;
  function setupStage() {
    stage.innerHTML = "";
    const linkLayer = svgEl("g", {});
    stage.appendChild(linkLayer);
    const n = 9;
    for (let i = 1; i <= n; i++) {
      const p = houseForDevice(i, n);
      const blind = trafficRows && trafficRows.some(d => d.device_id === i && d.device_locally_blind_to_mirai);
      linkLayer.appendChild(deviceGlyph(p, blind));
    }
    particleLayer = svgEl("g", { id: "demo-particle-layer" });
    stage.appendChild(particleLayer);
    const guardGlow = svgEl("circle", { cx: GUARD.x, cy: GUARD.y, r: 24, fill: "var(--color-accent)", opacity: .08 });
    stage.appendChild(guardGlow);
    stage.appendChild(svgEl("circle", {
      cx: GUARD.x, cy: GUARD.y, r: 15,
      fill: "var(--color-panel-raised)", stroke: "var(--color-accent)", "stroke-width": 1.6,
    }));
    stage.appendChild(svgEl("circle", { cx: GUARD.x, cy: GUARD.y, r: 3, fill: "var(--color-accent)" }));
    stage.appendChild(svgEl("text", {
      x: GUARD.x, y: GUARD.y + 32, "text-anchor": "middle",
      fill: "var(--color-fg-muted)", "font-size": 12, "font-family": "var(--font-mono)",
    })).textContent = "guard";
  }

  function spawnNext() {
    if (!trafficRows || !running) return;
    const row = trafficRows[spawnIdx % trafficRows.length];
    spawnIdx++;
    const from = houseForDevice(row.device_id, 9);
    const special = row.device_locally_blind_to_mirai && row.is_mirai;
    const color = !row.is_attack ? "var(--color-fg-muted)" : (row.correct ? "var(--color-ok)" : "var(--color-bad)");

    const dot = svgEl("circle", { cx: from.x, cy: from.y, r: special ? 5.5 : 4, fill: color });
    dot.style.transition = "cx 900ms linear, cy 900ms linear, opacity 280ms linear 620ms";
    dot.style.opacity = "1";
    particleLayer.appendChild(dot);

    let ring = null;
    if (special) {
      ring = svgEl("circle", { cx: from.x, cy: from.y, r: 8, fill: "none", stroke: color, "stroke-width": 1.5 });
      ring.style.transition = dot.style.transition;
      particleLayer.appendChild(ring);
    }
    // Set the start position synchronously above, then move to the end
    // position one frame later so the browser registers the "from" state
    // first and actually animates the transition instead of jumping.
    requestAnimationFrame(function () {
      dot.setAttribute("cx", GUARD.x); dot.setAttribute("cy", GUARD.y);
      dot.style.opacity = "0";
      if (ring) { ring.setAttribute("cx", GUARD.x); ring.setAttribute("cy", GUARD.y); ring.style.opacity = "0"; }
    });
    setTimeout(function () { dot.remove(); if (ring) ring.remove(); }, 950);

    stats.total++; if (row.correct) stats.correct++;
    if (row.is_attack) stats.attacks++;
    if (special) { stats.blindTotal++; if (row.predicted_label !== "benign") stats.blindCaught++; }
    const totalEl = document.querySelector("#stat-total"), accEl = document.querySelector("#stat-acc");
    if (totalEl) totalEl.textContent = fmtInt(stats.total);
    if (accEl) accEl.textContent = fmtPct0(stats.correct / stats.total);

    if (nowEl) {
      const verdict = !row.is_attack ? "benign" : (row.correct ? "caught ✓" : "missed (named the wrong attack family)");
      const blindNote = special ? " — this device has zero Mirai of its own" : "";
      nowEl.innerHTML = `<b>${row.device_name}</b> &rarr; actual: <b>${row.actual_label}</b>, model said: <b>${row.predicted_label}</b> (${verdict})${blindNote}`;
    }
  }

  dash.onData(function () {
    const r = dash.data("sample_traffic");
    if (r.status !== "ok" || trafficRows) return;
    trafficRows = r.data.slice(0, 60);
    const totalEl = document.querySelector("#stat-total");
    if (totalEl) totalEl.textContent = "0";
    setupStage();
    setInterval(spawnNext, 420);
  });

  playBtn.addEventListener("click", function () {
    running = !running;
    playBtn.textContent = running ? "Pause" : "Play";
  });
  restartBtn.addEventListener("click", function () {
    spawnIdx = 0; stats = { total: 0, correct: 0, attacks: 0, blindCaught: 0, blindTotal: 0 };
    running = true; playBtn.textContent = "Pause";
    if (particleLayer) particleLayer.innerHTML = "";
    const totalEl = document.querySelector("#stat-total"), accEl = document.querySelector("#stat-acc");
    if (totalEl) totalEl.textContent = "0";
    if (accEl) accEl.textContent = "—";
  });
})();
