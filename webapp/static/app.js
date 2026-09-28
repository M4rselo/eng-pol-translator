(function () {
  "use strict";

  const state = {
    version: "v1_5",
    selfRef: "na",
    addrRef: "na",
    alpha: 0.6,
    numK: 5,
    activeTab: "translate",
    lastTranslation: null,
    translateAbort: null,
    tokenizeAbort: null,
    lastCompare: null,
    compareGroupBy: "self",
  };

  const versionSelect = document.getElementById("version-select");
  const refButtonsEl = document.getElementById("ref-buttons");
  const alphaSlider = document.getElementById("alpha-slider");
  const alphaVal = document.getElementById("alpha-val");
  const numkInput = document.getElementById("numk-input");
  const srcText = document.getElementById("src-text");
  const tgtText = document.getElementById("tgt-text");
  const confBadge = document.getElementById("confidence-badge");
  const wordconfToggle = document.getElementById("wordconf-toggle");
  const wordLegend = document.getElementById("word-legend");
  const tokenCounter = document.getElementById("token-counter");
  const tokenChips = document.getElementById("token-chips");
  const tgtChips = document.getElementById("tgt-chips");
  const examplesRow = document.getElementById("examples-row");
  const statusMsg = document.getElementById("status-msg");

  const insightStatus = document.getElementById("insight-status");
  const insightMap = document.getElementById("insight-map");
  const prefixAttnMap = document.getElementById("prefix-attn-map");

  const compareRunBtn = document.getElementById("compare-run");
  const compareStatus = document.getElementById("compare-status");
  const compareResults = document.getElementById("compare-results");
  const compareGroupbyEl = document.getElementById("compare-groupby");

  // ---------------------------------------------------------------------
  // Reference-token buttons: 3 for self-only versions (v1_2/v1_3),
  // 3 + 4 for versions that also have addressee tokens (v1_4/v1_5).
  // Buttons show the actual token spelling (<self_f>, <addr_na>, ...).
  // ---------------------------------------------------------------------
  function renderRefButtons() {
    const cfg = UI_CONFIGS[state.version];
    refButtonsEl.innerHTML = "";
    state.selfRef = "na";
    state.addrRef = "na";

    refButtonsEl.appendChild(buildRefGroup("Speaker:", "self", cfg.self_opts));
    if (cfg.has_addr) {
      refButtonsEl.appendChild(buildRefGroup("Addressee:", "addr", cfg.addr_opts));
    }
  }

  function buildRefGroup(label, group, opts) {
    const wrap = document.createElement("div");
    wrap.className = "ref-group";
    const title = document.createElement("span");
    title.className = "ref-group-title";
    title.textContent = label;
    wrap.appendChild(title);

    opts.forEach((opt) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = `<${group}_${opt}>`;
      btn.className = "ref-btn" + (opt === "na" ? " active" : "");
      btn.addEventListener("click", () => {
        if (group === "self") state.selfRef = opt; else state.addrRef = opt;
        wrap.querySelectorAll(".ref-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        scheduleTranslate();
      });
      wrap.appendChild(btn);
    });
    return wrap;
  }

  // ---------------------------------------------------------------------
  // Live token counter: a cheap, model-free /api/tokenize call, debounced
  // much shorter than translation itself since it does no beam search.
  // Shown as a badge (same visual weight as the confidence badge) with
  // traffic-light coloring as the count approaches the model's limit.
  // ---------------------------------------------------------------------
  let tokenizeDebounce = null;

  function scheduleTokenize() {
    if (tokenizeDebounce) clearTimeout(tokenizeDebounce);
    tokenizeDebounce = setTimeout(runTokenize, 120);
  }

  async function runTokenize() {
    const text = srcText.value;
    if (!text.trim()) {
      tokenCounter.hidden = true;
      tokenChips.innerHTML = "";
      return;
    }
    if (state.tokenizeAbort) state.tokenizeAbort.abort();
    const controller = new AbortController();
    state.tokenizeAbort = controller;
    try {
      const resp = await fetch("/api/tokenize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: state.version, text }),
        signal: controller.signal,
      });
      const data = await resp.json();
      if (!resp.ok) return;

      const ratio = data.count / data.max_tokens;
      const level = ratio > 1 ? "tc-over" : ratio >= 0.8 ? "tc-warn" : "tc-ok";
      tokenCounter.hidden = false;
      tokenCounter.className = "token-counter " + level;
      tokenCounter.textContent = `${data.count}/${data.max_tokens} tokens`;

      tokenChips.innerHTML = data.tokens
        .map((t) => `<span class="token-chip${t.over_limit ? " over-limit" : ""}" title="id ${t.id}">${escapeHtml(t.text)}</span>`)
        .join("");
    } catch (err) {
      if (err.name !== "AbortError") { /* silent -- this is a background nicety, not core functionality */ }
    }
  }

  // ---------------------------------------------------------------------
  // Translate tab: realtime, debounced, cancels its own stale requests.
  // Never touches Comparison -- that only runs on an explicit click.
  // ---------------------------------------------------------------------
  let debounceTimer = null;

  function scheduleTranslate() {
    scheduleTokenize();
    examplesRow.hidden = srcText.value.trim().length > 0;

    if (debounceTimer) clearTimeout(debounceTimer);
    const text = srcText.value.trim();
    if (!text) {
      tgtText.textContent = "";
      confBadge.hidden = true;
      wordLegend.hidden = true;
      tgtChips.innerHTML = "";
      return;
    }
    // Spinner replaces whatever is currently in the Polish box -- it is
    // swapped back out by runTranslate() itself once a result (or error)
    // comes back, never toggled via [hidden] (see style.css for why).
    tgtText.innerHTML = '<span class="spinner-inline" aria-label="Translating"></span>';
    confBadge.hidden = true;
    debounceTimer = setTimeout(runTranslate, 300);
  }

  async function runTranslate() {
    const text = srcText.value.trim();
    if (state.translateAbort) state.translateAbort.abort();
    const controller = new AbortController();
    state.translateAbort = controller;

    try {
      const resp = await fetch("/api/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version: state.version, text,
          self_ref: state.selfRef, addr_ref: state.addrRef,
          alpha: state.alpha, num_k: state.numK,
        }),
        signal: controller.signal,
      });
      const data = await resp.json();
      if (!resp.ok) {
        statusMsg.textContent = "Error: " + data.error;
        tgtText.textContent = "";
        return;
      }
      statusMsg.textContent = "";
      state.lastTranslation = data;
      renderOutput(data);
      renderConfidence(data);
      renderTgtChips(data);
    } catch (err) {
      if (err.name === "AbortError") return; // superseded by a newer keystroke, not a real error
      statusMsg.textContent = "Error: " + err.message;
      if (state.translateAbort === controller) tgtText.textContent = "";
    }
  }

  function renderConfidence(data) {
    confBadge.hidden = false;
    confBadge.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
  }

  // Returns a hex color (not rgb(...)) specifically so `${color}33`-style
  // alpha suffixes elsewhere (a valid #RRGGBBAA trick) actually work --
  // appending "33" to "rgb(217,83,79)" is not valid CSS and was silently
  // dropped by the browser (found while redesigning the Comparison cards).
  function confColor(pct) {
    const t = Math.max(0, Math.min(1, pct / 100));
    const r = Math.round(217 - t * 180);
    const g = Math.round(83 + t * 140);
    const b = Math.round(79 + t * 40);
    return "#" + [r, g, b].map((x) => x.toString(16).padStart(2, "0")).join("");
  }

  // Renders the translation into #tgt-text -- either as plain text, or,
  // when the toggle is on, as the same sentence with each word colored by
  // its own confidence (in place, not as a separate list next to it).
  // Hovering a word dims its siblings so the tooltip % is unambiguous.
  function renderOutput(data) {
    const words = data.words;
    if (!wordconfToggle.checked || !words || !words.length) {
      wordLegend.hidden = true;
      tgtText.textContent = data.translation;
      return;
    }
    wordLegend.hidden = false;
    tgtText.innerHTML = words
      .map((w) => {
        const color = confColor(w.confidence);
        return `<span class="word-chip" style="border-bottom-color:${color};background-color:${color}33">` +
               `${escapeHtml(w.word)}<span class="chip-pct">${w.confidence.toFixed(1)}%</span></span>`;
      })
      .join(" ");

    const chips = tgtText.querySelectorAll(".word-chip");
    chips.forEach((chip) => {
      chip.addEventListener("mouseenter", () => {
        chips.forEach((c) => { if (c !== chip) c.classList.add("dimmed"); });
      });
      chip.addEventListener("mouseleave", () => {
        chips.forEach((c) => c.classList.remove("dimmed"));
      });
    });
  }

  wordconfToggle.addEventListener("change", () => {
    if (state.lastTranslation) renderOutput(state.lastTranslation);
  });

  // Raw BPE sub-word pieces of the *target* sentence -- always shown,
  // mirroring the English token-chips row (same idea, other side, no
  // separate toggle needed since it's just visual segmentation, no ids).
  function renderTgtChips(data) {
    if (!data.subwords || !data.subwords.length) {
      tgtChips.innerHTML = "";
      return;
    }
    tgtChips.innerHTML = data.subwords
      .map((s) => `<span class="token-chip">${escapeHtml(s.text)}</span>`)
      .join("");
  }

  // ---------------------------------------------------------------------
  // Example sentences -- verified against the live model to each show a
  // clean, distinct gender-context contrast. Hidden as soon as there is
  // any input, shown again once the box is cleared (see scheduleTranslate).
  // ---------------------------------------------------------------------
  document.querySelectorAll(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      srcText.value = chip.dataset.sample;
      scheduleTranslate();
    });
  });

  // ---------------------------------------------------------------------
  // Attention Map tab: fetched once when the tab is opened, never from
  // realtime typing.
  // ---------------------------------------------------------------------
  // Light end matches the confidence-badge blue (#e8f0fe) so the heatmap
  // reads as "part of the same app", not an arbitrary spreadsheet gradient.
  // Real attention rows are mostly small fractions spread across many
  // columns, so a *linear* map crams almost every cell into the same pale
  // corner of the scale -- a sqrt curve (gamma 0.5) pulls the low/mid
  // range apart so genuinely different weights actually look different,
  // and the darker end is pushed further down for more usable range.
  // Reverted the sqrt/darker-end contrast experiment -- it read as too
  // heavy/dark. Back to the plain light-blue linear scale that matched
  // the rest of the app's palette (confidence-badge blue at the light end).
  function attnColor(w) {
    const t = Math.max(0, Math.min(1, w));
    const r = Math.round(232 - t * 209);
    const g = Math.round(240 - t * 162);
    const b = Math.round(254 - t * 88);
    return `rgb(${r},${g},${b})`;
  }

  async function runInsight() {
    const text = srcText.value.trim();
    if (!text) {
      insightMap.innerHTML = "<p class='hint'>Type a sentence in the Translate tab first.</p>";
      prefixAttnMap.innerHTML = "";
      return;
    }
    // A spinner instead of red status text -- it barely takes any time,
    // but a red line still reads as "something went wrong" for that split
    // second regardless of how briefly it's visible.
    insightStatus.innerHTML = '<span class="spinner-inline"></span>';
    try {
      const resp = await fetch("/api/insight", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          version: state.version, text,
          self_ref: state.selfRef, addr_ref: state.addrRef,
          alpha: state.alpha, num_k: state.numK,
        }),
      });
      const data = await resp.json();
      insightStatus.innerHTML = "";
      if (!resp.ok) {
        insightMap.innerHTML = `<p class='hint'>Error: ${data.error}</p>`;
        return;
      }
      renderAttention(data.attention);
      renderPrefixAttention(data.prefix_attention);
    } catch (err) {
      insightStatus.innerHTML = "";
      insightMap.innerHTML = `<p class='hint'>Error: ${escapeHtml(err.message)}</p>`;
    }
  }

  function renderGenericAttnTable(container, cols, rows, matrix) {
    if (!matrix || !matrix.length) {
      container.innerHTML = "<p class='hint'>No attention data.</p>";
      return;
    }
    let html = '<table class="attn-table"><thead><tr><th></th>';
    cols.forEach((c) => { html += `<th>${escapeHtml(c)}</th>`; });
    html += "</tr></thead><tbody>";
    for (let r = 0; r < matrix.length; r++) {
      html += `<tr><th>${escapeHtml(rows[r] || "")}</th>`;
      for (let c = 0; c < matrix[r].length; c++) {
        const w = matrix[r][c];
        html += `<td style="background:${attnColor(w)}" title="${(w * 100).toFixed(1)}%"></td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    // Inner scroll div (not the outer .table-wrap) owns overflow-x, so
    // the outer container's rounded corners+border aren't fought by the
    // scrollbar -- see style.css.
    container.innerHTML = `<div class="attn-scroll">${html}</div>`;
  }

  function renderAttention(attn) {
    if (!attn) { insightMap.innerHTML = "<p class='hint'>No attention data.</p>"; return; }
    renderGenericAttnTable(insightMap, attn.src_tokens, attn.tgt_tokens, attn.matrix);
  }

  // pattn.matrix is (tgt_word x prefix_token) -- only 2-3 prefix tokens, so
  // that orientation makes a tall, narrow, awkward table. Transposed, the
  // few prefix tokens become row labels and the (potentially many) target
  // words become column headers -- a short, wide table instead.
  function transpose(matrix) {
    if (!matrix || !matrix.length) return matrix;
    return matrix[0].map((_, c) => matrix.map((row) => row[c]));
  }

  function renderPrefixAttention(pattn) {
    if (!pattn) { prefixAttnMap.innerHTML = ""; return; }
    renderGenericAttnTable(prefixAttnMap, pattn.tgt_tokens, pattn.prefix_tokens, transpose(pattn.matrix));
  }

  // ---------------------------------------------------------------------
  // Comparison tab: only runs when the button is clicked. Never wired to
  // realtime typing or tab-switching, on purpose.
  // ---------------------------------------------------------------------
  async function runCompare() {
    const text = srcText.value.trim();
    if (!text) {
      compareResults.innerHTML = "<p class='hint'>Type a sentence in the Translate tab first.</p>";
      return;
    }
    // All combinations are already known from UI_CONFIGS (we don't need
    // the response to know *which* cards will exist) -- so the full card
    // grid, grouped exactly like the real result will be, appears
    // immediately with a spinner in place of each translation, instead of
    // a red status line that reads like an error.
    const cfg = UI_CONFIGS[state.version];
    compareGroupbyEl.hidden = !cfg.has_addr;
    renderCompareSkeleton(cfg, true);
    compareStatus.textContent = "";
    compareRunBtn.disabled = true;
    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ version: state.version, text, alpha: state.alpha, num_k: state.numK }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        compareResults.innerHTML = `<p class='hint'>Error: ${data.error}</p>`;
        return;
      }
      state.lastCompare = data;
      renderCompare(data);
    } catch (err) {
      compareResults.innerHTML = `<p class='hint'>Error: ${err.message}</p>`;
    } finally {
      compareRunBtn.disabled = false;
    }
  }

  // A flat grid of 12 near-identical cards made it hard to tell which
  // token actually varies, or that the rows are grouped/sorted at all.
  // Fix: group cards by whichever axis is *not* the current groupBy (an
  // explicit section per group, e.g. one per speaker gender), color
  // self-chips vs addr-chips differently, and give every chip a fixed
  // min-width so a short translation never makes its chip *look* smaller
  // than one on a card with a longer sentence next to it.
  function refChipHtml(kind, value) {
    const cls = kind === "self" ? "ref-chip-self" : "ref-chip-addr";
    return `<span class="ref-chip ${cls}">&lt;${kind}_${value}&gt;</span>`;
  }

  function compareCardHtml(row, chipHtml) {
    const color = confColor(row.confidence);
    return `<div class="compare-card">
        <div class="compare-tokens">${chipHtml}</div>
        <div class="compare-translation">${escapeHtml(row.translation)}</div>
        <div class="compare-conf" style="background:${color}22;color:${color}">${row.confidence.toFixed(1)}%</div>
      </div>`;
  }

  // withSpinner=false (tab-switch, before the button is clicked): an empty
  // card frame, so the grid/chips are already in their final layout, but
  // nothing here should read as "loading" since no request is in flight
  // yet. withSpinner=true (right after the click): same frame, now with a
  // spinner standing in for the translation until the response arrives.
  function compareSkeletonCardHtml(chipHtml, withSpinner) {
    return `<div class="compare-card">
        <div class="compare-tokens">${chipHtml}</div>
        <div class="compare-translation">${withSpinner ? '<span class="spinner-inline"></span>' : ""}</div>
      </div>`;
  }

  // Same grouping rules as renderCompare(), but built from UI_CONFIGS's
  // known self/addr options instead of a server response -- so the grid
  // (and which card is which) is already fully laid out the instant you
  // look at the tab, before the request even resolves.
  function renderCompareSkeleton(cfg, withSpinner) {
    if (!cfg.has_addr) {
      const cards = cfg.self_opts
        .map((s) => compareSkeletonCardHtml(refChipHtml("self", s), withSpinner))
        .join("");
      compareResults.innerHTML = `<div class="compare-group-cards">${cards}</div>`;
      return;
    }

    const outerKind = state.compareGroupBy;
    const innerKind = outerKind === "self" ? "addr" : "self";
    const outerOpts = outerKind === "self" ? cfg.self_opts : cfg.addr_opts;
    const innerOpts = innerKind === "self" ? cfg.self_opts : cfg.addr_opts;
    const outerLabel = outerKind === "self" ? "Speaker" : "Addressee";

    compareResults.innerHTML = outerOpts
      .map((outerVal) => {
        const cards = innerOpts.map((innerVal) => compareSkeletonCardHtml(refChipHtml(innerKind, innerVal), withSpinner)).join("");
        return `<div class="compare-group">
            <div class="compare-group-title">${outerLabel}: ${refChipHtml(outerKind, outerVal)}</div>
            <div class="compare-group-cards">${cards}</div>
          </div>`;
      })
      .join("");
  }

  function renderCompare(data) {
    if (!data.has_addr) {
      const cards = data.rows
        .map((row) => compareCardHtml(row, refChipHtml("self", row.self_ref)))
        .join("");
      compareResults.innerHTML = `<div class="compare-group-cards">${cards}</div>`;
      return;
    }

    // state.compareGroupBy picks which axis becomes the section header
    // ("outer") vs. which one varies per card within it ("inner").
    const outerKind = state.compareGroupBy; // 'self' (default) or 'addr'
    const innerKind = outerKind === "self" ? "addr" : "self";
    const outerField = outerKind === "self" ? "self_ref" : "addr_ref";
    const innerField = innerKind === "self" ? "self_ref" : "addr_ref";
    const outerLabel = outerKind === "self" ? "Speaker" : "Addressee";

    const groups = new Map();
    data.rows.forEach((row) => {
      const key = row[outerField];
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    });

    compareResults.innerHTML = Array.from(groups.entries())
      .map(([outerVal, rows]) => {
        const cards = rows
          .map((row) => compareCardHtml(row, refChipHtml(innerKind, row[innerField])))
          .join("");
        return `<div class="compare-group">
            <div class="compare-group-title">${outerLabel}: ${refChipHtml(outerKind, outerVal)}</div>
            <div class="compare-group-cards">${cards}</div>
          </div>`;
      })
      .join("");
  }

  document.querySelectorAll(".groupby-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.compareGroupBy = btn.dataset.group;
      document.querySelectorAll(".groupby-btn").forEach((b) => b.classList.toggle("active", b === btn));
      if (state.lastCompare) renderCompare(state.lastCompare);
      else if (!compareGroupbyEl.hidden) renderCompareSkeleton(UI_CONFIGS[state.version], false);
    });
  });

  // ---------------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------------
  function switchTab(name) {
    state.activeTab = name;
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + name));
    if (name === "insight") runInsight();
    // The card grid (with its token chips) should already be there the
    // moment you look at the tab, not only after clicking the button --
    // only the translations themselves need the click.
    if (name === "comparison" && !state.lastCompare) {
      const cfg = UI_CONFIGS[state.version];
      compareGroupbyEl.hidden = !cfg.has_addr;
      renderCompareSkeleton(cfg, false);
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------------
  versionSelect.addEventListener("change", () => {
    state.version = versionSelect.value;
    renderRefButtons();
    scheduleTranslate();
    // Stale Comparison results (and its groupBy toggle) belong to whatever
    // version they were computed for -- clear them rather than leave them
    // silently mismatched with the newly selected version.
    state.lastCompare = null;
    state.compareGroupBy = "self";
    document.querySelectorAll(".groupby-btn").forEach((b) => b.classList.toggle("active", b.dataset.group === "self"));
    compareGroupbyEl.hidden = true;
    compareResults.innerHTML = "";
  });

  alphaSlider.addEventListener("input", () => {
    state.alpha = parseFloat(alphaSlider.value);
    alphaVal.textContent = state.alpha.toFixed(1);
    scheduleTranslate();
  });

  numkInput.addEventListener("change", () => {
    state.numK = Math.max(1, Math.min(20, parseInt(numkInput.value, 10) || 5));
    numkInput.value = state.numK;
    scheduleTranslate();
  });

  srcText.addEventListener("input", scheduleTranslate);

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  compareRunBtn.addEventListener("click", runCompare);

  // Sync state.version from the actual <select> (its default -- whichever
  // option has `selected` in the HTML, or the browser's first-option
  // fallback) instead of trusting the hardcoded initial value above; the
  // two silently disagreeing was exactly the v1_2-shows-addressee-buttons
  // bug on first load.
  state.version = versionSelect.value;
  renderRefButtons();
  scheduleTranslate();
})();
