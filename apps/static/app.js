/* ============================================================
   OptiGuard Operator Console — app.js  (Step 14)
   No build step, no CDN. Runs from file served by serve.py.
   ============================================================ */

'use strict';

// ── State ──────────────────────────────────────────────────────
const state = {
  result:       null,   // last full result payload from /analyse
  H:            0,
  W:            0,
  scaleMin:     -1.5,
  scaleMax:     1.5,
  dither:       true,
  // Per-pixel arrays (flat H×W, filled after scan)
  crlbFlat:     null,
  sigmaFlat:    null,
  ratioFlat:    null,
  failT1aFlat:  null,
  failT1bFlat:  null,
  failT4Flat:   null,
  riskFlat:     null,
  sgCenterFlat: null,
  ogCenterFlat: null,
};

// ── DOM refs ────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const materialSelect  = $('material-select');
const exposureSelect  = $('exposure-select');
const indexInput      = $('index-input');
const scanBtn         = $('scan-btn');
const btnLabel        = scanBtn.querySelector('.btn-label');
const btnSpinner      = scanBtn.querySelector('.btn-spinner');
const ditherCheck     = $('dither-check');

const decisionBand    = $('decision-band');
const decisionBadge   = $('decision-badge');
const decisionRat     = $('decision-rationale');

const canvasSG        = $('canvas-sg');
const canvasOG        = $('canvas-og');
const ctxSG           = canvasSG.getContext('2d');
const ctxOG           = canvasOG.getContext('2d');

const placeholderSG   = $('placeholder-sg');
const placeholderOG   = $('placeholder-og');

// ── Init ────────────────────────────────────────────────────────
fetchConfig();

ditherCheck.addEventListener('change', () => {
  state.dither = ditherCheck.checked;
  if (state.result) renderMaps();
});

scanBtn.addEventListener('click', runScan);

canvasOG.addEventListener('click', e => onPixelClick(e, canvasOG));
canvasSG.addEventListener('click', e => onPixelClick(e, canvasSG));

// Check for ?replay= query param (demo replay mode)
(function checkReplay() {
  const params = new URLSearchParams(window.location.search);
  if (params.has('replay')) {
    const path = params.get('replay');
    fetch(`/result/${path}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) updateConsole(data); })
      .catch(() => {});
  }
})();

// ── Config fetch ────────────────────────────────────────────────
async function fetchConfig() {
  try {
    const res  = await fetch('/health');
    const data = await res.json();
    // Also fetch /config for material list (old api.py compat + new serve.py)
    const cfgRes  = await fetch('/config').catch(() => null);
    if (cfgRes && cfgRes.ok) {
      const cfg = await cfgRes.json();
      materialSelect.innerHTML = '';
      cfg.materials.forEach(mat => {
        const opt = document.createElement('option');
        opt.value = mat;
        opt.textContent = mat.replace(/_/g, ' ');
        if (mat === 'silicon') opt.selected = true;
        materialSelect.appendChild(opt);
      });
    } else {
      // serve.py does not expose /config; fall back to known materials
      ['silicon','amorphous_silicon','sic_4h','gan','si_doublet','si_broad'].forEach(mat => {
        const opt = document.createElement('option');
        opt.value = mat;
        opt.textContent = mat.replace(/_/g, ' ');
        if (mat === 'silicon') opt.selected = true;
        materialSelect.appendChild(opt);
      });
    }
  } catch (e) {
    console.error('Config fetch failed', e);
  }
}

// ── Run scan ────────────────────────────────────────────────────
async function runScan() {
  setBusy(true);
  try {
    const body = {
      material:   materialSelect.value,
      index:      parseInt(indexInput.value) || 6,
      exposure_s: parseFloat(exposureSelect.value) || 0.1,
    };

    // Try /analyse (serve.py), fall back to /scan (api.py)
    let res = await fetch('/analyse', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      res = await fetch('/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
    }
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateConsole(data);
  } catch (e) {
    console.error('Scan failed', e);
    decisionBadge.textContent = 'ERROR';
    decisionRat.textContent   = 'Scan failed — check server console.';
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  scanBtn.disabled = busy;
  btnLabel.classList.toggle('hidden', busy);
  btnSpinner.classList.toggle('hidden', !busy);
}

// ── Update full console ─────────────────────────────────────────
function updateConsole(data) {
  state.result = data;

  // Normalise payload: serve.py wraps in {meta, summary, maps, sg_baseline}
  // api.py uses flat {summary, risk_map, ood_score_map, crlb_map, shape}
  const isNewFormat = !!data.maps;

  const summary = data.summary;
  const shape   = isNewFormat ? data.meta?.shape : data.shape;
  const H = state.H = shape ? shape[0] : 64;
  const W = state.W = shape ? shape[1] : 64;

  // ── Flatten maps ──
  const riskGrid   = isNewFormat ? data.maps.risk_class    : data.risk_map;
  const crlbGrid   = isNewFormat ? data.maps.crlb          : data.crlb_map;
  const sigmaGrid  = isNewFormat ? data.maps.sigma_scaled  : null;
  const ratioGrid  = isNewFormat ? data.maps.precision_ratio : null;
  const t1aGrid    = isNewFormat ? data.maps.fail_t1a      : null;
  const t1bGrid    = isNewFormat ? data.maps.fail_t1b      : null;
  const t4Grid     = isNewFormat ? data.maps.fail_t4       : null;
  const sgCGrid    = data.sg_baseline?.sg_center_map       ?? null;
  const ogCGrid    = isNewFormat ? data.maps.center        : null;

  state.riskFlat    = flattenGrid(riskGrid, H, W);
  state.crlbFlat    = flattenNumGrid(crlbGrid, H, W);
  state.sigmaFlat   = flattenNumGrid(sigmaGrid, H, W);
  state.ratioFlat   = flattenNumGrid(ratioGrid, H, W);
  state.failT1aFlat = flattenNumGrid(t1aGrid, H, W);
  state.failT1bFlat = flattenNumGrid(t1bGrid, H, W);
  state.failT4Flat  = flattenNumGrid(t4Grid, H, W);
  state.sgCenterFlat = flattenNumGrid(sgCGrid, H, W);
  state.ogCenterFlat = flattenNumGrid(ogCGrid, H, W);

  // Compute shared colour scale from actual fitted centers (Δ cm⁻¹)
  const centers = (state.ogCenterFlat || state.sgCenterFlat).filter(v => v !== null && isFinite(v));
  if (centers.length > 0) {
    const sorted = [...centers].sort((a, b) => a - b);
    const p2  = sorted[Math.floor(sorted.length * 0.02)];
    const p98 = sorted[Math.floor(sorted.length * 0.98)];
    state.scaleMin = p2;
    state.scaleMax = p98;
  }

  // ── Render maps ──
  renderMaps();

  // ── Decision badge ──
  const nExploit  = summary.n_exploit  ?? 0;
  const nErasure  = summary.n_feature_erasure ?? 0;
  const nHalluc   = summary.n_hallucination ?? 0;
  const nOod      = summary.n_ood ?? 0;
  let decision, badgeClass, rationale;

  if (nExploit > 0 || nOod > 0) {
    decision   = 'WITHHELD';
    badgeClass = 'badge--withheld';
    if (nOod > 0) {
        rationale = `${nOod} pixel${nOod > 1 ? 's' : ''} out-of-distribution (e.g. linewidth inconsistent with instrument prior). Peak model may not apply. Certification withheld.`;
    } else {
        rationale  = `${nExploit} pixel${nExploit > 1 ? 's' : ''} claim precision the photons cannot support. Certification withheld.`;
    }
  } else if (nErasure > 0 || nHalluc > 0) {
    decision   = 'REVIEW';
    badgeClass = 'badge--review';
    rationale  = `${nErasure} feature-erasure pixel${nErasure !== 1 ? 's' : ''} require re-scan at higher exposure.`;
  } else {
    decision   = 'CERTIFIED';
    badgeClass = 'badge--certified';
    rationale  = `${summary.n_pass} pixels certified. All measurements are photon-budget consistent.`;
  }

  decisionBadge.textContent = decision;
  decisionBadge.className   = 'decision-badge ' + badgeClass;
  decisionRat.textContent   = rationale;

  // ── Panel stats ──
  const mat      = materialSelect.value;
  const exposure = parseFloat(exposureSelect.value) || 0.1;
  $('scan-meta').textContent =
    `${mat.replace(/_/g, ' ')} · ${H}×${W} · ${exposure.toFixed(2)} s/pt · 532 nm`;

  const total = H * W;
  $('og-certified').textContent = summary.n_pass ?? '—';
  $('og-review').textContent    = (summary.n_feature_erasure ?? 0) + (summary.n_hallucination ?? 0);
  $('og-withheld').textContent  = summary.n_exploit ?? 0;

  // SG baseline (Now Spatial Gauss)
  const sg = data.sg_baseline;
  if (sg) {
    $('sg-rmse').textContent  = sg.rmse_center_cm1 != null ? sg.rmse_center_cm1.toFixed(4) : '—';
    $('sg-gain').textContent  = summary.speedup_factor != null ? summary.speedup_factor.toFixed(1) : '—';
    $('sg-false').textContent = '0';  // SG doesn't flag; Step 7 proved 0 false features
    $('sg-footnote').textContent =
      `Spatial Gaussian Denoiser (σ=1.0). ${nErasure} of ${total} pixel${nErasure !== 1 ? 's' : ''} ` +
      `not supported by the photons collected at ${exposure.toFixed(2)} s.`;
  }

  // Recommendation
  const nFlagged = summary.n_flagged_points ?? nErasure;
  const hybrid   = summary.cost_hybrid_s;
  const slow     = summary.cost_slow_scan_s;
  const planAct  = summary.planner_action;

  if (nFlagged > 0 && planAct) {
    const recExp = planAct.replace(/_/g, ' ').toLowerCase();
    $('rec-text').textContent =
      `Re-scan ${nFlagged} flagged point${nFlagged !== 1 ? 's' : ''} at ${recExp}.`;
  } else if (nFlagged === 0) {
    $('rec-text').textContent = 'No re-scan required — map is fully certified.';
  } else {
    $('rec-text').textContent = '—';
  }

  if (hybrid != null) {
    $('time-hybrid').textContent = fmtTime(hybrid);
    $('time-slow').textContent   = fmtTime(slow);
  }

  // Scale bar labels
  $('scale-min').textContent = state.scaleMin.toFixed(2);
  $('scale-max').textContent = state.scaleMax.toFixed(2);

  // Show canvases
  placeholderSG.classList.add('hidden');
  placeholderOG.classList.add('hidden');
}

// ── Render both maps ────────────────────────────────────────────
function renderMaps() {
  const { H, W, scaleMin, scaleMax, dither } = state;
  if (!H || !W) return;

  // Left: SG stress — always smooth
  renderStressMap(ctxSG, H, W, state.sgCenterFlat,
                  scaleMin, scaleMax, false);

  // Right: OG certified map with optional dither on uncertified pixels
  renderCertifiedMap(ctxOG, H, W, state.ogCenterFlat,
                     state.riskFlat, state.crlbFlat, scaleMin, scaleMax, dither);
}

// Render a smooth stress map (shared colour scale)
function renderStressMap(ctx, H, W, valFlat, vMin, vMax, _dither) {
  const img  = ctx.createImageData(W, H);
  const data = img.data;
  const rng  = vMax - vMin || 1;

  for (let i = 0; i < H * W; i++) {
    const v   = valFlat ? valFlat[i] : null;
    const t   = v !== null && isFinite(v) ? Math.max(0, Math.min(1, (v - vMin) / rng)) : 0.5;
    const rgb = stressColor(t);
    data[i*4]   = rgb[0];
    data[i*4+1] = rgb[1];
    data[i*4+2] = rgb[2];
    data[i*4+3] = 255;
  }
  blit(ctx, img, W, H);
}

// Render certified map: smooth where CERTIFIED, dithered where WITHHELD/REVIEW
function renderCertifiedMap(ctx, H, W, valFlat, riskFlat, crlbFlat, vMin, vMax, dither) {
  const img  = ctx.createImageData(W, H);
  const data = img.data;
  const rng  = vMax - vMin || 1;

  for (let i = 0; i < H * W; i++) {
    const risk = riskFlat ? riskFlat[i] : 'PASS';
    let v = valFlat ? valFlat[i] : null;
    if (v === null || !isFinite(v)) v = (vMin + vMax) / 2;

    const certified = (risk === 'PASS');
    if (!certified && dither) {
      // Dither by CRLB: add noise ±crlb/2 to visualise uncertainty
      const crlb = (crlbFlat && crlbFlat[i] !== null && isFinite(crlbFlat[i]))
        ? crlbFlat[i] : (vMax - vMin) * 0.1;
      const noiseMag = Math.min(crlb * 0.5, rng * 0.25);
      v += (Math.random() * 2 - 1) * noiseMag;
    }

    const t   = Math.max(0, Math.min(1, (v - vMin) / rng));
    let rgb = stressColor(t);

    // Hatch uncertified in non-dither mode (every 3rd pixel darkened)
    if (!certified && !dither) {
      const row = Math.floor(i / W), col = i % W;
      if ((row + col) % 3 === 0) rgb = rgb.map(c => Math.floor(c * 0.55));
    }

    data[i*4]   = rgb[0];
    data[i*4+1] = rgb[1];
    data[i*4+2] = rgb[2];
    data[i*4+3] = 255;
  }
  blit(ctx, img, W, H);
}

// Diverging colour scale: green (low) → warm white (mid) → red (high)
function stressColor(t) {
  // t = 0 → certified green, t = 0.5 → near-white, t = 1 → withheld red
  if (t < 0.5) {
    const s = t * 2;              // 0→1
    return [
      lerp(31,  245, s),          // R: 31 → 245
      lerp(122, 240, s),          // G: 122 → 240
      lerp(76,  230, s),          // B: 76 → 230
    ];
  } else {
    const s = (t - 0.5) * 2;     // 0→1
    return [
      lerp(245, 158, s),          // R: 245 → 158
      lerp(240, 43,  s),          // G: 240 → 43
      lerp(230, 37,  s),          // B: 230 → 37
    ];
  }
}

function lerp(a, b, t) { return Math.round(a + (b - a) * t); }

function blit(ctx, imgData, W, H) {
  const tmp = document.createElement('canvas');
  tmp.width = W; tmp.height = H;
  tmp.getContext('2d').putImageData(imgData, 0, 0);
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(tmp, 0, 0, W, H, 0, 0, cw, ch);
}

// ── Pixel click → evidence row ──────────────────────────────────
function onPixelClick(e, canvas) {
  if (!state.result) return;
  const rect  = canvas.getBoundingClientRect();
  const xFrac = (e.clientX - rect.left)  / rect.width;
  const yFrac = (e.clientY - rect.top)   / rect.height;
  const col   = Math.floor(xFrac * state.W);
  const row   = Math.floor(yFrac * state.H);
  if (col < 0 || row < 0 || col >= state.W || row >= state.H) return;
  showEvidence(row, col);
}

function showEvidence(row, col) {
  const idx   = row * state.W + col;
  const crlb  = safeVal(state.crlbFlat,   idx);
  const sigma = safeVal(state.sigmaFlat,  idx);
  const ratio = safeVal(state.ratioFlat,  idx);
  const t1a   = state.failT1aFlat ? state.failT1aFlat[idx] : null;
  const t1b   = state.failT1bFlat ? state.failT1bFlat[idx] : null;
  const t4    = state.failT4Flat  ? state.failT4Flat[idx]  : null;
  const risk  = state.riskFlat    ? state.riskFlat[idx]    : null;

  $('ev-coords').textContent = `${col}, ${row}`;

  // Photons: sum of crlb window (proxy); serve.py doesn't expose per-pixel photons
  // Use crlb to back-derive ~N: CRLB ≈ fwhm/(2*sqrt(N)) → N ≈ (fwhm/(2*crlb))²
  const fwhm = 3.5;
  const approxPhotons = crlb != null && crlb > 0 && crlb < 999
    ? Math.round((fwhm / (2 * crlb)) ** 2)
    : null;
  $('ev-photons').textContent  = approxPhotons != null ? approxPhotons.toLocaleString() : '—';
  $('ev-crlb').textContent     = crlb != null && crlb < 999 ? crlb.toFixed(4) : '—';
  $('ev-sigma').textContent    = sigma != null ? sigma.toFixed(4) : '—';
  $('ev-ratio').textContent    = ratio != null && isFinite(ratio) ? ratio.toFixed(2) : '—';

  setGateBadge('ev-t1a', t1a);
  setGateBadge('ev-t1b', t1b);
  setGateBadge('ev-t4',  t4);

  // Verdict in operator language
  let verdict = 'Measurement certified — photon budget consistent.';
  if (risk === 'EXPLOIT' || (t1a && t1b)) {
    verdict = `χ² deficit + precision below floor. ${sigma != null ? 'Claimed σ ' + sigma.toFixed(4) + ' cm⁻¹' : ''} violates information limit. Certification withheld.`;
  } else if (risk === 'FEATURE_ERASURE' || t4) {
    verdict = 'Peak shift from spatial background erased by processing. Re-scan this point at higher exposure.';
  } else if (risk === 'HALLUCINATION') {
    verdict = 'Restored spectrum inconsistent with raw photon counts. Possible hallucination.';
  } else if (risk === 'REVIEW') {
    verdict = 'Out-of-distribution signal detected — model assumption may not hold here.';
  } else if (t1a && !t1b) {
    verdict = 'Variance suppressed below Poisson floor, but within legitimate pooling credit.';
  }
  $('ev-verdict').textContent = verdict;
}

function setGateBadge(id, failVal) {
  const el = $(id);
  if (failVal === null) { el.className = 'gate-badge'; return; }
  const failed = failVal === 1 || failVal === true;
  el.className = 'gate-badge ' + (failed ? 'fail' : 'pass');
}

function safeVal(arr, idx) {
  if (!arr) return null;
  const v = arr[idx];
  return (v !== null && v !== undefined && isFinite(v)) ? v : null;
}

// ── Helpers ─────────────────────────────────────────────────────
function flattenGrid(grid, H, W) {
  if (!grid) return null;
  const out = new Array(H * W);
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++)
      out[r * W + c] = grid[r]?.[c] ?? null;
  return out;
}

function flattenNumGrid(grid, H, W) {
  if (!grid) return null;
  const out = new Float32Array(H * W);
  for (let r = 0; r < H; r++)
    for (let c = 0; c < W; c++) {
      const v = grid[r]?.[c];
      out[r * W + c] = (v !== null && v !== undefined) ? v : NaN;
    }
  return out;
}

function fmtTime(seconds) {
  if (seconds == null) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m} min ${s} s` : `${s} s`;
}
