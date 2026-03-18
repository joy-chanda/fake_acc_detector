/**
 * InstaCheck — Gaussian Naïve Bayes Classifier
 * Trained on 785 real Instagram profile records.
 *
 * Features (12):
 *   edge_followed_by, edge_follow, username_length, username_has_number,
 *   full_name_has_number, full_name_length, is_private, is_joined_recently,
 *   has_channel, is_business_account, has_guides, has_external_url
 */

// ─────────────────────────────────────────────
// MODEL PARAMETERS  (derived from dataset)
// Gaussian NB for continuous features, Bernoulli for binary.
// Fake class = 1 (≈ 50.3%), Real class = 0 (≈ 49.7%)
// ─────────────────────────────────────────────
const MODEL = {
  priors: { fake: 0.503, real: 0.497 },

  // Continuous features: [mean, variance] per class {fake, real}
  // Variances kept wide enough to avoid numerical underflow.
  // edge_followed_by / edge_follow / full_name_length are ONLY used
  // when the user has opened Advanced Options (advancedOpen flag).
  continuous: {
    // Always-used (derived from username string)
    username_length:   { fake: [11.8, 15.0], real: [7.8,  9.0] },
    // Advanced-only (require user-supplied values)
    edge_followed_by:  { fake: [0.004, 0.0016], real: [0.28, 0.055] },
    edge_follow:       { fake: [0.47,  0.070],  real: [0.21, 0.035] },
    full_name_length:  { fake: [4.5,   40.0],   real: [10.8, 18.0]  },
  },

  // Binary features: P(feature=1 | class) {fake, real}
  binary: {
    username_has_number:   { fake: 0.75, real: 0.30 },
    // The following are advanced-only (all default to 0 when panel closed)
    full_name_has_number:  { fake: 0.20, real: 0.08 },
    is_private:            { fake: 0.24, real: 0.42 },
    is_joined_recently:    { fake: 0.36, real: 0.08 },
    has_channel:           { fake: 0.02, real: 0.08 },
    is_business_account:   { fake: 0.05, real: 0.22 },
    has_guides:            { fake: 0.01, real: 0.08 },
    has_external_url:      { fake: 0.05, real: 0.26 },
  },
};

// Features that should only be scored when Advanced Options are open.
// These have zero discriminatory power at their default values.
const ADVANCED_CONTINUOUS = new Set(['edge_followed_by','edge_follow','full_name_length']);
const ADVANCED_BINARY     = new Set(['full_name_has_number','is_private','is_joined_recently',
                                      'has_channel','is_business_account','has_guides','has_external_url']);

// Tracks whether the user has opened the Advanced panel this session
let advancedOpen = false;

// Human-readable feature labels
const FEATURE_LABELS = {
  edge_followed_by:    'Follower Ratio',
  edge_follow:         'Following Ratio',
  username_length:     'Username Length',
  full_name_length:    'Full Name Length',
  username_has_number: 'Username Has Number',
  full_name_has_number:'Full Name Has Number',
  is_private:          'Private Account',
  is_joined_recently:  'Joined Recently',
  has_channel:         'Has Channel',
  is_business_account: 'Business Account',
  has_guides:          'Has Guides',
  has_external_url:    'External URL',
};

// ─────────────────────────────────────────────
// GAUSSIAN PDF
// ─────────────────────────────────────────────
function gaussianPDF(x, mean, variance) {
  const eps = 1e-9;
  const v = variance + eps;
  return (1 / Math.sqrt(2 * Math.PI * v)) * Math.exp(-((x - mean) ** 2) / (2 * v));
}

// ─────────────────────────────────────────────
// FEATURE EXTRACTION FROM USERNAME STRING
// ─────────────────────────────────────────────
function extractUsername(raw) {
  raw = raw.trim();
  // Handle URLs: instagram.com/username or @username
  const urlMatch = raw.match(/instagram\.com\/([A-Za-z0-9_.]+)/i);
  if (urlMatch) return urlMatch[1];
  return raw.replace(/^@/, '');
}

function buildFeatures(username, opts = {}) {
  return {
    // Always derived from the username string itself
    username_length:     username.length,
    username_has_number: /\d/.test(username) ? 1 : 0,
    // Advanced-only — only populated when advancedOpen is true
    ...(advancedOpen ? {
      edge_followed_by:    parseFloat(opts.edge_followed_by),
      edge_follow:         parseFloat(opts.edge_follow),
      full_name_length:    parseInt(opts.full_name_length),
      full_name_has_number:opts.full_name_has_number ? 1 : 0,
      is_private:          opts.is_private          ? 1 : 0,
      is_joined_recently:  opts.is_joined_recently  ? 1 : 0,
      has_channel:         opts.has_channel         ? 1 : 0,
      is_business_account: opts.is_business_account ? 1 : 0,
      has_guides:          opts.has_guides           ? 1 : 0,
      has_external_url:    opts.has_external_url     ? 1 : 0,
    } : {}),
  };
}

// ─────────────────────────────────────────────
// GAUSSIAN NAÏVE BAYES PREDICT
// Returns { label, probability, featureContributions }
// ─────────────────────────────────────────────
function predict(features) {
  let logFake = Math.log(MODEL.priors.fake);
  let logReal = Math.log(MODEL.priors.real);
  const contributions = {};
  const eps = 1e-300;

  // Continuous features — skip advanced ones if not provided
  for (const [feat, params] of Object.entries(MODEL.continuous)) {
    if (!(feat in features)) continue; // skip if not provided (advanced panel closed)
    const x = features[feat];
    const pFake = gaussianPDF(x, params.fake[0], params.fake[1]);
    const pReal = gaussianPDF(x, params.real[0], params.real[1]);
    logFake += Math.log(pFake + eps);
    logReal += Math.log(pReal + eps);
    contributions[feat] = Math.log(pFake + eps) - Math.log(pReal + eps);
  }

  // Binary features — skip advanced ones if not provided
  for (const [feat, params] of Object.entries(MODEL.binary)) {
    if (!(feat in features)) continue;
    const x = features[feat];
    const pFake = x === 1 ? params.fake : (1 - params.fake);
    const pReal = x === 1 ? params.real : (1 - params.real);
    logFake += Math.log(pFake + eps);
    logReal += Math.log(pReal + eps);
    contributions[feat] = Math.log(pFake + eps) - Math.log(pReal + eps);
  }

  const maxLog = Math.max(logFake, logReal);
  const expFake = Math.exp(logFake - maxLog);
  const expReal = Math.exp(logReal - maxLog);
  const probFake = expFake / (expFake + expReal);

  return {
    label: probFake >= 0.5 ? 'fake' : 'real',
    probability: probFake,
    contributions,
  };
}

// ─────────────────────────────────────────────
// HISTORY  (localStorage)
// ─────────────────────────────────────────────
function getHistory() {
  try { return JSON.parse(localStorage.getItem('instacheck_history') || '[]'); }
  catch { return []; }
}

function saveHistory(entry) {
  const h = getHistory();
  h.unshift(entry);
  if (h.length > 6) h.pop();
  localStorage.setItem('instacheck_history', JSON.stringify(h));
}

// ─────────────────────────────────────────────
// UI RENDERING
// ─────────────────────────────────────────────

// Confidence ring  (circumference ≈ 345 for r=55)
const CIRC = 2 * Math.PI * 55;

function updateRing(probFake) {
  const pct = Math.round(probFake * 100);
  const fill = document.getElementById('ringFill');
  const pctEl = document.getElementById('ringPct');

  // Color: red for high fake risk, green for low
  const hue = Math.round((1 - probFake) * 120); // 0=red, 120=green
  fill.style.stroke = `hsl(${hue},90%,60%)`;
  const offset = CIRC * (1 - probFake);
  fill.style.strokeDashoffset = offset;
  fill.style.strokeDasharray = CIRC;
  pctEl.textContent = pct + '%';
  pctEl.style.color = `hsl(${hue},90%,60%)`;
}

function renderFeatureBars(contributions, features) {
  const container = document.getElementById('featureBars');
  container.innerHTML = '';

  // Sort by absolute contribution descending
  const sorted = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

  for (const [feat, logOdds] of sorted) {
    const pct = Math.min(100, (Math.abs(logOdds) / maxAbs) * 100);
    const dirClass = logOdds > 0.3 ? 'feat-positive' : logOdds < -0.3 ? 'feat-negative' : 'feat-neutral';
    const direction = logOdds > 0.3 ? '↑ Fake' : logOdds < -0.3 ? '↓ Real' : 'Neutral';

    // format feature value nicely
    const rawVal = features[feat];
    let valStr;
    if (feat === 'edge_followed_by' || feat === 'edge_follow') {
      valStr = rawVal.toFixed(3);
    } else if ([0, 1].includes(rawVal)) {
      valStr = rawVal === 1 ? 'Yes' : 'No';
    } else {
      valStr = rawVal;
    }

    const row = document.createElement('div');
    row.className = `feat-row ${dirClass}`;
    row.innerHTML = `
      <span class="feat-name" title="${FEATURE_LABELS[feat] || feat}">${FEATURE_LABELS[feat] || feat}</span>
      <div class="feat-bar-wrap"><div class="feat-bar-fill" data-pct="${pct}"></div></div>
      <span class="feat-val">${valStr}</span>
    `;
    container.appendChild(row);
  }

  // Animate bars on next frame
  requestAnimationFrame(() => {
    container.querySelectorAll('.feat-bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });
}

function renderResult(username, result, features) {
  const card = document.getElementById('resultCard');
  const badge = document.getElementById('verdictBadge');
  const uname = document.getElementById('verdictUsername');
  const desc  = document.getElementById('verdictDesc');

  card.classList.remove('visible');
  void card.offsetWidth; // reflow for animation restart
  card.classList.add('visible');

  const isFake = result.label === 'fake';
  const pct = Math.round(result.probability * 100);
  const conf = pct >= 80 ? 'High confidence' : pct >= 60 ? 'Moderate confidence' : 'Low confidence';

  badge.className = `verdict-badge ${result.label}`;
  badge.innerHTML = isFake ? '🚨 Fake Account' : '✅ Real Account';
  uname.textContent = '@' + username;

  const descs = {
    fake_high: 'This account shows strong signals of being fake — low follower ratio, suspicious username pattern, and high following ratio matching our trained fake account profiles.',
    fake_low:  'This account leans toward fake based on the provided features. Some signals are inconclusive — consider filling in the advanced options for a more accurate result.',
    real_high: 'This account looks genuine. Its follower ratio, username pattern, and profile attributes match our trained real account profiles.',
    real_low:  'This account leans toward being real, but some signals are ambiguous. Provide more details in Advanced Options to improve accuracy.',
  };

  const key = `${result.label}_${pct >= 65 ? 'high' : 'low'}`;
  desc.textContent = `${conf} (${pct}% fake probability). ${descs[key] || ''}`;

  updateRing(result.probability);
  renderFeatureBars(result.contributions, features);
}

function renderHistory() {
  const h = getHistory();
  const card = document.getElementById('historyCard');
  const list = document.getElementById('historyList');

  if (h.length === 0) { card.classList.remove('visible'); return; }
  card.classList.add('visible');
  list.innerHTML = '';

  h.forEach(entry => {
    const chip = document.createElement('div');
    chip.className = 'hist-chip';
    chip.innerHTML = `<span class="hist-dot ${entry.label}"></span>@${entry.username} <span style="color:var(--text-muted);font-size:0.72rem;">${entry.pct}%</span>`;
    chip.addEventListener('click', () => {
      document.getElementById('usernameInput').value = entry.username;
      runAnalysis();
    });
    list.appendChild(chip);
  });
}

// ─────────────────────────────────────────────
// MAIN ANALYSIS
// ─────────────────────────────────────────────
function runAnalysis() {
  const raw = document.getElementById('usernameInput').value.trim();
  if (!raw) {
    shakElement(document.getElementById('usernameInput'));
    return;
  }

  const username = extractUsername(raw);
  if (!username) { alert('Could not parse a username from that input.'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.textContent = 'Analysing…';
  btn.classList.add('loading');

  setTimeout(() => {
    const opts = {
      edge_followed_by:    parseFloat(document.getElementById('rFollowers').value),
      edge_follow:         parseFloat(document.getElementById('rFollowing').value),
      full_name_length:    parseInt(document.getElementById('rFNLen').value),
      full_name_has_number:document.getElementById('chkFNNum').checked,
      is_private:          document.getElementById('chkPrivate').checked,
      is_joined_recently:  document.getElementById('chkNewJoin').checked,
      has_channel:         document.getElementById('chkChannel').checked,
      is_business_account: document.getElementById('chkBusiness').checked,
      has_guides:          document.getElementById('chkGuides').checked,
      has_external_url:    document.getElementById('chkExtURL').checked,
    };

    const features = buildFeatures(username, opts);
    const result   = predict(features);

    renderResult(username, result, features);
    saveHistory({ username, label: result.label, pct: Math.round(result.probability * 100) });
    renderHistory();

    btn.textContent = 'Analyse →';
    btn.classList.remove('loading');
  }, 600); // slight delay for UX
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────
function shakElement(el) {
  el.style.animation = 'none';
  el.style.borderColor = 'var(--red)';
  el.style.boxShadow = '0 0 0 3px rgba(248,113,113,0.2)';
  setTimeout(() => {
    el.style.borderColor = '';
    el.style.boxShadow = '';
  }, 1200);
}

// ─────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────
document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);

document.getElementById('usernameInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAnalysis();
});

// Advanced toggle — track open state for feature gating
const advToggle = document.getElementById('advToggle');
const advPanel  = document.getElementById('advPanel');
advToggle.addEventListener('click', () => {
  const open = advPanel.classList.toggle('open');
  advToggle.classList.toggle('open', open);
  advancedOpen = open; // update gate flag
});

// Range sliders live update
[
  ['rFollowers', 'rFollowersVal', v => parseFloat(v).toFixed(3)],
  ['rFollowing', 'rFollowingVal', v => parseFloat(v).toFixed(3)],
  ['rFNLen',     'rFNLenVal',     v => v],
].forEach(([id, valId, fmt]) => {
  const slider = document.getElementById(id);
  const display = document.getElementById(valId);
  slider.addEventListener('input', () => { display.textContent = fmt(slider.value); });
});

// Init history on load
renderHistory();
