/**
 * InstaCheck — Live Backend Gaussian Naïve Bayes Classifier
 */

const MODEL = {
  priors: { fake: 0.503, real: 0.497 },
  continuous: {
    username_length:   { fake: [11.8, 15.0], real: [7.8,  9.0] },
    edge_followed_by:  { fake: [0.004, 0.0016], real: [0.28, 0.055] },
    edge_follow:       { fake: [0.47,  0.070],  real: [0.21, 0.035] },
    full_name_length:  { fake: [4.5,   40.0],   real: [10.8, 18.0]  },
  },
  binary: {
    username_has_number:   { fake: 0.75, real: 0.30 },
    full_name_has_number:  { fake: 0.20, real: 0.08 },
    is_private:            { fake: 0.24, real: 0.42 },
    is_joined_recently:    { fake: 0.36, real: 0.08 },
    has_channel:           { fake: 0.02, real: 0.08 },
    is_business_account:   { fake: 0.05, real: 0.22 },
    has_guides:            { fake: 0.01, real: 0.08 },
    has_external_url:      { fake: 0.05, real: 0.26 },
  },
};

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

function gaussianPDF(x, mean, variance) {
  const eps = 1e-9;
  const v = variance + eps;
  return (1 / Math.sqrt(2 * Math.PI * v)) * Math.exp(-((x - mean) ** 2) / (2 * v));
}

function extractUsername(raw) {
  raw = raw.trim();
  const urlMatch = raw.match(/instagram\.com\/([A-Za-z0-9_.]+)/i);
  if (urlMatch) return urlMatch[1];
  return raw.replace(/^@/, '');
}

function predict(features) {
  let logFake = Math.log(MODEL.priors.fake);
  let logReal = Math.log(MODEL.priors.real);
  const contributions = {};
  const eps = 1e-300;

  for (const [feat, params] of Object.entries(MODEL.continuous)) {
    const x = features[feat] || 0;
    const pFake = gaussianPDF(x, params.fake[0], params.fake[1]);
    const pReal = gaussianPDF(x, params.real[0], params.real[1]);
    logFake += Math.log(pFake + eps);
    logReal += Math.log(pReal + eps);
    contributions[feat] = Math.log(pFake + eps) - Math.log(pReal + eps);
  }

  for (const [feat, params] of Object.entries(MODEL.binary)) {
    const x = features[feat] || 0;
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

  return { label: probFake >= 0.5 ? 'fake' : 'real', probability: probFake, contributions };
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
const CIRC = 2 * Math.PI * 55;

function updateRing(probFake) {
  const pct = Math.round(probFake * 100);
  const fill = document.getElementById('ringFill');
  const pctEl = document.getElementById('ringPct');
  const hue = Math.round((1 - probFake) * 120); 
  fill.style.stroke = `hsl(${hue},90%,60%)`;
  fill.style.strokeDashoffset = CIRC * (1 - probFake);
  fill.style.strokeDasharray = CIRC;
  pctEl.textContent = pct + '%';
  pctEl.style.color = `hsl(${hue},90%,60%)`;
}

function renderFeatureBars(contributions, features) {
  const container = document.getElementById('featureBars');
  container.innerHTML = '';
  const sorted = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxAbs = Math.max(...sorted.map(([, v]) => Math.abs(v)), 0.01);

  for (const [feat, logOdds] of sorted) {
    if (feat === '_display') continue;
    
    const pct = Math.min(100, (Math.abs(logOdds) / maxAbs) * 100);
    const dirClass = logOdds > 0.3 ? 'feat-positive' : logOdds < -0.3 ? 'feat-negative' : 'feat-neutral';

    let valStr = features[feat];
    if (feat === 'edge_followed_by' || feat === 'edge_follow') valStr = valStr.toFixed(3);
    else if ([0, 1].includes(valStr)) valStr = valStr === 1 ? 'Yes' : 'No';

    const row = document.createElement('div');
    row.className = `feat-row ${dirClass}`;
    row.innerHTML = `
      <span class="feat-name" title="${FEATURE_LABELS[feat] || feat}">${FEATURE_LABELS[feat] || feat}</span>
      <div class="feat-bar-wrap"><div class="feat-bar-fill" style="width: 0%" data-pct="${pct}"></div></div>
      <span class="feat-val">${valStr}</span>
    `;
    container.appendChild(row);
  }

  requestAnimationFrame(() => {
    container.querySelectorAll('.feat-bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });
}

function formatNumber(num) {
  if (num >= 1000000) return (num/1000000).toFixed(1) + 'M';
  if (num >= 1000) return (num/1000).toFixed(1) + 'K';
  return num;
}

function renderResult(username, result, features) {
  const card = document.getElementById('resultCard');
  const badge = document.getElementById('verdictBadge');
  const uname = document.getElementById('verdictUsername');
  const desc  = document.getElementById('verdictDesc');

  card.classList.remove('visible');
  void card.offsetWidth; 
  card.classList.add('visible');

  // Populate profile header
  const _d = features._display;
  document.getElementById('profAvatar').src = _d.avatar || 'data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=';
  document.getElementById('profName').textContent = _d.name || ('@' + username);
  document.getElementById('profFollowers').textContent = formatNumber(_d.followers);
  document.getElementById('profFollowing').textContent = formatNumber(_d.following);
  document.getElementById('profPosts').textContent = formatNumber(_d.posts);
  document.getElementById('profBio').textContent = _d.bio || 'No biography provided.';

  const isFake = result.label === 'fake';
  const pct = Math.round(result.probability * 100);
  const conf = pct >= 80 ? 'High confidence' : pct >= 60 ? 'Moderate confidence' : 'Low confidence';

  badge.className = `verdict-badge ${result.label}`;
  badge.innerHTML = isFake ? '🚨 Fake Account' : '✅ Real Account';
  uname.textContent = '@' + username;

  desc.textContent = `${conf} (${pct}% fake probability). Based on real-time data fetched from Instagram.`;

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
async function runAnalysis() {
  const raw = document.getElementById('usernameInput').value.trim();
  if (!raw) {
    shakElement(document.getElementById('usernameInput'));
    return;
  }

  const username = extractUsername(raw);
  if (!username) { alert('Could not parse a username from that input.'); return; }

  const btn = document.getElementById('analyzeBtn');
  btn.textContent = 'Fetching…';
  btn.classList.add('loading');

  try {
    const response = await fetch(`http://localhost:3000/api/scrape/${username}`);
    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || 'Failed to fetch profile.');
    }
    
    const features = await response.json();
    const result = predict(features);

    renderResult(username, result, features);
    saveHistory({ username, label: result.label, pct: Math.round(result.probability * 100) });
    renderHistory();
  } catch (err) {
    alert('Error connecting to backend API: ' + err.message);
  } finally {
    btn.textContent = 'Analyse →';
    btn.classList.remove('loading');
  }
}

function shakElement(el) {
  el.style.animation = 'none';
  el.style.borderColor = 'var(--red)';
  el.style.boxShadow = '0 0 0 3px rgba(248,113,113,0.2)';
  setTimeout(() => {
    el.style.borderColor = '';
    el.style.boxShadow = '';
  }, 1200);
}

document.getElementById('analyzeBtn').addEventListener('click', runAnalysis);
document.getElementById('usernameInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAnalysis();
});

renderHistory();
