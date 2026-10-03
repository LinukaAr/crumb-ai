/**
 * Crumb – app.js
 * Vanilla JS frontend. No external requests after Chart.js is vendored.
 */

/* ---- API helpers ---- */
const API = {
  async upload(file) {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }
    return res.json();
  },

  async forecast(item, days) {
    const res = await fetch(`/api/forecast?item=${encodeURIComponent(item)}&days=${days}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Forecast failed');
    return res.json();
  },

  async anomalies(item) {
    const res = await fetch(`/api/anomalies?item=${encodeURIComponent(item)}`);
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Anomaly fetch failed');
    return res.json();
  },

  async chat(message) {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Chat failed');
    return res.json();
  },

  async health() {
    const res = await fetch('/api/health');
    if (!res.ok) return null;
    return res.json();
  },
};

/* ---- State ---- */
let state = {
  items: [],
  selectedItem: null,
  forecastDays: 14,
  chart: null,
};

/* ---- DOM refs ---- */
const $ = id => document.getElementById(id);
const uploadArea   = $('upload-area');
const fileInput    = $('file-input');
const itemSelect   = $('item-select');
const horizonInput = $('horizon-input');
const horizonVal   = $('horizon-val');
const tomorrowBig  = $('tomorrow-big');
const tomorrowRange = $('tomorrow-range');
const tomorrowMethod = $('tomorrow-method');
const chartTitle   = $('chart-title');
const chartMeta    = $('chart-meta');
const anomalyList  = $('anomaly-list');
const chatMessages = $('chat-messages');
const chatInput    = $('chat-input');
const sendBtn      = $('send-btn');
const ollamaStatus = $('ollama-status');
const modelBadge   = $('model-badge');
const methodBadge  = $('method-badge');

/* ---- Upload ---- */
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleUpload(file);
});
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { if (fileInput.files[0]) handleUpload(fileInput.files[0]); });

async function handleUpload(file) {
  uploadArea.querySelector('p').textContent = 'Uploading…';
  try {
    const data = await API.upload(file);
    state.items = data.items || [];
    populateItemSelect(state.items);
    uploadArea.querySelector('p').textContent = `✓ Loaded ${state.items.length} items`;
    if (state.items.length > 0) {
      state.selectedItem = state.items[0];
      itemSelect.value = state.selectedItem;
      await refreshAll();
    }
  } catch (err) {
    uploadArea.querySelector('p').textContent = '✗ ' + err.message;
  }
}

function populateItemSelect(items) {
  itemSelect.innerHTML = items.map(it => `<option value="${esc(it)}">${esc(it)}</option>`).join('');
}

itemSelect.addEventListener('change', async () => {
  state.selectedItem = itemSelect.value;
  await refreshAll();
});

/* ---- Horizon slider ---- */
horizonInput.addEventListener('input', () => {
  state.forecastDays = +horizonInput.value;
  horizonVal.textContent = state.forecastDays + ' days';
});
horizonInput.addEventListener('change', async () => { await refreshAll(); });

/* ---- Chart ---- */
function initChart() {
  const ctx = $('forecast-chart').getContext('2d');
  state.chart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12 } },
        tooltip: { callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${Math.round(ctx.parsed.y)} units`,
        }},
      },
      scales: {
        x: {
          ticks: { maxRotation: 45, font: { size: 11 } },
          grid: { color: '#f0e8df' },
        },
        y: {
          beginAtZero: true,
          grid: { color: '#f0e8df' },
          title: { display: true, text: 'Units sold' },
        },
      },
    },
  });
}

function updateChart(data) {
  const { history, forecast: fc, item } = data;
  const histLabels = history.dates;
  const fcLabels   = fc.dates;
  const allLabels  = [...histLabels, ...fcLabels];

  // History values (undefined for forecast range)
  const histValues = [...history.units, ...Array(fcLabels.length).fill(null)];
  // Forecast median (null for history range)
  const medianValues = [...Array(histLabels.length).fill(null), ...fc.median];
  // Upper band
  const upperValues = [...Array(histLabels.length).fill(null), ...fc.upper];
  // Lower band
  const lowerValues = [...Array(histLabels.length).fill(null), ...fc.lower];

  state.chart.data.labels = allLabels;
  state.chart.data.datasets = [
    {
      label: 'History',
      data: histValues,
      borderColor: '#7a6652',
      backgroundColor: 'transparent',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.3,
      spanGaps: false,
    },
    {
      label: 'Forecast (median)',
      data: medianValues,
      borderColor: '#c97d2e',
      backgroundColor: 'transparent',
      borderWidth: 2.5,
      borderDash: [5, 3],
      pointRadius: 3,
      tension: 0.3,
      spanGaps: false,
    },
    {
      label: 'Upper (p90)',
      data: upperValues,
      borderColor: 'transparent',
      backgroundColor: 'rgba(201,125,46,0.12)',
      fill: '+1',
      pointRadius: 0,
      tension: 0.3,
      spanGaps: false,
    },
    {
      label: 'Lower (p10)',
      data: lowerValues,
      borderColor: 'transparent',
      backgroundColor: 'rgba(201,125,46,0.12)',
      fill: false,
      pointRadius: 0,
      tension: 0.3,
      spanGaps: false,
    },
  ];
  state.chart.update();
}

/* ---- Tomorrow card ---- */
function updateTomorrow(data) {
  if (!data.forecast || !data.forecast.dates.length) return;
  const idx = 0; // first forecast day = tomorrow
  const med  = data.forecast.median[idx];
  const lo   = data.forecast.lower[idx];
  const hi   = data.forecast.upper[idx];
  const item = data.item;
  tomorrowBig.textContent = `${med} units`;
  tomorrowRange.textContent = `Range: ${lo}–${hi} units`;
  tomorrowMethod.textContent = data.method === 'baseline_fallback'
    ? '⚠ Limited data – baseline estimate'
    : `Model: ${data.interval_method === 'tabpfn_quantile' ? 'TabPFN (quantile)' : 'TabPFN + residuals'}`;
  chartTitle.textContent = item;
  if (data.backtest_metrics && data.backtest_metrics.mae_tabpfn) {
    const mae = data.backtest_metrics.mae_tabpfn;
    chartMeta.textContent = `Backtest MAE: ${mae != null ? mae.toFixed(1) : '–'} units | Model: ${data.model}`;
  }
  methodBadge.textContent = data.method === 'baseline_fallback' ? 'Baseline' : 'TabPFN';
  methodBadge.className = `badge ${data.method === 'baseline_fallback' ? 'badge-warn' : 'badge-ok'}`;
}

/* ---- Anomalies ---- */
function renderAnomalies(data) {
  const closures = new Set(data.possible_closure_dates || []);
  const anomalies = data.anomalies || [];

  if (!anomalies.length && !closures.size) {
    anomalyList.innerHTML = '<li style="color:var(--muted);font-size:.85rem;">No anomalies detected.</li>';
    return;
  }

  const closureItems = [...closures].map(d => `
    <li class="anomaly-item anomaly-closure">
      <span class="anomaly-date">${d}</span>
      <span class="anomaly-text">Possible shop closure (all items at 0)</span>
    </li>`);

  const anomalyItems = anomalies.map(a => `
    <li class="anomaly-item anomaly-${a.direction}">
      <span class="anomaly-date">${a.date}</span>
      <span class="anomaly-text">${esc(a.label)}</span>
    </li>`);

  anomalyList.innerHTML = [...closureItems, ...anomalyItems].join('');
}

/* ---- Refresh all panels ---- */
async function refreshAll() {
  if (!state.selectedItem) return;
  try {
    const [fcData, anomData] = await Promise.all([
      API.forecast(state.selectedItem, state.forecastDays),
      API.anomalies(state.selectedItem),
    ]);
    updateChart(fcData);
    updateTomorrow(fcData);
    renderAnomalies(anomData);
  } catch (err) {
    console.error('Refresh error', err);
  }
}

/* ---- Chat ---- */
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChat();
  }
});
sendBtn.addEventListener('click', sendChat);
// Auto-resize textarea
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 140) + 'px';
});

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  chatInput.value = '';
  chatInput.style.height = '';
  sendBtn.disabled = true;

  appendMsg('user', msg);
  const thinkingEl = appendMsg('assistant', '<span class="spinner"></span> Thinking…', null);

  try {
    const data = await API.chat(msg);
    thinkingEl.remove();
    appendMsg('assistant', data.answer, data.tool);
  } catch (err) {
    thinkingEl.remove();
    appendMsg('assistant', '⚠ ' + err.message, null);
  } finally {
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

function appendMsg(role, html, tool) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg-${role}`;
  if (tool) {
    const badge = document.createElement('div');
    badge.className = 'msg-tool-badge';
    badge.textContent = '🔧 ' + tool;
    wrap.appendChild(badge);
  }
  const content = document.createElement('div');
  content.innerHTML = html;
  wrap.appendChild(content);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

/* ---- Health check ---- */
async function pollHealth() {
  const data = await API.health().catch(() => null);
  if (!data) { ollamaStatus.className = 'badge badge-err'; ollamaStatus.textContent = 'Ollama: unreachable'; return; }
  const ok = data.ollama?.ollama_running;
  const modelOk = data.ollama?.model_ready;
  if (!ok) {
    ollamaStatus.className = 'badge badge-err';
    ollamaStatus.textContent = 'Ollama: offline';
  } else if (!modelOk) {
    ollamaStatus.className = 'badge badge-warn';
    ollamaStatus.textContent = `Ollama: model not ready (${data.model})`;
  } else {
    ollamaStatus.className = 'badge badge-ok';
    ollamaStatus.textContent = 'Ollama: ready';
  }
  if (data.model) modelBadge.textContent = data.model;
}

/* ---- Util ---- */
function esc(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

/* ---- Init ---- */
document.addEventListener('DOMContentLoaded', () => {
  initChart();
  horizonVal.textContent = horizonInput.value + ' days';
  pollHealth();
  setInterval(pollHealth, 30000);
  appendMsg('assistant', '👋 Hi! Upload a CSV to get started, then ask me anything — "How many croissants should I bake Saturday?" or "Did anything odd happen last month?"', null);
});
