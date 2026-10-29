/**
 * Web UI of the VBS 2027 interactive video search tool (docs/ARCHITECTURE.md §7).
 *
 * One ES module, no build step, no third-party code. It talks to the FastAPI server through the
 * JSON API of §6. With `?mock=1` in the page URL every /api call is answered by ./mock.js, so the
 * interface can be developed and demonstrated without a server.
 *
 * Sections: helpers → API and interaction log → state → top bar (status, DRES) → query builder →
 * results (virtualised) → video view (players, score curve, keyframes) → actions (submit, basket,
 * VQA) → drawer → keyboard → start-up.
 *
 * Keyboard actions (S, E, B, A) apply to whatever the operator used last: the result card under the
 * pointer or focused, or the current moment of the video view after stepping, seeking or opening.
 *
 * Submissions go through a guard (§7): /api/verify first; "ok" submits at once, "warn"/"fail" shows a
 * banner and submits only on a second press within 5 s. `?print=1` switches on the "presentation"
 * class (larger type, thicker score marks, no history) for screenshots that are printed small.
 */

/** Working name of the tool; used for the page title and the top bar. */
const PRODUCT_NAME = 'SceneChain';

const PAGE_PARAMS = new URLSearchParams(window.location.search);
const MOCK = PAGE_PARAMS.get('mock') === '1';
const PRESENTATION = PAGE_PARAMS.get('print') === '1';
const mock = MOCK ? await import('./mock.js') : null;

const SVG_NS = 'http://www.w3.org/2000/svg';
const MAX_SCENES = 4;
const TOP_K = 300;
const HISTORY_LIMIT = 20;
const HISTORY_KEY = 'vbs.queryHistory.v1';
const RESULTS_STACK_LIMIT = 10;
const QUERY_INFO_LIMIT = 64;
const VQA_MAX_FRAMES = 5;
const LOG_FLUSH_MS = 5000;
const LOG_QUEUE_LIMIT = 5000;
/** §6: at most 1000 events per /api/log call; sendBeacon payloads share a browser quota of ~64 KB. */
const LOG_BATCH_LIMIT = 1000;
const LOG_BEACON_BYTES = 48000;
const DRES_POLL_MS = 10000;
const TOAST_LIMIT = 4;
const MODES = ['frames', 'videos', 'chains'];
const MODE_LABELS = { frames: 'Frames', videos: 'Videos', chains: 'Chains' };
const MODE_ICONS = { frames: 'grid', videos: 'rows', chains: 'chain' };
/** Dash pattern per scene so that curves stay distinguishable in greyscale print. */
const SCENE_DASHES = ['', '6 3', '1.5 2.5', '9 3 2 3'];
/** Result grid geometry in CSS pixels; row heights are derived from these values. */
const GRID = {
  pad: 10, gap: 8, meta: 20, border: 2, minCard: 168, overscanRows: 3,
  chainHead: 28, chainPad: 10, chainGap: 46, minChainThumb: 150, maxChainThumb: 236,
  minVideoThumb: 150, stripGap: 6, scrollbar: 12,
};
const CURVE = { minHeight: 90, left: 40, right: 14, top: 12, bottom: 20, dot: 4, labelSize: 10, swatch: 22 };
if (PRESENTATION) {
  // Must match the `.presentation` overrides in style.css (meta row, group head, chain gap).
  Object.assign(GRID, { meta: 26, minCard: 220, chainHead: 34, chainGap: 58, minChainThumb: 180, maxChainThumb: 300, minVideoThumb: 190 });
  Object.assign(CURVE, { minHeight: 120, left: 54, right: 18, top: 16, bottom: 26, dot: 5.5, labelSize: 13, swatch: 32 });
}
/** Submission guard: time for the confirming second press, and parallel checks for the AVS basket. */
const GUARD_CONFIRM_MS = 5000;
const GUARD_SAME_MOMENT_S = 5;
const VERIFY_CONCURRENCY = 5;
/** The server's check has an 8 s deadline (verify.BUDGET_S); past this the UI stops waiting and warns. */
const VERIFY_TIMEOUT_MS = 10000;
/** A submission just sent is not sent again by another press for the same target within this time. */
const GUARD_REPEAT_BLOCK_MS = 1500;
/** Buttons that submit: they never keep focus after a click, and Space / Enter on them never confirm a warning. */
const SUBMIT_BUTTONS = '#video-submit, #basket-submit, #vqa-submit, .card-action';
const GUARD_VIEW = {
  checking: { label: 'Checking before submitting', icon: 'shield' },
  warn: { label: 'Check warns', icon: 'alert' },
  fail: { label: 'Check failed', icon: 'x' },
};
/** A verify reason that says the video was already judged wrong (used when no `rejected` flag is sent). */
const REJECTED_REASON = /already judged wrong|judged wrong|rejected/i;
const CHAT_KEY = 'vbs.kisc.v1';
const CURVE_STATUS_MS = 2600;
const CARD_ACTIONS = [
  ['submit', 'send', 'Submit this frame (S)'],
  ['similar', 'similar', 'More like this (E)'],
  ['basket', 'basket', 'Add to AVS basket (B)'],
  ['vqa', 'vqa', 'Ask VQA about this frame (A)'],
];
const VERDICT_VIEW = {
  sending: { label: 'Sending', icon: 'clock', toast: 'info' },
  correct: { label: 'Correct', icon: 'check', toast: 'success' },
  wrong: { label: 'Wrong', icon: 'x', toast: 'error' },
  pending: { label: 'Awaiting judgement', icon: 'clock', toast: 'warning' },
  accepted: { label: 'Accepted', icon: 'check', toast: 'info' },
  offline: { label: 'Not sent', icon: 'alert', toast: 'warning' },
  error: { label: 'Failed', icon: 'alert', toast: 'error' },
};
const TOAST_ICONS = { success: 'check', error: 'x', warning: 'alert', info: 'info' };
const PLACEHOLDER_IMAGE = `data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 90"><rect width="160" height="90" fill="#e3e6ea"/>'
  + '<g fill="none" stroke="#9aa2ac" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round">'
  + '<rect x="63" y="29" width="34" height="28" rx="3"/><circle cx="72.5" cy="38" r="3"/><path d="m97 51-9-8-15 14"/></g></svg>',
)}`;

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

const $ = (selector, root = document) => root.querySelector(selector);

/** Create an HTML element; `text` sets textContent, `on<event>` adds a listener, null/false are skipped. */
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'text') node.textContent = value;
    else if (key.startsWith('on') && typeof value === 'function') node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  node.append(...children.flat().filter((child) => child !== null && child !== undefined && child !== false));
  return node;
}

function svgEl(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, String(value));
  }
  node.append(...children.filter((child) => child !== null && child !== undefined));
  return node;
}

function icon(name, extraClass = '') {
  return svgEl('svg', { class: `icon ${extraClass}`.trim(), 'aria-hidden': 'true' }, svgEl('use', { href: `#i-${name}` }));
}

function sceneChip(scene, label = `S${scene + 1}`) {
  return el('span', { class: `chip s${scene + 1}`, text: label });
}

function fmtTime(seconds) {
  const total = Math.max(0, Math.floor(seconds + 1e-6));
  const h = Math.floor(total / 3600);
  const mm = String(Math.floor((total % 3600) / 60)).padStart(2, '0');
  const ss = String(total % 60).padStart(2, '0');
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function fmtTimePrecise(seconds) {
  const safe = Math.max(0, seconds);
  return `${fmtTime(safe)}.${Math.floor(((safe % 1) * 10) + 1e-6) % 10}`;
}

const fmtScore = (score) => (Number.isFinite(score) ? score.toFixed(3) : '–');
const fmtGap = (seconds) => `+${seconds < 10 ? Math.round(seconds * 10) / 10 : Math.round(seconds)} s`;
const fmtClock = (ms) => new Date(ms).toLocaleTimeString('en-GB', { hour12: false });
const fmtCount = (n) => n.toLocaleString('en-US');
const clamp = (value, lo, hi) => Math.min(hi, Math.max(lo, value));
const round2 = (value) => Math.round(value * 100) / 100;

function chunk(items, size) {
  const out = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

function loadStored(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

function store(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage blocked or full: the history is a per-browser convenience, the session keeps working.
  }
}

function isTyping(target) {
  return target instanceof HTMLElement && (target.isContentEditable || target.matches('input, textarea, select'));
}

const round3 = (value) => Math.round(value * 1000) / 1000;

/** Map `items` through the async `fn` with at most `limit` calls in flight; keeps the order. */
async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next;
      next += 1;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

/** Copy text to the clipboard; falls back to a hidden textarea where the async API is blocked. */
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const area = el('textarea', { class: 'copy-buffer', 'aria-hidden': 'true', readonly: true });
    area.value = text;
    document.body.append(area);
    area.select();
    let copied = false;
    try {
      copied = document.execCommand('copy');
    } catch {
      copied = false;
    }
    area.remove();
    return copied;
  }
}

// ---------------------------------------------------------------------------------------------
// API and interaction log
// ---------------------------------------------------------------------------------------------

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

function timeoutError(timeoutMs) {
  const error = new ApiError(`no answer within ${timeoutMs / 1000} s`, 0);
  error.timedOut = true;
  return error;
}

/**
 * GET when `body` is undefined, else POST JSON. Resolves to the parsed payload; throws ApiError.
 * `timeoutMs` (optional) aborts the request and throws an ApiError with `timedOut` set.
 */
async function api(path, body, { timeoutMs = 0 } = {}) {
  const method = body === undefined ? 'GET' : 'POST';
  let status;
  let payload = null;
  if (mock) {
    const call = mock.mockFetch(method, path, body);
    ({ status, body: payload } = await (timeoutMs
      ? Promise.race([call, new Promise((_, reject) => setTimeout(() => reject(timeoutError(timeoutMs)), timeoutMs))])
      : call));
  } else {
    const controller = timeoutMs ? new AbortController() : null;
    const timer = controller ? setTimeout(() => controller.abort(), timeoutMs) : 0;
    try {
      let response;
      try {
        response = await fetch(path, {
          method,
          headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller?.signal,
        });
      } catch (error) {
        if (controller?.signal.aborted) throw timeoutError(timeoutMs);
        throw new ApiError(`server unreachable (${error.message})`, 0);
      }
      status = response.status;
      try {
        payload = await response.json();
      } catch {
        if (controller?.signal.aborted) throw timeoutError(timeoutMs);
        payload = null;
      }
    } finally {
      clearTimeout(timer);
    }
  }
  if (status < 200 || status >= 300) throw new ApiError(payload?.error ?? `HTTP ${status} from ${path}`, status);
  if (payload === null || typeof payload !== 'object') throw new ApiError(`invalid JSON from ${path}`, status);
  return payload;
}

const logQueue = [];
let logFlushing = false;

/** Queue one interaction event in the DRES QueryEvent shape; `value` is sent as a string. */
function logEvent(category, type, value) {
  logQueue.push({ timestamp_ms: Date.now(), category, type, value: typeof value === 'string' ? value : JSON.stringify(value) });
  if (logQueue.length > LOG_QUEUE_LIMIT) logQueue.splice(0, logQueue.length - LOG_QUEUE_LIMIT);
}

/** Send the queue in batches of at most LOG_BATCH_LIMIT events (the server reads no more per call). */
async function flushLog() {
  if (logFlushing || !logQueue.length) return;
  logFlushing = true;
  try {
    while (logQueue.length) {
      const events = logQueue.splice(0, LOG_BATCH_LIMIT);
      let response;
      try {
        response = await api('/api/log', { events });
      } catch (error) {
        logQueue.unshift(...events);
        if (logQueue.length > LOG_QUEUE_LIMIT) logQueue.splice(0, logQueue.length - LOG_QUEUE_LIMIT);
        toast('warning', 'Interaction log not delivered', `${error.message}. Events are kept and retried every 5 s.`, { key: 'log-error' });
        return;
      }
      const accepted = Number(response.accepted);
      if (Number.isFinite(accepted) && accepted < events.length) {
        toast('warning', 'Interaction events rejected', `The server accepted ${accepted} of ${events.length} events.`, { key: 'log-rejected' });
      }
    }
  } finally {
    logFlushing = false;
  }
}

function flushLogOnPageHide() {
  if (!logQueue.length) return;
  if (mock) {
    mock.mockFetch('POST', '/api/log', { events: logQueue.splice(0) });
    return;
  }
  // Beacons of at most LOG_BATCH_LIMIT events and LOG_BEACON_BYTES each; what the browser refuses stays
  // queued (a page restored from the back-forward cache sends it with the next flush).
  const encoder = new TextEncoder();
  while (logQueue.length) {
    let count = 0;
    let bytes = 16;
    while (count < logQueue.length && count < LOG_BATCH_LIMIT) {
      const size = encoder.encode(JSON.stringify(logQueue[count])).length + 1;
      if (count && bytes + size > LOG_BEACON_BYTES) break;
      bytes += size;
      count += 1;
    }
    const events = logQueue.slice(0, count);
    if (!navigator.sendBeacon('/api/log', new Blob([JSON.stringify({ events })], { type: 'application/json' }))) return;
    logQueue.splice(0, count);
  }
}

function toast(kind, title, detail = '', { key = null, duration = 4500 } = {}) {
  const box = $('#toasts');
  if (key) box.querySelector(`[data-key="${CSS.escape(key)}"]`)?.remove();
  const node = el('div', { class: `toast ${kind}`, role: kind === 'error' ? 'alert' : 'status', 'data-key': key },
    el('span', { class: 'toast-icon' }, icon(TOAST_ICONS[kind])),
    el('div', { class: 'toast-text' },
      el('div', { class: 'toast-title', text: title }),
      detail ? el('div', { class: 'toast-detail', text: detail }) : null),
    el('button', { type: 'button', class: 'icon-button small', 'aria-label': 'Dismiss notification', onclick: () => node.remove() }, icon('x')));
  box.append(node);
  while (box.children.length > TOAST_LIMIT) box.firstElementChild.remove();
  setTimeout(() => node.remove(), duration);
}

// ---------------------------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------------------------

/**
 * FrameRef: {id, row, video_id, frame_idx, time, thumb, score, scene (index or null),
 *            entry (index of its video group or chain, or null), queryId}
 * Results:  {kind: "search"|"similar", queryId, mode, scenes?, window?, source?, tookMs,
 *            refs: FrameRef[], entries: (number | {videoId, score, rank, ids})[]}
 */
const state = {
  status: null,
  scenes: [''],
  lastScene: 0,
  window: [0, 30],
  mode: 'frames',
  searchSeq: 0,
  results: null,
  resultsStack: [],
  layoutRows: [],
  refPos: new Map(),
  activeRef: null,
  hoveredRef: null,
  keyTarget: null,
  video: null,
  basket: [],
  vqa: { frames: [], suggestion: null, asking: false },
  submissions: [],
  history: loadHistory(),
  dres: { connected: false, user: null, evaluationId: null, evaluations: [], task: null, pollError: null },
  drawerOpen: false,
  drawerTab: 'basket',
  queryTab: 'query',
  /** Pending submission guard: {kind, videoId, time, key, request, meta, check, phase, result, deadline, timer}. */
  guard: null,
  /** Submissions in flight or just sent: {kind, videoId, time, key, inFlight, until}; see GUARD_REPEAT_BLOCK_MS. */
  sent: [],
  /** Videos known to be judged wrong in the current task (DRES verdicts and /api/verify answers). */
  rejectedVideos: new Set(),
  rejectedTask: null,
  chat: loadChat(),
};

/** Labels, texts and kind ("text" | "similar") of the scenes behind a query id. */
const queryInfo = new Map();
let submissionSeq = 0;
let videoSeq = 0;
let vqaSeq = 0;
let videoScoresSeq = 0;
let curveStatusTimer = 0;

function rememberQuery(queryId, labels, texts, kind = 'text') {
  queryInfo.set(queryId, { labels, texts, kind });
  if (queryInfo.size > QUERY_INFO_LIMIT) queryInfo.delete(queryInfo.keys().next().value);
}

// ---------------------------------------------------------------------------------------------
// Top bar: collection status and DRES
// ---------------------------------------------------------------------------------------------

async function loadStatus() {
  try {
    const status = await api('/api/status');
    state.status = status;
    const { collection } = status;
    $('#collection-text').textContent = `${collection.name} · ${fmtCount(collection.n_keyframes)} keyframes · ${fmtCount(collection.n_videos)} videos`;
    $('#collection-info').title = `Encoder ${status.encoder}, ${collection.dim}-d embeddings`;
    state.dres.connected = Boolean(status.dres.connected);
    state.dres.user = status.dres.user;
    state.dres.evaluationId = status.dres.evaluation_id;
    renderVqa();
    renderDres();
    if (state.dres.connected) await refreshEvaluations();
  } catch (error) {
    $('#collection-text').textContent = 'Server unreachable';
    toast('error', 'Cannot reach the search server', error.message, { duration: 10000 });
  }
}

function taskOf(evaluation) {
  const task = evaluation?.task;
  if (!task) return null;
  // DRES v2 names: ApiClientTaskTemplateInfo.name/taskType, ApiEvaluationState.timeLeft (seconds).
  return {
    /** DRES task run id; a re-run of the same task template gets a new one (mock mode has none). */
    id: task.task_id ?? task.taskId ?? null,
    name: task.name ?? task.taskName ?? null,
    type: task.taskType ?? task.type ?? null,
    remaining: [task.remaining_s, task.timeLeft].find((value) => Number.isFinite(value)) ?? null,
    receivedAt: Date.now(),
  };
}

async function refreshEvaluations() {
  try {
    const { evaluations } = await api('/api/dres/evaluations');
    state.dres.evaluations = evaluations;
    state.dres.pollError = null;
    const known = evaluations.some((evaluation) => evaluation.id === state.dres.evaluationId);
    if (!known) {
      state.dres.evaluationId = null;
      const active = evaluations.find((evaluation) => String(evaluation.status).toUpperCase() === 'ACTIVE');
      if (active) {
        await selectEvaluation(active.id);
        return;
      }
    }
    state.dres.task = taskOf(evaluations.find((evaluation) => evaluation.id === state.dres.evaluationId));
  } catch (error) {
    if (state.dres.pollError !== error.message) toast('warning', 'DRES evaluations unavailable', error.message, { key: 'dres-poll' });
    state.dres.pollError = error.message;
  }
  renderDres();
}

async function selectEvaluation(evaluationId) {
  if (!evaluationId) return;
  try {
    const response = await api('/api/dres/select', { evaluation_id: evaluationId });
    state.dres.evaluationId = response.evaluation_id;
    state.dres.task = taskOf(state.dres.evaluations.find((evaluation) => evaluation.id === response.evaluation_id));
  } catch (error) {
    toast('error', 'Could not select the evaluation', error.message);
  }
  renderDres();
}

function renderDres() {
  const { connected, user, evaluations, evaluationId, pollError } = state.dres;
  const pill = $('#dres-pill');
  pill.classList.toggle('is-connected', connected);
  pill.classList.toggle('is-warning', connected && Boolean(pollError));
  pill.title = pollError ? `DRES: ${pollError}` : 'DRES connection';
  $('#dres-label').textContent = connected ? `DRES · ${user}` : 'DRES not connected';
  pill.setAttribute('aria-label', connected ? `DRES connected as ${user}. Open login dialog.` : 'DRES not connected. Open login dialog.');

  const select = $('#evaluation-select');
  const options = evaluations.map((evaluation) => {
    const status = String(evaluation.status).toLowerCase();
    return el('option', { value: evaluation.id, text: status === 'active' ? evaluation.name : `${evaluation.name} (${status})` });
  });
  if (!evaluationId) options.unshift(el('option', { value: '', text: evaluations.length ? 'Select evaluation' : 'No evaluation' }));
  select.replaceChildren(...options);
  select.value = evaluationId ?? '';
  select.disabled = !connected || !evaluations.length;
  // Wrong verdicts belong to one task: forget them when the evaluation or its task changes.
  const taskKey = `${evaluationId ?? ''}|${state.dres.task?.id ?? state.dres.task?.name ?? ''}`;
  if (state.rejectedTask !== null && state.rejectedTask !== taskKey && state.rejectedVideos.size) {
    state.rejectedVideos.clear();
    refreshCardFlags();
    renderBasket();
  }
  state.rejectedTask = taskKey;
  renderTask();
}

function renderTask() {
  const box = $('#task-info');
  const task = state.dres.connected ? state.dres.task : null;
  box.hidden = !task;
  if (!task) return;
  $('#task-name').textContent = [task.name, task.type].filter(Boolean).join(' · ') || 'Current task';
  if (task.remaining === null) {
    $('#task-remaining').textContent = '';
    box.classList.remove('is-urgent');
    return;
  }
  const left = Math.max(0, task.remaining - (Date.now() - task.receivedAt) / 1000);
  $('#task-remaining').textContent = `${fmtTime(left)} left`;
  box.classList.toggle('is-urgent', left <= 30);
}

function openLoginDialog() {
  $('#login-error').hidden = true;
  if (!$('#login-user').value && state.dres.user) $('#login-user').value = state.dres.user;
  $('#login-dialog').showModal();
  ($('#login-user').value ? $('#login-password') : $('#login-user')).focus();
}

async function submitLogin(event) {
  event.preventDefault();
  const body = { username: $('#login-user').value.trim(), password: $('#login-password').value };
  const url = $('#login-url').value.trim();
  if (url) body.url = url;
  const button = $('#login-submit');
  button.disabled = true;
  try {
    const response = await api('/api/dres/login', body);
    Object.assign(state.dres, { connected: Boolean(response.connected), user: response.user, evaluationId: null, evaluations: [], task: null });
    $('#login-password').value = '';
    $('#login-dialog').close();
    toast('success', 'Connected to DRES', `Signed in as ${response.user}.`);
    renderDres();
    await refreshEvaluations();
  } catch (error) {
    $('#login-error').textContent = error.message;
    $('#login-error').hidden = false;
  } finally {
    button.disabled = false;
  }
}

function bindTopbar() {
  $('#dres-pill').addEventListener('click', openLoginDialog);
  $('#login-form').addEventListener('submit', submitLogin);
  $('#login-cancel').addEventListener('click', () => $('#login-dialog').close());
  $('#evaluation-select').addEventListener('change', (event) => selectEvaluation(event.target.value));
  $('#help-button').addEventListener('click', openHelp);
  $('#help-close').addEventListener('click', () => $('#help-dialog').close());
  for (const dialog of document.querySelectorAll('dialog')) {
    dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
  }
}

function openHelp() {
  $('#help-dialog').showModal();
}

// ---------------------------------------------------------------------------------------------
// Query builder
// ---------------------------------------------------------------------------------------------

const thenText = () => `then, ${state.window[0]}–${state.window[1]} s later`;

function renderScenes() {
  const list = $('#scene-list');
  const count = state.scenes.length;
  list.replaceChildren(...state.scenes.flatMap((text, k) => {
    const textarea = el('textarea', {
      id: `scene-${k}`,
      rows: count === 1 ? 4 : 2,
      spellcheck: 'false',
      placeholder: k === 0
        ? 'Describe what is visible, or paste a whole description and split it'
        : 'What is visible next',
    });
    textarea.value = text;
    textarea.addEventListener('input', () => { state.scenes[k] = textarea.value; });
    textarea.addEventListener('focus', () => { state.lastScene = k; });
    const box = el('div', { class: `scene s${k + 1}` },
      el('div', { class: 'scene-head' },
        sceneChip(k),
        el('label', { for: textarea.id, text: count === 1 ? 'Scene' : `Scene ${k + 1}` }),
        el('span', { class: 'spacer' }),
        el('button', {
          type: 'button', class: 'icon-button small', title: 'Clear text', 'aria-label': `Clear scene ${k + 1}`,
          onclick: () => { state.scenes[k] = ''; textarea.value = ''; textarea.focus(); },
        }, icon('x')),
        k > 0 ? el('button', {
          type: 'button', class: 'icon-button small', title: 'Remove scene', 'aria-label': `Remove scene ${k + 1}`,
          onclick: () => removeScene(k),
        }, icon('minus')) : null),
      textarea);
    if (k === 0) return [box];
    return [el('div', { class: 'then-connector', 'aria-hidden': 'true' }, icon('arrow-down'), el('span', { class: 'then-text', text: thenText() })), box];
  }));
  $('#add-scene').disabled = count >= MAX_SCENES;
  $('#window-field').disabled = count < 2;
  $('#window-field').title = count < 2 ? 'Applies when the query has two or more scenes' : '';
}

function focusScene(k) {
  const textarea = $(`#scene-${clamp(k, 0, state.scenes.length - 1)}`);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function addScene() {
  if (state.scenes.length >= MAX_SCENES) return;
  state.scenes.push('');
  if (state.mode === 'frames') setMode('chains', { rerun: false });
  renderScenes();
  focusScene(state.scenes.length - 1);
}

function removeScene(k) {
  state.scenes.splice(k, 1);
  if (state.scenes.length === 1 && state.mode === 'chains') setMode('frames', { rerun: false });
  renderScenes();
  focusScene(k - 1);
}

async function splitDescription() {
  const text = state.scenes[0].trim();
  if (!text) {
    toast('info', 'Nothing to split', 'Paste the full description into the first scene box.');
    focusScene(0);
    return;
  }
  const button = $('#split-button');
  button.disabled = true;
  try {
    const response = await api('/api/decompose', { text });
    const scenes = response.scenes.map((scene) => scene.trim()).filter(Boolean);
    if (scenes.length > MAX_SCENES) scenes.splice(MAX_SCENES - 1, scenes.length, scenes.slice(MAX_SCENES - 1).join(' '));
    state.scenes = scenes.length ? scenes : [text];
    if (state.scenes.length > 1 && state.mode === 'frames') setMode('chains', { rerun: false });
    renderScenes();
    logEvent('TEXT', 'decompose', { text, scenes: state.scenes, source: response.source, model: response.model });
    toast('info', state.scenes.length > 1 ? `Split into ${state.scenes.length} scenes` : 'No scene boundary found',
      response.source === 'llm' ? `Language model ${response.model}` : 'Rule-based split on sentences and time cue words');
  } catch (error) {
    toast('error', 'Split failed', error.message);
  } finally {
    button.disabled = false;
  }
}

function renderWindow() {
  const [lo, hi] = state.window;
  const minInput = $('#window-min');
  const maxInput = $('#window-max');
  minInput.value = lo;
  maxInput.value = hi;
  const range = Number(maxInput.max);
  $('#window-fill').style.left = `${(lo / range) * 100}%`;
  $('#window-fill').style.right = `${100 - (hi / range) * 100}%`;
  $('#window-readout').textContent = `${lo}–${hi} s`;
  for (const node of document.querySelectorAll('.then-text')) node.textContent = thenText();
}

function onWindowInput(changed) {
  let lo = Number($('#window-min').value);
  let hi = Number($('#window-max').value);
  if (lo >= hi) {
    if (changed === 'min') lo = hi - 1;
    else hi = lo + 1;
  }
  const range = Number($('#window-max').max);
  hi = clamp(hi, 1, range);
  lo = clamp(lo, 0, hi - 1);
  state.window = [lo, hi];
  renderWindow();
}

function renderMode() {
  for (const button of document.querySelectorAll('#mode-toggle [data-mode]')) {
    button.setAttribute('aria-pressed', String(button.dataset.mode === state.mode));
  }
}

function setMode(mode, { rerun }) {
  if (!MODES.includes(mode) || mode === state.mode) return;
  state.mode = mode;
  renderMode();
  logEvent('FILTER', 'resultMode', mode);
  if (rerun && state.results?.kind === 'search') runSearch();
}

function loadHistory() {
  const stored = loadStored(HISTORY_KEY, []);
  if (!Array.isArray(stored)) return [];
  return stored.filter((entry) => entry
    && Array.isArray(entry.scenes) && entry.scenes.length >= 1 && entry.scenes.length <= MAX_SCENES
    && entry.scenes.every((scene) => typeof scene === 'string')
    && Array.isArray(entry.window) && entry.window.length === 2 && entry.window.every(Number.isFinite)
    && MODES.includes(entry.mode) && Number.isFinite(entry.ts)).slice(0, HISTORY_LIMIT);
}

function pushHistory(request) {
  const key = JSON.stringify([request.scenes, request.window, request.mode]);
  state.history = [
    { scenes: request.scenes, window: request.window, mode: request.mode, ts: Date.now() },
    ...state.history.filter((entry) => JSON.stringify([entry.scenes, entry.window, entry.mode]) !== key),
  ].slice(0, HISTORY_LIMIT);
  store(HISTORY_KEY, state.history);
  renderHistory();
}

function renderHistory() {
  $('#history-count').textContent = state.history.length ? String(state.history.length) : '';
  const items = state.history.map((entry) => el('li', {},
    el('button', {
      type: 'button', class: 'history-item', title: 'Restore this query and search again',
      'aria-label': `Restore query: ${entry.scenes.join(', then ')}`,
      onclick: () => restoreHistory(entry),
    },
    el('span', { class: 'history-mode' }, icon(MODE_ICONS[entry.mode])),
    el('span', { class: 'history-scenes' }, entry.scenes.map((scene, k) => el('span', { class: 'history-scene' },
      entry.scenes.length > 1 ? sceneChip(k) : null, el('span', { class: 'history-text', text: scene })))),
    el('span', { class: 'history-time', text: fmtClock(entry.ts).slice(0, 5) }))));
  $('#history-list').replaceChildren(...(items.length ? items : [el('li', { class: 'history-empty', text: 'Queries you run appear here.' })]));
}

function restoreHistory(entry) {
  state.scenes = [...entry.scenes];
  state.window = [...entry.window];
  state.mode = entry.mode;
  renderScenes();
  renderWindow();
  renderMode();
  runSearch();
}

function bindQueryBuilder() {
  $('#add-scene').addEventListener('click', addScene);
  $('#split-button').addEventListener('click', splitDescription);
  $('#search-button').addEventListener('click', runSearch);
  $('#window-min').addEventListener('input', () => onWindowInput('min'));
  $('#window-max').addEventListener('input', () => onWindowInput('max'));
  for (const input of [$('#window-min'), $('#window-max')]) {
    input.addEventListener('change', () => logEvent('FILTER', 'temporalWindow', `${state.window[0]}-${state.window[1]}`));
  }
  for (const button of document.querySelectorAll('#mode-toggle [data-mode]')) {
    button.addEventListener('click', () => setMode(button.dataset.mode, { rerun: true }));
  }
}

/** Replace the scene boxes (at most MAX_SCENES; the rest is merged into the last) and fit the mode. */
function setScenes(scenes, fallback) {
  const clean = scenes.map((scene) => String(scene).trim()).filter(Boolean);
  if (clean.length > MAX_SCENES) clean.splice(MAX_SCENES - 1, clean.length, clean.slice(MAX_SCENES - 1).join(' '));
  state.scenes = clean.length ? clean : [fallback];
  if (state.scenes.length > 1 && state.mode === 'frames') setMode('chains', { rerun: false });
  if (state.scenes.length === 1 && state.mode === 'chains') setMode('frames', { rerun: false });
  renderScenes();
}

function setQueryTab(tab) {
  state.queryTab = tab;
  for (const button of document.querySelectorAll('.query-tabs [role="tab"]')) {
    const selected = button.dataset.qtab === tab;
    button.setAttribute('aria-selected', String(selected));
    $(`#qpane-${button.dataset.qtab}`).hidden = !selected;
  }
}

/** Focus the main text box of the visible query tab (key `/`). */
function focusQuery() {
  if (state.queryTab === 'chat') $('#chat-input').focus();
  else focusScene(state.lastScene);
}

// ---------------------------------------------------------------------------------------------
// KIS-C conversation
// ---------------------------------------------------------------------------------------------

const ROLE_VIEW = {
  team: { chip: 'Q', label: 'Team question', next: 'team question' },
  oracle: { chip: 'A', label: 'Oracle answer', next: 'oracle answer' },
};

function loadChat() {
  const empty = { initial: '', turns: [], draft: '', result: null, updating: false };
  const stored = loadStored(CHAT_KEY, null);
  if (!stored || typeof stored !== 'object') return empty;
  const turns = Array.isArray(stored.turns)
    ? stored.turns.filter((turn) => turn && ROLE_VIEW[turn.role] && typeof turn.text === 'string').map(({ role, text }) => ({ role, text }))
    : [];
  const saved = stored.result;
  const result = saved && Array.isArray(saved.scenes) && typeof saved.next_question === 'string'
    ? { scenes: saved.scenes.filter((scene) => typeof scene === 'string'), next_question: saved.next_question, model: typeof saved.model === 'string' ? saved.model : null, applied: saved.applied !== false }
    : null;
  return {
    ...empty,
    initial: typeof stored.initial === 'string' ? stored.initial : '',
    draft: typeof stored.draft === 'string' ? stored.draft : '',
    turns,
    result,
  };
}

function saveChat() {
  const { initial, turns, draft, result } = state.chat;
  store(CHAT_KEY, { initial, turns, draft, result });
}

/** Ctrl+Enter alternates: after a team question comes the oracle's answer, otherwise a question. */
function nextChatRole() {
  return state.chat.turns.at(-1)?.role === 'team' ? 'oracle' : 'team';
}

function renderChat({ turns: withTurns = false } = {}) {
  const { turns, result, updating } = state.chat;
  if (withTurns) renderChatTurns();
  const questions = turns.filter((turn) => turn.role === 'team').length;
  $('#chat-count').textContent = String(turns.length);
  $('#chat-turns-info').textContent = turns.length ? `${questions} Q · ${turns.length - questions} A` : '';
  $('#chat-next-role').textContent = ROLE_VIEW[nextChatRole()].next;
  $('#chat-update').disabled = updating;
  $('#chat-update-label').textContent = updating ? 'Updating scenes…' : 'Update scenes';
  $('#chat-result').hidden = !result;
  if (!result) return;
  $('#chat-model').textContent = result.model ?? 'rule-based';
  $('#chat-apply').hidden = result.applied !== false || !result.scenes.length;
  $('#chat-scenes').replaceChildren(...result.scenes.map((text, k) => el('li', {},
    el('button', {
      type: 'button', class: 'chat-scene', title: 'Edit this scene in the Query tab',
      onclick: () => { setQueryTab('query'); focusScene(k); },
    }, sceneChip(k), el('span', { class: 'chat-scene-text', text })))));
  $('#chat-next-question').textContent = result.next_question || 'No question suggested.';
  $('#chat-copy').disabled = !result.next_question;
  $('#chat-use').disabled = !result.next_question;
}

function renderChatTurns() {
  const { turns } = state.chat;
  $('#chat-turns').replaceChildren(...(turns.length ? turns.map((turn, i) => {
    const view = ROLE_VIEW[turn.role];
    const area = el('textarea', { class: 'turn-text', rows: 1, spellcheck: 'false', 'aria-label': `${view.label} ${i + 1}` });
    area.value = turn.text;
    area.addEventListener('input', () => {
      turn.text = area.value;
      saveChat();
    });
    return el('li', { class: `turn ${turn.role}` },
      el('button', {
        type: 'button', class: `role-chip ${turn.role}`, title: `${view.label}. Click to switch the role.`,
        'aria-label': `${view.label}; switch role`,
        onclick: () => {
          turn.role = turn.role === 'team' ? 'oracle' : 'team';
          saveChat();
          renderChat({ turns: true });
        },
      }, view.chip),
      area,
      el('button', {
        type: 'button', class: 'icon-button small', title: 'Remove this turn', 'aria-label': `Remove turn ${i + 1}`,
        onclick: () => {
          state.chat.turns = state.chat.turns.filter((other) => other !== turn);
          saveChat();
          renderChat({ turns: true });
        },
      }, icon('x')));
  }) : [el('li', { class: 'chat-empty', text: 'No turns yet. Type a question for the oracle below, then its answer.' })]));
}

function addChatTurn(role) {
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text) {
    input.focus();
    return;
  }
  state.chat.turns.push({ role, text });
  state.chat.draft = '';
  input.value = '';
  logEvent('TEXT', 'chatTurn', { role, text });
  saveChat();
  renderChat({ turns: true });
  const scroller = $('#chat-scroll');
  scroller.scrollTop = scroller.scrollHeight;
  input.focus();
  input.scrollIntoView({ block: 'nearest' });
}

/** POST /api/chat, put the scenes into the scene boxes, search again and show the next question. */
async function updateFromChat() {
  const chat = state.chat;
  if (chat.updating) return;
  const initial = chat.initial.trim();
  const turns = chat.turns.map(({ role, text }) => ({ role, text: text.trim() })).filter((turn) => turn.text);
  const scenes = state.scenes.map((scene) => scene.trim()).filter(Boolean);
  if (!initial && !turns.length && !scenes.length) {
    toast('info', 'Nothing to update from', 'Type the initial description of the task first.');
    $('#chat-initial').focus();
    return;
  }
  const results = state.results;
  const queryId = results?.kind === 'search' ? results.queryId : null;
  // The reply can take many seconds; if the operator searches or edits the scenes meanwhile, it must
  // not overwrite that work.
  const searchSeq = state.searchSeq;
  const scenesBefore = JSON.stringify(state.scenes);
  chat.updating = true;
  renderChat();
  try {
    const request = { initial, turns, scenes, query_id: queryId };
    let response = await api('/api/chat', request);
    if (response.query_known === false && queryId) {
      // The server evicted the query, so the model saw no candidate videos: renew it and ask again.
      const fresh = await renewQueryId(results);
      if (fresh) response = await api('/api/chat', { ...request, query_id: fresh });
    }
    const replyScenes = (Array.isArray(response.scenes) ? response.scenes : []).map((scene) => String(scene).trim()).filter(Boolean);
    const nextQuestion = String(response.next_question ?? '').trim();
    const stale = state.searchSeq !== searchSeq || state.results !== results || JSON.stringify(state.scenes) !== scenesBefore;
    if (stale) {
      chat.result = { scenes: replyScenes, next_question: nextQuestion, model: response.model ?? null, applied: false };
      saveChat();
      logEvent('TEXT', 'chatUpdate', { initial, turns, scenes: replyScenes, next_question: nextQuestion, model: chat.result.model, applied: false });
      toast('info', 'Scenes from the dialogue not applied', 'The query changed while waiting for the reply. Click "Use these scenes" in the KIS-C tab to apply them.', { key: 'chat-stale', duration: 6000 });
      return;
    }
    setScenes(replyScenes, scenes.length ? scenes.join(' ') : initial || turns.at(-1).text);
    chat.result = { scenes: [...state.scenes], next_question: nextQuestion, model: response.model ?? null, applied: true };
    saveChat();
    logEvent('TEXT', 'chatUpdate', { initial, turns, scenes: state.scenes, next_question: chat.result.next_question, model: chat.result.model });
    runSearch();
  } catch (error) {
    toast('error', 'Could not update the scenes', error.message);
  } finally {
    chat.updating = false;
    renderChat();
  }
}

/** Put the scenes of a reply that arrived after the query had changed into the scene boxes and search. */
function applyChatScenes() {
  const result = state.chat.result;
  if (!result?.scenes.length) return;
  setScenes(result.scenes, result.scenes.join(' '));
  result.scenes = [...state.scenes];
  result.applied = true;
  saveChat();
  renderChat();
  logEvent('TEXT', 'chatApply', { scenes: state.scenes });
  runSearch();
}

async function copyNextQuestion() {
  const text = state.chat.result?.next_question;
  if (!text) return;
  const copied = await copyText(text);
  logEvent('TEXT', 'chatCopyQuestion', text);
  toast(copied ? 'success' : 'warning', copied ? 'Question copied' : 'The browser blocked copying', copied ? text : 'Select the question and copy it by hand.', { key: 'chat-copy', duration: 2500 });
}

function useNextQuestion() {
  const text = state.chat.result?.next_question;
  if (!text) return;
  const input = $('#chat-input');
  input.value = text;
  state.chat.draft = text;
  saveChat();
  input.focus();
  input.setSelectionRange(text.length, text.length);
}

function clearChat() {
  logEvent('TEXT', 'chatClear', { turns: state.chat.turns.length });
  state.chat = { initial: '', turns: [], draft: '', result: null, updating: false };
  $('#chat-initial').value = '';
  $('#chat-input').value = '';
  saveChat();
  renderChat({ turns: true });
  $('#chat-initial').focus();
}

function bindChat() {
  for (const button of document.querySelectorAll('.query-tabs [role="tab"]')) {
    button.addEventListener('click', () => setQueryTab(button.dataset.qtab));
  }
  $('#chat-initial').value = state.chat.initial;
  $('#chat-input').value = state.chat.draft;
  $('#chat-initial').addEventListener('input', (event) => {
    state.chat.initial = event.target.value;
    saveChat();
  });
  $('#chat-input').addEventListener('input', (event) => {
    state.chat.draft = event.target.value;
    saveChat();
  });
  $('#chat-add-team').addEventListener('click', () => addChatTurn('team'));
  $('#chat-add-oracle').addEventListener('click', () => addChatTurn('oracle'));
  $('#chat-update').addEventListener('click', updateFromChat);
  $('#chat-copy').addEventListener('click', copyNextQuestion);
  $('#chat-apply').addEventListener('click', applyChatScenes);
  $('#chat-use').addEventListener('click', useNextQuestion);
  $('#chat-clear').addEventListener('click', clearChat);
  // Ctrl+Enter: in the input box it adds the turn; elsewhere in the tab it updates the scenes.
  $('#qpane-chat').addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' || !(event.ctrlKey || event.metaKey)) return;
    event.preventDefault();
    event.stopPropagation();
    if (event.target.id === 'chat-input') addChatTurn(nextChatRole());
    else updateFromChat();
  });
}

// ---------------------------------------------------------------------------------------------
// Search and results
// ---------------------------------------------------------------------------------------------

function setResultsLoading(loading) {
  $('#results-progress').hidden = !loading;
  $('#results-scroller').setAttribute('aria-busy', String(loading));
}

async function runSearch() {
  const scenes = state.scenes.map((scene) => scene.trim()).filter(Boolean);
  if (!scenes.length) {
    toast('info', 'Nothing to search', 'Describe at least one scene.');
    focusScene(0);
    return;
  }
  const request = { scenes, window: [...state.window], mode: state.mode, top_k: TOP_K };
  // Leave the text box so that single-key shortcuts act on the results straight away (/ returns).
  if (document.activeElement?.closest('.query-panel') && isTyping(document.activeElement)) document.activeElement.blur();
  const seq = ++state.searchSeq;
  setResultsLoading(true);
  logEvent('TEXT', scenes.length > 1 ? 'temporalQuery' : 'textQuery', request);
  try {
    const response = await api('/api/search', request);
    if (seq !== state.searchSeq) return;
    pushHistory(request);
    rememberQuery(response.query_id, scenes.map((_, k) => `S${k + 1}`), scenes);
    showResults(prepareResults({
      kind: 'search', queryId: response.query_id, mode: response.mode, scenes, window: request.window, tookMs: response.took_ms, timings: response.timings ?? null,
    }, response[response.mode] ?? []));
  } catch (error) {
    if (seq === state.searchSeq) toast('error', 'Search failed', error.message);
  } finally {
    if (seq === state.searchSeq) setResultsLoading(false);
  }
}

async function runSimilar(ref) {
  const request = { video_id: ref.video_id, frame_idx: ref.frame_idx, top_k: TOP_K };
  const seq = ++state.searchSeq;
  setResultsLoading(true);
  logEvent('IMAGE', 'similarFrames', request);
  try {
    const response = await api('/api/similar', request);
    if (seq !== state.searchSeq) return;
    rememberQuery(response.query_id, ['Similarity'], [`Similarity to ${ref.video_id} at ${fmtTime(ref.time)}`], 'similar');
    showResults(prepareResults({
      kind: 'similar', queryId: response.query_id, mode: 'frames', source: ref, tookMs: response.took_ms,
    }, response.frames));
  } catch (error) {
    if (seq === state.searchSeq) toast('error', 'More-like-this failed', error.message);
  } finally {
    if (seq === state.searchSeq) setResultsLoading(false);
  }
}

/** Flatten frames / video groups / chains into FrameRefs plus mode-specific entries. */
function prepareResults(base, items) {
  const refs = [];
  const addRef = (frame, videoId, scene, entry) => {
    const id = refs.length;
    refs.push({
      id, row: frame.row, video_id: videoId, frame_idx: frame.frame_idx, time: frame.time,
      thumb: frame.thumb, score: frame.score, scene, entry, queryId: base.queryId,
    });
    return id;
  };
  let entries;
  if (base.mode === 'frames') {
    entries = items.map((frame) => addRef(frame, frame.video_id, null, null));
  } else if (base.mode === 'videos') {
    entries = items.map((group, i) => ({
      videoId: group.video_id, score: group.score, rank: i + 1,
      ids: group.hits.map((hit) => addRef(hit, group.video_id, null, i)),
    }));
  } else {
    entries = items.map((chain, i) => ({
      videoId: chain.video_id, score: chain.score, rank: i + 1,
      ids: chain.frames.map((frame, k) => addRef(frame, chain.video_id, k, i)),
    }));
  }
  return { ...base, refs, entries };
}

function showResults(results) {
  if (state.results) {
    state.resultsStack.push(state.results);
    if (state.resultsStack.length > RESULTS_STACK_LIMIT) state.resultsStack.shift();
  }
  applyResults(results);
  logEvent('OTHER', 'resultList', {
    query_id: results.queryId,
    mode: results.mode,
    count: results.entries.length,
    top: results.refs.slice(0, 20).map((ref) => `${ref.video_id}@${ref.frame_idx}`),
  });
}

function applyResults(results) {
  state.results = results;
  state.activeRef = null;
  state.hoveredRef = null;
  if (state.keyTarget === 'card') setKeyTarget(state.video ? 'video' : null);
  $('#results-scroller').scrollTop = 0;
  renderResults();
  syncVideoWithResults();
}

/**
 * Re-run the request behind a result list whose query id the server no longer holds (§6 keeps the
 * last 32), so score curves and chat candidates work again. Resolves to the new id, or null.
 */
function renewQueryId(results) {
  if (!results) return Promise.resolve(null);
  if (!results.renewing) {
    results.renewing = (async () => {
      const old = results.queryId;
      try {
        let response;
        if (results.kind === 'search') {
          response = await api('/api/search', { scenes: results.scenes, window: results.window, mode: results.mode, top_k: TOP_K });
        } else if (results.kind === 'similar' && results.source) {
          response = await api('/api/similar', { video_id: results.source.video_id, frame_idx: results.source.frame_idx, top_k: TOP_K });
        } else {
          return null;
        }
        const info = queryInfo.get(old);
        if (info) rememberQuery(response.query_id, info.labels, info.texts, info.kind);
        results.queryId = response.query_id;
        for (const ref of results.refs) if (ref.queryId === old) ref.queryId = response.query_id;
        logEvent('OTHER', 'queryRenewed', { old_query_id: old, query_id: response.query_id });
        return response.query_id;
      } catch (error) {
        toast('warning', 'The query expired on the server', `Search again for score curves (${error.message}).`, { key: 'query-expired' });
        return null;
      } finally {
        results.renewing = null;
      }
    })();
  }
  return results.renewing;
}

function goBack() {
  const previous = state.resultsStack.pop();
  if (!previous) return;
  logEvent('BROWSING', 'resultsBack', { query_id: previous.queryId });
  applyResults(previous);
}

/** Minimal list virtualisation: fixed-height rows, only rows near the viewport are in the DOM. */
class VirtualRows {
  constructor(scroller, content) {
    this.scroller = scroller;
    this.content = content;
    this.count = 0;
    this.rowHeight = 1;
    this.gap = 0;
    this.offset = 0;
    this.renderRow = null;
    this.rendered = new Map();
    this.frame = 0;
    scroller.addEventListener('scroll', () => this.schedule(), { passive: true });
  }

  reset({ count, rowHeight, gap, offset, renderRow }) {
    Object.assign(this, { count, rowHeight, gap, offset, renderRow });
    this.rendered.clear();
    this.content.replaceChildren();
    this.content.style.height = `${count ? 2 * offset + count * rowHeight - gap : 0}px`;
    this.update();
  }

  clear() {
    this.reset({ count: 0, rowHeight: 1, gap: 0, offset: 0, renderRow: null });
  }

  schedule() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.update();
    });
  }

  update() {
    if (!this.count) return;
    const top = this.scroller.scrollTop - this.content.offsetTop - this.offset;
    const first = Math.max(0, Math.floor(top / this.rowHeight) - GRID.overscanRows);
    const last = Math.min(this.count - 1, Math.floor((top + this.scroller.clientHeight) / this.rowHeight) + GRID.overscanRows);
    for (const [i, node] of this.rendered) {
      if (i < first || i > last) {
        node.remove();
        this.rendered.delete(i);
      }
    }
    for (let i = first; i <= last; i += 1) {
      if (this.rendered.has(i)) continue;
      const node = this.renderRow(i);
      node.classList.add('vrow');
      node.style.top = `${this.offset + i * this.rowHeight}px`;
      node.style.height = `${this.rowHeight - this.gap}px`;
      this.content.append(node);
      this.rendered.set(i, node);
    }
  }

  reveal(i) {
    const top = this.content.offsetTop + this.offset + i * this.rowHeight;
    const bottom = top + this.rowHeight - this.gap;
    if (top < this.scroller.scrollTop) this.scroller.scrollTop = top - this.offset;
    else if (bottom > this.scroller.scrollTop + this.scroller.clientHeight) this.scroller.scrollTop = bottom - this.scroller.clientHeight + this.offset;
    this.update();
  }
}

const virtualRows = new VirtualRows($('#results-scroller'), $('#results-content'));

function renderResults() {
  renderResultsHeader();
  const results = state.results;
  $('#results-empty').hidden = Boolean(results);
  if (!results) {
    virtualRows.clear();
    return;
  }
  if (!results.entries.length) {
    virtualRows.clear();
    $('#results-content').style.height = 'auto';
    $('#results-content').append(el('div', { class: 'empty-state inline' },
      el('p', { class: 'empty-title', text: 'No results' }),
      el('p', { text: results.mode === 'chains'
        ? 'No video contains the scenes in this order within the time window. Widen the window or rephrase a scene.'
        : 'Try a broader description.' })));
    return;
  }
  layoutResults();
}

function renderResultsHeader() {
  const results = state.results;
  $('#results-back').hidden = !state.resultsStack.length;
  const summary = $('#results-summary');
  if (!results) {
    $('#results-title').textContent = 'Results';
    summary.replaceChildren();
    $('#results-stats').textContent = '';
    return;
  }
  if (results.kind === 'similar') {
    $('#results-title').textContent = 'More like this';
    summary.replaceChildren(
      el('img', { class: 'summary-thumb', src: results.source.thumb, alt: '' }),
      el('span', { class: 'vid', text: results.source.video_id }),
      el('span', { class: 'muted', text: `at ${fmtTime(results.source.time)}` }));
  } else {
    $('#results-title').textContent = MODE_LABELS[results.mode];
    summary.replaceChildren(...results.scenes.flatMap((text, k) => [
      k ? icon('arrow-right', 'summary-arrow') : null,
      el('span', { class: 'summary-scene', title: text }, sceneChip(k), el('span', { class: 'summary-text', text })),
    ]).filter(Boolean));
  }
  const noun = { frames: 'frames', videos: 'videos', chains: 'chains' }[results.mode];
  const windowText = results.mode === 'chains' && results.scenes.length > 1 ? ` · gap ${results.window[0]}–${results.window[1]} s` : '';
  $('#results-stats').textContent = `${fmtCount(results.entries.length)} ${noun}${windowText} · ${results.tookMs} ms`;
  const t = results.timings;
  $('#results-stats').title = t ? `Server ${results.tookMs} ms: text encoding ${t.encode_ms} ms, similarity scan ${t.scan_ms} ms, ranking ${t.rank_ms} ms${t.cached_scenes ? ` · ${t.cached_scenes} cached ${t.cached_scenes === 1 ? 'scene' : 'scenes'}` : ''}` : '';
}

function layoutResults() {
  const results = state.results;
  const content = $('#results-content');
  const width = content.clientWidth;
  if (!results || !results.entries.length || !width) return;
  const { pad, gap, meta, border } = GRID;
  const inner = width - 2 * pad;
  let rows;
  let rowHeight;
  let renderRow;
  if (results.mode === 'frames') {
    const perRow = Math.max(1, Math.floor((inner + gap) / (GRID.minCard + gap)));
    const cardWidth = Math.floor((inner - (perRow - 1) * gap) / perRow);
    const thumbHeight = Math.round((cardWidth * 9) / 16);
    rows = chunk(results.entries, perRow);
    rowHeight = thumbHeight + meta + border + gap;
    renderRow = (i) => el('div', { class: 'frames-row' },
      rows[i].map((id) => frameCard(results.refs[id], { width: cardWidth, thumbHeight, showVideo: true })));
    state.layoutRows = rows;
  } else if (results.mode === 'videos') {
    const stripWidth = inner - border - 2 * GRID.chainPad;
    const visible = Math.max(1, Math.floor((stripWidth + GRID.stripGap) / (GRID.minVideoThumb + GRID.stripGap)));
    const thumbWidth = Math.floor((stripWidth - (visible - 1) * GRID.stripGap) / visible);
    const thumbHeight = Math.round((thumbWidth * 9) / 16);
    rows = results.entries.map((entry) => [entry]);
    rowHeight = border + GRID.chainHead + thumbHeight + meta + border + GRID.scrollbar + gap;
    renderRow = (i) => el('div', { class: 'groups-row' }, videoGroup(results, results.entries[i], { thumbWidth, thumbHeight }));
    state.layoutRows = results.entries.map((entry) => entry.ids);
  } else {
    const n = results.entries[0].ids.length;
    const chainMin = n * GRID.minChainThumb + (n - 1) * GRID.chainGap + 2 * GRID.chainPad + border;
    const perRow = Math.max(1, Math.floor((inner + gap) / (chainMin + gap)));
    const chainWidth = Math.floor((inner - (perRow - 1) * gap) / perRow);
    const thumbWidth = Math.min(GRID.maxChainThumb, Math.floor((chainWidth - border - 2 * GRID.chainPad - (n - 1) * GRID.chainGap) / n));
    const thumbHeight = Math.round((thumbWidth * 9) / 16);
    rows = chunk(results.entries, perRow);
    rowHeight = border + GRID.chainHead + thumbHeight + meta + border + GRID.chainPad + gap;
    renderRow = (i) => el('div', { class: 'chains-row' },
      rows[i].map((entry) => chainBox(results, entry, { chainWidth, thumbWidth, thumbHeight })));
    state.layoutRows = rows.map((row) => row.flatMap((entry) => entry.ids));
  }
  state.refPos = new Map();
  state.layoutRows.forEach((ids, r) => ids.forEach((id, c) => state.refPos.set(id, [r, c])));
  virtualRows.reset({ count: rows.length, rowHeight, gap, offset: pad, renderRow });
}

function basketKey(item) {
  return `${item.video_id}#${item.frame_idx}`;
}

function inBasket(ref) {
  const key = basketKey(ref);
  return state.basket.some((item) => basketKey(item) === key);
}

function frameCard(ref, { width, thumbHeight, showVideo }) {
  const active = state.keyTarget === 'card' && state.activeRef === ref.id;
  const rejected = state.rejectedVideos.has(ref.video_id);
  const classes = ['card', inBasket(ref) ? 'in-basket' : '', active ? 'is-active' : '', rejected ? 'is-rejected' : ''].filter(Boolean).join(' ');
  const sceneText = ref.scene === null ? '' : `, scene ${ref.scene + 1}`;
  return el('div', {
    class: classes, role: 'button', tabindex: '0', 'data-ref': ref.id, style: `width:${width}px`,
    'aria-label': `${ref.video_id} at ${fmtTime(ref.time)}${sceneText}, score ${fmtScore(ref.score)}`,
  },
  el('div', { class: 'thumb', style: `height:${thumbHeight}px` },
    el('img', { src: ref.thumb, alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' }),
    el('div', { class: 'flags' },
      ref.scene === null ? null : sceneChip(ref.scene),
      el('span', { class: 'basket-flag', title: 'In the AVS basket' }, icon('basket')),
      el('span', { class: 'rejected-flag', title: 'A submission in this video was judged wrong in the current task' }, icon('x'))),
    el('span', { class: 'badge', text: fmtScore(ref.score) }),
    el('div', { class: 'card-actions' }, CARD_ACTIONS.map(([action, iconName, label]) => el('button', {
      type: 'button', class: 'card-action', tabindex: '-1', 'data-action': action, title: label, 'aria-label': label,
    }, icon(iconName))))),
  el('div', { class: 'meta' },
    el('span', { class: showVideo ? 'vid' : 'time', text: showVideo ? ref.video_id : fmtTime(ref.time) }),
    el('span', { class: showVideo ? 'time' : 'frame-no', text: showVideo ? fmtTime(ref.time) : `frame ${ref.frame_idx}` })));
}

function groupHead(entry, detail, scoreLabel) {
  return el('div', { class: 'group-head' },
    el('span', { class: 'rank', text: String(entry.rank) }),
    el('span', { class: 'vid', text: entry.videoId }),
    el('span', { class: 'muted', text: detail }),
    el('span', { class: 'spacer' }),
    el('span', { class: 'score-label', text: scoreLabel }),
    el('span', { class: 'badge static', text: fmtScore(entry.score) }));
}

function chainBox(results, entry, { chainWidth, thumbWidth, thumbHeight }) {
  const refs = entry.ids.map((id) => results.refs[id]);
  const frames = refs.flatMap((ref, k) => [
    k ? el('div', { class: 'chain-gap', style: `height:${thumbHeight}px`, title: `${fmtGap(ref.time - refs[k - 1].time)} after scene ${k}` },
      icon('arrow-right'), el('span', { text: fmtGap(ref.time - refs[k - 1].time) })) : null,
    frameCard(ref, { width: thumbWidth, thumbHeight, showVideo: false }),
  ]).filter(Boolean);
  const span = refs.length > 1 ? `${fmtTime(refs[0].time)}–${fmtTime(refs.at(-1).time)}` : fmtTime(refs[0].time);
  return el('div', {
    class: 'group chain', style: `width:${chainWidth}px`, role: 'group',
    'aria-label': `Chain ${entry.rank}: ${entry.videoId} ${span}, chain score ${fmtScore(entry.score)}`,
  }, groupHead(entry, span, refs.length > 1 ? 'chain' : 'score'), el('div', { class: 'chain-frames' }, frames));
}

function videoGroup(results, entry, { thumbWidth, thumbHeight }) {
  const refs = entry.ids.map((id) => results.refs[id]);
  return el('div', { class: 'group video-group', role: 'group', 'aria-label': `Video ${entry.rank}: ${entry.videoId}, best score ${fmtScore(entry.score)}` },
    groupHead(entry, `${refs.length} ${refs.length === 1 ? 'match' : 'matches'}`, 'best'),
    el('div', { class: 'video-strip' }, refs.map((ref) => frameCard(ref, { width: thumbWidth, thumbHeight, showVideo: false }))));
}

function refOfCard(card) {
  return state.results?.refs[Number(card.dataset.ref)] ?? null;
}

function setKeyTarget(target) {
  state.keyTarget = target;
  $('#video-panel').classList.toggle('is-target', target === 'video' && Boolean(state.video));
  for (const card of document.querySelectorAll('#results-content .card.is-active')) card.classList.remove('is-active');
  if (target === 'card' && state.activeRef !== null) {
    $(`#results-content .card[data-ref="${state.activeRef}"]`)?.classList.add('is-active');
  }
}

function setActiveRef(id) {
  state.activeRef = id;
  setKeyTarget('card');
}

function moveCardFocus(card, key) {
  const position = state.refPos.get(Number(card.dataset.ref));
  if (!position) return;
  const rows = state.layoutRows;
  let [r, c] = position;
  if (key === 'ArrowLeft') {
    if (c > 0) c -= 1;
    else if (r > 0) { r -= 1; c = rows[r].length - 1; }
  } else if (key === 'ArrowRight') {
    if (c < rows[r].length - 1) c += 1;
    else if (r < rows.length - 1) { r += 1; c = 0; }
  } else if (key === 'ArrowUp' && r > 0) {
    r -= 1;
    c = Math.min(c, rows[r].length - 1);
  } else if (key === 'ArrowDown' && r < rows.length - 1) {
    r += 1;
    c = Math.min(c, rows[r].length - 1);
  }
  virtualRows.reveal(r);
  $(`#results-content .card[data-ref="${rows[r][c]}"]`)?.focus();
}

function refreshCardFlags() {
  for (const card of document.querySelectorAll('#results-content .card')) {
    const ref = refOfCard(card);
    card.classList.toggle('in-basket', Boolean(ref && inBasket(ref)));
    card.classList.toggle('is-rejected', Boolean(ref && state.rejectedVideos.has(ref.video_id)));
  }
}

function performAction(action, ref) {
  if (action === 'open') openVideo(ref);
  else if (action === 'submit') submitKis(ref);
  else if (action === 'similar') runSimilar(ref);
  else if (action === 'basket') addToBasket(ref);
  else if (action === 'vqa') addToVqa(ref);
  else throw new Error(`unknown card action "${action}"`);
}

function bindResults() {
  const content = $('#results-content');
  content.addEventListener('click', (event) => {
    const card = event.target.closest('.card');
    const ref = card && refOfCard(card);
    if (!ref) return;
    const actionButton = event.target.closest('[data-action]');
    if (actionButton) {
      setActiveRef(ref.id);
      performAction(actionButton.dataset.action, ref);
    } else {
      // Arrow keys now step the video; move focus there so the clicked card keeps no focus ring.
      $('#video-panel').focus({ preventScroll: true });
      performAction('open', ref);
    }
  });
  content.addEventListener('mouseover', (event) => {
    const card = event.target.closest('.card');
    if (!card) return;
    const id = Number(card.dataset.ref);
    if (id === state.hoveredRef) return;
    state.hoveredRef = id;
    setActiveRef(id);
  });
  content.addEventListener('mouseout', (event) => {
    if (!event.relatedTarget?.closest?.('.card')) state.hoveredRef = null;
  });
  content.addEventListener('focusin', (event) => {
    const card = event.target.closest('.card');
    if (card) setActiveRef(Number(card.dataset.ref));
  });
  $('#results-back').addEventListener('click', goBack);
  let lastWidth = 0;
  new ResizeObserver(() => {
    const width = content.clientWidth;
    if (width !== lastWidth) {
      lastWidth = width;
      layoutResults();
    } else {
      virtualRows.update();
    }
  }).observe($('#results-scroller'));
}

// ---------------------------------------------------------------------------------------------
// Video view: players
// ---------------------------------------------------------------------------------------------

let youtubeApi = null;

/** Load the YouTube IFrame API once, on first use. */
function loadYouTubeApi() {
  if (window.YT?.Player) return Promise.resolve(window.YT);
  if (!youtubeApi) {
    youtubeApi = new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('the YouTube IFrame API did not load within 10 s')), 10000);
      window.onYouTubeIframeAPIReady = () => {
        clearTimeout(timer);
        resolve(window.YT);
      };
      const script = el('script', { src: 'https://www.youtube.com/iframe_api', async: true });
      script.addEventListener('error', () => {
        clearTimeout(timer);
        reject(new Error('the YouTube IFrame API could not be loaded'));
      });
      document.head.append(script);
    }).catch((error) => {
      youtubeApi = null;
      throw error;
    });
  }
  return youtubeApi;
}

class FilePlayer {
  constructor(host, src, startTime, { onTime, onState, onError }) {
    this.kind = 'Video file';
    this.pendingSeek = startTime;
    this.video = el('video', { src, controls: true, preload: 'auto', playsinline: true });
    // Positions requested before the metadata arrives are applied once seeking is possible.
    this.video.addEventListener('loadedmetadata', () => {
      this.video.currentTime = this.pendingSeek;
      this.pendingSeek = null;
    }, { once: true });
    this.video.addEventListener('timeupdate', () => {
      if (this.pendingSeek === null) onTime(this.video.currentTime);
    });
    this.video.addEventListener('play', onState);
    this.video.addEventListener('pause', onState);
    this.video.addEventListener('error', () => {
      onError(new Error(`the video file could not be played (media error ${this.video.error?.code ?? 'unknown'})`));
    }, { once: true });
    host.replaceChildren(this.video);
  }

  get playing() { return !this.video.paused; }

  seek(time) {
    if (this.pendingSeek !== null) this.pendingSeek = time;
    else this.video.currentTime = time;
  }

  toggle() {
    if (this.video.paused) this.video.play();
    else this.video.pause();
  }

  destroy() {
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
  }
}

class YouTubePlayer {
  constructor(host, videoId, startTime, { onTime, onState, onError }) {
    this.kind = 'YouTube';
    this.player = null;
    this.pendingSeek = null;
    this.lastTime = startTime;
    this.state = -1;
    this.poll = 0;
    this.destroyed = false;
    const mount = el('div', { class: 'youtube-mount' });
    host.replaceChildren(mount);
    loadYouTubeApi().then((YT) => {
      if (this.destroyed) return;
      this.player = new YT.Player(mount, {
        videoId,
        width: '100%',
        height: '100%',
        playerVars: { start: Math.floor(startTime), rel: 0, playsinline: 1 },
        events: {
          onReady: () => {
            if (this.pendingSeek !== null) this.seek(this.pendingSeek);
          },
          onStateChange: (event) => {
            this.state = event.data;
            onState();
          },
          onError: (event) => onError(new Error(`YouTube player error ${event.data}`)),
        },
      });
      this.poll = setInterval(() => {
        // Only playing (1), paused (2) and buffering (3) report a meaningful position.
        if (![1, 2, 3].includes(this.state) || typeof this.player.getCurrentTime !== 'function') return;
        const time = this.player.getCurrentTime();
        if (Math.abs(time - this.lastTime) > 0.05) {
          this.lastTime = time;
          onTime(time);
        }
      }, 250);
    }, onError);
  }

  get playing() { return this.state === 1; }

  seek(time) {
    this.lastTime = time;
    if (this.player?.seekTo) this.player.seekTo(time, true);
    else this.pendingSeek = time;
  }

  toggle() {
    if (!this.player?.playVideo) return;
    if (this.playing) this.player.pauseVideo();
    else this.player.playVideo();
  }

  destroy() {
    this.destroyed = true;
    clearInterval(this.poll);
    this.player?.destroy?.();
  }
}

/** Fallback player: shows the keyframe at the current time and advances in real time when playing. */
class SlideshowPlayer {
  constructor(host, keyframes, duration, startTime, { onTime, onState }, note) {
    this.kind = 'Keyframe slideshow';
    this.keyframes = keyframes;
    this.duration = duration;
    this.time = startTime;
    this.onTime = onTime;
    this.onState = onState;
    this.timer = 0;
    this.shown = -1;
    this.image = el('img', { class: 'slideshow-image', alt: 'Keyframe at the current time' });
    host.replaceChildren(this.image, el('span', { class: 'player-note', text: note }));
    this.render();
  }

  get playing() { return Boolean(this.timer); }

  seek(time) {
    this.time = time;
    this.render();
  }

  render() {
    const index = keyframeIndexAt(this.keyframes, this.time);
    if (index !== this.shown && this.keyframes[index]) {
      this.image.src = this.keyframes[index].thumb;
      this.shown = index;
    }
  }

  toggle() {
    if (this.timer) {
      this.stop();
      return;
    }
    if (this.time >= this.duration) this.time = 0;
    let last = performance.now();
    this.timer = setInterval(() => {
      const now = performance.now();
      this.time = Math.min(this.duration, this.time + (now - last) / 1000);
      last = now;
      this.render();
      this.onTime(this.time);
      if (this.time >= this.duration) this.stop();
    }, 100);
    this.onState();
  }

  stop() {
    clearInterval(this.timer);
    this.timer = 0;
    this.onState();
  }

  destroy() {
    clearInterval(this.timer);
    this.timer = 0;
  }
}

// ---------------------------------------------------------------------------------------------
// Video view: state, stepping, score curve, keyframe strip
// ---------------------------------------------------------------------------------------------

/** Index of the last keyframe at or before `time` (0 when the time precedes all keyframes). */
function keyframeIndexAt(keyframes, time) {
  let lo = 0;
  let hi = keyframes.length - 1;
  let found = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (keyframes[mid].time <= time + 1e-3) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

function nearestKeyframe(keyframes, time) {
  const i = keyframeIndexAt(keyframes, time);
  const next = keyframes[i + 1];
  return next && next.time - time < time - keyframes[i].time ? next : keyframes[i];
}

function videoDuration(video) {
  return Math.max(video.info.duration || 0, video.keyframes.at(-1)?.time ?? 0);
}

function videoPath(videoId, queryId) {
  const path = `/api/video/${encodeURIComponent(videoId)}`;
  return queryId ? `${path}?query_id=${encodeURIComponent(queryId)}` : path;
}

function chainOf(ref) {
  const results = state.results;
  if (!results || results.mode !== 'chains' || ref.entry === null || ref.queryId !== results.queryId) return null;
  const refs = results.entries[ref.entry].ids.map((id) => results.refs[id]);
  return refs.length > 1 ? refs : null;
}

async function openVideo(ref) {
  logEvent('BROWSING', 'openVideo', { video_id: ref.video_id, frame_idx: ref.frame_idx, time: ref.time });
  let queryId = ref.queryId ?? null;
  const chain = chainOf(ref);
  const current = state.video;
  if (current && current.info.video_id === ref.video_id && current.queryId === queryId) {
    current.chain = chain;
    setKeyTarget('video');
    renderCurve();
    renderStrip();
    seekTo(ref.time, { log: false });
    return;
  }
  const seq = ++videoSeq;
  $('#video-progress').hidden = false;
  try {
    let data = await api(videoPath(ref.video_id, queryId));
    const results = state.results;
    if (data.query_known === false && seq === videoSeq && results && results.queryId === queryId) {
      const fresh = await renewQueryId(results);
      if (fresh && seq === videoSeq) {
        queryId = fresh;
        data = await api(videoPath(ref.video_id, fresh));
      }
    }
    if (seq !== videoSeq) return;
    const previous = state.video;
    const sameVideo = previous?.info.video_id === ref.video_id;
    if (previous && !sameVideo) previous.player.destroy();
    state.video = {
      info: data.video, keyframes: data.keyframes, queryId, chain, time: ref.time, keyframeIndex: -1,
      player: sameVideo ? previous.player : null, curve: null,
    };
    $('#video-empty').hidden = true;
    $('#video-body').hidden = false;
    $('#video-close').hidden = false;
    if (!sameVideo) state.video.player = createPlayer(state.video, ref.time);
    setKeyTarget('video');
    renderVideoHeader();
    renderStrip();
    renderCurve();
    seekTo(ref.time, { log: false });
    renderPlayButton();
  } catch (error) {
    if (seq === videoSeq) toast('error', 'Could not open the video', error.message);
  } finally {
    if (seq === videoSeq) $('#video-progress').hidden = true;
  }
}

/** The chain of `videoId` in chain results that covers `time`, else the one nearest to it. */
function chainNear(results, videoId, time) {
  if (!results || results.mode !== 'chains') return null;
  let best = null;
  let bestGap = Infinity;
  for (const entry of results.entries) {
    if (entry.videoId !== videoId || entry.ids.length < 2) continue;
    const refs = entry.ids.map((id) => results.refs[id]);
    const gap = Math.max(0, refs[0].time - time, time - refs.at(-1).time);
    if (gap < bestGap) {
      best = refs;
      bestGap = gap;
    }
  }
  return best;
}

/**
 * Fast reformulation (§7): when a new result list arrives while a video is open, re-fetch that
 * video's per-scene scores for the new query and redraw curve and keyframe bars in place. The
 * player, the playhead and the video view stay as they are.
 */
async function syncVideoWithResults() {
  const video = state.video;
  const results = state.results;
  let queryId = results?.queryId ?? null;
  if (!video || !queryId || video.queryId === queryId) return;
  const seq = ++videoScoresSeq;
  try {
    let data = await api(videoPath(video.info.video_id, queryId));
    if (data.query_known === false && seq === videoScoresSeq && state.results === results) {
      // The server evicted this query (Back to an old list): renew it instead of dropping the curves.
      const fresh = await renewQueryId(results);
      if (fresh) {
        queryId = fresh;
        data = await api(videoPath(video.info.video_id, fresh));
      }
    }
    if (seq !== videoScoresSeq || state.video !== video) return;
    Object.assign(video, { keyframes: data.keyframes, queryId, chain: chainNear(results, video.info.video_id, video.time), keyframeIndex: -1 });
    renderStrip();
    renderCurve();
    updateVideoTime();
    const scenes = sceneCount(video);
    showCurveStatus(data.query_known === false
      ? 'The query expired on the server; search again for score curves'
      : scenes
        ? `Rescored for the new query · ${scenes} ${scenes === 1 ? 'curve' : 'curves'} · position kept`
        : 'No scores for this video under the new query');
    logEvent('BROWSING', 'rescoreVideo', { video_id: video.info.video_id, query_id: queryId, time: round2(video.time) });
  } catch (error) {
    if (seq === videoScoresSeq) toast('warning', 'Score curve not updated', error.message);
  }
}

function showCurveStatus(text) {
  const block = $('#curve-block');
  $('#curve-status').textContent = text;
  block.classList.remove('is-refreshed');
  // Restart the highlight animation when two refreshes follow each other quickly.
  void block.offsetWidth;
  block.classList.add('is-refreshed');
  clearTimeout(curveStatusTimer);
  curveStatusTimer = setTimeout(() => {
    block.classList.remove('is-refreshed');
    $('#curve-status').textContent = '';
  }, CURVE_STATUS_MS);
}

function createPlayer(video, startTime) {
  const host = $('#player');
  const hooks = { onTime: onPlayerTime, onState: renderPlayButton, onError: (error) => fallbackToSlideshow(video, error) };
  if (video.info.file) return new FilePlayer(host, video.info.file, startTime, hooks);
  if (video.info.youtube_id) return new YouTubePlayer(host, video.info.youtube_id, startTime, hooks);
  return new SlideshowPlayer(host, video.keyframes, videoDuration(video), startTime, hooks, 'Keyframe slideshow · no video source');
}

function fallbackToSlideshow(video, error) {
  if (state.video !== video || video.player instanceof SlideshowPlayer) return;
  const failed = video.player.kind;
  video.player.destroy();
  video.player = new SlideshowPlayer($('#player'), video.keyframes, videoDuration(video), video.time,
    { onTime: onPlayerTime, onState: renderPlayButton }, `Keyframe slideshow · ${failed} unavailable`);
  renderVideoHeader();
  renderPlayButton();
  toast('warning', `${failed} player unavailable`, `${error.message}. Showing keyframes instead.`);
}

function onPlayerTime(time) {
  if (!state.video) return;
  state.video.time = time;
  updateVideoTime();
}

function closeVideo() {
  const video = state.video;
  if (!video) return;
  videoSeq += 1;
  video.player.destroy();
  $('#player').replaceChildren();
  state.video = null;
  $('#video-body').hidden = true;
  $('#video-empty').hidden = false;
  $('#video-close').hidden = true;
  $('#video-title').textContent = 'Video';
  $('#video-subtitle').textContent = '';
  setKeyTarget(state.activeRef === null ? null : 'card');
  logEvent('BROWSING', 'closeVideo', { video_id: video.info.video_id });
}

function renderVideoHeader() {
  const video = state.video;
  $('#video-title').textContent = video.info.video_id;
  const subtitle = [video.info.title, fmtTime(videoDuration(video)), `${video.info.fps} fps`].filter(Boolean).join(' · ');
  $('#video-subtitle').textContent = subtitle;
  $('#video-subtitle').title = subtitle;
  $('#strip-info').textContent = `${fmtCount(video.keyframes.length)} keyframes · ${video.player?.kind ?? ''}`;
}

function renderPlayButton() {
  const playing = Boolean(state.video?.player?.playing);
  const button = $('#play-toggle');
  button.querySelector('use').setAttribute('href', playing ? '#i-pause' : '#i-play');
  button.setAttribute('aria-label', playing ? 'Pause' : 'Play');
}

function seekTo(time, { log = true } = {}) {
  const video = state.video;
  if (!video) return;
  video.time = clamp(time, 0, videoDuration(video));
  video.player.seek(video.time);
  updateVideoTime();
  if (log) logEvent('BROWSING', 'seek', { video_id: video.info.video_id, time: round2(video.time) });
}

function stepKeyframe(direction) {
  const video = state.video;
  const target = direction > 0
    ? video.keyframes.find((kf) => kf.time > video.time + 0.05)
    : video.keyframes.findLast((kf) => kf.time < video.time - 0.05);
  if (target) seekTo(target.time);
}

function stepSeconds(delta) {
  seekTo(state.video.time + delta);
}

function updateVideoTime() {
  const video = state.video;
  const index = keyframeIndexAt(video.keyframes, video.time);
  const keyframe = video.keyframes[index];
  $('#video-time').textContent = `${fmtTimePrecise(video.time)} / ${fmtTime(videoDuration(video))}`;
  $('#video-kf').textContent = keyframe ? `keyframe ${index + 1}/${video.keyframes.length} · frame ${keyframe.frame_idx}` : '';
  if (video.curve) video.curve.marker.setAttribute('transform', `translate(${video.curve.x(video.time).toFixed(1)} 0)`);
  if (index !== video.keyframeIndex) {
    video.keyframeIndex = index;
    const strip = $('#keyframe-strip');
    strip.querySelector('.kf.is-current')?.classList.remove('is-current');
    const tile = strip.querySelector(`.kf[data-i="${index}"]`);
    if (tile) {
      tile.classList.add('is-current');
      if (tile.offsetTop < strip.scrollTop) strip.scrollTop = tile.offsetTop - 6;
      else if (tile.offsetTop + tile.offsetHeight > strip.scrollTop + strip.clientHeight) {
        strip.scrollTop = tile.offsetTop + tile.offsetHeight - strip.clientHeight + 6;
      }
    }
    renderCurveReadout(keyframe, null);
  }
}

function sceneCount(video) {
  return video.keyframes.find((kf) => Array.isArray(kf.scores))?.scores.length ?? 0;
}

function scoreRange(video) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const kf of video.keyframes) {
    for (const score of kf.scores ?? []) {
      if (Number.isFinite(score)) {
        lo = Math.min(lo, score);
        hi = Math.max(hi, score);
      }
    }
  }
  return lo <= hi ? [lo, hi] : null;
}

function niceStep(span, maxTicks, steps) {
  return steps.find((step) => span / step <= maxTicks) ?? steps.at(-1);
}

function renderCurve() {
  const video = state.video;
  const box = $('#curve');
  const width = Math.max(240, Math.floor(box.clientWidth));
  const height = Math.max(CURVE.minHeight, Math.floor(box.clientHeight));
  const { left, right, top, bottom } = CURVE;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const duration = videoDuration(video) || 1;
  const x = (t) => left + clamp(t / duration, 0, 1) * plotWidth;
  const scenes = sceneCount(video);
  const range = scenes ? scoreRange(video) : null;

  const grid = svgEl('g', { class: 'curve-grid' });
  const labels = svgEl('g', { class: 'curve-labels' });
  const timeStep = niceStep(duration, Math.max(2, Math.floor(plotWidth / 72)), [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]);
  for (let t = 0; t <= duration + 1e-6; t += timeStep) {
    const px = Math.round(x(t)) + 0.5;
    grid.append(svgEl('line', { x1: px, x2: px, y1: top, y2: top + plotHeight }));
    const anchor = t === 0 ? 'start' : px > width - right - 16 ? 'end' : 'middle';
    labels.append(svgEl('text', { x: px, y: height - 5, 'text-anchor': anchor }, fmtTime(t)));
  }

  const layers = [];
  if (range) {
    const pad = Math.max((range[1] - range[0]) * 0.08, 0.004);
    const lo = range[0] - pad;
    const hi = range[1] + pad;
    const y = (score) => top + (1 - (score - lo) / (hi - lo)) * plotHeight;
    const scoreStep = niceStep(hi - lo, 4, [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1]);
    const decimals = scoreStep < 0.01 ? 3 : 2;
    for (let s = Math.ceil(lo / scoreStep) * scoreStep; s <= hi + 1e-9; s += scoreStep) {
      const py = Math.round(y(s)) + 0.5;
      grid.append(svgEl('line', { x1: left, x2: left + plotWidth, y1: py, y2: py }));
      labels.append(svgEl('text', { x: left - 6, y: py + 3.5, 'text-anchor': 'end' }, s.toFixed(decimals)));
    }
    const lines = svgEl('g', { class: 'curve-lines' });
    for (let k = scenes - 1; k >= 0; k -= 1) {
      const points = video.keyframes
        .filter((kf) => Number.isFinite(kf.scores?.[k]))
        .map((kf) => `${x(kf.time).toFixed(1)},${y(kf.scores[k]).toFixed(1)}`)
        .join(' ');
      lines.append(svgEl('polyline', { class: `curve-line s${k + 1}`, points, 'stroke-dasharray': SCENE_DASHES[k] || null }));
    }
    layers.push(lines);
    const bounds = { left, right: width - right, top, bottom: top + plotHeight };
    // Keep labels off the position marker as drawn now, so a dot at the playhead gets a side label.
    const markerX = x(video.time);
    const placed = [{ x0: markerX - 1.5, x1: markerX + 1.5, y0: top - 10, y1: top + plotHeight }];
    const dots = svgEl('g', { class: 'curve-dots' });
    const labelled = new Set();
    for (const [k, ref] of (video.chain ?? []).entries()) {
      const score = video.keyframes.find((kf) => kf.row === ref.row)?.scores?.[k] ?? ref.score;
      if (!Number.isFinite(score) || k >= scenes) continue;
      const cx = x(ref.time);
      const cy = y(score);
      placed.push({ x0: cx - CURVE.dot, x1: cx + CURVE.dot, y0: cy - CURVE.dot, y1: cy + CURVE.dot });
      dots.append(svgEl('circle', { class: `curve-dot s${k + 1}`, cx: cx.toFixed(1), cy: cy.toFixed(1), r: CURVE.dot }));
      labelled.add(k);
    }
    for (const [k, ref] of (video.chain ?? []).entries()) {
      if (!labelled.has(k)) continue;
      const score = video.keyframes.find((kf) => kf.row === ref.row)?.scores?.[k] ?? ref.score;
      dots.append(curveLabel(x(ref.time), y(score), `S${k + 1}`, `curve-dot-label s${k + 1}`, placed, bounds));
    }
    if (PRESENTATION) {
      // Direct labels at each curve's peak so that scenes stay identifiable in greyscale print.
      for (let k = 0; k < scenes; k += 1) {
        if (labelled.has(k)) continue;
        let peak = null;
        for (const kf of video.keyframes) {
          if (Number.isFinite(kf.scores?.[k]) && (!peak || kf.scores[k] > peak.scores[k])) peak = kf;
        }
        if (peak) dots.append(curveLabel(x(peak.time), y(peak.scores[k]), `S${k + 1}`, `curve-dot-label s${k + 1}`, placed, bounds));
      }
    }
    layers.push(dots);
  } else {
    labels.append(svgEl('text', { class: 'curve-empty', x: left + plotWidth / 2, y: top + plotHeight / 2, 'text-anchor': 'middle' },
      'No query scores for this video yet: run a search and open a result'));
  }

  const hover = svgEl('line', { class: 'curve-hover', x1: 0, x2: 0, y1: top, y2: top + plotHeight, visibility: 'hidden' });
  const marker = svgEl('g', { class: 'curve-marker' },
    svgEl('line', { x1: 0, x2: 0, y1: top - 2, y2: top + plotHeight }),
    svgEl('path', { d: `M -5 ${top - 9} h 10 l -5 7 z` }));
  const svg = svgEl('svg', {
    class: 'curve-svg', width, height, viewBox: `0 0 ${width} ${height}`, role: 'img',
    'aria-label': scenes ? `Score of ${scenes === 1 ? 'the query' : `${scenes} scenes`} over the video; click to seek` : 'Video timeline; click to seek',
  },
  grid,
  svgEl('line', { class: 'curve-axis', x1: left, x2: left + plotWidth, y1: top + plotHeight + 0.5, y2: top + plotHeight + 0.5 }),
  labels, ...layers, hover, marker);
  box.replaceChildren(svg);
  video.curve = { x, left, plotWidth, duration, marker, hover };
  marker.setAttribute('transform', `translate(${x(video.time).toFixed(1)} 0)`);
  renderLegend(video, scenes);
  renderCurveReadout(video.keyframes[keyframeIndexAt(video.keyframes, video.time)], null);
}

/**
 * A scene label next to the point (cx, cy): above, below, right or left, whichever first stays
 * inside the plot without covering an earlier label or chain dot (boxes collected in `placed`).
 */
function curveLabel(cx, cy, text, className, placed, bounds) {
  const size = CURVE.labelSize;
  const w = size * 0.66 * text.length + 4;
  const h = size + 2;
  const gap = CURVE.dot + 3;
  const side = gap + w / 2;
  const candidates = [
    [cx, cy - gap], [cx + side, cy - gap], [cx - side, cy - gap], [cx + side, cy + h / 2 - 2],
    [cx - side, cy + h / 2 - 2], [cx, cy + gap + h - 2], [cx + side, cy + gap + h - 2], [cx - side, cy + gap + h - 2],
  ];
  const boxOf = ([bx, by]) => {
    const nx = clamp(bx, bounds.left + w / 2, bounds.right - w / 2);
    const ny = clamp(by, bounds.top + h - 2, bounds.bottom + 2);
    return { x: nx, y: ny, x0: nx - w / 2, x1: nx + w / 2, y0: ny - h + 2, y1: ny + 2 };
  };
  const overlaps = (a) => placed.some((b) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1);
  const box = candidates.map(boxOf).find((candidate) => !overlaps(candidate)) ?? boxOf(candidates[0]);
  placed.push(box);
  return svgEl('text', { class: className, x: box.x.toFixed(1), y: box.y.toFixed(1), 'text-anchor': 'middle' }, text);
}

function renderLegend(video, scenes) {
  const info = queryInfo.get(video.queryId);
  $('#curve-legend').replaceChildren(...Array.from({ length: scenes }, (_, k) => el('span', {
    class: 'legend-item', title: info?.texts[k] ?? '',
  },
  svgEl('svg', { class: 'legend-swatch', width: CURVE.swatch, height: 10, 'aria-hidden': 'true' },
    svgEl('line', { class: `curve-line s${k + 1}`, x1: 1, x2: CURVE.swatch - 1, y1: 5, y2: 5, 'stroke-dasharray': SCENE_DASHES[k] || null })),
  el('span', { text: info?.labels[k] ?? `S${k + 1}` }))));
}

function renderCurveReadout(keyframe, hoverTime) {
  const readout = $('#curve-readout');
  if (!keyframe) {
    readout.textContent = '';
    return;
  }
  const info = queryInfo.get(state.video.queryId);
  const scores = (keyframe.scores ?? []).map((score, k) => `${info?.labels[k] ?? `S${k + 1}`} ${fmtScore(score)}`);
  readout.textContent = [fmtTime(hoverTime ?? keyframe.time), ...scores].join('  ');
  readout.classList.toggle('is-hover', hoverTime !== null);
}

function curveTimeAt(event) {
  const curve = state.video?.curve;
  const svg = $('#curve svg');
  if (!curve || !svg) return null;
  const px = event.clientX - svg.getBoundingClientRect().left;
  return clamp((px - curve.left) / curve.plotWidth, 0, 1) * curve.duration;
}

function renderStrip() {
  const video = state.video;
  const range = scoreRange(video);
  const chainRows = new Map((video.chain ?? []).map((ref, k) => [ref.row, k]));
  const width = (score) => (range && range[1] > range[0] ? clamp(((score - range[0]) / (range[1] - range[0])) * 100, 2, 100) : 0);
  $('#keyframe-strip').replaceChildren(...video.keyframes.map((kf, i) => {
    const scene = chainRows.get(kf.row);
    return el('button', {
      type: 'button', class: 'kf', 'data-i': i, tabindex: '-1',
      title: `Keyframe ${i + 1} · ${fmtTime(kf.time)} · frame ${kf.frame_idx}`,
      'aria-label': `Keyframe ${i + 1} at ${fmtTime(kf.time)}`,
    },
    el('span', { class: 'kf-thumb' },
      el('img', { src: kf.thumb, alt: '', loading: 'lazy', decoding: 'async', draggable: 'false' }),
      el('span', { class: 'kf-time', text: fmtTime(kf.time) }),
      scene === undefined ? null : sceneChip(scene)),
    kf.scores ? el('span', { class: 'kf-bars', 'aria-hidden': 'true' },
      kf.scores.map((score, k) => {
        const bar = el('span', { class: `kf-bar s${k + 1}`, style: `width:${width(score).toFixed(1)}%` });
        // Presentation: a label per bar (bars also get the curve's dash pattern in style.css).
        return PRESENTATION && kf.scores.length > 1
          ? el('span', { class: `kf-bar-row s${k + 1}` }, el('span', { class: 'kf-bar-label', text: `S${k + 1}` }), el('span', { class: 'kf-bar-track' }, bar))
          : bar;
      })) : null);
  }));
  video.keyframeIndex = -1;
}

function videoMomentRef() {
  const video = state.video;
  const keyframe = video.keyframes[keyframeIndexAt(video.keyframes, video.time)];
  return {
    id: null, row: keyframe.row, video_id: video.info.video_id, frame_idx: keyframe.frame_idx, time: video.time,
    thumb: keyframe.thumb, score: null, scene: null, entry: null, queryId: video.queryId,
  };
}

function bindVideo() {
  const onVideo = (handler) => () => {
    if (!state.video) return;
    setKeyTarget('video');
    handler();
  };
  $('#video-close').addEventListener('click', closeVideo);
  $('#kf-prev').addEventListener('click', onVideo(() => stepKeyframe(-1)));
  $('#kf-next').addEventListener('click', onVideo(() => stepKeyframe(1)));
  $('#sec-back').addEventListener('click', onVideo(() => stepSeconds(-1)));
  $('#sec-fwd').addEventListener('click', onVideo(() => stepSeconds(1)));
  $('#play-toggle').addEventListener('click', onVideo(() => state.video.player.toggle()));
  $('#video-submit').addEventListener('click', onVideo(() => submitKis(videoMomentRef())));
  $('#video-vqa').addEventListener('click', onVideo(() => addToVqa(videoMomentRef())));
  $('#video-basket').addEventListener('click', onVideo(() => addToBasket(videoMomentRef())));
  $('#video-similar').addEventListener('click', onVideo(() => runSimilar(videoMomentRef())));
  $('#keyframe-strip').addEventListener('click', (event) => {
    const tile = event.target.closest('.kf');
    if (!tile || !state.video) return;
    setKeyTarget('video');
    seekTo(state.video.keyframes[Number(tile.dataset.i)].time);
  });

  const curveBox = $('#curve');
  let scrubbing = false;
  const scrubTo = (event) => {
    const time = curveTimeAt(event);
    if (time === null) return;
    const video = state.video;
    seekTo(nearestKeyframe(video.keyframes, time).time, { log: false });
  };
  curveBox.addEventListener('pointerdown', (event) => {
    if (!state.video || event.button !== 0) return;
    scrubbing = true;
    curveBox.setPointerCapture(event.pointerId);
    setKeyTarget('video');
    scrubTo(event);
  });
  curveBox.addEventListener('pointermove', (event) => {
    const video = state.video;
    const time = curveTimeAt(event);
    if (!video || time === null) return;
    if (scrubbing) scrubTo(event);
    const keyframe = nearestKeyframe(video.keyframes, time);
    video.curve.hover.setAttribute('transform', `translate(${video.curve.x(keyframe.time).toFixed(1)} 0)`);
    video.curve.hover.setAttribute('visibility', 'visible');
    renderCurveReadout(keyframe, keyframe.time);
  });
  const endScrub = () => {
    if (!scrubbing || !state.video) return;
    scrubbing = false;
    logEvent('BROWSING', 'seek', { video_id: state.video.info.video_id, time: round2(state.video.time) });
  };
  curveBox.addEventListener('pointerup', endScrub);
  curveBox.addEventListener('pointercancel', endScrub);
  curveBox.addEventListener('pointerleave', () => {
    const video = state.video;
    if (!video?.curve) return;
    video.curve.hover.setAttribute('visibility', 'hidden');
    renderCurveReadout(video.keyframes[keyframeIndexAt(video.keyframes, video.time)], null);
  });
  new ResizeObserver(() => { if (state.video) renderCurve(); }).observe(curveBox);
}

// ---------------------------------------------------------------------------------------------
// Actions: submissions, AVS basket, VQA
// ---------------------------------------------------------------------------------------------

/**
 * DRES verdict of a /api/submit response: raw.submission first; a response that is not ok is a
 * rejection whatever its text says (412 "Maximum number of correct submissions (1) exceeded" contains
 * the word "correct"); the status text is read only for ok responses without a verdict.
 */
function verdictOf(response) {
  const judged = String(response.raw?.submission ?? '').toUpperCase();
  const text = String(response.status ?? '').toLowerCase();
  if (judged === 'CORRECT') return 'correct';
  if (judged === 'WRONG') return 'wrong';
  if (['INDETERMINATE', 'UNDECIDABLE'].includes(judged)) return 'pending';
  if (!response.ok) return /not connected/.test(text) ? 'offline' : 'error';
  if (/\bcorrect\b/.test(text) && !/incorrect|not correct/.test(text)) return 'correct';
  if (/\bwrong\b|incorrect/.test(text)) return 'wrong';
  if (/indeterminate|undecidable|pending|awaiting/.test(text)) return 'pending';
  return 'accepted';
}

/**
 * Send one submission, record it and report the DRES verdict. Callers go through guardedSubmit;
 * `check` is the guard's verdict ("ok" | "warn" | "fail" | "none") and `confirmed` whether the
 * operator confirmed it after a warning.
 */
async function submit(request, { kind, target, thumb, check = null, confirmed = false }) {
  const entry = { id: ++submissionSeq, ts: Date.now(), kind, target, thumb, check, confirmed, verdict: 'sending', status: 'Sending…' };
  state.submissions.unshift(entry);
  renderSubmissions();
  logEvent('OTHER', `submit${kind}`, request);
  let response = null;
  try {
    response = await api('/api/submit', request);
    entry.verdict = verdictOf(response);
    entry.status = response.status || (response.ok ? 'accepted' : 'rejected');
  } catch (error) {
    entry.verdict = 'error';
    entry.status = error.message;
  }
  if (entry.verdict === 'wrong' && request.kind === 'kis') markRejected(request.video_id);
  renderSubmissions();
  const view = VERDICT_VIEW[entry.verdict];
  const detail = entry.verdict === 'offline'
    ? `${kind} ${target} was logged locally only. Click the DRES status in the top bar to sign in.`
    : `${kind} ${target} · ${entry.status}`;
  toast(view.toast, view.label, detail, { duration: 6000 });
  return response;
}

// ---------------------------------------------------------------------------------------------
// Submission guard (§7): verify first, confirm warnings with a second press within 5 s
// ---------------------------------------------------------------------------------------------

/** Scenes a moment is checked against: the text query behind the frame, else the scene boxes. */
function verifyScenes(ref) {
  const info = ref.queryId ? queryInfo.get(ref.queryId) : null;
  if (info?.kind === 'text') return [...info.texts];
  return state.scenes.map((scene) => scene.trim()).filter(Boolean);
}

/** POST /api/verify; never throws (an unreachable check becomes a warning). */
async function checkMoment(scenes, videoId, time) {
  try {
    const response = await api('/api/verify', { scenes, video_id: videoId, time }, { timeoutMs: VERIFY_TIMEOUT_MS });
    const verdict = ['ok', 'warn', 'fail'].includes(response.verdict) ? response.verdict : 'warn';
    const reasons = Array.isArray(response.reasons) ? response.reasons.map(String).filter(Boolean) : [];
    if (verdict !== 'ok' && !reasons.length) reasons.push('The check gave no reason.');
    // Non-blocking notes (e.g. a wrong submission elsewhere in this video): shown with the reasons, or as a toast after "ok".
    const notes = Array.isArray(response.notes) ? response.notes.map(String).filter(Boolean) : [];
    if (verdict !== 'ok') reasons.push(...notes);
    const rejected = typeof response.rejected === 'boolean'
      ? response.rejected
      : verdict === 'fail' && reasons.some((reason) => REJECTED_REASON.test(reason));
    if (rejected) markRejected(videoId);
    return { verdict, reasons, notes, model: response.model ?? null, took_ms: Number.isFinite(response.took_ms) ? response.took_ms : null, rejected };
  } catch (error) {
    const reason = error.timedOut
      ? `The check did not answer within ${VERIFY_TIMEOUT_MS / 1000} s; the moment was not verified.`
      : `The check is unavailable: ${error.message}`;
    return { verdict: 'warn', reasons: [reason], model: null, took_ms: null, rejected: false, unavailable: true };
  }
}

/** AVS: no per-item verdicts, only a warning when an item lies at a moment already judged wrong (the server blocks moments, not videos). */
async function checkBasket(items) {
  const started = performance.now();
  const answers = await mapLimit(items, VERIFY_CONCURRENCY, (item) => checkMoment([], item.video_id, round3(item.time)));
  const rejectedItems = items.filter((_, i) => answers[i].rejected);
  const unavailable = answers.filter((answer) => answer.unavailable).length;
  const tookMs = Math.round(performance.now() - started);
  if (unavailable) {
    toast('warning', 'Basket check incomplete', `${unavailable} of ${items.length} shots could not be checked against earlier verdicts.`, { key: 'basket-check' });
  }
  if (!rejectedItems.length) return { verdict: 'ok', reasons: [], model: null, took_ms: tookMs, rejectedItems };
  const videos = [...new Set(rejectedItems.map((item) => item.video_id))];
  return {
    verdict: 'warn',
    reasons: [`${rejectedItems.length} of ${items.length} shots lie at moments already judged wrong in ${videos.length === 1 ? 'video' : 'videos'} ${videos.join(', ')}`],
    model: null,
    took_ms: tookMs,
    rejectedItems,
  };
}

function markRejected(videoId) {
  if (state.rejectedVideos.has(videoId)) return;
  state.rejectedVideos.add(videoId);
  refreshCardFlags();
  renderBasket();
}

/** Whether a new request is the one the pending guard waits to have confirmed. */
function sameSubmission(guard, spec) {
  if (guard.kind !== spec.kind) return false;
  if (spec.kind === 'KIS') return guard.videoId === spec.videoId && Math.abs(guard.time - spec.time) <= GUARD_SAME_MOMENT_S;
  return guard.key === spec.key;
}

function guardSubject(spec) {
  if (spec.kind === 'KIS') return { kind: 'kis', video_id: spec.videoId, time: spec.time };
  if (spec.kind === 'QA') return { kind: 'text', text: spec.request.text, video_id: spec.videoId ?? null, time: spec.time ?? null };
  return { kind: 'avs', items: spec.request.items.length };
}

function logDecision(spec, decision) {
  logEvent('OTHER', 'submitDecision', { ...guardSubject(spec), verdict: spec.result?.verdict ?? null, decision });
}

/**
 * Submit through the guard. spec: {kind: "KIS"|"QA"|"AVS", videoId?, time?, key?, request,
 * meta: {kind, target, thumb}, check: (() => Promise<result>) | null, after?: (response) => void}.
 * A second call for the same submission while its warning is shown confirms it.
 */
async function guardedSubmit(spec) {
  const pending = state.guard;
  if (pending && sameSubmission(pending, spec)) {
    if (pending.phase !== 'checking') {
      resolveGuard(pending, 'confirmed');
    } else {
      toast('info', 'Still checking', 'Wait for the check, or click "Submit without check" in the banner.', { key: 'guard-wait', duration: 2500 });
    }
    return;
  }
  // A second press right after an instant "ok" must not send the same answer again (DRES counts it).
  const recent = recentSubmission(spec);
  if (recent) {
    toast('info', recent.inFlight ? 'Already sent, waiting for the verdict' : 'Just sent',
      `${spec.meta.kind} ${spec.meta.target} was not sent again.`, { key: 'guard-repeat', duration: 2500 });
    return;
  }
  if (pending) dropGuard('superseded');
  if (!spec.check) {
    logDecision(spec, 'submitted-unchecked');
    const response = await sendSubmission(spec, { ...spec.meta, check: 'none' });
    spec.after?.(response);
    return;
  }
  const guard = { ...spec, phase: 'checking', result: null, deadline: 0, timer: 0 };
  state.guard = guard;
  renderGuard();
  const result = await spec.check();
  if (state.guard !== guard) return;
  guard.result = result;
  logEvent('OTHER', 'verify', {
    ...guardSubject(guard), verdict: result.verdict, reasons: result.reasons, model: result.model, took_ms: result.took_ms,
    ...(result.rejectedItems ? { rejected_videos: [...new Set(result.rejectedItems.map((item) => item.video_id))] } : {}),
  });
  if (result.verdict === 'ok') {
    if (result.notes?.length) toast('info', 'Earlier wrong submission in this video', result.notes[0], { key: 'verify-note', duration: 6000 });
    resolveGuard(guard, 'submitted');
    return;
  }
  guard.phase = result.verdict;
  guard.deadline = Date.now() + GUARD_CONFIRM_MS;
  guard.timer = setTimeout(() => { if (state.guard === guard) dropGuard('expired'); }, GUARD_CONFIRM_MS);
  renderGuard();
}

async function resolveGuard(guard, decision) {
  clearTimeout(guard.timer);
  if (state.guard === guard) state.guard = null;
  renderGuard();
  logDecision(guard, decision);
  const check = guard.result?.verdict ?? 'none';
  const response = await sendSubmission(guard, { ...guard.meta, check, confirmed: decision === 'confirmed' });
  guard.after?.(response);
}

/** submit() plus the record that blocks a repeated press for the same target while in flight and just after. */
async function sendSubmission(spec, options) {
  const record = { kind: spec.kind, videoId: spec.videoId, time: spec.time, key: spec.key, inFlight: true, until: 0 };
  state.sent.push(record);
  try {
    return await submit(spec.request, options);
  } finally {
    record.inFlight = false;
    record.until = Date.now() + GUARD_REPEAT_BLOCK_MS;
  }
}

function recentSubmission(spec) {
  const now = Date.now();
  state.sent = state.sent.filter((record) => record.inFlight || record.until > now);
  return state.sent.find((record) => sameSubmission(record, spec)) ?? null;
}

/** "Submit without check" while the check is still running (the operator is sure, or the network is slow). */
function skipPendingCheck() {
  const guard = state.guard;
  if (guard && guard.phase === 'checking') resolveGuard(guard, 'skipped-check');
}

/** End the pending guard without submitting: "cancelled", "expired", "superseded" or "removed-rejected". */
function dropGuard(decision) {
  const guard = state.guard;
  if (!guard) return;
  clearTimeout(guard.timer);
  state.guard = null;
  renderGuard();
  logDecision(guard, decision);
  if (decision === 'cancelled') {
    toast('info', 'Submission cancelled', `${guard.meta.kind} ${guard.meta.target} was not sent.`, { key: 'guard', duration: 2500 });
  } else if (decision === 'expired') {
    toast('info', 'Submission not sent', `No second press within ${GUARD_CONFIRM_MS / 1000} s for ${guard.meta.kind} ${guard.meta.target}.`, { key: 'guard', duration: 3500 });
  }
}

function confirmPendingGuard() {
  const guard = state.guard;
  if (guard && guard.phase !== 'checking') resolveGuard(guard, 'confirmed');
}

function renderGuard() {
  const box = $('#guard');
  const guard = state.guard;
  box.hidden = !guard;
  if (!guard) {
    box.replaceChildren();
    return;
  }
  const checking = guard.phase === 'checking';
  const view = GUARD_VIEW[guard.phase];
  const result = guard.result;
  box.className = `guard ${guard.phase}`;
  box.setAttribute('role', checking ? 'status' : 'alert');
  const rejectedItems = result?.rejectedItems ?? [];
  const facts = checking ? [] : [
    `Submit anyway within ${GUARD_CONFIRM_MS / 1000} s`,
    result.model,
    Number.isFinite(result.took_ms) ? `checked in ${fmtCount(result.took_ms)} ms` : null,
  ].filter(Boolean);
  box.replaceChildren(
    el('div', { class: 'guard-main' },
      el('span', { class: 'guard-icon' }, icon(view.icon)),
      guard.meta.thumb ? el('img', { class: 'guard-thumb', src: guard.meta.thumb, alt: '' }) : null,
      el('div', { class: 'guard-text' },
        el('div', { class: 'guard-title' },
          el('span', { class: 'guard-verdict', text: view.label }),
          el('span', { class: 'kind-tag', text: guard.meta.kind }),
          el('span', { class: 'guard-target', text: guard.meta.target })),
        checking
          ? el('div', { class: 'guard-note', text: guard.kind === 'AVS'
            ? `Looking up earlier wrong verdicts for ${guard.request.items.length} shots…`
            : 'Comparing the frames around this moment with the description…' })
          : el('ul', { class: 'guard-reasons' }, result.reasons.map((reason) => el('li', { text: reason }))),
        facts.length ? el('div', { class: 'guard-note' }, facts.flatMap((fact, i) => [i ? ' · ' : null, el('span', { class: 'guard-fact', text: fact })])) : null),
      el('div', { class: 'guard-actions' },
        rejectedItems.length ? el('button', {
          type: 'button', class: 'btn small',
          onclick: () => {
            state.basket = state.basket.filter((item) => !rejectedItems.includes(item));
            logEvent('OTHER', 'basketRemove', { items: rejectedItems.map((item) => ({ video_id: item.video_id, frame_idx: item.frame_idx })), reason: 'judged wrong' });
            renderBasket();
            dropGuard('removed-rejected');
          },
        }, icon('trash'), `Remove ${rejectedItems.length}`) : null,
        checking
          ? el('button', { type: 'button', class: 'btn small guard-skip', onclick: skipPendingCheck },
            icon('send'), guard.kind === 'AVS' ? 'Submit all without check' : 'Submit without check')
          : el('button', { type: 'button', class: 'btn small guard-confirm', onclick: confirmPendingGuard },
            icon('send'), guard.kind === 'AVS' ? 'Submit all anyway' : 'Submit anyway', el('kbd', { text: 'S' })),
        el('button', { type: 'button', class: 'btn small', onclick: () => dropGuard('cancelled') }, 'Cancel', el('kbd', { text: 'Esc' })))),
    el('div', { class: `guard-timer${checking ? ' is-checking' : ''}`, 'aria-hidden': 'true' },
      el('span', { class: 'guard-timer-bar', style: checking ? null : `animation-duration:${Math.max(0, guard.deadline - Date.now())}ms` })));
}

function submitKis(ref) {
  const time = round3(ref.time);
  return guardedSubmit({
    kind: 'KIS',
    videoId: ref.video_id,
    time,
    request: { kind: 'kis', video_id: ref.video_id, time },
    meta: { kind: 'KIS', target: `${ref.video_id} @ ${fmtTimePrecise(ref.time)}`, thumb: ref.thumb },
    check: () => checkMoment(verifyScenes(ref), ref.video_id, time),
  });
}

function checkCell(entry) {
  if (!entry.check) return el('span', { class: 'muted', text: '–' });
  if (entry.check === 'none') return el('span', { class: 'check-tag none', text: 'not checked' });
  if (entry.check === 'ok') return el('span', { class: 'check-tag ok', text: 'passed' });
  return el('span', { class: `check-tag ${entry.check}`, text: `${entry.check}, confirmed` });
}

function renderSubmissions() {
  const submissions = state.submissions;
  const sent = submissions.filter((entry) => ['correct', 'wrong', 'pending', 'accepted'].includes(entry.verdict));
  $('#count-total').textContent = String(sent.length);
  $('#count-correct').textContent = String(submissions.filter((entry) => entry.verdict === 'correct').length);
  $('#count-wrong').textContent = String(submissions.filter((entry) => entry.verdict === 'wrong').length);
  $('#log-count').textContent = String(submissions.length);
  $('#log-empty').hidden = submissions.length > 0;
  $('#log-body').replaceChildren(...submissions.map((entry) => {
    const view = VERDICT_VIEW[entry.verdict];
    return el('tr', {},
      el('td', { class: 'mono', text: fmtClock(entry.ts) }),
      el('td', {}, el('span', { class: 'kind-tag', text: entry.kind })),
      el('td', { class: 'log-target' },
        entry.thumb ? el('img', { class: 'log-thumb', src: entry.thumb, alt: '' }) : null,
        el('span', { text: entry.target })),
      el('td', {}, checkCell(entry)),
      el('td', {}, el('span', { class: `verdict ${entry.verdict}` }, icon(view.icon), view.label)),
      el('td', { class: 'log-message', text: entry.status, title: entry.status }));
  }));
  const latest = submissions[0];
  $('#drawer-latest').replaceChildren(...(latest ? [
    el('span', { class: 'muted', text: 'Last submission' }),
    el('span', { text: `${latest.kind} ${latest.target}` }),
    el('span', { class: `verdict ${latest.verdict}` }, icon(VERDICT_VIEW[latest.verdict].icon), VERDICT_VIEW[latest.verdict].label),
  ] : []));
}

function miniCard(item, { onOpen, onRemove, removeLabel }) {
  const rejected = state.rejectedVideos.has(item.video_id);
  return el('div', { class: rejected ? 'mini-card is-rejected' : 'mini-card', title: rejected ? 'A submission in this video was judged wrong in the current task' : null },
    el('button', { type: 'button', class: 'mini-thumb', title: 'Open in the video view', 'aria-label': `Open ${item.video_id} at ${fmtTime(item.time)}`, onclick: onOpen },
      el('img', { src: item.thumb, alt: '', loading: 'lazy', draggable: 'false' })),
    el('button', { type: 'button', class: 'mini-remove', title: removeLabel, 'aria-label': `${removeLabel}: ${item.video_id} at ${fmtTime(item.time)}`, onclick: onRemove }, icon('x')),
    el('div', { class: 'meta' }, el('span', { class: 'vid', text: item.video_id }), el('span', { class: 'time', text: fmtTime(item.time) })));
}

function openStoredFrame(item) {
  openVideo({ ...item, id: null, score: null, scene: null, entry: null, queryId: state.results?.queryId ?? null });
}

function addToBasket(ref) {
  if (inBasket(ref)) {
    toast('info', 'Already in the basket', `${ref.video_id} @ ${fmtTime(ref.time)}`, { key: 'basket', duration: 2000 });
    return;
  }
  const item = { video_id: ref.video_id, frame_idx: ref.frame_idx, time: ref.time, thumb: ref.thumb, row: ref.row };
  state.basket.push(item);
  logEvent('OTHER', 'basketAdd', { video_id: item.video_id, frame_idx: item.frame_idx, time: round2(item.time) });
  renderBasket();
  toast('info', `Basket: ${state.basket.length} ${state.basket.length === 1 ? 'shot' : 'shots'}`, `Added ${ref.video_id} @ ${fmtTime(ref.time)}`, { key: 'basket', duration: 2000 });
}

function removeFromBasket(item) {
  state.basket = state.basket.filter((other) => other !== item);
  logEvent('OTHER', 'basketRemove', { video_id: item.video_id, frame_idx: item.frame_idx });
  renderBasket();
}

async function submitBasket() {
  if (!state.basket.length) return;
  const items = [...state.basket];
  await guardedSubmit({
    kind: 'AVS',
    key: `avs:${items.map(basketKey).sort().join(',')}`,
    request: { kind: 'avs', items: items.map((item) => ({ video_id: item.video_id, time: round3(item.time) })) },
    meta: { kind: 'AVS', target: `${items.length} ${items.length === 1 ? 'shot' : 'shots'}`, thumb: items[0].thumb },
    check: () => checkBasket(items),
    after: (response) => {
      if (!response?.ok) return;
      state.basket = state.basket.filter((item) => !items.includes(item));
      renderBasket();
    },
  });
}

function renderBasket() {
  const { basket } = state;
  $('#basket-count').textContent = String(basket.length);
  $('#basket-submit').disabled = !basket.length;
  $('#basket-clear').disabled = !basket.length;
  const videos = new Set(basket.map((item) => item.video_id)).size;
  const rejected = basket.filter((item) => state.rejectedVideos.has(item.video_id)).length;
  const rejectedNote = rejected ? ` ${rejected} ${rejected === 1 ? 'comes' : 'come'} from videos judged wrong.` : '';
  $('#basket-note').replaceChildren(...(basket.length
    ? [`${basket.length} ${basket.length === 1 ? 'shot' : 'shots'} from ${videos} ${videos === 1 ? 'video' : 'videos'}. Submitting sends all of them at once.${rejectedNote}`]
    : ['Press ', el('kbd', { text: 'B' }), ' on a result or in the video view to collect shots for an ad-hoc search task.']));
  $('#basket-list').replaceChildren(...basket.map((item) => miniCard(item, {
    onOpen: () => openStoredFrame(item), onRemove: () => removeFromBasket(item), removeLabel: 'Remove from basket',
  })));
  refreshCardFlags();
}

function addToVqa(ref) {
  const frames = state.vqa.frames.filter((frame) => basketKey(frame) !== basketKey(ref));
  frames.push({ video_id: ref.video_id, frame_idx: ref.frame_idx, time: ref.time, thumb: ref.thumb, row: ref.row });
  state.vqa.frames = frames.slice(-VQA_MAX_FRAMES);
  logEvent('OTHER', 'vqaFrame', { video_id: ref.video_id, frame_idx: ref.frame_idx });
  renderVqa();
  openDrawer('vqa');
  $('#vqa-question').focus();
}

async function askVqa() {
  const question = $('#vqa-question').value.trim();
  if (!question || !state.vqa.frames.length) {
    toast('info', 'VQA needs a question and a frame', 'Press A on a result or in the video view to attach frames.');
    return;
  }
  const frames = state.vqa.frames.map(({ video_id: videoId, frame_idx: frameIdx }) => ({ video_id: videoId, frame_idx: frameIdx }));
  // A newer question supersedes this one; a late reply must not replace the newer answer or one the
  // operator typed while waiting.
  const seq = ++vqaSeq;
  const answerBox = $('#vqa-answer');
  const answerBefore = answerBox.value;
  state.vqa.asking = true;
  renderVqa();
  logEvent('TEXT', 'vqaQuestion', { question, frames });
  try {
    const response = await api('/api/vqa', { question, frames });
    if (seq !== vqaSeq) return;
    const changed = answerBox.value !== answerBefore || $('#vqa-question').value.trim() !== question;
    state.vqa.suggestion = { ...response, kept: changed };
    if (!changed) answerBox.value = response.answer;
  } catch (error) {
    if (seq === vqaSeq) toast('error', 'VQA failed', error.message);
  } finally {
    if (seq === vqaSeq) {
      state.vqa.asking = false;
      renderVqa();
    }
  }
}

function submitVqaAnswer() {
  const text = $('#vqa-answer').value.trim();
  if (!text) {
    toast('info', 'Nothing to submit', 'Type or ask for an answer first.');
    $('#vqa-answer').focus();
    return;
  }
  // The question stands in for the description; the check looks at the most recently attached frame.
  const question = $('#vqa-question').value.trim();
  const frame = state.vqa.frames.at(-1) ?? null;
  const time = frame ? round3(frame.time) : null;
  guardedSubmit({
    kind: 'QA',
    key: `text:${text.toLowerCase()}`,
    videoId: frame?.video_id ?? null,
    time,
    request: { kind: 'text', text },
    meta: { kind: 'QA', target: `"${text}"`, thumb: frame?.thumb ?? null },
    check: frame && question ? () => checkMoment([question], frame.video_id, time) : null,
  });
}

function renderVqa() {
  const { frames, suggestion, asking } = state.vqa;
  const llm = state.status?.llm;
  const available = !llm || llm.gemini || llm.openai;
  $('#vqa-count').textContent = String(frames.length);
  $('#vqa-frames').replaceChildren(...(frames.length
    ? frames.map((frame) => miniCard(frame, {
      onOpen: () => openStoredFrame(frame),
      onRemove: () => { state.vqa.frames = state.vqa.frames.filter((other) => other !== frame); renderVqa(); },
      removeLabel: 'Remove from VQA',
    }))
    : [el('p', { class: 'pane-note', text: 'No frames attached.' })]));
  const ask = $('#vqa-ask');
  ask.disabled = asking || !available;
  $('#vqa-ask-label').textContent = asking ? 'Asking…' : 'Ask model';
  $('#vqa-model').textContent = !available ? '(no VQA model configured on the server)' : suggestion ? `(${suggestion.model})` : '';
  $('#vqa-evidence').textContent = !suggestion
    ? 'Ask a question to get a suggested answer with its visual evidence.'
    : suggestion.kept
      ? `Suggested "${suggestion.answer}" (not filled in: the question or answer changed while asking). ${suggestion.evidence}`
      : suggestion.evidence;
  $('#vqa-evidence').classList.toggle('is-empty', !suggestion);
}

// ---------------------------------------------------------------------------------------------
// Drawer
// ---------------------------------------------------------------------------------------------

function openDrawer(tab) {
  state.drawerTab = tab;
  state.drawerOpen = true;
  renderDrawer();
}

function renderDrawer() {
  const drawer = $('#drawer');
  drawer.dataset.open = String(state.drawerOpen);
  for (const tab of drawer.querySelectorAll('[role="tab"]')) {
    const selected = tab.dataset.tab === state.drawerTab;
    tab.setAttribute('aria-selected', String(selected && state.drawerOpen));
    $(`#pane-${tab.dataset.tab}`).hidden = !selected;
  }
  const toggle = $('#drawer-toggle');
  toggle.setAttribute('aria-expanded', String(state.drawerOpen));
  toggle.setAttribute('aria-label', state.drawerOpen ? 'Collapse drawer' : 'Expand drawer');
  toggle.querySelector('use').setAttribute('href', state.drawerOpen ? '#i-chevron-down' : '#i-chevron-up');
}

function bindDrawer() {
  for (const tab of document.querySelectorAll('#drawer [role="tab"]')) {
    tab.addEventListener('click', () => {
      if (state.drawerOpen && state.drawerTab === tab.dataset.tab) {
        state.drawerOpen = false;
        renderDrawer();
      } else {
        openDrawer(tab.dataset.tab);
      }
    });
  }
  $('#drawer-toggle').addEventListener('click', () => {
    state.drawerOpen = !state.drawerOpen;
    renderDrawer();
  });
  $('#basket-submit').addEventListener('click', submitBasket);
  $('#basket-clear').addEventListener('click', () => {
    logEvent('OTHER', 'basketClear', { count: state.basket.length });
    state.basket = [];
    renderBasket();
  });
  $('#vqa-ask').addEventListener('click', askVqa);
  $('#vqa-question').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      askVqa();
    }
  });
  $('#vqa-submit').addEventListener('click', submitVqaAnswer);
  $('#vqa-answer').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.ctrlKey && !event.metaKey) {
      event.preventDefault();
      submitVqaAnswer();
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------------------------

/** The frame that S / E / B / A act on: the last used card, or the current moment of the video. */
function keyTargetRef() {
  if (state.keyTarget === 'card' && state.activeRef !== null && state.results) return state.results.refs[state.activeRef];
  return state.video ? videoMomentRef() : null;
}

const KEY_ACTIONS = { s: 'submit', e: 'similar', b: 'basket', a: 'vqa' };

function onKeyDown(event) {
  const dialogOpen = Boolean(document.querySelector('dialog[open]'));
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    if (dialogOpen) return;
    event.preventDefault();
    runSearch();
    return;
  }
  if (dialogOpen) return;
  const typing = isTyping(event.target);
  if (event.key === 'Escape') {
    if (state.guard) dropGuard('cancelled');
    else if (typing) event.target.blur();
    else closeVideo();
    return;
  }
  if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
  const card = event.target instanceof Element ? event.target.closest('#results-content .card') : null;
  const key = event.key;

  if (key === '/') {
    event.preventDefault();
    focusQuery();
  } else if (key === '?') {
    event.preventDefault();
    openHelp();
  } else if (key === '1' || key === '2' || key === '3') {
    setMode(MODES[Number(key) - 1], { rerun: true });
  } else if (key === 'ArrowLeft' || key === 'ArrowRight') {
    if (state.video) {
      event.preventDefault();
      setKeyTarget('video');
      const direction = key === 'ArrowRight' ? 1 : -1;
      if (event.shiftKey) stepSeconds(direction);
      else stepKeyframe(direction);
    } else if (card) {
      event.preventDefault();
      moveCardFocus(card, key);
    }
  } else if (key === 'ArrowUp' || key === 'ArrowDown') {
    if (card) {
      event.preventDefault();
      moveCardFocus(card, key);
    }
  } else if (key === 'Enter') {
    if (card && !event.repeat) {
      event.preventDefault();
      openVideo(refOfCard(card));
    } else if (state.guard && event.target.closest?.(SUBMIT_BUTTONS)) {
      event.preventDefault(); // the browser's click on a focused Submit button would confirm the warning
    }
  } else if (key === ' ') {
    // Space is play/pause whenever a video is open, also on a focused button: the default click of a
    // focused "Submit" button would otherwise confirm a warned submission.
    if (state.video) {
      event.preventDefault();
      setKeyTarget('video');
      state.video.player.toggle();
    } else if (state.guard && event.target.closest?.(SUBMIT_BUTTONS)) {
      event.preventDefault();
    }
  } else if (KEY_ACTIONS[key.toLowerCase()] && !event.shiftKey && !event.repeat) {
    const action = KEY_ACTIONS[key.toLowerCase()];
    const pending = state.guard;
    const ref = keyTargetRef();
    // A pending answer or basket warning is confirmed by S; a KIS warning only for the same video
    // (performAction → submitKis → guardedSubmit decides), otherwise S checks the new target.
    if (action === 'submit' && pending && pending.phase !== 'checking' && (pending.kind !== 'KIS' || !ref)) {
      event.preventDefault();
      confirmPendingGuard();
      return;
    }
    if (!ref) {
      toast('info', 'Nothing selected', 'Point at a result or open a video first.', { key: 'no-target', duration: 2500 });
      return;
    }
    event.preventDefault();
    performAction(action, ref);
  }
}

// ---------------------------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------------------------

function init() {
  document.title = PRODUCT_NAME;
  $('#brand-name').textContent = PRODUCT_NAME;
  document.documentElement.classList.toggle('presentation', PRESENTATION);
  $('#mock-tag').hidden = !MOCK;
  document.addEventListener('error', (event) => {
    const image = event.target;
    if (image instanceof HTMLImageElement && image.src !== PLACEHOLDER_IMAGE) {
      image.src = PLACEHOLDER_IMAGE;
      image.classList.add('is-placeholder');
    }
  }, true);
  document.addEventListener('keydown', onKeyDown);
  // Submit buttons never take focus on click, so a later Space or Enter cannot press them again.
  document.addEventListener('mousedown', (event) => {
    if (event.target instanceof Element && event.target.closest(SUBMIT_BUTTONS)) event.preventDefault();
  });
  bindTopbar();
  bindQueryBuilder();
  bindResults();
  bindVideo();
  bindDrawer();
  bindChat();

  setQueryTab(state.queryTab);
  renderChat({ turns: true });
  renderGuard();
  renderScenes();
  renderWindow();
  renderMode();
  renderHistory();
  renderResults();
  renderBasket();
  renderVqa();
  renderSubmissions();
  renderDrawer();
  renderDres();

  setInterval(flushLog, LOG_FLUSH_MS);
  window.addEventListener('pagehide', flushLogOnPageHide);
  setInterval(renderTask, 1000);
  setInterval(() => { if (state.dres.connected) refreshEvaluations(); }, DRES_POLL_MS);
  loadStatus();
}

init();
