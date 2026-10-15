// Activity page: a webcam (MediaPipe Pose, in the browser) teaches the WiFi
// classifier; the server scores it honestly on a separately recorded test session.
//
// Adapted from the wifi-sensing-kit movement page by Andris Gauracs (MIT licence)
// https://github.com/andrisgauracs/wifi-sensing-kit
//
// Privacy: video frames and landmarks stay in this page. The only thing sent to
// the server is a movement label string (see postLabel).

import { apiService } from '../../services/api.service.js';
import { MovementClassifier, MOVES, UNSURE, BONES, JOINTS } from './movement-rules.js';

// Pinned exact versions; the wasm runtime must match the JS bundle version.
const MEDIAPIPE_VERSION = '0.10.14';
const MEDIAPIPE_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
// The model weights are not on npm; Google hosts the official, versioned (/1/) file.
const POSE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

const LABEL_POST_MS = 200;
const STATUS_POLL_MS = 1000;
const API = '/api/v1/activity';
const NAMES = Object.fromEntries(MOVES.map((m) => [m.label, m.name]));
const pretty = (l) => NAMES[l] || l;
const pct = (x) => `${(100 * x).toFixed(1)}%`;

const $ = (id) => document.getElementById(id);
const video = $('video');
const skeleton = $('skeleton');

/** Build an element with text content (never innerHTML: labels come from the server). */
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
const setError = (id, err) => { $(id).textContent = err ? String(err.message || err) : ''; };

// ---------------------------------------------------------------- camera ----

const classifier = new MovementClassifier();
let landmarker = null;
let camLabel = UNSURE;
let lastPost = 0;
let running = false;

function drawSkeleton(lm) {
  const ctx = skeleton.getContext('2d');
  skeleton.width = video.videoWidth || 640;
  skeleton.height = video.videoHeight || 480;
  ctx.clearRect(0, 0, skeleton.width, skeleton.height);
  if (!lm) return;
  const visible = (i) => (lm[i].visibility ?? 1) > 0.5;
  ctx.strokeStyle = '#b9c0cf';
  ctx.fillStyle = '#eef1f8';
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  for (const [a, b] of BONES) {
    if (!visible(a) || !visible(b)) continue;
    ctx.beginPath();
    ctx.moveTo(lm[a].x * skeleton.width, lm[a].y * skeleton.height);
    ctx.lineTo(lm[b].x * skeleton.width, lm[b].y * skeleton.height);
    ctx.stroke();
  }
  for (const i of JOINTS) {
    if (!visible(i)) continue;
    ctx.beginPath();
    ctx.arc(lm[i].x * skeleton.width, lm[i].y * skeleton.height, 6, 0, 2 * Math.PI);
    ctx.fill();
  }
}

/** Only the label string leaves the browser. "unsure" is never sent. */
function postLabel(label, now) {
  if (label === UNSURE || now - lastPost < LABEL_POST_MS) return;
  lastPost = now;
  apiService.post(`${API}/label`, { label }).catch((e) => console.debug('label post failed:', e.message));
}

function onFrame(lm, now) {
  const label = classifier.push(lm, now);
  if (label !== null) {
    camLabel = label;
    $('cam-label').textContent = label === UNSURE ? 'not sure' : pretty(label);
  }
  if (!$('stage').classList.contains('covered')) postLabel(camLabel, now);
}

async function startCamera() {
  $('btn-camera').disabled = true;
  setError('cam-error', null);
  try {
    $('cam-status').textContent = 'loading pose model...';
    const { PoseLandmarker, FilesetResolver } = await import(`${MEDIAPIPE_BASE}/vision_bundle.mjs`);
    const fileset = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_BASE}/wasm`);
    landmarker = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: 'GPU' },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    await video.play();
    $('cam-status').textContent = 'pose model running locally';
    running = true;
    let lastT = -1;
    const tick = () => {
      if (!running) return;
      const now = performance.now();
      if (video.readyState >= 2 && video.currentTime !== lastT) {
        lastT = video.currentTime;
        const res = landmarker.detectForVideo(video, now);
        const lm = res.landmarks && res.landmarks[0];
        drawSkeleton(lm);
        onFrame(lm, now);
      }
      requestAnimationFrame(tick);
    };
    tick();
    $('btn-camera').textContent = 'Camera on';
  } catch (e) {
    $('cam-status').textContent = 'camera off';
    setError('cam-error', `Camera / pose model error: ${e.message}`);
    $('btn-camera').disabled = false;
  }
}

$('btn-camera').addEventListener('click', startCamera);
$('btn-cover').addEventListener('click', () => {
  const on = $('stage').classList.toggle('covered');
  $('btn-cover').textContent = on ? 'Show camera' : 'Cover camera';
});

// -------------------------------------------------------------- sessions ----

async function call(fn, errId) {
  setError(errId, null);
  try {
    return await fn();
  } catch (e) {
    setError(errId, e);
    return null;
  }
}

const startSession = (kind) => call(() => apiService.post(`${API}/session/start`, { kind }), 'session-error').then(refresh);
const stopSession = () => call(() => apiService.post(`${API}/session/stop`), 'session-error').then(refresh);
const deleteSession = (id) =>
  call(() => apiService.delete(`${API}/sessions/${encodeURIComponent(id)}`), 'session-error').then(refresh);

$('btn-train').addEventListener('click', () => startSession('train'));
$('btn-test').addEventListener('click', () => startSession('test'));
$('btn-stop').addEventListener('click', stopSession);

$('btn-model').addEventListener('click', async () => {
  const r = await call(() => apiService.post(`${API}/train`, { k: Number($('k').value) || 7 }), 'model-error');
  if (r) refresh();
});

$('btn-eval').addEventListener('click', async () => {
  setError('eval-error', null);
  const r = await call(() => apiService.post(`${API}/evaluate`, { k: Number($('k').value) || 7 }), 'eval-error');
  if (r && r.evaluation) renderEvaluation(r.evaluation);
});

function renderSessions(s) {
  const body = $('sessions').querySelector('tbody');
  body.replaceChildren();
  const recId = s.recording && s.recording.id;
  for (const x of s.sessions) {
    const tr = el('tr');
    tr.append(el('td', x.id), el('td', x.kind), el('td', String(x.windows)));
    tr.append(el('td', Object.entries(x.labels).map(([l, n]) => `${pretty(l)} ${n}`).join(', ') || '-'));
    const td = el('td');
    const del = el('button', 'delete', 'btn small');
    del.addEventListener('click', () => deleteSession(x.id));
    del.disabled = x.id === recId;
    td.append(del);
    tr.append(td);
    body.append(tr);
  }
  const rec = s.recording;
  $('btn-stop').disabled = !rec;
  $('btn-train').disabled = !!rec;
  $('btn-test').disabled = !!rec;
  $('rec-banner').textContent = rec
    ? `Recording ${rec.kind} session: ${rec.windows} windows saved, ${rec.skipped} skipped (${rec.no_label} no confident camera label, ${rec.no_wifi} no WiFi)`
    : '';
  const usable = s.links.filter((l) => l.usable).length;
  $('links-info').textContent = `${usable} of ${s.links.length} links streaming at >= ${s.min_link_rate_hz} frames/s`;
}

// ------------------------------------------------------------ evaluation ----

function renderEvaluation(ev) {
  $('eval-empty').classList.add('hidden');
  $('eval').classList.remove('hidden');
  const v = $('eval-verdict');
  v.replaceChildren(
    el('b', pct(ev.accuracy)),
    document.createTextNode(` on ${ev.n_test} held-out test windows (${ev.test_sessions.join(', ')}) `),
    el('span', ev.evidence_tag, 'tag'),
    document.createElement('br'),
    document.createTextNode(
      `guessing: ${pct(ev.chance_baseline)} (1 / ${ev.labels.length} classes) | always "${pretty(ev.majority_class)}": ${pct(ev.majority_baseline)}`
        + ` | trained on ${ev.n_train} windows, k=${ev.k}${ev.subsampled ? ' (subsampled)' : ''}`,
    ),
  );
  const t = $('confusion');
  t.replaceChildren();
  const head = el('tr');
  head.append(el('th', 'camera \\ WiFi'));
  for (const l of ev.labels) head.append(el('th', pretty(l)));
  head.append(el('th', 'recall'));
  t.append(head);
  for (const truth of ev.labels) {
    const row = ev.confusion[truth];
    if (!row) continue;
    const tr = el('tr');
    tr.append(el('td', pretty(truth)));
    for (const pred of ev.labels) {
      const n = row[pred] || 0;
      tr.append(el('td', String(n), n === 0 ? 'zero' : pred === truth ? 'hit' : 'miss'));
    }
    tr.append(el('td', pct(ev.per_class[truth].recall)));
    t.append(tr);
  }
  const li = $('link-importance');
  li.replaceChildren();
  if (!ev.link_importance.length) {
    const tr = el('tr');
    tr.append(el('td', 'needs 2 or more links'));
    li.append(tr);
  } else {
    const h = el('tr');
    h.append(el('th', 'link'), el('th', 'accuracy without'), el('th', 'drop'));
    li.append(h);
    for (const x of [...ev.link_importance].sort((a, b) => b.accuracy_drop - a.accuracy_drop)) {
      const tr = el('tr');
      tr.append(el('td', x.link), el('td', pct(x.accuracy_without)), el('td', pct(x.accuracy_drop)));
      li.append(tr);
    }
  }
  $('eval-note').textContent = ev.excluded_sessions.length
    ? `${ev.note} Excluded (different link set): ${ev.excluded_sessions.join(', ')}.`
    : ev.note;
}

// ------------------------------------------------------------ live WiFi ----

function renderLive(p) {
  const ans = $('wifi-answer');
  const bars = $('wifi-bars');
  if (!p.best) {
    ans.textContent = '-';
    ans.className = 'big-answer unsure';
    $('wifi-sub').textContent = {
      no_model: 'no model yet: record a training session and press Train model',
      waiting_for_links: 'waiting for WiFi links',
    }[p.status] || p.status;
    bars.replaceChildren();
    return;
  }
  const sure = p.status === 'ok';
  ans.textContent = sure ? pretty(p.best) : 'Not sure';
  ans.className = sure ? 'big-answer' : 'big-answer unsure';
  $('wifi-sub').textContent = `${Math.round(100 * p.confidence)}%${sure ? '' : ` | maybe ${pretty(p.best)}`} | last ${p.window_secs} s`;
  bars.replaceChildren();
  for (const [label, prob] of Object.entries(p.probs)) {
    const row = el('div', undefined, label === p.best ? 'bar-row top' : 'bar-row');
    const bar = document.createElement('progress');
    bar.max = 1;
    bar.value = prob;
    row.append(el('div', pretty(label), 'name'), bar, el('div', `${Math.round(100 * prob)}%`, 'val'));
    bars.append(row);
  }
}

async function refresh() {
  try {
    const s = await apiService.get(`${API}/status`);
    renderSessions(s);
    const live = await apiService.get(`${API}/live`);
    renderLive(live.prediction);
  } catch (e) {
    $('wifi-sub').textContent = `server unreachable: ${e.message}`;
  }
}

// Self-scheduling poll: the next request starts only after the previous one finished.
async function pollLoop() {
  await refresh();
  setTimeout(pollLoop, STATUS_POLL_MS);
}
pollLoop();
