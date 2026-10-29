/**
 * Offline stand-in for the VBS tool HTTP API (docs/ARCHITECTURE.md §6).
 *
 * app.js loads this module only when the page URL carries `?mock=1` and then answers every /api
 * call through `mockFetch`. Nothing here talks to a network. Responses are fabricated but
 * internally consistent: every query gets a small generative score model (a few Gaussian "hot
 * spots" per scene plus hashed noise over each video's keyframes), and frame lists, video groups,
 * temporal chains and the per-scene score curves of the video view are all read from that model.
 * The chain search runs the same backward dynamic programme as `search.temporal_search`.
 *
 * Results are deterministic: the same scenes and window always give the same ranking. Video ids
 * and titles are invented; thumbnails point to /frames/<video_id>/<nnn>.jpg, and when no frame
 * server answers, app.js shows a placeholder image instead.
 */

const COLLECTION = { name: 'mock-demo', n_keyframes: 7139, n_videos: 36, dim: 1152 };
const ENCODER = 'google/siglip2-so400m-patch14-384';
const LLM_MODEL = 'gemini-2.5-flash';
const DRES_DEFAULT_URL = 'http://localhost:8080';

/** Simulated server latency per route, in milliseconds. */
const LATENCY_MS = { search: 260, similar: 180, video: 90, decompose: 450, vqa: 900, verify: 700, chat: 800, dres: 150, other: 40 };

/** [video_id, keyframes, fps, duration in seconds] of the videos that results are drawn from. */
const VIDEO_TABLE = [
  ['demo_01', 307, 30, 1257], ['demo_02', 292, 30, 1152], ['demo_03', 313, 30, 1293],
  ['demo_04', 87, 25, 368], ['demo_05', 156, 25, 521], ['demo_06', 149, 30, 507],
  ['demo_07', 374, 29.97, 1137], ['demo_08', 401, 25, 1428], ['demo_09', 356, 25, 1145],
  ['demo_10', 125, 25, 292], ['demo_11', 132, 25, 313], ['demo_12', 139, 25, 324],
  ['demo_13', 154, 25, 327], ['demo_14', 153, 25, 328], ['demo_15', 148, 25, 323],
  ['demo_16', 155, 25, 323], ['demo_17', 178, 25, 317], ['demo_18', 155, 25, 307],
  ['demo_19', 159, 25, 323], ['demo_20', 168, 25, 307], ['demo_21', 167, 25, 323],
  ['demo_22', 144, 25, 302], ['demo_23', 170, 25, 317], ['demo_24', 173, 25, 302],
  ['demo_25', 164, 25, 302], ['demo_26', 178, 25, 307], ['demo_27', 163, 25, 297],
  ['demo_28', 174, 25, 333], ['demo_29', 157, 25, 287], ['demo_30', 167, 25, 303],
  ['demo_31', 323, 25, 599], ['demo_32', 431, 25, 1123], ['demo_33', 445, 25, 1075],
  ['demo_34', 41, 25, 138], ['demo_35', 29, 25, 62], ['demo_36', 112, 25, 268],
];

const TITLES = [
  'Evening news bulletin', 'Morning regional report', 'Weekend current affairs',
  'Science and technology weekly', 'Travel and culture magazine', 'Agriculture and weather report',
];

/** Number of videos per text query that contain a planted, correctly ordered event. */
const RELEVANT_VIDEOS = 8;
const QUERY_CACHE_SIZE = 32;
const VIDEO_TOP_FRAMES = 500;

class MockHttpError extends Error {
  constructor(status, message) {
    super(message);
    this.name = 'MockHttpError';
    this.status = status;
  }
}

/** 32-bit FNV-1a followed by the MurmurHash3 finaliser, so that similar keys decorrelate. */
function hash(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** Deterministic pseudo-random number in [0, 1) derived from a key. */
function unit(key) {
  return hash(key) / 4294967296;
}

const round = (value, digits) => Number(value.toFixed(digits));
const delay = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

const VIDEOS = buildVideos();
const VIDEO_BY_ID = new Map(VIDEOS.map((video) => [video.id, video]));

function buildVideos() {
  let rowOffset = 0;
  return VIDEO_TABLE.map(([id, n, fps, duration]) => {
    const step = duration / (n - 1);
    const times = new Float64Array(n);
    for (let i = 1; i < n - 1; i += 1) {
      times[i] = i * step + (unit(`${id}:t:${i}`) - 0.5) * 0.6 * step;
    }
    times[n - 1] = duration;
    const h = hash(id);
    const video = {
      id, n, fps, duration, rowOffset, times,
      title: `${TITLES[h % TITLES.length]}, episode ${(h >>> 8) % 90 + 10}`,
    };
    rowOffset += n;
    return video;
  });
}

function thumbUrl(video, i) {
  return `/frames/${video.id}/${String(i + 1).padStart(3, '0')}.jpg`;
}

function frameObject(video, i, score) {
  return {
    row: video.rowOffset + i,
    video_id: video.id,
    frame_idx: Math.round(video.times[i] * video.fps),
    time: round(video.times[i], 2),
    score: round(score, 4),
    thumb: thumbUrl(video, i),
  };
}

// ---------------------------------------------------------------------------------------------
// Query score models
// ---------------------------------------------------------------------------------------------

const queries = new Map();
const scoreCache = new Map();
let querySequence = 0;

function registerQuery(spec) {
  querySequence += 1;
  const id = `q${String(querySequence).padStart(4, '0')}-${hash(spec.key).toString(16).padStart(8, '0').slice(0, 6)}`;
  queries.set(id, spec);
  if (queries.size > QUERY_CACHE_SIZE) queries.delete(queries.keys().next().value);
  return id;
}

function textSpec(scenes) {
  const key = scenes.map((scene) => scene.toLowerCase().replace(/\s+/g, ' ')).join(' | ');
  const ranked = VIDEOS.map((video) => [unit(`${key}|rank|${video.id}`), video.id]).sort((a, b) => a[0] - b[0]);
  const rank = new Map(ranked.slice(0, RELEVANT_VIDEOS).map(([, id], r) => [id, r]));
  return { kind: 'text', key, n: scenes.length, rank };
}

function similarSpec(video, i) {
  return { kind: 'similar', key: `similar|${video.id}|${i}`, n: 1, video: video.id, index: i };
}

/** Per-scene score arrays (one Float64Array per scene) of `video` under the query `spec`. */
function sceneScores(spec, video) {
  const cacheKey = `${spec.key}#${video.id}`;
  let scores = scoreCache.get(cacheKey);
  if (!scores) {
    scores = spec.kind === 'text' ? textScores(spec, video) : similarScores(spec, video);
    scoreCache.set(cacheKey, scores);
    if (scoreCache.size > 400) scoreCache.delete(scoreCache.keys().next().value);
  }
  return scores;
}

function textScores(spec, video) {
  const u = (name) => unit(`${spec.key}|${video.id}|${name}`);
  const spots = Array.from({ length: spec.n }, () => []);
  const r = spec.rank.get(video.id);
  if (r !== undefined) {
    const strength = 1 - r * 0.07;
    let t = 6 + u('start') * Math.max(1, video.duration - 12 - 26 * (spec.n - 1));
    for (let k = 0; k < spec.n; k += 1) {
      if (k > 0) t += 4 + u(`gap${k}`) * 18;
      spots[k].push({ t, amp: (0.09 + 0.05 * u(`amp${k}`)) * strength, sigma: 1.5 + 2.5 * u(`sigma${k}`) });
    }
  }
  for (let k = 0; k < spec.n; k += 1) {
    const extra = Math.floor(u(`extra${k}`) * 3);
    for (let h = 0; h < extra; h += 1) {
      spots[k].push({
        t: u(`xt${k}.${h}`) * video.duration,
        amp: 0.02 + 0.06 * u(`xa${k}.${h}`),
        sigma: 2 + 5 * u(`xs${k}.${h}`),
      });
    }
  }
  return spots.map((list, k) => curve(video, list, 0.035 + 0.03 * u(`base${k}`), 0.024, `${spec.key}|${video.id}|${k}`));
}

function similarScores(spec, video) {
  const u = (name) => unit(`${spec.key}|${video.id}|${name}`);
  if (video.id === spec.video) {
    const centre = video.times[spec.index];
    const values = curve(video, [{ t: centre, amp: 0.34, sigma: 8 }], 0.58, 0.05, spec.key);
    values[spec.index] = 1;
    return [values];
  }
  const spots = [];
  const extra = Math.floor(u('extra') * 3);
  for (let h = 0; h < extra; h += 1) {
    spots.push({ t: u(`xt${h}`) * video.duration, amp: 0.05 + 0.15 * u(`xa${h}`), sigma: 3 + 5 * u(`xs${h}`) });
  }
  return [curve(video, spots, 0.42 + 0.1 * u('base'), 0.05, `${spec.key}|${video.id}`)];
}

function curve(video, spots, base, noise, noiseKey) {
  const values = new Float64Array(video.n);
  for (let i = 0; i < video.n; i += 1) {
    const t = video.times[i];
    let s = base + noise * (unit(`${noiseKey}|${i}`) - 0.5);
    for (const spot of spots) s += spot.amp * Math.exp(-((t - spot.t) ** 2) / (2 * spot.sigma * spot.sigma));
    values[i] = s;
  }
  return values;
}

// ---------------------------------------------------------------------------------------------
// Ranking (mirrors search.py semantics on the mock score model)
// ---------------------------------------------------------------------------------------------

/** Best frames over the pool; with several scenes a keyframe scores its best scene. */
function rankFrames(spec, topK) {
  const hits = [];
  for (const video of VIDEOS) {
    const scores = sceneScores(spec, video);
    for (let i = 0; i < video.n; i += 1) {
      let best = -Infinity;
      for (const s of scores) best = Math.max(best, s[i]);
      hits.push([best, video, i]);
    }
  }
  hits.sort((a, b) => b[0] - a[0]);
  return hits.slice(0, topK).map(([score, video, i]) => frameObject(video, i, score));
}

function groupByVideo(frames, maxVideos, framesPerVideo) {
  const groups = new Map();
  for (const frame of frames) {
    let group = groups.get(frame.video_id);
    if (!group) {
      group = { video_id: frame.video_id, score: frame.score, hits: [] };
      groups.set(frame.video_id, group);
    }
    if (group.hits.length < framesPerVideo) group.hits.push(frame);
  }
  return [...groups.values()]
    .slice(0, maxVideos)
    .map((group) => ({ ...group, hits: group.hits.sort((a, b) => a.time - b.time) }));
}

function temporalSearch(spec, window, topK) {
  const [wMin, wMax] = window;
  const n = spec.n;
  const chains = [];
  for (const video of VIDEOS) {
    const scores = sceneScores(spec, video);
    const { times } = video;
    const next = [];
    let best = Float64Array.from(scores[n - 1]);
    for (let k = n - 2; k >= 0; k -= 1) {
      const current = new Float64Array(video.n).fill(-Infinity);
      const argmax = new Int32Array(video.n).fill(-1);
      for (let i = 0; i < video.n; i += 1) {
        let m = -Infinity;
        for (let j = i + 1; j < video.n && times[j] <= times[i] + wMax; j += 1) {
          if (times[j] > times[i] + wMin && best[j] > m) {
            m = best[j];
            argmax[i] = j;
          }
        }
        if (m > -Infinity) current[i] = scores[k][i] + m;
      }
      next.unshift(argmax);
      best = current;
    }
    const starts = [];
    for (let i = 0; i < video.n; i += 1) if (best[i] > -Infinity) starts.push([best[i] / n, i]);
    starts.sort((a, b) => b[0] - a[0]);
    const taken = [];
    for (const [score, first] of starts) {
      if (taken.length === 3) break;
      const path = [first];
      for (let k = 0; k < n - 1; k += 1) path.push(next[k][path[k]]);
      const span = [times[path[0]], times[path[n - 1]]];
      if (taken.some((other) => span[0] <= other[1] && other[0] <= span[1])) continue;
      taken.push(span);
      chains.push({
        video_id: video.id,
        score: round(score, 4),
        frames: path.map((i, k) => {
          const { row, frame_idx, time, thumb } = frameObject(video, i, scores[k][i]);
          return { row, frame_idx, time, score: round(scores[k][i], 4), thumb };
        }),
      });
    }
  }
  chains.sort((a, b) => b.score - a.score);
  return chains.slice(0, topK);
}

// ---------------------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------------------

const dres = { connected: false, user: null, url: null, evaluationId: null, loginAt: 0 };

function requireObject(body) {
  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    throw new MockHttpError(400, 'request body must be a JSON object');
  }
  return body;
}

function status() {
  return {
    collection: COLLECTION,
    encoder: ENCODER,
    dres: { connected: dres.connected, user: dres.user, evaluation_id: dres.evaluationId },
    llm: { gemini: true, openai: true },
  };
}

function search(body) {
  const { scenes, window = [0, 30], top_k: topK = 300, filter_videos: filterVideos } = requireObject(body);
  if (!Array.isArray(scenes) || scenes.length < 1 || scenes.length > 4
      || scenes.some((scene) => typeof scene !== 'string' || !scene.trim())) {
    throw new MockHttpError(400, 'scenes must be a list of 1 to 4 non-empty strings');
  }
  if (!Array.isArray(window) || window.length !== 2 || !(window[0] <= window[1])) {
    throw new MockHttpError(400, 'window must be [min_s, max_s] with min_s <= max_s');
  }
  if (filterVideos !== undefined) throw new MockHttpError(400, 'the mock server does not implement filter_videos');
  const mode = body.mode ?? (scenes.length === 1 ? 'frames' : 'chains');
  const spec = textSpec(scenes.map((scene) => scene.trim()));
  let payload;
  if (mode === 'frames') payload = { frames: rankFrames(spec, topK) };
  else if (mode === 'videos') payload = { videos: groupByVideo(rankFrames(spec, VIDEO_TOP_FRAMES), 60, 8) };
  else if (mode === 'chains') payload = { chains: temporalSearch(spec, window, topK) };
  else throw new MockHttpError(400, `unknown mode "${mode}"`);
  const rankMs = 30 + (hash(spec.key) % 60);
  const timings = { encode_ms: 80 * scenes.length, scan_ms: 10 * scenes.length, rank_ms: rankMs, cached_scenes: 0 };
  return { query_id: registerQuery(spec), took_ms: timings.encode_ms + timings.scan_ms + rankMs, mode, timings, ...payload };
}

function similar(body) {
  const { video_id: videoId, frame_idx: frameIdx, top_k: topK = 300 } = requireObject(body);
  const video = VIDEO_BY_ID.get(videoId);
  if (!video) throw new MockHttpError(404, `unknown video "${videoId}"`);
  let index = 0;
  for (let i = 0; i < video.n; i += 1) {
    if (Math.abs(video.times[i] * video.fps - frameIdx) < Math.abs(video.times[index] * video.fps - frameIdx)) index = i;
  }
  const spec = similarSpec(video, index);
  return { query_id: registerQuery(spec), took_ms: 60 + (hash(spec.key) % 40), frames: rankFrames(spec, topK) };
}

function videoView(videoId, queryId) {
  const video = VIDEO_BY_ID.get(videoId);
  if (!video) throw new MockHttpError(404, `unknown video "${videoId}"`);
  const spec = queryId ? queries.get(queryId) : undefined;
  const scores = spec ? sceneScores(spec, video) : null;
  return {
    video: {
      video_id: video.id, fps: video.fps, duration: video.duration, title: video.title, youtube_id: null, file: null,
    },
    keyframes: Array.from({ length: video.n }, (_, i) => {
      const { row, frame_idx, time, thumb } = frameObject(video, i, 0);
      return { row, frame_idx, time, thumb, scores: scores ? scores.map((s) => round(s[i], 4)) : null };
    }),
  };
}

const CUE_WORDS = /\b(?:then|afterwards|after that|later|next|finally|followed by|before)\b/i;

function decompose(body) {
  const { text } = requireObject(body);
  if (typeof text !== 'string' || !text.trim()) throw new MockHttpError(400, 'text must be a non-empty string');
  const scenes = text
    .split(/(?<=[.!?;])\s+/)
    .flatMap((sentence) => sentence.split(CUE_WORDS))
    .map((part) => part.replace(/^[\s,;:]*(?:and\s+)?/i, '').replace(/[\s,;:]*(?:and)?[\s,;:]*$/i, '').trim())
    .filter((part) => part.length >= 3);
  return { scenes: scenes.length ? scenes : [text.trim()], source: 'rule', model: null };
}

function vqa(body) {
  const { question, frames } = requireObject(body);
  if (typeof question !== 'string' || !question.trim()) throw new MockHttpError(400, 'question must be a non-empty string');
  if (!Array.isArray(frames) || frames.length < 1 || frames.length > 5) throw new MockHttpError(400, 'frames must hold 1 to 5 frames');
  const first = frames[0];
  const where = `${first.video_id}, frame ${first.frame_idx}`;
  if (/how many|number of/i.test(question)) {
    return { answer: '3', evidence: `Three people stand side by side in the centre of the image (${where}).`, model: LLM_MODEL };
  }
  if (/colou?r/i.test(question)) {
    return { answer: 'red', evidence: `The object in question is painted red and fills the lower left of the image (${where}).`, model: LLM_MODEL };
  }
  return { answer: 'a wooden desk', evidence: `A wooden desk with a lamp is visible behind the speaker (${where}).`, model: LLM_MODEL };
}

function dresLogin(body) {
  const { url, username, password } = requireObject(body);
  if (!username || !password) throw new MockHttpError(400, 'username and password are required');
  Object.assign(dres, { connected: true, user: username, url: url || DRES_DEFAULT_URL, evaluationId: null, loginAt: Date.now() });
  return { connected: true, user: username };
}

function requireDres() {
  if (!dres.connected) throw new MockHttpError(409, 'not connected to DRES');
}

function evaluations() {
  requireDres();
  const duration = 420;
  const elapsed = Math.floor((Date.now() - dres.loginAt) / 1000) + 131;
  return {
    evaluations: [
      {
        id: '7f3c2a10-mock-kis', name: 'Practice session: known-item search', status: 'ACTIVE',
        task: { name: 'kis-textual-practice-04', taskType: 'Textual KIS', duration, timeLeft: duration - (elapsed % duration) },
      },
      { id: 'b81e44d9-mock-avs', name: 'Practice session: ad-hoc video search', status: 'CREATED', task: null },
    ],
  };
}

function dresSelect(body) {
  requireDres();
  const { evaluation_id: evaluationId } = requireObject(body);
  if (!evaluations().evaluations.some((evaluation) => evaluation.id === evaluationId)) {
    throw new MockHttpError(404, `unknown evaluation "${evaluationId}"`);
  }
  dres.evaluationId = evaluationId;
  return { evaluation_id: evaluationId };
}

const VERDICTS = {
  CORRECT: 'Submission correct!',
  WRONG: 'Submission incorrect! Try again',
  INDETERMINATE: 'Submission received, awaiting judgement',
};

/** KIS moments judged WRONG in this mock session: video id -> [seconds] (the server keeps this per DRES task). */
const judgedWrong = new Map();
/** Mirrors verify.WRONG_RADIUS_S: a moment this close to a wrong one is that moment. */
const WRONG_RADIUS_S = 5;

function submit(body) {
  const request = requireObject(body);
  let verdict;
  if (request.kind === 'kis') {
    if (!VIDEO_BY_ID.has(request.video_id)) throw new MockHttpError(404, `unknown video "${request.video_id}"`);
    const time = request.time ?? (request.start_ms + request.end_ms) / 2000;
    if (!Number.isFinite(time)) throw new MockHttpError(400, 'kis submissions need "time" or "start_ms" and "end_ms"');
    verdict = ['CORRECT', 'WRONG', 'WRONG', 'CORRECT', 'INDETERMINATE'][hash(`${request.video_id}|${Math.round(time)}`) % 5];
    if (verdict === 'WRONG' && dres.connected && dres.evaluationId) {
      judgedWrong.set(request.video_id, [...(judgedWrong.get(request.video_id) ?? []), time]);
    }
  } else if (request.kind === 'text') {
    if (typeof request.text !== 'string' || !request.text.trim()) throw new MockHttpError(400, 'text must be a non-empty string');
    verdict = ['CORRECT', 'WRONG', 'INDETERMINATE'][hash(request.text.trim().toLowerCase()) % 3];
  } else if (request.kind === 'avs') {
    if (!Array.isArray(request.items) || !request.items.length) throw new MockHttpError(400, 'items must be a non-empty list');
    verdict = 'INDETERMINATE';
  } else {
    throw new MockHttpError(400, `unknown submission kind "${request.kind}"`);
  }
  if (!dres.connected) return { ok: false, status: 'not connected', raw: null };
  if (!dres.evaluationId) throw new MockHttpError(409, 'no DRES evaluation selected');
  return {
    ok: true,
    status: VERDICTS[verdict],
    raw: { status: true, submission: verdict, description: VERDICTS[verdict] },
  };
}

/** What the mock received, for browser tests: interaction events and requests with timings. */
export const receivedEvents = [];
export const requestLog = [];
const RECORD_LIMIT = 2000;

function log(body) {
  const { events } = requireObject(body);
  if (!Array.isArray(events)) throw new MockHttpError(400, 'events must be a list');
  receivedEvents.push(...events);
  if (receivedEvents.length > RECORD_LIMIT) receivedEvents.splice(0, receivedEvents.length - RECORD_LIMIT);
  return { accepted: events.length };
}

/** A keyframe score above this counts as "the scene is visible" in the mock score model. */
const VISIBLE_SCORE = 0.105;

const shorten = (text, max = 48) => (text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text);

function mmss(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

/**
 * Pre-submission check. Mirrors verify.verify_moment: "fail" when `time` is within WRONG_RADIUS_S of a
 * moment judged wrong (other moments of that video only get a note), otherwise the frames at `time`
 * and two keyframes either side are compared with the scenes.
 * Here "comparing" reads the mock score model: a scene counts as visible when one of those five
 * keyframes scores above VISIBLE_SCORE. Empty `scenes` checks the verdict history only.
 */
function verify(body) {
  const { scenes, video_id: videoId, time } = requireObject(body);
  if (!Array.isArray(scenes) || scenes.length > 4 || scenes.some((scene) => typeof scene !== 'string')) {
    throw new MockHttpError(400, 'scenes must be a list of at most 4 strings');
  }
  const video = VIDEO_BY_ID.get(videoId);
  if (!video) throw new MockHttpError(404, `unknown video "${videoId}"`);
  if (!Number.isFinite(time)) throw new MockHttpError(400, 'time must be a number of seconds');
  const wrong = judgedWrong.get(videoId) ?? [];
  const near = wrong.filter((at) => Math.abs(at - time) <= WRONG_RADIUS_S);
  if (near.length) {
    return {
      verdict: 'fail',
      reasons: [`${videoId} was already judged wrong for this task at ${near.map(mmss).join(', ')}; ${mmss(time)} is within ${WRONG_RADIUS_S} s of it.`],
      model: null,
      took_ms: 2,
      rejected: true,
      notes: [],
    };
  }
  const notes = wrong.length
    ? [`${videoId} already had a wrong submission in this task at ${wrong.map(mmss).join(', ')}; this moment is farther away and was checked as usual.`]
    : [];
  return { ...verifyFrames(scenes, video, videoId, time), rejected: false, notes };
}

/** The frame comparison of verify(), for a moment the verdict history does not block. */
function verifyFrames(scenes, video, videoId, time) {
  const texts = scenes.map((scene) => scene.trim()).filter(Boolean);
  if (!texts.length) return { verdict: 'ok', reasons: [], model: null, took_ms: 1 };
  const tookMs = 520 + (hash(`${videoId}|${time}|took`) % 700);
  let index = 0;
  for (let i = 0; i < video.n; i += 1) if (Math.abs(video.times[i] - time) < Math.abs(video.times[index] - time)) index = i;
  const lo = Math.max(0, index - 2);
  const hi = Math.min(video.n - 1, index + 2);
  if (texts.length === 1 && /\?\s*$/.test(texts[0])) {
    // A VQA question as the description: the mock only sometimes finds the asked-about content.
    const seen = hash(`${texts[0].toLowerCase()}|${videoId}|${index}`) % 3 !== 0;
    return seen
      ? { verdict: 'ok', reasons: [], model: LLM_MODEL, took_ms: tookMs }
      : { verdict: 'warn', reasons: [`What the question asks about is not clearly visible in the 5 frames around ${mmss(time)}.`], model: LLM_MODEL, took_ms: tookMs };
  }
  const scores = sceneScores(textSpec(texts), video);
  const best = scores.map((values) => Math.max(...values.slice(lo, hi + 1)));
  if (best.some((score) => score >= VISIBLE_SCORE)) return { verdict: 'ok', reasons: [], model: LLM_MODEL, took_ms: tookMs };
  const reasons = texts.slice(0, 3).map((text, k) => `Scene ${k + 1} "${shorten(text)}" is not visible in the 5 frames around ${mmss(time)}.`);
  return { verdict: 'warn', reasons, model: LLM_MODEL, took_ms: tookMs };
}

const STOP_WORDS = new Set(('a an the and or of in on at to is are was were be been it its this that there with for from by as '
  + 'what which who whom how many much does do did can you see any some into onto than then').split(' '));

const contentWords = (text) => new Set(text.toLowerCase().match(/[a-z]+/g)?.filter((word) => word.length > 2 && !STOP_WORDS.has(word)) ?? []);

/** Questions that usually separate candidate videos; the mock suggests the first one not yet asked. */
const NEXT_QUESTIONS = [
  'Is there readable text or a channel logo on screen in the first scene?',
  'How many people are visible when the first scene starts?',
  'Is the first scene filmed indoors or outdoors?',
  'Does the camera move during the second scene, or is it still?',
  'What colour is the largest object in the second scene?',
  'Is a news presenter shown in a studio before these scenes?',
];

/**
 * KIS-C dialogue → scenes, as conversation.update_from_dialogue does with a language model:
 * scenes come from the initial description (or the current scene boxes); every informative oracle
 * answer is attached to the scene that shares most words with its question, or to all scenes when
 * none does (setting, weather, time of day). Bare yes/no answers add nothing.
 */
function chat(body) {
  const { initial = '', turns = [], scenes = [] } = requireObject(body);
  if (typeof initial !== 'string') throw new MockHttpError(400, 'initial must be a string');
  if (!Array.isArray(turns) || turns.some((turn) => !turn || !['team', 'oracle'].includes(turn.role) || typeof turn.text !== 'string')) {
    throw new MockHttpError(400, 'turns must be a list of {"role": "team"|"oracle", "text": str}');
  }
  if (!Array.isArray(scenes) || scenes.some((scene) => typeof scene !== 'string')) throw new MockHttpError(400, 'scenes must be a list of strings');
  let base = initial.trim() ? decompose({ text: initial }).scenes : scenes.map((scene) => scene.trim()).filter(Boolean);
  if (!base.length) throw new MockHttpError(400, 'initial or scenes must describe at least one scene');
  if (base.length > 4) base = [...base.slice(0, 3), base.slice(3).join(' ')];
  const out = base.map((scene) => scene.replace(/[.\s]+$/, ''));
  let question = '';
  for (const turn of turns) {
    const text = turn.text.trim();
    if (turn.role === 'team') {
      question = text;
      continue;
    }
    const fact = text.replace(/^(?:yes|no)\b[,.!]?\s*/i, '').replace(/[.!\s]+$/, '');
    if (!fact) continue;
    const words = contentWords(question);
    const overlap = out.map((scene) => [...contentWords(scene)].filter((word) => words.has(word)).length);
    const top = Math.max(...overlap);
    const targets = top > 0 ? [overlap.lastIndexOf(top)] : out.map((_, k) => k);
    const wearing = /\b(\w+) wearing\b/i.exec(question);
    const phrase = wearing ? `${wearing[1].toLowerCase()} wearing ${fact.charAt(0).toLowerCase()}${fact.slice(1)}` : `${fact.charAt(0).toLowerCase()}${fact.slice(1)}`;
    for (const k of targets) out[k] = `${out[k]}, ${phrase}`;
    question = '';
  }
  const asked = turns.filter((turn) => turn.role === 'team').map((turn) => turn.text.trim().toLowerCase());
  const start = hash(`${initial}|${turns.length}`) % NEXT_QUESTIONS.length;
  const nextQuestion = Array.from({ length: NEXT_QUESTIONS.length }, (_, i) => NEXT_QUESTIONS[(start + i) % NEXT_QUESTIONS.length])
    .find((candidate) => !asked.includes(candidate.toLowerCase())) ?? NEXT_QUESTIONS[start];
  return { scenes: out, next_question: nextQuestion, model: LLM_MODEL };
}

function route(method, url, body) {
  const path = url.pathname;
  const videoMatch = /^\/api\/video\/([^/]+)$/.exec(path);
  if (method === 'GET' && videoMatch) {
    return [LATENCY_MS.video, () => videoView(decodeURIComponent(videoMatch[1]), url.searchParams.get('query_id'))];
  }
  const handlers = {
    'GET /api/status': [LATENCY_MS.other, status],
    'POST /api/search': [LATENCY_MS.search, () => search(body)],
    'POST /api/similar': [LATENCY_MS.similar, () => similar(body)],
    'POST /api/decompose': [LATENCY_MS.decompose, () => decompose(body)],
    'POST /api/vqa': [LATENCY_MS.vqa, () => vqa(body)],
    'POST /api/dres/login': [LATENCY_MS.dres, () => dresLogin(body)],
    'GET /api/dres/evaluations': [LATENCY_MS.dres, evaluations],
    'POST /api/dres/select': [LATENCY_MS.dres, () => dresSelect(body)],
    'POST /api/submit': [LATENCY_MS.dres, () => submit(body)],
    'POST /api/log': [LATENCY_MS.other, () => log(body)],
    'POST /api/verify': [LATENCY_MS.verify, () => verify(body)],
    'POST /api/chat': [LATENCY_MS.chat, () => chat(body)],
  };
  return handlers[`${method} ${path}`] ?? [LATENCY_MS.other, () => { throw new MockHttpError(404, `no route ${method} ${path}`); }];
}

/**
 * Answer one API request.
 * @param {string} method HTTP method.
 * @param {string} path Request path including the query string, e.g. "/api/video/demo_01?query_id=q0001".
 * @param {unknown} body Parsed JSON body, or undefined for GET.
 * @returns {Promise<{status: number, body: object}>} HTTP status and JSON payload ({"error"} on failure).
 */
export async function mockFetch(method, path, body) {
  const [latency, handler] = route(method, new URL(path, 'http://mock.invalid'), body);
  const record = { method, path, body, start: performance.now(), end: null };
  requestLog.push(record);
  if (requestLog.length > RECORD_LIMIT) requestLog.splice(0, requestLog.length - RECORD_LIMIT);
  await delay(latency);
  try {
    return { status: 200, body: handler() };
  } catch (error) {
    if (error instanceof MockHttpError) return { status: error.status, body: { error: error.message } };
    throw error;
  } finally {
    record.end = performance.now();
  }
}
