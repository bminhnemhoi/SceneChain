# SceneChain — architecture and API contract

This document is the design contract of SceneChain, the interactive tool the team brings to the
Video Browser Showdown 2027 (MMM 2027, Siem Reap, 5 January 2027). Every module and the web UI
are written against the contract below. If an implementation needs to deviate, change this file
first and say why in the change.

Status: research prototype (development started 17 Sep 2026). Backend modules, the FastAPI server,
the web UI, the submission guard and the KIS-C chat tab exist and are tested. Measurements quoted
below were taken on the authors' local copy of the AIC 2026 collection, which is not distributed
with this repository; `collections/aic2026` names that local copy.

## 1. Goals and non-goals

**Goals for the paper deadline (22 Sep 2026)**
1. A live, English, browser-based search tool over the existing AIC 2026 collection
   (873 videos, 177,321 keyframes, SigLIP-2 so400m-384 index) that supports the VBS task types:
   KIS-T (textual known-item), KIS-V (visual), VQA and AVS.
2. Multi-scene temporal queries ("scene A, then scene B") with automatic decomposition of a long
   textual description into ordered scenes, grounded in the AIC 2026 finding that 51% of real
   queries describe consecutive scenes.
3. A DRES v2 client (login, evaluation/task discovery, KIS/VQA/AVS submission, query and result
   logging) exercised against a mock server in tests.
4. A dataset-agnostic ingestion path (videos → shots → keyframes → embeddings → collection) so the
   same server runs on V3C / Marine Video Kit / GynSurg without code changes.

5. **Pre-submission verification** (added 17 Sep after reading the DRES scorers and the VBS 2026
   archive): DRES `KisTaskScorer` subtracts 100/1000 for every wrong submission before the first
   correct one, and VBS 2026 KIS-T saw 264 wrong vs 94 correct submissions. Before a KIS/VQA
   submission the tool runs a fast check (VLM contradiction check on frames around the moment,
   plus "this video was already judged wrong" from the DRES verdict history) and requires an
   explicit second key press when the check fails.
6. **KIS-C conversation panel**: the oracle's answers are typed into a chat; an LLM folds the dialogue
   into an updated scene list, re-runs the search, and suggests the next question that best separates
   the current top candidate videos.

**Non-goals now:** V3C-scale indexing (planned for Oct–Dec 2026; FAISS IVF-PQ / int8 is the planned
ANN path), multi-user collaboration, model training, autonomous agent mode (roadmap).

## 2. Hard rules

- Text embeddings reproduce the measured AIC 2026 query encoding: the SigLIP-2 text tower and its
  long-text chunking live in `scenechain/siglip_text.py` (vendored from the authors' AIC pipeline),
  the 4-prompt ensemble in `scenechain/encoder.py`.
- The collection layer must not assume AIC naming. A collection directory is described in §3.
- Nothing from the AIC organisers' data (videos, keyframes, embeddings, OCR/ASR, query files,
  answers) may enter the repository. The repository is public; test fixtures are synthetic.
- All files UTF-8 without BOM, LF line endings.
- Tests must not load the 817 MB index or the SigLIP-2 model and must not reach the network: use the
  synthetic collection fixture and the fake encoder from `tests/conftest.py`, and inject LLM /
  VQA / DRES clients (or monkeypatch `decompose.default_llm`, `vqa.default_providers`,
  `verify.default_providers`), because the defaults read the real keys from `.env`.
- Every user-visible string in the web UI is English.
- Secrets: API keys, the DRES password and the DRES session token are never printed or logged.
  `DresClient.login()` returns the `ApiUser` including `sessionId`; callers must not log that dict.
  Keep `urllib3` at WARNING (at DEBUG it prints full URLs including `?session=`).

## 3. Collection format (on disk)

A collection is a directory:

```
<collection>/
  collection.json        {"name": str, "model": "google/siglip2-so400m-patch14-384", "dim": 1152,
                          "embeddings": "embeddings.npy", "metadata": "metadata.json",
                          "frames_dir": "frames" | absolute path, "frame_url": optional URL template,
                          "videos": "videos.json", "invalid_rows": optional "invalid_rows.json",
                          "encode_frames_dir": optional, larger copies of the keyframes (same relative
                          names) read only by encode_images.py; the server keeps serving frames_dir}
  embeddings.npy         (N, D) float32 or float16, L2-normalised rows, row i <-> metadata[i]
  metadata.json          list of N objects: {"video_id": str, "n": int (1-based keyframe number within video),
                          "frame_idx": int, "pts_time": float seconds, "fps": float,
                          "frame_filename": str (relative to frames_dir/<video_id>/)}
  videos.json            {video_id: {"fps": float, "duration": float seconds, "n_frames": int,
                          "title": str, "youtube_id": str | null, "file": str | null (local video path/URL)}}
  frames/<video_id>/<frame_filename>   keyframe images (any size; the UI shows them scaled)
```

- `frame_url` template placeholders: `{video_id}`, `{frame_filename}`, `{frame_idx}`, `{n}`. Without
  a template the URL is `/frames/{video_id}/{frame_filename}`.
- Rows need not be contiguous per video nor sorted by time; the loader builds a per-video time order.
- `file`: an `http(s)://` URL is passed to the UI unchanged; anything else is a local path (relative
  to the collection directory) that the server streams at `/media/{video_id}` (with Range support).
- Frame file names and video ids used in `/frames` URLs must consist of letters, digits, `.`, `_`, `-`
  and must not start with `.` (the server rejects anything else, see §6).

The AIC adapter (`scenechain/ingest/build_from_aic.py`) writes `collection.json`, `videos.json` and a
`metadata.json` view into `collections/aic2026/` and points `embeddings`, `frames_dir` at the
existing files in `data/` (no copy of the 817 MB matrix; paths may be absolute). `collections/`,
`cache/` and `logs/` are gitignored. For AIC, `n_frames = last keyframe frame_idx + 1` and
`duration = n_frames / fps` (the YouTube length in the media-info zip is not used: it ends between
0.3 s before and 8.4 s after the last keyframe).

Rows marked invalid (first two keyframes of each AIC video, `data/blank_frame_indices.json`) are
excluded through the optional `"invalid_rows"` list. On AIC 2026: 1,903 invalid rows
(1,746 first-two keyframes + 168 blank frames, 11 in both).

## 4. Python package `scenechain`

| Module | Responsibility | Key API |
|---|---|---|
| `config.py` | settings from env / `.env` at repo root | `Settings(collection_dir, host, port, dres_url, dres_user, dres_password, dres_collection, gemini_key, openai_key, log_dir)` (frozen; secrets excluded from `repr`); `load_settings(env_file=REPO_ROOT/.env, environ=None, **overrides)`. Precedence: overrides (None ignored) > process env > `.env` > defaults; `.env` is parsed, never copied into `os.environ`. Variables: `VBS_COLLECTION` (default `collections/aic2026`), `VBS_HOST` (`127.0.0.1`), `VBS_PORT` (`8787`), `VBS_LOG_DIR` (`logs`), `DRES_URL`, `DRES_USER`, `DRES_PASSWORD`, `DRES_COLLECTION` (collection name sent with DRES answers; unset = not sent), `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENAI_API_KEY`. |
| `collection.py` | load a collection (mmap embeddings), video table, row lookup | `Collection.load(dir)` (raises `FileNotFoundError` / `CollectionError`); `.n`, `.dim`, `.n_videos`, `.name`, `.video_ids`; `.row(video_id, frame_idx) -> int|None`; `.rows_of(video_id) -> np.ndarray` (sorted by time; `KeyError` if unknown); `.video(video_id) -> dict`; `.video_position(video_id)`; `.video_id_of(row)`; `.meta(row)`; `.frame_path(row)`; `.frame_url(row) -> str`; `.embedding(row)` float32; `.similarities(vec) -> np.ndarray (N,)` float32, chunked; `.valid` read-only bool mask; arrays `times`, `frame_idx`, `n_in_video`, `row_video`, `time_order`, `video_starts`. |
| `encoder.py` | text → vector | `TextEncoder` protocol: `name`, `encode(text) -> (D,)`, `encode_chunks(text) -> list[(D,)]`, L2-normalised. `SiglipTextEncoder(model_name=DEFAULT_MODEL, device=None, engine=None)` wraps the `siglip_text.SiglipTextTower` text tower (the index is never opened) with the 4-prompt ensemble and the English text in both language slots; long text is chunked by `SiglipTextTower.chunk_text` (first chunk keeps the full text in its second slot, as in the AIC pipeline); all prompts of all chunks go through one forward pass; `engine=` injects a prepared `SiglipTextTower`. Mean of chunk similarities is done in `search.py`. `FakeEncoder` lives in tests only. |
| `llm.py` | shared LLM clients for decompose / vqa / verify / conversation | `LLMClient` protocol: `generate(prompt, parse, images: Sequence[bytes]=()) -> (parse(reply), model)`; a `ValueError` from `parse` counts as a failure of that model. `GeminiClient(api_key, models=GEMINI_MODELS, client=None, timeout_s=45, max_output_tokens=2048)`: model order `GEMINI_MODELS` (`DEFAULT_MODEL` then `FALLBACK_MODELS`), JSON output, temperature 0, automatic function calling off, never sleeps or retries (any failure moves to the next model; a model at its daily quota is skipped for the rest of the process); raises `LLMError`. `OpenAIClient(api_key, model="gpt-5.2", client=None)` (needs the `openai` SDK, not installed on the team laptop; `sdk_available()`); `optional_openai(key)`; `extract_json_object(text)`. |
| `search.py` | all ranking logic, pure numpy, no I/O | see §5 |
| `decompose.py` | long description → ordered scenes | `decompose(text, *, llm=None, cache_dir=cache/decompose) -> {"scenes": [str], "source": "llm"|"rule", "model": str|None}`; `llm=None` uses `default_llm()` (one Gemini client per process from the settings, `None` without a key). Disk cache keyed by sha1(stripped text), entries carry `prompt_version`; only LLM splits are cached; `cache_dir=None` disables it. Blank text → `scenes: []`. Rule fallback (`split_scenes_by_rule`): a new scene starts only at a temporal cue (then, afterwards, after that, later, next, finally, followed by, before); a sentence without a cue is more detail of the current scene; "before that/this" puts the new scene in front of the previous one; anything past 4 scenes is merged into the 4th. Never raises. |
| `vqa.py` | answer a question about one or more frames | `answer(question, image_paths, *, providers=None) -> {"answer", "evidence", "model"}`; `providers=None` uses `default_providers()` (Gemini, then OpenAI when key and SDK exist). Images re-encoded as JPEG, long side ≤ 1024. An empty or non-JSON reply counts as a failure (next model); an empty answer is never returned (always guess — AIC lesson). Raises `ValueError` (empty question, not 1..5 images), `FileNotFoundError`, `VQAUnavailable` (no provider configured or none answered). |
| `dres.py` | DRES v2 REST client | Follows `doc/oas-client.json` at DRES tag `v2.0.5` (commit `bd600c8d`); every path/schema used is identical in release 2.0.4, which was run live. `DresClient(base_url, timeout=10.0, http=None)`; `.login(user, pw) -> ApiUser` (incl. `sessionId`; a refused or failed login keeps the previous session, which is replaced only on success); `.logout()`; `normalize_base_url(url)` (the form `base_url` is stored in); properties `logged_in`, `username`, `role`; `.evaluations() -> list[dict]`; `.task_info(evaluation_id) -> dict|None` (`None` when DRES answers 404 "no active task"; an unknown evaluation still raises); `.evaluation_state(evaluation_id)` (`taskStatus`, `taskId`, `timeLeft`, `timeElapsed` in seconds); `.submit(evaluation_id, answer_sets)`; `.submit_kis(evaluation_id, media_item, start_ms, end_ms, collection=None)`; `.submit_text(evaluation_id, text)`; `.submit_avs(evaluation_id, items: list[(media_item, start_ms, end_ms)], collection=None)` (one answer set per shot); `.log_query(evaluation_id, events, timestamp_ms=None)`; `.log_results(evaluation_id, result_log)`. Module-level payload builders `query_event(category, type, value, timestamp_ms)`, `result_log(answers, sort_type=, result_set_availability=, events=, timestamp_ms=)`, `temporal_answer`, `text_answer`. Returns parsed JSON; raises `DresError(status, description, method, path)` (token and password removed); invalid arguments raise `ValueError` before any request; nothing is retried (a repeated submission counts twice); no v1 GET submission fallback (no 2.0.x release has it). |
| `verify.py` | pre-submission check | `verify_moment(collection, scenes, video_id, time_s, rejected=frozenset(), *, providers=None, cache_dir=cache/verify, budget_s=BUDGET_S) -> {"verdict": "ok"|"warn"|"fail", "reasons": [str], "model": str|None, "took_ms": int, "rejected": bool}`. `rejected` (argument): videos DRES judged WRONG for the current task, a set of ids or ids mapped to the submitted times. "fail" with `rejected: true` and the reason "<video> was already judged wrong for this task (submitted at m:ss.s)" if the video is in it (instant, no model call). No scene text (KIS-V, AVS basket items) → "ok" after the history check, no model call. Otherwise the valid keyframe nearest to `time_s` and up to 2 valid keyframes on each side (JPEG, long side 768) go to the VLM, which is asked to list concrete, visible contradictions between the ordered scenes and the frames (not "is this correct?", which invites false agreement; a scene simply not visible is not a contradiction). Contradictions → "warn" with them as reasons (≤ 5); none → "ok". `providers=None` uses `verify.default_providers()` (`verify.gemini_client(key)`: the first `MAX_MODELS = 2` Gemini models with a 10 s timeout per request, the API's minimum, then OpenAI). **One deadline for the whole check** (`BUDGET_S = 8` s; measured median 2.1 s, p90 4.4 s, 7.3 s for a process's first call): the providers run in a worker thread, no further provider is started once the budget is spent, and past the deadline the answer is "warn" ("did not answer within 8 s") while a late reply still goes to the cache (a model fallback chain of 6 × 12 s had held a KIS submission for over a minute). Model replies cached by (collection root and name, scenes, video, and for each keyframe sent: row, image path, size, mtime, time), so a rebuilt collection or another collection with the same ids never reuses a verdict about other frames (`PROMPT_VERSION` 2). Never raises: no model, unknown video, no images or any failure → "warn" with the reason. |
| `conversation.py` | KIS-C dialogue → scenes | `update_from_dialogue(initial, turns: list[{"role": "team"|"oracle", "text"}], current_scenes, top_videos: list[{"video_id", "caption_hint", "image_path"?}] = (), *, llm=None, cache_dir=cache/conversation) -> {"scenes": [str], "next_question": str, "model": str|None}`. `llm=None` uses `decompose.default_llm()`. The LLM gets the initial description, the dialogue, the current scenes and up to 5 candidates (hint text plus, when `image_path` exists, that keyframe as an image) and returns 1–4 ordered self-contained scenes and one short English question whose answer best separates the candidates. Model replies are cached. Rule fallback (`rule_update`): keeps the current scenes (or rule-splits `initial`), appends oracle answers that are not a bare yes/no/unknown and not already contained to the last scene, and picks the first generic discriminating question whose topic the team has not asked yet. Never raises. |
| `interaction_log.py` | buffer UI events and result lists; flush to DRES and to `log_dir/*.jsonl` | `InteractionLog(log_dir, dres_client=None, *, evaluation_id=None, collection_name=None, clock=...)`; `evaluation_id` and `dres_client` are settable attributes. `.event(kind, payload, *, category=None, timestamp_ms=None)` (category defaults from `KIND_CATEGORIES`, else `OTHER`; unknown category → `ValueError`); `.results(query_payload, items, *, sort_type="score", availability="top")` (items: frame hits, chains or video groups, best first); `.flush() -> {events, results, dres_sent, dres_error}`. Local files `events.jsonl` and `results.jsonl` are written first; a DRES failure keeps the local log and those records are not re-sent. |
| `server.py` | FastAPI app factory + CLI | `create_app(settings, collection=None, encoder=None, llm=None, vqa_providers=None, dres_client=None, *, warm_up=True, web_dir=scenechain/web, cache_dir=cache) -> FastAPI`. Defaults: `Collection.load(settings.collection_dir)`, `SiglipTextEncoder()`, LLM clients built from `settings` (not from a fresh `load_settings()`, so the server uses exactly the keys `/api/status` reports): one `GeminiClient` for decompose, chat and VQA, `optional_openai` added for VQA, `verify.gemini_client` (2 models, 10 s per request, `max_output_tokens=512`) plus `optional_openai` for verify; without a Gemini key a `NullLLM` makes decompose/chat use their rule fallbacks and VQA answers 503. An injected `vqa_providers` serves both VQA and verify. A `DresClient` is created at login. Startup (lifespan): one full similarity scan (pages the mmap in), one encoder call and the creation of the verify Gemini client (imports `google.genai`, several seconds, so the first check is not the slow one), then a log flush every 5 s; final flush on shutdown. `mimetypes.add_type('text/javascript', '.js')`. CLI: `python -m scenechain.server [--collection DIR] [--host H] [--port N] [--device cpu|cuda] [--log-dir DIR]`. Endpoints §6. |

## 5. Search semantics (`search.py`)

All functions take a `Collection` and precomputed similarity vectors, so they are testable with
synthetic data.

- `text_similarities(collection, encoder, text) -> np.ndarray`: mean over chunk similarities
  (one chunk for short text). Invalid rows set to `-inf`. Empty text → `ValueError`.
- `rank_frames(sims, collection, top_k=500) -> list[FrameHit]`:
  `FrameHit = {row, video_id, frame_idx, time, score}` best first (ties: lower row); invalid and
  non-finite rows are never returned.
- `group_by_video(hits, max_videos=60, frames_per_video=8) -> list[VideoGroup]`:
  `VideoGroup = {video_id, score (max of its hits), hits[...] sorted by time}` ordered by score
  (ties: first appearance). Each group keeps its `frames_per_video` best hits.
- `temporal_search(collection, scene_sims: list[np.ndarray], window_s=(0.0, 30.0), top_k=200,
  candidate_videos=300, scene_norm=None) -> list[ChainHit]`:
  ordered scenes q1..qn. `scene_norm="z"` (opt-in, added 29 Oct 2026 for the camera-ready chain
  evaluation; the server does not pass it) z-normalises each scene's similarities over the
  keyframes of the candidate videos before the DP; scores are then z-scores; ignored for n = 1. Candidate videos = union of the top `candidate_videos` videos of each
  scene by best keyframe. Within a video, keyframes sorted by `pts_time`. Chain score is computed by
  dynamic programming backwards over scenes: `best_n(i) = s_n(i)`;
  `best_k(i) = s_k(i) + max_{j : t_i + w_min < t_j <= t_i + w_max} best_{k+1}(j)`, with `-inf` when no
  successor exists. Final score = `best_1(i) / n` (mean so that it is comparable across n).
  Return chains `ChainHit = {video_id, score, frames: [{row, frame_idx, time, score}] (one per scene)}`,
  at most 3 chains per video, best first; two chains of the same video never overlap in their
  `[first time, last time]` span (removes near-duplicates sharing a successor); ties go to the lower
  row of the first frame. The window must satisfy `0 <= min < max` (min exclusive, max inclusive),
  else `ValueError`. Implementation: all candidate videos at once (searchsorted successor windows +
  sparse-table range-maximum), no Python loop over keyframes.
  n = 1 wraps `rank_frames` exactly (no per-video limit). Must run in < 1.5 s for n = 3 on 177k rows on
  CPU given precomputed sims (measured: 2 scenes 0.05 s, 3 scenes 0.08 s on synthetic 177k rows;
  51–69 ms median on the real AIC sims).
- `similar_frames(collection, row, top_k=200, exclude_same_video=False) -> list[FrameHit]`:
  query by example using the image embedding of `row`; never returns `row` itself.
- `video_timeline(collection, video_id, scene_sims: list[np.ndarray]|None) -> dict`:
  `{"video": {...}, "keyframes": [{row, frame_idx, time, valid, scores}]}`, all keyframes in time
  order; `scores` is `null` without sims, else one value per scene with `null` for non-finite
  (invalid) rows, so the result is JSON-safe. Unknown video → `KeyError`.
- `submission_segment(collection, video_id, time_s, half_width_s=0.0) -> (start_ms, end_ms)`:
  clamp to [0, duration]. **The default is a point** (`start_ms == end_ms`): DRES judges a temporal
  answer correct only when the whole `[start, end]` lies inside the target segment (confirmed on
  live DRES 2.0.4 with a 20–30 s target: windows 18–22 s and 28–32 s WRONG, points 19.999 s and
  30.001 s WRONG, point 25 s CORRECT), so any width only adds the risk of crossing an edge.

## 6. HTTP API (FastAPI, JSON)

Base path `/api`. Errors: HTTP 4xx/5xx with exactly `{"error": str}` — also for framework errors
(unknown path 404, wrong method 405, schema/JSON errors 422). Status codes used: 400 semantic
validation, 401 DRES login refused, 404 unknown video / keyframe / evaluation / path, 405, 409 not
connected to DRES, 422 request does not match the schema, 502 DRES call failed, 503 no VQA model.
All times in seconds unless the name ends in `_ms`. Request bodies ignore unknown fields.

| Method, path | Request | Response |
|---|---|---|
| `GET /api/status` | – | `{"collection": {"name", "n_keyframes", "n_videos", "dim"}, "encoder": str, "dres": {"connected": bool, "user": str|null, "evaluation_id": str|null}, "llm": {"gemini": bool, "openai": bool}}` — `gemini` = a Gemini key is set; `openai` = key set and SDK installed. Both false disables the VQA button. |
| `POST /api/search` | `{"scenes": [str] (1..4, non-empty), "window": [min_s, max_s] (default [0, 30], 0 <= min < max), "mode": "frames"|"videos"|"chains" (default: "frames" if 1 scene else "chains"), "top_k": int (1..2000, default 300), "filter_videos": [str] optional (non-empty; unknown id → 404)}` | `{"query_id": str, "took_ms": int, "timings": {"encode_ms", "scan_ms", "rank_ms", "cached_scenes"}, "mode", "frames": [FrameHit + {"thumb"}]` or `"videos": [VideoGroup]` (hits carry `thumb`; `min(60, top_k)` videos from the best `max(top_k, 1000)` frames, 8 per video) or `"chains": [ChainHit]}` (frames carry `thumb`). `mode` is always echoed (the UI reads `response[response.mode]`). Every frame object carries `row`, `frame_idx`, `time`, `score`, `thumb`; chain `frames[].row` uses the same row numbers as `/api/video` keyframes. In frames/videos mode with several scenes a keyframe is ranked by its best scene score. Encodings of the last 16 scene texts are reused, so editing one scene re-encodes only that scene. `timings` (all int): `encode_ms` time in the text model, `scan_ms` similarity scan over the matrix (both 0 for cached scenes), `rank_ms` ranking/grouping/chain search plus thumbnail URLs, `cached_scenes` scenes served from the text cache; the UI shows them as the tooltip of the result count. |
| `POST /api/decompose` | `{"text": str}` | `{"scenes": [str], "source": "llm"|"rule", "model": str|null}` |
| `POST /api/similar` | `{"video_id": str, "frame_idx": int, "top_k": int (default 300)}` | `{"query_id", "took_ms", "frames": [...]}`; 404 for an unknown video or a frame that is not a keyframe. The `query_id` holds the example's similarity vector (one "scene"). |
| `GET /api/video/{video_id}` | query param `query_id` optional (to attach per-scene scores of that query; an unknown or evicted id is ignored) | `{"video": {video_id, fps, duration, n_frames, title, youtube_id, file}, "keyframes": [{row, frame_idx, time, valid, thumb, scores: [float|null per scene]|null}], "query_known": bool|null}`; a local `file` is rewritten to `/media/{video_id}` (null if missing). `query_known` is null without `query_id`, false for an unknown or evicted one (the UI then re-runs that result list's request for a fresh id instead of silently dropping the curves). |
| `POST /api/vqa` | `{"question": str, "frames": [{"video_id", "frame_idx"}] (1..5)}` | `{"answer", "evidence", "model"}`; 400 empty question / wrong frame count, 404 not a keyframe, 503 no model configured or none answered |
| `POST /api/dres/login` | `{"url"?: str, "username"?: str, "password"?: str}` (missing values fall back to `DRES_URL` / `DRES_USER` / `DRES_PASSWORD`; without `url` the connected server is used; **the stored `DRES_USER` / `DRES_PASSWORD` are used only when the target equals `DRES_URL` after normalisation**, so a caller-named host never receives them) | `{"connected": true, "user": str}`; 400 no URL or credentials, 401 refused, 502 DRES unreachable. A successful login clears the selected evaluation; a failed one leaves the working session and evaluation as they were. |
| `GET /api/dres/evaluations` | – | `{"evaluations": [{"id", "name", "status", "type", "task": {...}|null}]}`; for `ACTIVE` evaluations `task` = `{"name", "taskGroup", "taskType", "duration"}` from `task_info` plus `{"status" (taskStatus), "task_id", "remaining_s" (timeLeft while RUNNING, else null)}` from `evaluation_state`; 409 when not connected, 502 when DRES fails |
| `POST /api/dres/select` | `{"evaluation_id": str}` | `{"evaluation_id"}`; 404 if DRES does not list it (accepted unchecked when the list call fails) |
| `POST /api/submit` | `{"kind": "kis", "video_id", "time" (s) or "start_ms"+"end_ms"}` or `{"kind": "text", "text"}` or `{"kind": "avs", "items": [{"video_id", "time" or "start_ms"+"end_ms"}] (1..500)}` | `{"ok": bool, "status": str, "raw": object|null}`. KIS and AVS items given by `time` are sent as **points**: `start_ms == end_ms == round(time*1000)` clamped to [0, duration] (§5); explicit `start_ms`/`end_ms` are sent as given (0 <= start <= end). AVS sends one answer set per item. The collection name is sent only when `DRES_COLLECTION` is set. With a DRES verdict: `ok` = DRES `status`, `status` = DRES `description` (e.g. "Submission correct!"), `raw` = the DRES response (`submission` = CORRECT/WRONG/INDETERMINATE/UNDECIDABLE). Not connected: `{"ok": false, "status": "not connected", "raw": null}`; no evaluation: `status` "no DRES evaluation selected"; DRES rejection: `{"ok": false, "status": <DresError.description>, "raw": {"http_status", "description"}}` (412 = duplicate or limit reached, 500 = no running task / unknown media item). Every attempt with a valid body is appended to `log_dir/submissions.jsonl` (request, what was sent, response). A KIS verdict WRONG adds the video (and time) to the rejected list of the current evaluation and task. |
| `POST /api/log` | `{"events": [{"timestamp_ms", "category", "type", "value"}]}` (≤ 1000; also accepts a `sendBeacon` JSON blob) | `{"accepted": int}` — events without `type` or with an unknown category are skipped; only the first 1000 events of a call are read, so the UI sends batches of at most 1000 (beacons also at most 48 KB) |
| `POST /api/verify` | `{"scenes": [str], "video_id", "time"}` | `{"verdict": "ok"|"warn"|"fail", "reasons": [str], "model", "took_ms", "rejected": bool}` (§4 `verify.py`); `scenes` may be empty (only the verdict history is checked); 404 unknown video. `rejected` = KIS submissions judged WRONG in the selected evaluation whose DRES task run id (`evaluation_state.taskId`, fetched when the WRONG verdict arrives) equals the last observed one (updated by `/api/dres/evaluations` and `/api/dres/select`), so the list resets when a new task starts. |
| `POST /api/chat` | `{"initial": str, "turns": [{"role", "text"}], "scenes": [str], "query_id": str|null}` | `{"scenes": [str], "next_question": str, "model", "query_known": bool|null}`; candidates = the best 5 videos of the cached query (by chain for several scenes, by best frame otherwise) with their title as hint and the best keyframe as image; `query_known` as in `/api/video` (false: no candidates were used) |
| `GET /frames/{video_id}/{filename}` | – | the keyframe image (`Cache-Control: max-age=86400`). Both segments must match `[A-Za-z0-9_-][A-Za-z0-9._-]*` without `..`, the extension must be an image type, the video must be in the collection and the resolved path must stay under `frames_dir`; otherwise 400/404 `{"error"}`. |
| `GET /media/{video_id}` | – | the local video file of a video whose `file` is a local path (Range requests supported) |
| `GET /` | – | static web UI from `scenechain/web` (`/?mock=1` runs it against the in-browser mock API) |

`query_id` is a server-side LRU cache key (last 32 queries) holding the scene similarity vectors, so
`/api/video/{id}?query_id=` can draw score curves without re-encoding.

Media item naming for DRES: `video_id` as in `metadata.json` (for V3C this is the 5-digit video id).
The collection name sent to DRES is `DRES_COLLECTION` when set (normally `collection.json:name`); by
default none is sent, and DRES then requires the item name to be unique across its collections.

**DRES facts the server relies on** (live DRES 2.0.4 and the backend sources, see
`docs/DRES_NOTES.md`):
- Session: the client sends `?session=<sessionId>` on every call and never relies on cookies (DRES
  re-issues `SESSIONID` as `Secure`, which a cookie jar drops over plain HTTP). An old token gets 401.
- KIS containment: see §5; submit points. AVS task types with `MAP_TO_SEGMENT` (the DRES preset)
  refuse with HTTP 500 an answer whose start and end fall in different predefined shots; points avoid it.
- Verdicts: a repeated submission → 412 duplicate; a further correct one → 412 "Maximum number of
  correct submissions (1) exceeded"; unknown item, no running task or ended task → 500; AVS → 202
  INDETERMINATE (stored as one submission with one answer set per shot). `text_answer` trims
  whitespace before sending.
- `task_info` keeps returning the ended task; running/ended and the countdown come from
  `evaluation_state` (`taskStatus`, `timeLeft` in seconds; `timeElapsed` is negative during the
  start countdown).
- DRES returns 200 for logs with extra or misnamed fields, so logging mistakes are silent; only an
  unknown category or a non-string value gives 500. Result log values `sortType="score"` and
  `resultSetAvailability="top"` still need confirming against the VBS 2027 participant instructions.

## 7. Web UI (`scenechain/web`, vanilla JS, no build step)

Layout (single page, dark-on-light, 1920×1080 target, usable at 1366×768):

- **Top bar:** tool name ("SceneChain"), collection name + size (hidden below
  1500 px width), DRES status pill (click → login dialog), evaluation selector (after login the first
  `ACTIVE` evaluation is selected automatically; evaluations are polled every 10 s), current task name
  and remaining time when DRES provides them (read from `evaluations[].task` as `name|taskName`,
  `taskType|type`, `remaining_s|timeLeft` in seconds, counted down locally), submission counter.
- **Left panel — query builder:** scene boxes (1–4), each a textarea with its own clear button; "+ then"
  adds a scene; a "Split description" button calls `/api/decompose` on the first box and fills the
  scene boxes; window slider (min/max seconds); result mode toggle (Frames / Videos / Chains);
  history of the last 20 queries (kept in `localStorage` key `vbs.queryHistory.v1`; clicking an entry
  restores the query and runs it again). Adding a second scene switches Frames → Chains; removing
  scenes back to one switches Chains → Frames.
- **Centre — results:** virtualised grid of thumbnails. Frames mode: cards with score, video id,
  mm:ss. Videos mode: one row per video with its best frames in time order (thumbnails sized so whole
  cards fill the row; the rest scrolls horizontally). Chains mode: one box per chain showing
  scene 1 → scene 2 → … frames with arrows, time gaps and the chain score; several chain boxes share
  a row when they fit (2 per row at 1920 px with 2 scenes). A Back stack keeps the last 10 result
  lists; more-like-this results label their score curve "Similarity".
- **Card actions** (hover + keys): a single click, `Enter` or double-click opens the video view at that
  frame; `S` submit KIS at this frame; `E` more-like-this (`/api/similar`); `B` add to AVS basket;
  `A` ask VQA about this frame. `S/E/B/A` act on whatever was used last: the card under the pointer or
  focused, or the current moment of the video view (after opening, stepping, clicking the curve or the
  keyframe grid); the target is highlighted.
- **Right panel — video view:** player (HTML5 `<video>` when `file` is set, else YouTube IFrame API when
  `youtube_id` is set, else keyframe slideshow); frame-accurate controls (←/→ one keyframe,
  Shift+←/→ one second, Space play/pause); **score curve** of each scene over the video timeline (SVG,
  one line per scene, current position marker); full keyframe strip; buttons "Submit this moment",
  "Ask VQA", "Add to basket".
- **Bottom drawer:** AVS basket (thumbnails, remove, "Submit all"), VQA panel (question, answer
  suggestion with evidence, editable answer, "Submit answer"; `Enter` in the question asks, `Enter` in
  the answer submits), submission log with DRES verdicts (classified from `raw.submission`; a response
  with `ok: false` is a rejection whatever its text, e.g. 412 "Maximum number of correct submissions
  (1) exceeded"; the status text is read only for ok responses without a verdict; "not connected" is
  shown as "Not sent"; the VQA answer is sent as kind `text` and
  shown as "QA").
- **Submission guard**: `S` (or "Submit this moment") first calls
  `/api/verify`; "ok" submits at once; "warn"/"fail" shows the reasons in a small banner and submits
  only on a second `S` within 5 s (or a click on "Submit anyway"). While the check runs, `S` for the
  same target only shows a hint, the banner offers "Submit without check" (logged as decision
  `skipped-check`, shown as "not checked"), and the UI stops waiting after 10 s (a "warn" that a second
  `S` confirms). A submission in flight, or sent less than 1.5 s ago, is not sent again by another
  press for the same target (KIS: same video within 5 s; QA/AVS: same answer or basket). Submit buttons
  take no focus on click, and `Space`/`Enter` on a focused submit button never confirm a warning
  (`Space` is play/pause whenever a video is open). The "judged wrong" flags reset when the DRES task
  run id (`task.task_id`) changes, falling back to the task name only without an id (mock).
- **KIS-C chat tab** in the left panel: initial description, oracle/team
  turns, "Update scenes" (calls `/api/chat` with the current `query_id`, fills the scene boxes, re-runs
  the search) and the suggested next question. If the operator searched or edited the scenes while
  waiting, the reply is kept in the tab with a "Use these scenes" button instead of replacing the
  query; if the server reports `query_known: false`, the query is renewed and the chat asked again.
- **Stale replies and expired queries:** a VQA reply to a question that was superseded is dropped, and
  a reply never overwrites an answer the operator typed (or a question edited) while waiting (the
  suggestion is shown with its evidence instead). When `/api/video` reports `query_known: false` for
  the current result list, the UI re-runs that list's request for a fresh `query_id` and redraws the
  curves.
- **Fast reformulation:** editing a scene box and pressing `Ctrl+Enter` keeps the video view open and
  refreshes its score curves in place (the VBS 2026 log analysis attributes success to iterative,
  high-frequency query reformulation with result inspection). The server re-encodes only changed scenes.
- A KIS submission sends `{kind: "kis", video_id, time}` (seconds rounded to ms), never
  `start_ms/end_ms`; AVS sends `items: [{video_id, time}]`.
- Every search, result list, open-video, submit and basket action goes to `/api/log` (batched every
  5 s in calls of at most 1000 events, plus `sendBeacon`s of at most 1000 events / 48 KB on
  `pagehide`; a refused beacon leaves its events queued) so the DRES interaction logs are complete. `value` is always
  a string. Categories used: TEXT, IMAGE, FILTER, BROWSING, OTHER. Types used: textQuery,
  temporalQuery, similarFrames, decompose, resultList, resultMode, temporalWindow, resultsBack,
  openVideo, closeVideo, seek, submitKIS, submitAVS, submitQA, basketAdd, basketRemove, basketClear,
  vqaFrame, vqaQuestion.
- Keyboard: `/` focus query, `Ctrl+Enter` search (and moves focus out of the box so `S/E/B/A` work at
  once), `1/2/3` switch mode (re-runs the current query in the new mode), arrow keys move between result
  cards when no video is open, `Esc` closes dialogs, else leaves a focused text box, else closes the
  video view, `?` help overlay. Letters typed into text boxes never fire shortcuts.
- Mock mode: `/?mock=1` (a "Mock API" badge appears); mock responses never set `youtube_id` or `file`.

## 8. Ingestion (`scenechain/ingest`)

- `extract_keyframes.py --videos DIR --out COLLECTION [--method auto|scenedetect|histogram|shots|uniform]
  [--fps 1] [--shots-dir DIR] [--max-gap S]`: shot boundaries from PySceneDetect ContentDetector
  (`auto` when installed), else a built-in HSV-histogram + frame-difference detector (hard cuts), or
  external `<video_id>.csv` files (`start_frame,end_frame`, inclusive; V3C master shot boundaries
  convert to this, see the module docstring); `uniform` samples by time. One to three keyframes per
  shot (middle, plus start/end for shots longer than 4 s), JPEG with long side 512 (quality 90);
  `--encode-long-side N` also writes `frames_encode/` copies for the encoder (`encode_frames_dir`).
  Frame numbers are OpenCV decode-order indices and `pts_time` is the decoder timestamp (correct
  for variable-frame-rate files). Writes `metadata.json`, `videos.json` and the non-embedding
  fields of `collection.json`, plus per-video records in `ingest/`; resumable per video (a record
  with the same parameters and input size is skipped). Changing the metadata removes the embedding
  entries from `collection.json`.
- `encode_images.py --collection COLLECTION [--batch 64] [--fp16] [--device cuda|cpu] [--float32]`:
  SigLIP-2 vision tower only (`google/siglip2-so400m-patch14-384`, same preprocessing and pooling as
  the notebook that built the authors' AIC 2026 index: checkpoint processor, squash to 384×384, pooled output, L2),
  float16 weights on CUDA; reads `encode_frames_dir` when present, else `frames_dir`; float32
  `embeddings_parts/part-NNNNN.npy` shards pinned by a manifest (row sha1, frames directory, shard
  size, encoder) and skipped when complete; final `embeddings.npy` is float16 unless
  `--float32`; updates `model`, `dim`, `embeddings` in `collection.json`; prints images/s.
  `--verify-against EMB.npy --verify-k K` re-encodes K random rows of an existing index and reports
  cosine parity. Measured: 1.00000 mean (min 0.99997) on 200 original AIC keyframes, but 0.886 on
  the 512 px JPEG mirror of the same frames, so encode from full-resolution frames whenever they
  exist (details and throughput in `docs/SCALE_UP_PLAN.md`).
- `build_from_aic.py --data <AIC data folder> --out collections/aic2026`: adapter described in §3 (about 2 s,
  copies nothing large; 873/873 YouTube ids found).
- `download_sample.py`: fetches three Blender open movies (CC BY, 480p WebM transcodes from Wikimedia
  Commons, 25 MB) into the gitignored `samples/` and writes `SOURCES.md` with URLs, licences and
  authors taken from the Commons API, for the end-to-end ingestion test (never commits media).

## 9. Tests (`tests`, pytest)

- `conftest.py`: synthetic collection (6 videos × 40 keyframes, D = 16, planted scene vectors,
  per-dimension noise `NOISE_SCALE = 0.10` so planted scenes beat every noise keyframe) written to
  `tmp_path`; `FakeEncoder` mapping known phrases to planted vectors.
- `test_search.py`: `rank_frames`, `group_by_video`, `temporal_search` (planted A→B chain found first;
  window respected; n=1 equals rank_frames; order matters: B→A not rewarded; brute-force comparison;
  row order and float16 independence), `similar_frames`, `video_timeline`, `submission_segment`
  (point by default, clamping); timing test on 177k synthetic rows, opt-in with `VBS_RUN_SLOW=1`.
- `test_decompose.py`, `test_vqa.py`: rule fallback, cache hit avoids a second call, model fallback,
  empty/non-JSON replies (mocked Gemini/OpenAI SDK objects).
- `test_dres.py`: client against a local mock implementing the DRES v2 paths used; request bodies and
  mock responses checked against schema fragments copied from the spec; error propagation. With
  `DRES_CLIENT_SPEC=<oas-client.json>` the fragments are also compared with a downloaded spec.
- `test_interaction_log.py`, `test_collection.py`, `test_encoder.py`, `test_config.py`,
  `test_build_from_aic.py`, `test_fixture_sanity.py`.
- `test_verify_conversation.py`: keyframe neighbourhood, ok/warn/fail, provider fallback, disk cache,
  never raises, one deadline across models and providers (a late reply is still cached), cache not
  shared between collections or rewritten frames; dialogue update with images and cache, rule fallback.
- `test_web_ui.py`: headless Chromium (Playwright) against the real app over the synthetic collection
  with fake models and scripted DRES; skipped without Playwright/Chromium or with `VBS_SKIP_UI=1`
  (about 30 s): Space/Enter never confirm a warning, no double submission after an instant "ok",
  a hanging check times out or is skipped, 412 max-correct is not "Correct", rejected flags reset on a
  task re-run, stale VQA and chat replies, an evicted query is renewed, log batches ≤ 1000 events.
- `test_server.py`: FastAPI `TestClient` over the synthetic collection + fake encoder, fake LLM and a
  `DresClient` with a scripted transport: every endpoint in §6 returns the documented shape; `{"error"}`
  on 400/404/405/409/422/503; frame path traversal rejected; `/api/submit` without DRES returns
  `ok=false` and logs locally; KIS and AVS are sent as points; a video judged WRONG makes `/api/verify`
  fail until the next task; chat returns scenes; the video timeline attaches per-scene scores of a
  `query_id`; scene encodings are reused; logs flush on shutdown.
- `test_ingest.py`: synthetic MJPG videos with colour-block scenes and known cuts (histogram and
  PySceneDetect detectors within ±2 frames), keyframe selection rules, uniform sampling by timestamp,
  external shot files, metadata / videos / collection schema, per-video resume, encoder copies;
  `encode_images` with a fake image encoder: float16 output, shard resume after an interruption,
  manifest mismatch, `encode_frames_dir`, missing frames, `--verify-against` parity, and an
  end-to-end video -> `Collection.load` check.
- Run: `python -m pytest tests -q` from the repo root (this repo's pytest config hides the summary
  line with `-q`; add `-o addopts=""` to see it). Must pass with no network and no GPU.

## 10. Running

```
python -m scenechain.ingest.build_from_aic --data data --out collections/aic2026
python -m scenechain.server --collection collections/aic2026 --port 8787 [--device cuda]
# open http://127.0.0.1:8787   (http://127.0.0.1:8787/?mock=1 for the UI without a backend)
```

Measured on the team laptop (i5-12450HX, 16 GB, CPU torch): `Collection.load` 0.46 s; SigLIP-2 text
model load 24.9 s; first similarity scan 3.1 s (page-in, done at startup), then ~31 ms; text encoding
~1 s per scene on CPU (a 3-scene query spends ~3 s encoding and < 0.1 s searching — run the encoder on
the GPU for live use); `rank_frames` top 500 in 22 ms.

End-to-end API latency on the real index (17 Sep, `docs/BENCHMARK.md`, median of 5 new queries): CPU
server 1.10 s for a 1-scene frames search (1.07 s of it encoding), 1.89 s for 2-scene chains, 2.60 s for
3-scene chains, 0.95 s when only one of three scenes changed; GPU server (`.venv-gpu`, `--device
cuda`) 0.10 s, 0.29 s, 0.38 s and 0.26 s. `/api/similar` 65 ms, `/api/video?query_id=` 4 ms; the LLM
endpoints (Gemini) 0.9–3.2 s median. The model loader contacts huggingface.co at start-up; set
`HF_HUB_OFFLINE=1` where there is no internet.
