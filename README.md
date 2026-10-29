# SceneChain

**Scene-chained interactive video search with verified submissions.**

SceneChain is an interactive known-item and question-answering video retrieval tool built for the
[Video Browser Showdown](https://videobrowsershowdown.org/) (VBS) 2027. About a third of the textual task descriptions in the
VBS archive narrate several shots in order, so SceneChain can search a description as an *ordered
chain of scenes*. A language model proposes the split (or the operator writes it), each scene is
embedded with SigLIP-2, and a dynamic program ranks temporally consistent chains inside each
video. The video view draws one similarity curve per scene along the timeline. Every wrong
submission costs points in VBS, so a *submission guard* asks a vision-language model for concrete
contradictions between the task text and the chosen moment before an answer reaches
[DRES](https://github.com/dres-dev/DRES). It also blocks moments close to an answer that DRES has
already judged wrong for the current task. The design follows measurements from the three preliminary rounds of the AI
Challenge HCMC 2026; see the paper cited [below](#citation).

> **Screenshot.** None is included yet. Every screenshot we have was taken on the AI Challenge
> HCMC 2026 collection, whose keyframes are organiser data and are not redistributed here. The
> paper's Fig. 1 shows the interface. You can see the full UI without any data by opening
> `http://127.0.0.1:8787/?mock=1` (see [Quick start](#quick-start)).

## Features by VBS task type

| Task | Workflow in SceneChain |
|---|---|
| **KIS-T** (textual known-item) | Paste the description and press **Split description** to get up to four ordered scenes, which you can edit. Use *Chains* mode, and add hints as they appear by editing scenes (`Ctrl+Enter` re-runs and keeps the open video, re-encoding only the changed scene). Open the best chain, pick the frame on the score curve, then press `S` (guarded). |
| **KIS-C** (conversational) | The *KIS-C* tab records the team's questions and the oracle's answers. **Update scenes** folds the dialogue into the scene list, re-runs the search and proposes one short question that would best separate the leading videos. The operator decides what to ask. |
| **VQA** | Locate the moment as for KIS-T, select up to five frames and press `A` to ask the assistant. It must answer *and* name the visual evidence. The answer stays editable, is guarded, and is sent to DRES as text. |
| **AVS** (ad-hoc) | Use *Frames* or *Videos* mode. `B` adds shots to a basket that shows how many distinct videos it covers (DRES counts each correct video once). Batch submission warns about videos already judged wrong and sends each shot as a time point. |

Also included:

- results in three modes: frames, videos (grouped) and chains;
- query-by-example from any keyframe (`E`);
- a frame-accurate player (local file, YouTube embed or keyframe slideshow) with a keyframe strip;
- a back stack of result lists and a query history;
- keyboard-driven operation (`?` shows all shortcuts);
- an in-browser mock API (`/?mock=1`) for trying the UI without a collection.

## Scene-chain scoring

A query is an ordered list of scenes $q_1,\dots,q_n$ ($1 \le n \le 4$), with a time window
$[w_{\min}, w_{\max}]$ between consecutive scenes (default 0 to 30 s). Let $s_k(i)$ be the
similarity of scene $k$ to keyframe $i$, with the keyframes of each video sorted by time $t_i$.
Chain scores are computed backwards:

$$b_n(i) = s_n(i), \qquad b_k(i) = s_k(i) + \max\{\, b_{k+1}(j) : t_i + w_{\min} < t_j \le t_i + w_{\max} \,\}$$

Chains are ranked by $b_1(i)/n$, so scores are comparable across chain lengths.

- **Candidate videos** are the union of the best videos of each scene.
- **Per video**, at most three non-overlapping chains are listed.
- **One scene** ($n = 1$) reduces exactly to plain frame ranking.

The implementation (`scenechain/search.py::temporal_search`) has no Python loop over keyframes. It
uses vectorised successor windows (`searchsorted`) and a sparse-table range maximum over all
candidate videos at once.

Long scene texts are split into chunks that fit the 64-token SigLIP-2 text window, and the chunk
similarities are averaged, so a frame must match the whole description.

**Status of the evidence.** On 58 two-scene test items over the AIC 2026 videos (scene texts
from the test set, tool defaults, no operator), chains listed the target moment among the first
ten results for 50% of queries, against 14% for the whole description as one query and 43% for
the second scene alone, and put the correct video among the first ten videos for 88% against 69%.
Under the same limit of three results per video, scene-2 retrieval matched chains within the
first 100 results. Most of the gain comes from retrieving the answer's scene on its own; the chain
adds mainly the right video. See the paper for confidence intervals and caveats.

## Submission guard

Pressing `S` (or *Submit this moment*) first calls `POST /api/verify`:

1. **Verdict history, per moment.** DRES judges a known-item answer by whether it lies inside one
   target segment, and the right video at the wrong moment is a common mistake. A WRONG verdict
   therefore blocks only a moment, not the whole video. The check fails at once, with no model call,
   when the new moment lies within `WRONG_RADIUS_S` (5 s) of a moment DRES already judged wrong for
   the current task run. Elsewhere in that video, a non-blocking note names the earlier wrong time,
   and the usual check runs. AVS shots are judged one by one, so other shots of a video are never
   blocked.
2. **Contradiction check.** The VLM sees the keyframe nearest the chosen time and two neighbours
   on each side. It is asked to *list concrete, visible contradictions* with the ordered scenes
   rather than whether the moment is correct, so agreement is never the default answer. A scene that
   is simply not visible does not count as a contradiction.
3. **Majority vote.** Verdicts flip between identical calls even at temperature 0, so the model is
   asked three times in parallel (`VOTES`) and the majority decides. The check answers as soon as
   the majority is known.
4. **One time budget.** The whole check has 8 s. A clean result submits immediately. Otherwise,
   or after the budget runs out, the reasons appear in a banner and a second `S` within 5 s submits
   anyway. *Submit without check* is always available. `Space` and `Enter` never confirm a warning.

The guard is a second opinion, not a gate, and the operator decides. On 40 test items over the
AIC 2026 videos, fixed before the run (majority of three, a Gemini Flash model through a proxy, 30 s
budget), it warned on 4 of 40 correct moments (10%) and on 26 of 39 moments of the right video at
least 60 s away (67%) and 26 of 40 best-ranked wrong videos (65%); 5% of repeated calls disagreed.
Those wrong moments are easier than real near misses, so the catch rates are upper bounds.

## DRES integration

`scenechain/dres.py` is a client for the DRES v2 client API. It was written against
`doc/oas-client.json` at DRES `v2.0.5` and rehearsed end to end against a locally started DRES
**2.0.4** release. It handles login, evaluation and task discovery, KIS/VQA/AVS submission, and
query-event and result logging. [`docs/DRES_NOTES.md`](docs/DRES_NOTES.md) has the details and a
rehearsal checklist. Observations the client relies on:

- **A temporal range must lie entirely inside the target segment.** With a 20 to 30 s target,
  windows of 18 to 22 s and 28 to 32 s, which straddle the target edge, were judged **WRONG**.
  Points at 19.999 s and 30.001 s were also WRONG, and a point at 25 s was CORRECT. SceneChain
  therefore submits known-item answers as a **single time point** (`start == end`), because any
  width only adds the risk of crossing an edge.
- **AVS tasks with the `MAP_TO_SEGMENT` preset** refuse (HTTP 500) an answer whose start and end
  fall in different predefined shots. Points avoid this.
- **Session token.** It is sent as `?session=<id>` on every call. DRES re-issues the cookie as
  `Secure`, and a cookie jar drops it over plain HTTP. The token and password are never logged,
  and `urllib3` is kept at WARNING because at DEBUG it would print full URLs.
- **Repeated submissions.** A repeated submission gets 412 (duplicate), and a further correct one
  gets 412 "Maximum number of correct submissions exceeded". Nothing is retried automatically,
  because a retry would count twice.
- **Logs.** DRES accepts query and result logs with misnamed fields with HTTP 200, so logging
  mistakes are silent. Every search, result list, video opening, verification and submission is
  also written locally to `logs/*.jsonl`.
- **Result-log values.** `sortType="score"` and `resultSetAvailability="top"` still need
  confirming against the VBS 2027 participant instructions.

## Installation

Requires Python 3.10 or newer (developed and tested on 3.12).

```bash
git clone https://github.com/bminhnemhoi/SceneChain.git
cd SceneChain
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[models,ingest,test]"
```

Extras:

| Extra | What for |
|---|---|
| *(core)* | server, search, DRES client, hosted-LLM features |
| `models` | SigLIP-2 text and image towers (`torch`, `transformers`). For GPU encoding, install a CUDA build of torch from [pytorch.org](https://pytorch.org) first. |
| `ingest` | keyframe extraction (`opencv-python-headless`, `scenedetect`) |
| `openai` | optional OpenAI(-compatible) fallback for the guard and VQA |
| `test`, `ui-test` | pytest, and Playwright for the headless-browser UI tests (`python -m playwright install chromium`) |

The SigLIP-2 checkpoint (`google/siglip2-so400m-patch14-384`) is downloaded from Hugging Face on
first use. After that, set `HF_HUB_OFFLINE=1` to start without network access.

## Quick start

**UI only, no data, no models:** run `python -m http.server -d scenechain/web 8000` and open
`http://127.0.0.1:8000/?mock=1`. The page then runs against an in-browser mock API with invented
videos.

**Your own videos:**

```bash
# optional: three short CC-BY Blender open movies (about 25 MB) into samples/
python -m scenechain.ingest.download_sample

# 1. shots -> keyframes -> metadata (PySceneDetect if installed, else a built-in cut detector;
#    --method uniform --fps 1 samples by time; --method shots --shots-dir DIR reads given shot boundaries)
python -m scenechain.ingest.extract_keyframes --videos samples --out collections/samples

# 2. SigLIP-2 image embeddings (resumable; fp16 on CUDA)
python -m scenechain.ingest.encode_images --collection collections/samples --device cpu

# 3. serve
python -m scenechain.server --collection collections/samples --port 8787 [--device cuda]
# open http://127.0.0.1:8787
```

A collection is a plain directory containing `collection.json`, `embeddings.npy`,
`metadata.json`, `videos.json` and `frames/<video_id>/...`. The format is specified in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §3. Use full-resolution frames for encoding where
they exist (`--encode-long-side`): re-encoding 512 px thumbnails lowered cosine parity with the
original index to 0.89 ([`docs/SCALE_UP_PLAN.md`](docs/SCALE_UP_PLAN.md)).
`scenechain/ingest/build_from_aic.py` is the adapter we used to wrap an existing keyframe index in
this format without copying it.

## Configuration

Settings come from command-line options first, then process environment variables, then a `.env`
file at the repository root, which is parsed but never copied into the environment. Copy
[`.env.example`](.env.example) to `.env`. Every variable is optional.

| Variable | Meaning |
|---|---|
| `VBS_COLLECTION` | collection directory (default `collections/default`) |
| `VBS_HOST`, `VBS_PORT` | bind address and port (default `127.0.0.1:8787`) |
| `VBS_LOG_DIR` | interaction and submission logs (default `logs/`) |
| `DRES_URL`, `DRES_USER`, `DRES_PASSWORD` | DRES server and credentials. The stored credentials are sent only to `DRES_URL`. |
| `DRES_COLLECTION` | collection name sent with answers (unset = not sent) |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | hosted VLM for Split, the guard, VQA and KIS-C |
| `OPENAI_API_KEY` | optional fallback for the guard and VQA (needs the `openai` extra) |
| `OPENAI_BASE_URL`, `OPENAI_MODEL` | optional OpenAI-compatible endpoint and model for that fallback |

Without a model key, every path degrades gracefully: Split and KIS-C fall back to rule-based
splitting, the guard shows a warning banner, and VQA offers a manual answer field. Model replies
are cached under `cache/`.

## Running the tests

```bash
pip install -e ".[ingest,test]" torch      # a CPU build of torch is enough
python -m pytest -q
```

The suite needs **no model weights, no GPU and no network**. It uses:

- a synthetic collection with planted scene vectors and a fake text encoder;
- synthetic videos with known cuts;
- fake LLM clients;
- a mock DRES server.

Some parts are opt-in or skip cleanly:

- the ingestion tests skip without OpenCV or torch;
- the headless-browser UI tests (`tests/test_web_ui.py`) skip without Playwright and Chromium, or
  with `VBS_SKIP_UI=1`;
- a timing test on 177k synthetic rows runs only with `VBS_RUN_SLOW=1`;
- a check of the DRES client against a downloaded OpenAPI spec runs only with
  `DRES_CLIENT_SPEC=<oas-client.json>`.

CI (`.github/workflows/tests.yml`) runs the suite without the UI tests on every push.

## Status and limitations

SceneChain is a research prototype prepared for VBS 2027, and we report it as such:

- **Collections.** All measurements so far were taken on the AI Challenge HCMC 2026 collection
  (873 Vietnamese TV videos, 177,321 keyframes). V3C, the Marine Video Kit and the laparoscopy
  collection **have not been ingested yet**.
- **Index size.** Search is an exact float scan over a memory-mapped matrix. V3C at one keyframe
  per master shot (about 4.1 M rows) would need about 19 GB of RAM in float32, and int8 or IVF-PQ
  indexes are still to be measured.
- **Evidence so far comes from AIC data.** On the 58 two-scene items of the authors'
  distribution-matched AIC test set, chain ranking (tool defaults, scene texts from the set) listed
  the target moment among the first ten results for 50% of queries, against 14% for the whole
  description as one query (paired difference +36 points, 95% bootstrap CI [+22, +50]) and 43% for
  the second scene alone (CI of the difference [−9, +22]). Within the first 100 results the second
  scene alone found the exact moment more often (71% vs. 59%), because at most three chains are
  listed per video. Effects on wrong submissions and time to the first correct answer are what we
  plan to measure at VBS 2027.
- **Missing channels.** There is no speech (ASR) or on-screen-text (OCR) channel yet, and no
  object or colour filters.
- **Hosted models.** Split, the guard, VQA and KIS-C call hosted models (Gemini, with an optional
  OpenAI fallback), so they depend on the network and on rate limits. The first call of a process
  is slower, so warm the server before a session. The Gemini model list in `scenechain/llm.py`
  reflects availability in August 2026. These features send keyframes to a hosted model, so use
  them only on collections whose data agreement permits it; a local VLM is planned.
- **Latency** on the authors' laptop (i5-12450HX, RTX 4050 6 GB): a three-scene chain query takes
  about 0.37 s with the text encoder on the GPU and about 2.6 s on the CPU. See
  [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

The documents in `docs/` are engineering records written during development. Where they mention
`collections/aic2026` or `data/`, they refer to the authors' local AIC 2026 working copy, which is
not part of this repository.

## Repository layout

```
scenechain/            Python package
  search.py            frame ranking, video grouping, scene-chain DP, timelines
  server.py            FastAPI app and CLI (python -m scenechain.server)
  verify.py            submission guard
  dres.py              DRES v2 client
  decompose.py         description -> ordered scenes (LLM, rule fallback)
  conversation.py      KIS-C dialogue -> scenes + next question
  vqa.py, llm.py       VQA assistant, Gemini/OpenAI clients
  encoder.py, siglip_text.py   SigLIP-2 text encoding (4-prompt ensemble, long-text chunking)
  collection.py, config.py, interaction_log.py
  ingest/              keyframe extraction, image encoding, sample download, index adapter
  web/                 single-page client (vanilla JS, no build step) and mock API
tests/                 pytest suite (synthetic fixtures only)
docs/                  architecture/API contract, DRES notes, benchmarks, scale-up plan
```

## Citation

If you use SceneChain, please cite:

```bibtex
@inproceedings{Ngo2027SceneChain,
  author    = {Ngo, Binh Minh and Le, Xuan Khanh and Ngo, Lam Tien},
  title     = {{SceneChain}: Scene-Chained Interactive Video Search with Verified Submissions},
  booktitle = {MultiMedia Modeling (MMM 2027)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2027}
}
```

Binh Minh Ngo, Xuan Khanh Le and Lam Tien Ngo are with the Faculty of Information Technology,
Ton Duc Thang University, Ho Chi Minh City, Vietnam. Machine-readable metadata is in
[`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE) © 2026 Binh Minh Ngo, Xuan Khanh Le, Lam Tien Ngo.

SceneChain uses the SigLIP-2 model (`google/siglip2-so400m-patch14-384`, Apache-2.0), which is
downloaded at run time and not redistributed. Sample videos fetched by `download_sample.py` keep
their own Creative Commons licences, which are recorded in `samples/SOURCES.md`.
