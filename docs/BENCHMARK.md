# API latency on the real AIC 2026 index

> Engineering record kept from development. The AIC 2026 index itself is not distributed with
> this repository; paths such as `collections/aic2026` refer to the authors' local working copy.

Measured on 17 Sep 2026 against the live server (`scenechain/server.py`) with the real index and the
real SigLIP-2 text model. Text encoding dominates every search on CPU; ranking itself stays well
under 0.2 s.

## Setup

| | |
|---|---|
| Machine | Team laptop: Intel Core i5-12450HX (12 logical CPUs), 16 GB RAM, NVIDIA GeForce RTX 4050 Laptop GPU 6 GB (driver 581.86), Windows 11 Pro |
| Index | AIC 2026 collection: 177,321 keyframes, 873 videos, 1,152-d float32 embeddings (817 MB, memory-mapped), 1,903 invalid rows |
| Text model | `google/siglip2-so400m-patch14-384`, float32, 4-prompt ensemble per scene (`encoder.py`) |
| CPU server | global Python 3.12.6, torch 2.13.0+cpu (8 threads), numpy 2.4.6, transformers 5.15.1, FastAPI 0.141.1, uvicorn 0.52.4 |
| GPU server | `.venv-gpu`: Python 3.12.6, torch 2.14.0+cu130, numpy 2.5.2, transformers 5.17.0, same FastAPI/uvicorn |
| LLM | Gemini API over the internet (`gemini-3.5-flash-lite` first; the model that answered is listed) |
| Client | `requests.Session` (keep-alive) on the same laptop; round trip includes JSON encoding and decoding |

Commands (repository root):

```
python -m scenechain.server --collection collections/aic2026 --port 8787 --device cpu --log-dir <scratch>/logs_cpu
python bench_api.py --base http://127.0.0.1:8787 --out bench_cpu.json

# GPU (optional row set, search only)
.venv-gpu/Scripts/python.exe -m scenechain.server --collection collections/aic2026 --port 8788 --device cuda --log-dir <scratch>/logs_gpu
python bench_api.py --base http://127.0.0.1:8788 --out bench_gpu.json --search-only
```

`bench_api.py` is a scratch script (not in the repository). Every row is one warm-up call and then
5 measured calls, run one after another. Each call uses an input the server has not seen before
(new scene texts, new moments, new dialogues), so neither the server's text cache nor the LLM disk
caches answer. With 5 runs the median is the 3rd value and p90 (nearest rank) is the slowest run.
`/api/search` responses now carry `timings`: `encode_ms` (text model), `scan_ms` (similarity scan over
all 177,321 rows), `rank_ms` (ranking, grouping or chain search, plus thumbnail URLs).

## CPU server (port 8787, `--device cpu`)

All times in milliseconds. "Server" is `took_ms` as reported by the server; the split columns are
medians.

| Request | Round trip median | Round trip p90 | Server median | Encode | Scan | Rank | Response |
|---|---:|---:|---:|---:|---:|---:|---|
| `/api/search` 1 scene, frames (top 300) | 1,096 | 1,312 | 1,091 | 1,067 | 23 | 3 | 39 KB |
| `/api/search` 1 scene, videos (60 videos) | 862 | 1,152 | 858 | 830 | 22 | 4 | 42–62 KB |
| `/api/search` 2 scenes, chains (300) | 1,892 | 2,470 | 1,884 | 1,796 | 44 | 53 | 82–84 KB |
| `/api/search` 3 scenes, chains (300) | 2,601 | 3,280 | 2,594 | 2,462 | 61 | 72 | 117 KB |
| `/api/search` 3 scenes, only scene 3 edited (2 cached) | 952 | 1,069 | 942 | 861 | 19 | 63 | 117 KB |
| `/api/similar` top 300 | 65 | 109 | 60 | – | – | – | 39 KB |
| `/api/video/{id}?query_id=` (3-scene scores, 269–491 keyframes) | 4 | 6 | – | – | – | – | 45–82 KB |
| `/api/verify` (LLM, 5 frames at 768 px) | 2,122 | 4,438 | 2,119 | – | – | – | verdicts: 4 ok, 1 warn |
| `/api/decompose` (LLM) | 947 | 1,101 | – | – | – | – | 2–3 scenes, all `source: llm` |
| `/api/chat` (LLM, 5 candidate keyframes) | 3,189 | 33,959 | – | – | – | – | see note 2 |
| `/api/vqa` (LLM, 1 frame) | 1,791 | 2,833 | – | – | – | – | – |

Text encoding is 95–98% of the search time on CPU: 1.07 s of 1.09 s for one scene, 2.46 s of 2.59 s
for three. Editing one scene of a three-scene query re-encodes only that scene and cuts the search
from 2.6 s to 0.95 s.

## GPU server (port 8788, `--device cuda`), search rows

Two passes with different texts. During pass 1 two short-lived Python processes from outside this
repository were running; they had exited before pass 2.

| Request | Pass 2 round trip median | Pass 2 p90 | Pass 2 server median | Encode | Scan | Rank | Pass 1 median / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/api/search` 1 scene, frames | 102 | 114 | 95 | 45 | 42 | 5 | 166 / 199 |
| `/api/search` 1 scene, videos | 116 | 146 | 111 | 47 | 49 | 8 | 138 / 178 |
| `/api/search` 2 scenes, chains | 291 | 360 | 281 | 101 | 85 | 82 | 387 / 480 |
| `/api/search` 3 scenes, chains | 377 | 387 | 366 | 136 | 124 | 109 | 579 / 731 |
| `/api/search` 3 scenes, only scene 3 edited | 260 | 308 | 243 | 60 | 41 | 117 | 263 / 356 |

On the GPU the text model takes 45 ms per scene, so a three-scene query answers in about 0.4 s
instead of 2.6 s. The similarity scan and the ranking run on the CPU in both servers, yet they were
slower in the GPU server (scan 41–48 ms per scene against 19–25 ms, chain ranking 77–165 ms against
47–127 ms). The venv has a newer numpy (2.5.2 against 2.4.6) and the server process had about
1.2 GB resident while RAM was short; which of the two causes it was not measured.

## Startup

| Step | CPU server | GPU server |
|---|---:|---:|
| `Collection.load` | 0.46 s | 0.39 s |
| SigLIP-2 model load | 46.3 s | 90.1 s |
| Warm-up similarity scan | 1.91 s | 2.21 s |
| Warm-up encoder call | 8.28 s | 1.73 s |
| Launch to first `/api/status` answer | 57 s | 96 s |

The model load was slow because available RAM stayed between 0.2 and 3.4 GB while other programs ran
(the builders measured 24.9 s for the same load earlier). The model loader also sent 28 (CPU) and 26
(GPU) HEAD/GET requests to huggingface.co at start-up; at the venue set `HF_HUB_OFFLINE=1` so the server
starts without internet.

## Notes

1. Load on the machine: other sessions' programs (a second web app, test runs of other projects, VS
   Code, browsers) kept total CPU use around 18–23% during the runs. The ingestion benchmark had
   finished before the CPU server was started, and no process of `.venv-gpu` other
   than this server was running during the GPU passes.
2. `/api/chat` p90: in run 4, `gemini-3.5-flash-lite` answered HTTP 429 Too Many Requests after
   0.6 s (later calls to that model succeeded, so it was not the daily quota) and the fallback `gemini-2.5-flash-lite` took 33.3 s to answer, so that call took 34.0 s. The
   other four chat calls took 2.4–3.3 s. The next calls went back to `gemini-3.5-flash-lite`.
3. The first `/api/verify` call of the process (warm-up) took 7.3 s; later calls 1.8–4.4 s.
4. The LLM rows depend on the network and on Google's load at that minute. They are not a property of
   the laptop.
5. Verify verdicts are not fully stable. For one AIC moment and the scene "a woman in a
   red dress interviews a farmer in a rice field", four live calls gave warn, warn, ok, ok (the woman
   wears a magenta top). With "yellow dress" instead of "red dress", eight live calls over three
   moments of the same video all gave warn; `real_guard.png` shows one of them.
6. The first query right after start-up can still be slow on CPU. On a third start of the CPU server
   (16:58: model load 21.5 s, warm-up scan 2.06 s, warm-up encoder
   call 11.54 s), the first operator search, a 2-scene chain query sent from the browser, took 9,247 ms,
   of which 9,080 ms was text encoding. The next requests of that session were normal (videos mode
   34 ms from cached encodings, more-like-this 131 ms).
