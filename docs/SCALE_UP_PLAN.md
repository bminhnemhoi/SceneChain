# Scale-up plan: from the AIC 2026 index to the VBS 2027 collections

Written 17 Sep 2026 (engineering record; `data/...` paths refer to the authors' local AIC 2026
working copy, which is not part of this repository). VBS 2027 runs at MMM 2027 on 5 Jan 2027. This plan sizes the work of indexing
the VBS collections with the ingestion code in `scenechain/ingest/` and the SigLIP-2 so400m-384 image
encoder, and schedules it for October to December 2026.

Every number carries one of four labels:

- **[measured]** on our hardware on 17 Sep 2026, with the command or script named;
- **[cited]** from the publication given in section 7;
- **[arithmetic]** computed only from measured and cited numbers (the formula is shown);
- **[estimate]** an assumption we have not measured yet, with the reason and the week in which a
  measurement replaces it.

Sizes use decimal units (1 GB = 10^9 bytes).

## 1. What was measured

### 1.1 Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB, driver 581.86, 75 W power cap as shown by `nvidia-smi` |
| CPU / RAM | 12 logical cores, 16 GB (shared with other work during the runs, see 1.5) |
| GPU software | isolated venv `.venv-gpu`: torch 2.14.0+cu130, torchvision 0.29.0+cu130, transformers 5.17.0, numpy 2.5.2 (index URL `https://download.pytorch.org/whl/cu130`, the CUDA 13.0 build listed for Windows on pytorch.org for PyTorch 2.14.0) |
| CPU software | global Python 3.12.6: torch 2.13.0+cpu, transformers 5.15.1, opencv-python-headless 5.0.0, PySceneDetect 0.7.1 |
| Model | `google/siglip2-so400m-patch14-384`, vision tower only (`SiglipVisionModel`), SDPA attention |

### 1.2 Image encoding throughput [measured]

`encode_images.py --collection <2,000 keyframes> --device cuda --batch B --workers 4 --restart`,
float16 weights, on 2,000 keyframes drawn at random (seed 2027) from the AIC 2026 collection, read
as the 512 px JPEG mirror in `data/frames` (28.2 KB mean file size), with the files read once
beforehand so that every run reads from the OS cache.

| Batch | Images/s, whole run | Images/s after the first batch | Peak VRAM allocated |
|---:|---:|---:|---:|
| 32 | 17.95 | 21.04 | 1,455 MB |
| 64 | 21.11 | **23.60** | 2,068 MB |
| 128 | 21.26 | 23.82 | 3,300 MB |
| 256 | out of memory | | |

- The GPU is the bottleneck, not JPEG decoding: the model alone on random input runs at
  25.6 images/s at batch 32 (`torch.randn` batches, 20 s window). Batches above 64 buy nothing.
- The whole-run rate includes spawning four worker processes on Windows and the first CUDA kernel
  selection (up to about 25 s for the first batch); for multi-hour runs the rate after the first batch applies.
- Run-to-run spread: an earlier batch-32 run (pinned host memory, prefetch 4) gave 18.15 / 23.18
  images/s, so treat differences below about 10% as noise (laptop thermals).
- Pinned host memory failed at batch 128 (`CUDA_ERROR_ALREADY_MAPPED` from `cuMemHostAlloc`), so
  `encode_images.py` does not pin memory; the host-to-device copy is not the bottleneck.
- CPU float32 (global torch, its default 8 intra-op threads): 0.27 to 0.37 images/s in the runs below, and as low as
  0.09 images/s while other processes loaded the machine. CPU encoding is for demo-size
  collections only: the 170 sample keyframes take about 8 minutes.

Planning rate used below: **23.6 images/s** (batch 64, after the first batch).

### 1.3 Parity with the AIC 2026 index [measured]

`encode_images.py --verify-against data/embeddings_siglip2_384.npy --verify-k K --seed 0`
re-encodes K random rows and compares them with the stored rows (built with
`notebooks/index-siglip2.ipynb`, written for a Colab T4, from the organisers' 1280x720 JPEG keyframes).

| Frames read | Device, precision | K | Mean cosine | Min | Share > 0.999 |
|---|---|---:|---:|---:|---:|
| original 1280x720 keyframes (downloaded for the sampled rows) | CUDA, fp16 | 200 | 1.00000 | 0.99997 | 1.000 |
| original 1280x720 keyframes | CPU, fp32 | 32 | 0.99999 | 0.99989 | 1.000 |
| 512 px mirror `data/frames` (PIL thumbnail, JPEG q85) | CUDA, fp16 | 200 | 0.88551 | 0.81206 | 0.000 |
| 512 px mirror | CPU, fp32 | 32 | 0.88203 | 0.76336 | 0.000 |

The preprocessing is therefore identical to the notebook, fp16 on CUDA changes nothing measurable,
and the only source of disagreement is the stored image. How the stored image matters (64 random
AIC keyframes, originals re-saved with OpenCV `INTER_AREA`, then encoded with the same processor):

| Stored as | Mean file size | Mean cosine to the index |
|---|---:|---:|
| long side 512, JPEG q90 (the `extract_keyframes.py` default) | 37 KB | 0.921 |
| long side 512, PNG | 197 KB | 0.986 |
| long side 768, JPEG q95 | 96 KB | 0.992 |
| long side 1024, JPEG q95 | 143 KB | 0.997 |
| long side 1024, PNG | 643 KB | 0.998 |
| 384x384 squashed with OpenCV, PNG | 213 KB | 0.993 |

Effect on retrieval, 24 invented generic English queries (for example "a boat on a river",
"fireworks at night") ranked over the same 2,000 keyframes with the index rows versus the
mirror-encoded rows: mean top-10 overlap 0.64 (lowest 0.3), mean top-50 overlap 0.62; the top-1 of
the index stays inside the mirror's top 10 for 24 of 24 queries (median rank 2, worst 8).

Consequence: encode full-resolution images wherever they exist. `extract_keyframes.py
--encode-long-side 1024` writes a quality-95 copy to `frames_encode/` that `encode_images.py`
reads through `encode_frames_dir`, while the server keeps serving the 512 px frames; for V3C the
dataset's own full-resolution PNG keyframes can be encoded directly (section 3).

### 1.4 Keyframe extraction speed [measured]

`extract_keyframes.py` on the three sample films from `download_sample.py` (Blender open movies,
480p VP9 WebM transcodes from Wikimedia Commons: 854 px wide, 24 fps, 9,298 frames, 402.9 s):

| Method | Decode passes | Wall time | Speed vs real time | Keyframes |
|---|---:|---:|---:|---:|
| `scenedetect` (ContentDetector, threshold 27) | 2 | 16.1 s | 25x | 170 from 118 shots |
| `histogram` (built-in) | 2 | 17.6 s | 23x | 200 from 142 shots |
| `shots` (external boundary files) | 1 | 3.6 s | 112x | 170 |
| `uniform --fps 0.5` | 1 | 3.7 s | 109x | 203 |

The single-pass methods only grab frames and convert the selected ones to images. With
`--encode-long-side 1024` the encoder copies of these films take 59 KB per keyframe (native
854 px, q95) against 18 KB for the 512 px JPEGs. On these films the 1-to-3-keyframes-per-shot rule
gave 1.44 keyframes per shot.

### 1.5 Conditions to keep in mind

The laptop was shared with other work during the day (a VBS server holding the AIC index, a local
LLM server that briefly used the GPU, other test runs). The GPU throughput table was taken before the
LLM server started. Later, with about 38 GB of the 43.5 GB Windows commit limit taken by other
processes, loading the checkpoint failed with an access violation in both Python environments, while
a numpy read of the same safetensors file worked; the likely cause is that torch maps the 4.5 GB
file copy-on-write, which Windows charges against the commit limit. One DataLoader run with four
workers also stalled under that pressure. Production runs need a machine with nothing else loaded
(or a larger page file).

## 2. Collections [cited]

| Collection | Videos | Duration | Shots / segments | Notes |
|---|---:|---:|---:|---|
| V3C1 | 7,475 | 1,000 h 23 min 50 s | 1,082,659 | Table 1 of [1] |
| V3C2 | 9,760 | 1,300 h 52 min 48 s | 1,425,454 | Table 1 of [1]; [2] reports 1,425,451 segments, about 146 per video, most common resolution 1280x720 then 1920x1080 |
| V3C3 | 11,215 | 1,500 h 8 min 57 s | 1,635,580 | Table 1 of [1] |
| V3C total | 28,450 | 3,801 h 25 min 35 s | 4,143,693 | videos 4.8 TB, 8.7 TB with keyframes and metadata [1]; one master-shot segmentation per video as TSV plus one full-resolution PNG keyframe and a 200 px thumbnail per segment [1] |
| Marine Video Kit, first shard | 1,379 | slightly above 12 h | single-shot clips | 2 s to 4.95 min, mean 29.9 s, recorded at 720p to 4K, 30 fps [3] |
| Marine Video Kit as used at VBS 2025 | 1,372 (MVK1) + 1,820 (MVK2) | over 28 h | single-shot clips | [4] |
| Gynecological laparoscopy collection used at VBS 2024 and 2025 | 72 laparoscopies | about 100 h | surgical video | [4] |

VBS 2027 has not announced its collections yet [estimate: the VBS 2025 set of V3C, MVK and the
laparoscopy collection]. V3C averages 3.30 s per segment (13,685,135 s / 4,143,693) [arithmetic].

## 3. Keyframes, GPU time, disk and RAM

Two sampling policies: one keyframe per V3C master-shot segment, and one keyframe every 2 s (duration
in seconds / 2). MVK clips are single shots [3] and we assume the laparoscopy recordings have no
useful shot structure either [estimate], so only the time-based policy is listed for them.

All columns are [arithmetic] from sections 1 and 2, except the T4 column and the JPEG column, which
are [estimate] as explained below the table.

| Collection | Policy | Keyframes | RTX 4050 GPU-h | Colab T4 GPU-h [estimate] | 512 px JPEG [estimate] | float16 embeddings |
|---|---|---:|---:|---:|---:|---:|
| V3C1 | 1 per shot | 1,082,659 | 12.7 | 12.7 to 18.2 | 41 GB | 2.49 GB |
| V3C1 | 1 per 2 s | 1,800,715 | 21.2 | 21.2 to 30.3 | 68 GB | 4.15 GB |
| V3C2 | 1 per shot | 1,425,454 | 16.8 | 16.8 to 24.0 | 54 GB | 3.28 GB |
| V3C2 | 1 per 2 s | 2,341,584 | 27.6 | 27.6 to 39.4 | 89 GB | 5.40 GB |
| V3C3 | 1 per shot | 1,635,580 | 19.3 | 19.3 to 27.5 | 62 GB | 3.77 GB |
| V3C3 | 1 per 2 s | 2,700,269 | 31.8 | 31.8 to 45.4 | 102 GB | 6.22 GB |
| **V3C total** | 1 per shot | **4,143,693** | **48.8** | 48.8 to 69.7 | 157 GB | **9.55 GB** |
| **V3C total** | 1 per 2 s | **6,842,568** | **80.5** | 80.5 to 115.1 | 259 GB | **15.77 GB** |
| MVK (VBS 2025, 28 h lower bound) | 1 per 2 s | 50,400 | 0.6 | 0.6 to 0.8 | 1.9 GB | 0.12 GB |
| Laparoscopy (100 h) | 1 per 2 s | 180,000 | 2.1 | 2.1 to 3.0 | 6.8 GB | 0.41 GB |

- **RTX 4050 GPU-h** = keyframes / 23.6 images/s / 3,600. Using the whole-run rate (21.11 images/s)
  instead adds 12%: 54.5 h for V3C at one per shot. Measured on 512 px JPEGs; decoding full-resolution
  PNG or 1024 px JPEG in the workers costs more CPU per image. With four workers the loader had
  headroom (the GPU-only rate is 25.6), but this is **[estimate]** until the week-40 measurement.
- **Colab T4 GPU-h [estimate]**: no T4 measurement exists (the AIC notebook run was not timed). We assume
  a T4 delivers 0.7 to 1.0 times the laptop RTX 4050's throughput on this fp16 ViT (both around 70 W,
  Turing versus Ada Lovelace). The first Colab session of week 40 replaces this with the same
  `encode_images.py --report` run on the same 2,000 keyframes.
- **512 px JPEG [estimate]**: keyframes x 37 KB, the measured mean for 1280x720 AIC keyframes saved
  as `extract_keyframes.py` does (section 1.3). V3C content differs, and the animated sample films gave
  18 KB, so the true figure may be lower. For one keyframe per shot, V3C's own 200 px thumbnails
  [1] can serve the result grid instead, with no extra disk.
- **Encoder copies**: at 143 KB (1024 px, q95, measured on AIC) V3C at one per 2 s would need about
  1.0 TB [arithmetic], which is why dense V3C sampling should decode frames in the encoder instead
  (work item W4).
- **float16 embeddings** = keyframes x 1,152 x 2 bytes; the shards written during encoding are float32,
  so the peak is three times this (15.77 GB x 3 = 47 GB for V3C at one per 2 s) until concatenation.

### 3.1 RAM for search

Search keeps the matrix in RAM (memory-mapped) and scores every row per query.

| V3C total | Exact, float16 | Exact, float32 | int8 scalar quantisation | IVF-PQ, 96-byte codes + 8-byte ids |
|---|---:|---:|---:|---:|
| 1 per shot (4,143,693 rows) | 9.55 GB | 19.09 GB | 4.77 GB | 0.43 GB + 0.30 GB centroids |
| 1 per 2 s (6,842,568 rows) | 15.77 GB | 31.53 GB | 7.88 GB | 0.71 GB + 0.30 GB centroids |

[arithmetic]: bytes per row are 1,152 x 2, x 4, x 1 and 96 + 8; centroids for 65,536 IVF lists are
65,536 x 1,152 x 4 bytes. Recall of int8 and IVF-PQ against exact search has not been measured
(faiss is not installed in any environment yet); it is work item W5 and nothing compressed goes on
stage without it.

Scan cost of one query over the 177,321-row AIC index on this laptop's CPU (numpy 2.4.6 with
OpenBLAS, best / median of 7 runs, scratch `scan_bench.py`) [measured]:

| Row format in RAM | 177,321 rows | V3C one per shot, 4,143,693 rows [estimate: linear in rows] |
|---|---:|---:|
| float32, one matrix product | 20 / 22 ms | 0.47 / 0.51 s |
| float16, converted to float32 in 200,000-row chunks per query | 335 / 384 ms | 7.8 / 9.0 s |
| int8, dequantised to float32 per query | 198 / 219 ms | 4.6 / 5.1 s |

The conversion, not the product, dominates the float16 and int8 rows. A multi-scene query scores
every scene, so these times multiply by the number of scenes. On this laptop neither compact format
is interactive at V3C scale with a plain numpy scan: the event machine needs either float32 in RAM
(19.1 GB for one keyframe per shot, so a 32 GB machine, about half a second per scene) or a
compressed index with SIMD distance code (faiss int8 or IVF-PQ), whose latency and recall W5 measures.

The metadata costs memory too. Parsing a `metadata.json` with the six contract fields for the
177,321 AIC rows takes 1.2 s and holds 81 MB of Python objects, 456 bytes per row (tracemalloc)
[measured]; for V3C at one keyframe per shot that is 1.9 GB and about 28 s at start-up [estimate:
linear in rows], in every process that loads the collection (server, encoder). W7 replaces it with
columnar arrays before V3C is loaded.

Memory alone: on the 16 GB laptop, float16 V3C at one keyframe per shot (9.55 GB) fits only with
little else running and one per 2 s (15.77 GB) does not fit; combined with the scan times above, the
realistic options are a 32 GB machine with float32 rows for one keyframe per shot, or faiss after W5.

## 4. How each collection is ingested

- **V3C, one keyframe per shot.** Use the official master-shot boundaries and the official
  full-resolution PNG keyframes: no video decoding at all. A small adapter (`build_from_v3c.py`, W2)
  writes `metadata.json` from the segment TSVs (the conversion to `start_frame,end_frame` is documented
  in `extract_keyframes.py`), points `encode_frames_dir` at the PNG keyframes and `frames_dir` at
  512 px JPEG thumbnails. The collection format requires the same relative file names in both
  directories, so W2 also decides how the `.png` originals and `.jpg` thumbnails are named (a name
  mapping in `encode_images.py`, or re-saved originals). DRES media items are the 5-digit V3C video ids.
- **V3C, denser.** `extract_keyframes.py --method shots --shots-dir <converted msb>` (one decode pass,
  112x real time on 480p VP9 [measured]; the most common V3C resolutions are 1280x720, then
  1920x1080 [2], so decoding will be slower [estimate, measured in week 40]) plus `--max-gap 2` for long shots.
- **Marine Video Kit.** Single-shot clips [3]: `--method uniform --fps 0.5`, with
  `--encode-long-side 1024`.
- **Laparoscopy videos.** Surgical recordings [4], assumed to have no useful shot cuts [estimate]:
  `--method uniform --fps 0.5`; check in week 45
  whether 1 fps is needed for instrument-level known-item tasks (360,000 keyframes, 4.2 GPU-h
  [arithmetic]).

## 5. Work items

| Id | Work | Size [estimate] |
|---|---|---|
| W1 | Timing runs: `encode_images.py --report` on a Colab T4 with the 2,000-keyframe set; decode speed of 20 V3C1 videos with `--method shots` and `--method scenedetect`; encoding throughput on V3C full-resolution PNGs | 1 day |
| W2 | `build_from_v3c.py`: segment TSV -> metadata, official PNGs as `encode_frames_dir`, thumbnails as `frames_dir`, resumable | 2 days |
| W3 | Nightly encoding runs with shard resume (the laptop GPU at 23.6 images/s is about 0.68 million keyframes per 8-hour night [arithmetic]) | see schedule |
| W4 | Encode straight from the video files (decode in DataLoader workers by `frame_idx`), so dense V3C sampling needs no encoder copies on disk | 3 days |
| W5 | Compressed search: faiss in its own venv; recall@100 of int8 and IVF-PQ against exact search with at least 50 invented queries per collection; query latency | 4 days |
| W6 | Multi-collection server start-up and DRES collection names per collection | 2 days |
| W7 | Columnar metadata (numpy arrays for `frame_idx`, `pts_time`, video index; string table for ids and file names) loaded alongside or instead of `metadata.json` for million-row collections; needs a contract change in §3 | 2 days |

## 6. Schedule, October to December 2026

| Dates | What | GPU budget [arithmetic unless noted] | Exit check |
|---|---|---|---|
| 28 Sep - 4 Oct | Request V3C access (data agreement), fetch MVK and the laparoscopy collection; W1 | Colab T4 session, about 1 h | T4 and V3C decode speeds measured; estimates in this file replaced |
| 5 - 18 Oct | W2 and W7; V3C1 at one keyframe per shot | 12.7 h on the RTX 4050: 2 nights | `--verify-against` style spot check on 200 V3C1 PNGs re-encoded; 20 invented queries give sensible top 10 |
| 19 Oct - 8 Nov | V3C2 and V3C3 at one keyframe per shot, split between laptop nights and Colab T4 sessions (shards make the split safe) | 36.1 h on the RTX 4050 (16.8 + 19.3), or 36 to 52 h on a T4 [estimate] | full V3C one-per-shot collection loads; exact search latency on 4.14 M rows measured |
| 9 - 22 Nov | MVK and laparoscopy at one keyframe per 2 s with encoder copies; W5 | 2.7 h | recall and latency table for faiss int8 / IVF-PQ; decision on the event machine (32 GB with float32 rows, or faiss) |
| 23 Nov - 13 Dec | W4 and W6; only if W5 and query tests show one keyframe per shot misses content: densify V3C to one per 2 s | up to 80.5 h on the RTX 4050 (about 10 nights) or 81 to 115 h on a T4 [estimate] | DRES rehearsal with every collection loaded |
| 14 Dec - 4 Jan | Freeze code and indexes; two copies of every collection on separate disks; novice rehearsal | none | tool starts offline from the copies |

Disk to reserve for the one-per-shot plan: 157 GB of 512 px thumbnails [estimate] (or none with the
official 200 px thumbnails) plus 9.55 GB of float16 embeddings, and 28.6 GB at the moment of
concatenation when all of V3C is one collection (19.09 GB of float32 shards + the float16 file)
[arithmetic], on top of the 4.8 TB of V3C videos if they are kept locally [1].

## 7. Sources

1. L. Rossetto, H. Schuldt, G. Awad, A. A. Butt. *V3C – A Research Video Collection.* MultiMedia
   Modeling (MMM 2019), pp. 349–360, 2019. doi:10.1007/978-3-030-05710-7_29,
   arXiv:1810.04401 (Table 1, section 3.1).
2. L. Rossetto, A. Bernstein, K. Schoeffmann. *Insights on the V3C2 Dataset.* arXiv:2105.01475, 2021.
3. Q.-T. Truong, T.-A. Vu, T.-S. Ha, J. Lokoč, Y. H. W. Tim, A. Joneja, S.-K. Yeung. *Marine Video
   Kit: A New Marine Video Dataset for Content-Based Analysis and Retrieval.* MultiMedia Modeling
   (MMM 2023), 2023. doi:10.1007/978-3-031-27077-2_42, arXiv:2209.11518.
4. L. Rossetto, K. Schoeffmann, C. Gurrin, J. Lokoč, W. Bailer. *Results of the 2025 Video
   Browser Showdown.* arXiv:2509.12000, 2025 (section 2, competition setup).

Measurement scripts not in the repository (scratch): the 2,000-keyframe benchmark collection
(`collection.json` with `frames_dir` pointing at `data/frames`), the model-only timing loop, the
storage-variant parity probe and the ranking-overlap probe. The commands for everything else are
in the section that reports the number.
