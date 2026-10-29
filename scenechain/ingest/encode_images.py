"""Keyframes -> SigLIP-2 image embeddings: the second stage of ingestion (ARCHITECTURE.md §3, §8).

Preprocessing and pooling are those of ``notebooks/index-siglip2.ipynb``, which built the AIC 2026
index (``data/embeddings_siglip2_384.npy``): the checkpoint's ``AutoProcessor`` (squash resize to
384x384 without cropping, rescale by 1/255, normalise with mean 0.5 / std 0.5), the pooled output of
the vision tower (its attention-pooling head, i.e. ``SiglipModel.get_image_features().pooler_output``)
and L2 normalisation. Only the vision tower is loaded (``SiglipVisionModel``); its output matches
``SiglipModel.get_image_features`` of the full checkpoint. On CUDA the tower runs with float16
weights by default; on CPU in float32.

Encoding mode (``--collection``) reads ``collection.json`` / ``metadata.json``, encodes the keyframes
in row order (from ``encode_frames_dir`` when the collection has one, else ``frames_dir``) and writes ``embeddings_parts/part-NNNNN.npy`` shards (float32) via atomic renames.
A rerun skips completed shards; ``embeddings_parts/manifest.json`` pins the rows (sha1 of every
``video_id/frame_filename``), the frames directory, the shard size and the encoder, and a mismatch is an error unless
``--restart`` is given. When all shards exist they are concatenated into ``embeddings.npy``
(float16 by default, ``--float32`` to keep full precision), ``encode_manifest.json`` records what was
encoded, ``collection.json`` gains ``model``, ``dim`` and ``embeddings``, and the shards are deleted
(``--keep-parts`` keeps them). A collection whose ``embeddings.npy`` already matches is left alone.

Verification mode (``--verify-against EMB.npy``) re-encodes ``--verify-k`` random rows of an existing
index (metadata and frames default to ``metadata.json`` and ``frames/`` next to the .npy file) and
reports the cosine between each re-encoded vector and its stored row. Parity is only meaningful
when the frames on disk are the images the index was built from: re-encoding a downscaled JPEG
copy of those images lowers the cosine by itself.

Both modes print progress with images/s; ``--report FILE`` also writes the numbers as JSON.
DataLoader workers use spawned processes, so the entry point is guarded by ``__main__``.

Usage::

    python -m scenechain.ingest.encode_images --collection collections/samples --device cpu --workers 2
    python -m scenechain.ingest.encode_images --collection collections/v3c1 --device cuda --batch 64
    python -m scenechain.ingest.encode_images --verify-against data/embeddings_siglip2_384.npy --verify-k 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"
PARTS_DIR = "embeddings_parts"
PARTS_MANIFEST = "manifest.json"
FINAL_MANIFEST = "encode_manifest.json"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_VERSION = 1


class EncodeError(RuntimeError):
    """A problem the user has to fix: missing files, mismatched shards, unavailable device."""


class ImageEncoder(Protocol):
    """What the encoding loop needs from a model; ``FakeImageEncoder`` in the tests implements it too."""

    dim: int
    device: str
    #: picklable callable PIL image -> (3, H, W) tensor, run inside DataLoader workers
    preprocess: Callable[[Image.Image], torch.Tensor]

    def encode(self, pixel_values: torch.Tensor) -> np.ndarray:
        """(B, 3, H, W) batch -> (B, dim) float32 L2-normalised rows."""
        ...

    def identity(self) -> dict:
        """JSON description of everything that changes the vectors (model, preprocessing, precision)."""
        ...


class SiglipPreprocess:
    """The checkpoint's image processor applied to one image (picklable for worker processes)."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]


class SiglipImageEncoder:
    """SigLIP / SigLIP-2 fixed-resolution vision tower, pooled output, L2-normalised."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, fp16: bool | None = None) -> None:
        from transformers import AutoConfig, AutoProcessor, SiglipVisionModel
        from transformers.utils import logging as hf_logging

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise EncodeError(f"CUDA requested but this torch build ({torch.__version__}) sees no CUDA device")
        fp16 = device.startswith("cuda") if fp16 is None else fp16
        if fp16 and not device.startswith("cuda"):
            raise EncodeError("float16 inference needs a CUDA device; use --no-fp16 on CPU")
        dtype = torch.float16 if fp16 else torch.float32
        verbosity = hf_logging.get_verbosity()
        # silence warnings about the text tower, whose config and weights are skipped on purpose
        hf_logging.set_verbosity_error()
        try:
            config = AutoConfig.from_pretrained(model_name)
            if config.model_type != "siglip":
                raise EncodeError(f"{model_name} is a {config.model_type!r} checkpoint; "
                                  "only fixed-resolution SigLIP / SigLIP-2 checkpoints are supported")
            processor = AutoProcessor.from_pretrained(model_name).image_processor
            self.model = SiglipVisionModel.from_pretrained(model_name, dtype=dtype).to(device).eval()
        finally:
            hf_logging.set_verbosity(verbosity)
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.dim = int(config.vision_config.hidden_size)
        self.preprocess = SiglipPreprocess(processor)
        self._processor_desc = {
            "class": type(processor).__name__,
            "size": dict(processor.size),
            "resample": int(processor.resample),
            "image_mean": list(processor.image_mean),
            "image_std": list(processor.image_std),
        }

    def encode(self, pixel_values: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            x = pixel_values.to(self.device, dtype=self.dtype)
            vecs = self.model(pixel_values=x).pooler_output.float()
            vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy()

    def identity(self) -> dict:
        return {
            "model": self.model_name,
            "pooling": "vision pooler_output, L2",
            "processor": self._processor_desc,
            "precision": "fp16" if self.dtype == torch.float16 else "fp32",
        }


class KeyframeDataset(Dataset):
    """Image files -> preprocessed tensors, in the given order."""

    def __init__(self, paths: Sequence[str], preprocess: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = list(paths)
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
        except OSError as exc:
            raise EncodeError(f"cannot read keyframe {path}: {exc}") from exc
        return self.preprocess(rgb)


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


def run_encoder(
    paths: Sequence[str],
    encoder: ImageEncoder,
    on_batch: Callable[[np.ndarray], None],
    *,
    batch_size: int,
    workers: int,
    log: Callable[[str], None] = print,
    progress_every_s: float = 10.0,
) -> dict:
    """Encode ``paths`` in order, handing each (B, dim) batch to ``on_batch``; return timing stats.

    ``images_per_s`` covers everything from the first worker start to the last batch;
    ``steady_images_per_s`` excludes the first batch (worker spawn, CUDA kernel selection).
    """
    total = len(paths)
    stats: dict[str, Any] = {"images": total, "batch_size": batch_size, "workers": workers, "device": encoder.device}
    if total == 0:
        return stats | {"seconds": 0.0, "images_per_s": 0.0, "steady_images_per_s": 0.0, "peak_vram_mb": None}
    cuda = encoder.device.startswith("cuda")
    loader = DataLoader(
        KeyframeDataset(paths, encoder.preprocess),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        # no pinned host memory: the encoder, not the copy, is the bottleneck, and large pinned batches
        # exhaust the page-locked memory Windows grants (CUDA_ERROR_ALREADY_MAPPED at batch 128)
        pin_memory=False,
        worker_init_fn=_worker_init if workers > 0 else None,
    )
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    done = 0
    first_batch = 0
    started = time.perf_counter()
    t_first = started
    last_report = started
    for pixel_values in loader:
        try:
            vecs = encoder.encode(pixel_values)
        except torch.cuda.OutOfMemoryError as exc:
            raise EncodeError(f"CUDA out of memory at batch size {batch_size}; use a smaller --batch") from exc
        if vecs.shape != (len(pixel_values), encoder.dim):
            raise EncodeError(f"encoder returned shape {vecs.shape}, expected ({len(pixel_values)}, {encoder.dim})")
        on_batch(vecs)
        done += len(vecs)
        now = time.perf_counter()
        if first_batch == 0:
            first_batch, t_first = done, now
        if now - last_report >= progress_every_s or done == total:
            rate = done / (now - started)
            eta_min = (total - done) / rate / 60 if rate > 0 else math.inf
            log(f"encoded {done:,}/{total:,} ({100 * done / total:.1f}%) | {rate:.1f} images/s | ETA {eta_min:.1f} min")
            last_report = now
    seconds = time.perf_counter() - started
    steady_s = time.perf_counter() - t_first
    stats.update({
        "seconds": round(seconds, 3),
        "images_per_s": round(done / seconds, 2),
        "steady_images_per_s": round((done - first_batch) / steady_s, 2) if done > first_batch and steady_s > 0 else None,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1) if cuda else None,
    })
    return stats


# --------------------------------------------------------------------------- collection encoding


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise EncodeError(f"missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, obj: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(json.dumps(obj, ensure_ascii=False, indent=1).encode("utf-8"))
    os.replace(tmp, path)


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def rows_fingerprint(metadata: Sequence[dict]) -> str:
    """sha1 over ``video_id/frame_filename`` of every row, in order."""
    digest = hashlib.sha1()
    for m in metadata:
        digest.update(f"{m['video_id']}/{m['frame_filename']}\n".encode("utf-8"))
    return digest.hexdigest()


def _shard_complete(path: Path, rows: int, dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    return arr.shape == (rows, dim) and arr.dtype == np.float32


def encode_collection(
    collection_dir: Path,
    encoder: ImageEncoder,
    *,
    batch_size: int = 64,
    shard_size: int = 10_000,
    workers: int = 0,
    float16: bool = True,
    keep_parts: bool = False,
    restart: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """Encode every keyframe of a collection (see module docstring); return a report dict."""
    collection_path = collection_dir / "collection.json"
    collection = _read_json(collection_path)
    metadata = _read_json(collection_dir / collection.get("metadata", "metadata.json"))
    if not metadata:
        raise EncodeError(f"{collection_dir} has no keyframes")
    frames_source = collection.get("encode_frames_dir") or collection.get("frames_dir", "frames")
    frames_dir = Path(frames_source)
    frames_dir = frames_dir if frames_dir.is_absolute() else collection_dir / frames_dir
    n = len(metadata)
    out_dtype = "float16" if float16 else "float32"
    expected = {
        "version": MANIFEST_VERSION,
        "n": n,
        "dim": encoder.dim,
        "rows_sha1": rows_fingerprint(metadata),
        "frames": frames_source,
        "shard_size": shard_size,
        "encoder": encoder.identity(),
    }
    report: dict[str, Any] = {"collection": str(collection_dir), "n": n, "dtype": out_dtype, "encoded": 0}

    final_path = collection_dir / EMBEDDINGS_FILE
    final_manifest = collection_dir / FINAL_MANIFEST
    parts_dir = collection_dir / PARTS_DIR
    if restart:
        shutil.rmtree(parts_dir, ignore_errors=True)
    elif final_path.is_file() and final_manifest.is_file():
        stored = _read_json(final_manifest)
        same = {k: v for k, v in stored.items() if k != "dtype"} == {k: v for k, v in expected.items() if k != "shard_size"}
        if same and stored.get("dtype") == out_dtype:
            log(f"{final_path} is up to date ({n} rows); nothing to encode")
            _register_embeddings(collection_path, collection, encoder)
            return report

    parts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = parts_dir / PARTS_MANIFEST
    if manifest_path.is_file():
        stored = _read_json(manifest_path)
        if stored != expected:
            differing = sorted(k for k in set(stored) | set(expected) if stored.get(k) != expected.get(k))
            raise EncodeError(
                f"shards in {parts_dir} were written for a different {', '.join(differing)}; "
                "rerun with --restart to discard them"
            )
    else:
        _write_json_atomic(manifest_path, expected)

    shards = [(k, a, min(a + shard_size, n)) for k, a in enumerate(range(0, n, shard_size))]
    shard_path = {k: parts_dir / f"part-{k:05d}.npy" for k, _, _ in shards}
    pending = [s for s in shards if not _shard_complete(shard_path[s[0]], s[2] - s[1], encoder.dim)]
    todo = sum(b - a for _, a, b in pending)
    log(f"{n:,} keyframes in {len(shards)} shards: {len(shards) - len(pending)} complete, {todo:,} images to encode "
        f"on {encoder.device} ({encoder.identity()['precision']}, batch {batch_size}, {workers} workers)")

    if pending:
        paths = [str(frames_dir / metadata[i]["video_id"] / metadata[i]["frame_filename"])
                 for _, a, b in pending for i in range(a, b)]
        queue = iter(pending)
        current = next(queue)
        buffer: list[np.ndarray] = []
        buffered = 0

        def on_batch(vecs: np.ndarray) -> None:
            nonlocal current, buffered, buffer
            offset = 0
            while offset < len(vecs):
                k, a, b = current
                take = vecs[offset: offset + (b - a) - buffered]
                buffer.append(take)
                buffered += len(take)
                offset += len(take)
                if buffered == b - a:
                    _save_npy_atomic(shard_path[k], np.concatenate(buffer).astype(np.float32, copy=False))
                    buffer, buffered = [], 0
                    following = next(queue, None)
                    if following is None:
                        if offset != len(vecs):
                            raise EncodeError("encoder produced more vectors than there are pending rows")
                        return
                    current = following

        report.update(run_encoder(paths, encoder, on_batch, batch_size=batch_size, workers=workers, log=log))
        report["encoded"] = todo
        missing = [k for k, a, b in shards if not _shard_complete(shard_path[k], b - a, encoder.dim)]
        if missing:
            raise EncodeError(f"shards {missing} are incomplete after encoding")

    tmp = collection_dir / (EMBEDDINGS_FILE + ".tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.dtype(out_dtype), shape=(n, encoder.dim))
    for k, a, b in shards:
        out[a:b] = np.load(shard_path[k])
    out.flush()
    del out
    try:
        os.replace(tmp, final_path)
    except PermissionError as exc:
        raise EncodeError(f"cannot replace {final_path} (is a server still memory-mapping it?): {exc}") from exc
    _write_json_atomic(final_manifest, {k: v for k, v in expected.items() if k != "shard_size"} | {"dtype": out_dtype})
    _register_embeddings(collection_path, collection, encoder)
    if not keep_parts:
        shutil.rmtree(parts_dir)
    log(f"wrote {final_path} ({n:,} x {encoder.dim}, {out_dtype})")
    return report


def _register_embeddings(collection_path: Path, collection: dict, encoder: ImageEncoder) -> None:
    updated = dict(collection) | {"model": encoder.identity()["model"], "dim": encoder.dim, "embeddings": EMBEDDINGS_FILE}
    if updated != collection:
        _write_json_atomic(collection_path, updated)


# --------------------------------------------------------------------------- verification


def verify_against(
    reference: Path,
    metadata_path: Path,
    frames_dir: Path,
    encoder: ImageEncoder,
    *,
    k: int = 200,
    seed: int = 0,
    batch_size: int = 64,
    workers: int = 0,
    log: Callable[[str], None] = print,
) -> dict:
    """Re-encode ``k`` random rows of an existing index and report cosine parity with the stored rows."""
    if not reference.is_file():
        raise EncodeError(f"reference embeddings not found: {reference}")
    stored = np.load(reference, mmap_mode="r")
    metadata = _read_json(metadata_path)
    if stored.ndim != 2 or stored.shape[0] != len(metadata):
        raise EncodeError(f"{reference} has shape {stored.shape} but {metadata_path} has {len(metadata)} rows")
    if stored.shape[1] != encoder.dim:
        raise EncodeError(f"{reference} has dimension {stored.shape[1]}, the encoder produces {encoder.dim}")
    rows = np.sort(np.random.default_rng(seed).choice(len(metadata), size=min(k, len(metadata)), replace=False))
    paths = [str(frames_dir / metadata[r]["video_id"] / metadata[r]["frame_filename"]) for r in rows]
    batches: list[np.ndarray] = []
    stats = run_encoder(paths, encoder, batches.append, batch_size=batch_size, workers=workers, log=log)
    fresh = np.concatenate(batches)
    old = np.asarray(stored[rows], dtype=np.float32)
    old /= np.linalg.norm(old, axis=1, keepdims=True)
    cos = (fresh * old).sum(axis=1)
    worst = np.argsort(cos)[:5]
    return {
        "reference": str(reference),
        "frames_dir": str(frames_dir),
        "k": int(len(rows)),
        "seed": seed,
        "encoder": encoder.identity(),
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "cosine_min": float(cos.min()),
        "cosine_p05": float(np.percentile(cos, 5)),
        "cosine_max": float(cos.max()),
        "share_above_0_999": float((cos > 0.999).mean()),
        "worst": [{"row": int(rows[i]), "video_id": metadata[rows[i]]["video_id"],
                   "frame_filename": metadata[rows[i]]["frame_filename"], "cosine": float(cos[i])} for i in worst],
    } | stats


# --------------------------------------------------------------------------- command line


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collection", type=Path, help="collection directory to encode")
    mode.add_argument("--verify-against", type=Path, metavar="EMB_NPY", help="existing index to check parity against")
    ap.add_argument("--verify-metadata", type=Path, help="metadata.json of the reference (default: next to EMB_NPY)")
    ap.add_argument("--verify-frames", type=Path,
                    help="frames directory of the reference (default: frames/ next to EMB_NPY)")
    ap.add_argument("--verify-k", type=int, default=200, help="number of random rows to re-encode (default 200)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --verify-against (default 0)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"checkpoint (default {DEFAULT_MODEL})")
    ap.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda if available, else cpu")
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                    help="float16 weights (default: on for CUDA, off for CPU)")
    ap.add_argument("--batch", type=int, default=64, help="batch size (default 64)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                    help="DataLoader worker processes, 0 = load in the main process (default min(4, CPUs))")
    ap.add_argument("--shard-size", type=int, default=10_000, help="rows per shard (default 10000)")
    ap.add_argument("--float32", action="store_true", help="store embeddings.npy as float32 instead of float16")
    ap.add_argument("--keep-parts", action="store_true", help="keep the shard files after concatenation")
    ap.add_argument("--restart", action="store_true", help="discard existing shards and encode from scratch")
    ap.add_argument("--report", type=Path, help="also write the run's numbers to this JSON file")
    args = ap.parse_args(argv)
    if args.batch <= 0 or args.shard_size <= 0 or args.workers < 0 or args.verify_k <= 0:
        ap.error("--batch, --shard-size and --verify-k must be positive, --workers non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    def log(line: str) -> None:
        print(line, flush=True)

    try:
        t0 = time.perf_counter()
        encoder = SiglipImageEncoder(args.model, device=args.device, fp16=args.fp16)
        log(f"loaded {args.model} vision tower on {encoder.device} ({encoder.identity()['precision']}) "
            f"in {time.perf_counter() - t0:.1f} s; torch {torch.__version__}")
        if args.collection is not None:
            report = encode_collection(args.collection, encoder, batch_size=args.batch, shard_size=args.shard_size,
                                       workers=args.workers, float16=not args.float32, keep_parts=args.keep_parts,
                                       restart=args.restart, log=log)
        else:
            ref = args.verify_against
            report = verify_against(ref, args.verify_metadata or ref.parent / "metadata.json",
                                    args.verify_frames or ref.parent / "frames", encoder, k=args.verify_k,
                                    seed=args.seed, batch_size=args.batch, workers=args.workers, log=log)
            log(f"parity over {report['k']} rows: mean {report['cosine_mean']:.5f}, median {report['cosine_median']:.5f}, "
                f"min {report['cosine_min']:.5f}, p05 {report['cosine_p05']:.5f}, "
                f"share > 0.999: {report['share_above_0_999']:.3f}")
    except EncodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if report.get("images_per_s") is not None:
        vram = f", peak VRAM {report['peak_vram_mb']} MB" if report.get("peak_vram_mb") is not None else ""
        log(f"throughput: {report['images_per_s']} images/s overall, {report['steady_images_per_s']} images/s "
            f"after the first batch{vram}")
    report["torch"] = torch.__version__
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_bytes(json.dumps(report, indent=1).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
