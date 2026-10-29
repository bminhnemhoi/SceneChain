"""The AIC adapter on a miniature AIC-shaped data folder (invented ids and titles)."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from scenechain.ingest import build_from_aic
from scenechain.collection import Collection


@pytest.fixture()
def aic_data(tmp_path):
    data = tmp_path / "data"
    (data / "frames").mkdir(parents=True)
    rows = []
    for vid, count, fps in (("K01_V001", 5, 25.0), ("K01_V002", 4, 30.0)):
        for n in range(1, count + 1):
            t = (n - 1) * 4.0
            rows.append({"video_id": vid, "n": n, "frame_filename": f"{n:03d}.jpg",
                         "frame_idx": int(t * fps), "pts_time": t, "fps": fps,
                         "rel_path": f"{vid}/{n:03d}.jpg", "abs_path": f"/somewhere/{vid}/{n:03d}.jpg"})
    (data / "metadata.json").write_text(json.dumps(rows), encoding="utf-8")
    emb = np.random.default_rng(0).standard_normal((len(rows), 8)).astype(np.float32)
    np.save(data / build_from_aic.EMBEDDINGS, emb / np.linalg.norm(emb, axis=1, keepdims=True))
    (data / build_from_aic.BLANK_FRAMES).write_text(json.dumps([3, 8]), encoding="utf-8")
    with zipfile.ZipFile(data / build_from_aic.MEDIA_INFO, "w") as zf:
        zf.writestr("media-info/K01_V001.json", json.dumps(
            {"title": "A test clip", "watch_url": "https://youtube.com/watch?v=abcDEF123_-"}))
        zf.writestr("media-info/K99_V999.json", json.dumps({"title": "unrelated"}))
    return data


def test_build_writes_a_loadable_collection(aic_data, tmp_path):
    out = tmp_path / "collections" / "aic"
    summary = build_from_aic.build(aic_data, out, name="aic-test")
    assert summary["rows"] == 9 and summary["videos"] == 2 and summary["youtube_ids"] == 1

    coll = Collection.load(out)
    assert coll.name == "aic-test" and coll.n == 9 and coll.dim == 8
    # n <= 2 of each video (rows 0, 1, 5, 6) plus blank frames 3 and 8
    assert np.flatnonzero(~coll.valid).tolist() == [0, 1, 3, 5, 6, 8]
    assert coll.video("K01_V001") == {"video_id": "K01_V001", "fps": 25.0, "duration": 16.04,
                                      "n_frames": 401, "title": "A test clip",
                                      "youtube_id": "abcDEF123_-", "file": None}
    assert coll.video("K01_V002")["title"] == "K01_V002" and coll.video("K01_V002")["youtube_id"] is None
    assert coll.frames_dir == (aic_data / "frames").resolve()

    spec = json.loads((out / "collection.json").read_text(encoding="utf-8"))
    assert spec["embeddings"] == (aic_data / build_from_aic.EMBEDDINGS).resolve().as_posix()
    assert spec["model"] == "google/siglip2-so400m-patch14-384"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert set(meta[0]) == set(build_from_aic.META_KEYS)
    raw = (out / "videos.json").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r\n" not in raw


def test_skip_first_n_and_missing_media_info(aic_data, tmp_path):
    (aic_data / build_from_aic.MEDIA_INFO).unlink()
    (aic_data / build_from_aic.BLANK_FRAMES).unlink()
    summary = build_from_aic.build(aic_data, tmp_path / "c", skip_first_n=0)
    assert summary["invalid_rows"] == 0 and summary["media_info"] is False
    assert Collection.load(tmp_path / "c").video("K01_V001")["title"] == "K01_V001"


def test_inconsistent_inputs_raise(aic_data, tmp_path):
    (aic_data / build_from_aic.BLANK_FRAMES).write_text("[99]", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        build_from_aic.build(aic_data, tmp_path / "c")
    np.save(aic_data / build_from_aic.EMBEDDINGS, np.zeros((3, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="rows"):
        build_from_aic.build(aic_data, tmp_path / "c")
    with pytest.raises(FileNotFoundError):
        build_from_aic.build(tmp_path / "nowhere", tmp_path / "c")


def test_cli(aic_data, tmp_path, capsys):
    assert build_from_aic.main(["--data", str(aic_data), "--out", str(tmp_path / "cli")]) == 0
    assert json.loads(capsys.readouterr().out)["rows"] == 9
