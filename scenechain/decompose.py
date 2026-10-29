"""Split a long textual description into the ordered visual scenes it describes.

Why: a known-item description often narrates several consecutive moments ("a man opens a door,
then sits down at a desk"), while one text embedding is matched against one keyframe. In AIC 2026,
28 of the 55 real queries of the first two rounds (51%) described consecutive scenes, and retrieving
scene B separately raised answer-keyframe recall from 53% to 76% on 66 two-scene queries of the
team's distribution-matched benchmark. Those numbers motivate the feature; they are not a
measurement of this module.

How: Gemini (model order of ``llm.GEMINI_MODELS``) is asked, at temperature 0 with JSON output, for 1-4
ordered, self-contained scene descriptions. Answers are cached on disk under ``cache/decompose/``
keyed by sha1 of the text, so repeating a query costs nothing and returns the same split. Without a
key, without network, or on an unusable reply, a rule-based splitter takes over. :func:`decompose`
never raises.

The rule splitter starts a new scene only at a temporal cue (then, afterwards, after that, later,
next, finally, followed by, before). A sentence without a cue is treated as more detail of the
current scene, because describers routinely spread one moment over several sentences and splitting
those would force the temporal search to find the details in different keyframes.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional, TypedDict

from .config import VBS_ROOT, load_settings
from .llm import GeminiClient, LLMClient, extract_json_object

MAX_SCENES = 4
CACHE_DIR = VBS_ROOT / "cache" / "decompose"
#: bump when the prompt changes so cached splits of the old prompt are ignored
PROMPT_VERSION = 1
MAX_INPUT_CHARS = 4000

PROMPT = """You prepare queries for a video retrieval engine that matches ONE text against ONE video keyframe.
Split the description below into the ordered visual scenes it describes.

Rules:
- Return 1 to 4 scenes, in the order in which they happen in the video.
- Start a new scene only when the description moves to a different moment, shot or place (for example "then", "afterwards", "next", "the camera cuts to"). All details of the same moment stay in one scene.
- Every scene must be self-contained: repeat the subject instead of using pronouns, and keep colours, counts, clothing, objects, places and visible text.
- Describe only what can be seen in a single frame; leave out sounds, speech and interpretation.
- Write each scene in English, in at most 40 words. Do not invent details.

Description:
\"\"\"{text}\"\"\"

Return only JSON of the form {{"scenes": ["first scene", "second scene"]}}."""


class DecomposeResult(TypedDict):
    scenes: list[str]
    source: Literal["llm", "rule"]
    model: Optional[str]


# ---------------------------------------------------------------------------------- rules
CUE_WORDS = ("then", "afterwards", "after that", "later", "next", "finally", "followed by", "before")
_CUE = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in sorted(CUE_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
#: cues that mark a transition even in the middle of a clause ("opens a door then sits down")
_STRONG_CUES = {"then", "afterwards", "after that", "followed by", "before"}
_CLAUSE_BOUNDARY = re.compile(r"(?:[,;:]|\band)\s*$", re.IGNORECASE)
#: "since then", "by then", "next to": not transitions
_NOT_A_CUE_BEFORE = re.compile(r"\b(?:since|by|until|till|from)\s*$", re.IGNORECASE)
_NOT_A_CUE_AFTER = re.compile(r"^\s*to\b", re.IGNORECASE)
#: "before that, ..." / "before this, ..." introduces an EARLIER scene
_REVERSAL = re.compile(r"^\s*(?:that|this)\b[\s,:]*", re.IGNORECASE)
_SENTENCES = re.compile(r"(?<=[.!?;])\s+|\n+")


def _clean(clause: str) -> str:
    text = re.sub(r"\s+", " ", clause).strip(" \t,;:.")
    text = re.sub(r"^(?:and|but|so)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(?:and|but|so)$", "", text, flags=re.IGNORECASE).strip(" ,;:.")
    return text[:1].upper() + text[1:]


def _join(first: str, second: str) -> str:
    return f"{first}. {second}"


def _cue_splits(sentence: str) -> list[tuple[Optional[str], str]]:
    """``[(cue or None, clause), ...]``: the sentence cut at every accepted temporal cue."""
    pieces: list[tuple[Optional[str], str]] = []
    cue: Optional[str] = None
    last = 0
    for m in _CUE.finditer(sentence):
        word = m.group(1).lower()
        before, after = sentence[last:m.start()], sentence[m.end():]
        at_start = not before.strip()
        if _NOT_A_CUE_BEFORE.search(before) or (word == "next" and _NOT_A_CUE_AFTER.match(after)):
            continue
        if not (at_start or word in _STRONG_CUES or _CLAUSE_BOUNDARY.search(before)):
            continue
        if not at_start:
            pieces.append((cue, before))
        cue, last = word, m.end()
    pieces.append((cue, sentence[last:]))
    return pieces


def split_scenes_by_rule(text: str) -> list[str]:
    """Deterministic scene split on temporal cue words; at most :data:`MAX_SCENES` scenes."""
    scenes: list[str] = []
    for sentence in _SENTENCES.split(text or ""):
        for cue, raw in _cue_splits(sentence):
            reversed_order = cue == "before" and bool(_REVERSAL.match(raw))
            clause = _clean(_REVERSAL.sub("", raw, count=1) if reversed_order else raw)
            if not clause:
                continue
            if cue is None and scenes:
                scenes[-1] = _join(scenes[-1], clause)
            elif reversed_order and scenes:
                scenes.insert(len(scenes) - 1, clause)
            else:
                scenes.append(clause)
    return _cap(scenes)


def _cap(scenes: list[str]) -> list[str]:
    if len(scenes) <= MAX_SCENES:
        return scenes
    return scenes[:MAX_SCENES - 1] + [". ".join(scenes[MAX_SCENES - 1:])]


# ------------------------------------------------------------------------------------ LLM
def _parse_scenes(reply: str) -> list[str]:
    scenes = extract_json_object(reply).get("scenes")
    if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
        raise ValueError("'scenes' is not a list of strings")
    cleaned = [s.strip() for s in scenes if s.strip()]
    if not cleaned:
        raise ValueError("no scenes in reply")
    return _cap(cleaned)


@lru_cache(maxsize=1)
def default_llm() -> Optional[LLMClient]:
    """The process-wide Gemini client from the settings, or ``None`` without a key."""
    key = load_settings().gemini_key
    return GeminiClient(key) if key else None


def _cache_path(cache_dir: Path, text: str) -> Path:
    return cache_dir / f"{hashlib.sha1(text.encode('utf-8')).hexdigest()}.json"


def _read_cache(path: Path) -> Optional[DecomposeResult]:
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("prompt_version") != PROMPT_VERSION:
        return None
    scenes = entry.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        return None
    return {"scenes": [str(s) for s in scenes], "source": "llm", "model": entry.get("model")}


def _write_cache(path: Path, text: str, result: DecomposeResult) -> None:
    entry = {"prompt_version": PROMPT_VERSION, "text": text, **result}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # a read-only disk costs a repeated call, never a failed query


def decompose(
    text: str,
    *,
    llm: Optional[LLMClient] = None,
    cache_dir: Optional[Path] = CACHE_DIR,
) -> DecomposeResult:
    """``{"scenes": [...], "source": "llm" | "rule", "model": str | None}``; never raises.

    ``llm`` defaults to :func:`default_llm`. ``cache_dir=None`` disables the disk cache. Blank text
    yields no scenes. Only LLM splits are cached, so a rule fallback is retried next time.
    """
    clean = (text or "").strip()
    if not clean:
        return {"scenes": [], "source": "rule", "model": None}
    try:
        path = _cache_path(cache_dir, clean) if cache_dir is not None else None
        cached = _read_cache(path) if path is not None else None
        if cached is not None:
            return cached
        client = llm if llm is not None else default_llm()
        if client is not None:
            scenes, model = client.generate(
                PROMPT.format(text=clean[:MAX_INPUT_CHARS]), _parse_scenes
            )
            result: DecomposeResult = {"scenes": scenes, "source": "llm", "model": model}
            if path is not None:
                _write_cache(path, clean, result)
            return result
    except Exception:  # noqa: BLE001 - the contract: fall back to rules, never fail a query
        pass
    return {"scenes": split_scenes_by_rule(clean) or [clean], "source": "rule", "model": None}
