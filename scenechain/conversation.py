"""KIS-C: fold the dialogue with the oracle into an updated scene list and suggest the next question.

In a conversational known-item task the team sees a short initial description and may ask the oracle
questions about the target video. Each answer should sharpen the query, and each question should be
the one whose answer best tells the current top candidates apart.

:func:`update_from_dialogue` gives an LLM the initial description, the dialogue so far, the current
ordered scenes and the top candidate videos (a text hint each and, when available, the best-matching
keyframe of each as an image). It returns the updated scenes (1-4, ordered, self-contained, as
:mod:`scenechain.decompose` writes them) and one short English question. Replies are cached on disk
under ``cache/conversation/``.

Without a model, or when the reply is unusable, a rule fallback keeps the current scenes, appends the
oracle's informative answers (not a bare "yes"/"no") to the last scene, and picks a generic
discriminating question that has not been asked yet. :func:`update_from_dialogue` never raises.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional, TypedDict

from . import decompose as decompose_module
from .config import VBS_ROOT
from .llm import LLMClient, extract_json_object
from .verify import load_jpeg

CACHE_DIR = VBS_ROOT / "cache" / "conversation"
#: bump when the prompt changes so cached replies of the old prompt are ignored
PROMPT_VERSION = 1
MAX_SCENES = decompose_module.MAX_SCENES
MAX_CANDIDATES = 5
MAX_TURNS = 30
MAX_QUESTION_WORDS = 25
MAX_INPUT_CHARS = 6000

PROMPT = """You help a team in a live conversational known-item video search competition.
The team sees a short description of ONE target video and may ask an oracle, who can see the target, questions about it.
A retrieval engine matches ONE text against ONE video keyframe, and a temporal search looks for the scenes in the given order.

Initial description:
\"\"\"{initial}\"\"\"

Dialogue so far (team questions and oracle answers, in order):
{dialogue}

Current ordered scene queries:
{scenes}

Current top candidate videos of the engine{image_note}:
{candidates}

Tasks:
1. Update the scene queries with everything the oracle has confirmed. Put each new detail into the scene it belongs to, remove details the oracle denied, and keep the order in which the scenes happen. Return 1 to 4 scenes. Every scene must be self-contained (repeat the subject instead of using pronouns), describe only what is visible in a single frame, be in English and use at most 40 words. Do not invent details.
2. Suggest ONE short English question (at most 15 words) for the oracle. It must be answerable by looking at the target video, must not repeat a question already asked, and its answer should best tell the candidate videos apart (for example a visible attribute that differs between them: setting, colours, number of people, on-screen text, time of day).

Return only JSON of the form {{"scenes": ["first scene", "second scene"], "next_question": "..."}}."""


class ConversationResult(TypedDict):
    scenes: list[str]
    next_question: str
    model: Optional[str]


#: generic questions of the rule fallback, each with words that show it was already asked
FALLBACK_QUESTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Is the scene indoors or outdoors?", ("indoor", "outdoor", "inside", "outside")),
    ("How many people are visible?", ("how many", "people", "person")),
    ("Is there any text or a logo on screen? What does it say?", ("text", "logo", "caption", "say")),
    ("What colours are the main person's clothes?", ("colour", "color", "clothes", "wearing")),
    ("Is it day or night?", ("day", "night", "dark")),
    ("Is the video a news report, an interview or a documentary?", ("news", "interview", "documentary")),
    ("What happens right before this moment?", ("before",)),
    ("What happens right after this moment?", ("after",)),
)

_BARE_ANSWER = re.compile(
    r"^\s*(?:yes|no|yeah|yep|nope|maybe|correct|right|wrong|true|false|i don'?t know|don'?t know|"
    r"unknown|not sure|no idea)\s*[.!]?\s*$",
    re.IGNORECASE,
)


def _clean_turns(turns: Sequence[Mapping]) -> list[dict]:
    out = []
    for turn in list(turns)[-MAX_TURNS:]:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role") or "").strip().lower()
        text = str(turn.get("text") or "").strip()
        if role in ("team", "oracle") and text:
            out.append({"role": role, "text": text})
    return out


def _sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip(" \t,;:.")
    return text[:1].upper() + text[1:]


def rule_update(
    initial: str, turns: Sequence[Mapping], current_scenes: Sequence[str]
) -> ConversationResult:
    """Deterministic fallback: append informative oracle answers to the last scene."""
    scenes = [s.strip() for s in current_scenes if isinstance(s, str) and s.strip()]
    if not scenes:
        base = (initial or "").strip()
        scenes = decompose_module.split_scenes_by_rule(base) or ([base] if base else [])
    clean = _clean_turns(turns)
    if scenes:
        known = " ".join(scenes).lower()
        for turn in clean:
            if turn["role"] != "oracle" or _BARE_ANSWER.match(turn["text"]):
                continue
            detail = _sentence(turn["text"])
            if detail and detail.lower() not in known:
                scenes[-1] = f"{scenes[-1].rstrip(' .')}. {detail}"
                known += " " + detail.lower()
    asked = " ".join(t["text"].lower() for t in clean if t["role"] == "team")
    question = next(
        (q for q, words in FALLBACK_QUESTIONS if not any(w in asked for w in words)),
        FALLBACK_QUESTIONS[0][0],
    )
    return {"scenes": scenes[:MAX_SCENES], "next_question": question, "model": None}


def _parse(reply: str) -> tuple[list[str], str]:
    obj = extract_json_object(reply)
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not all(isinstance(s, str) for s in scenes):
        raise ValueError("'scenes' is not a list of strings")
    cleaned = [s.strip() for s in scenes if s.strip()]
    if not cleaned:
        raise ValueError("no scenes in reply")
    if len(cleaned) > MAX_SCENES:
        cleaned = cleaned[:MAX_SCENES - 1] + [". ".join(cleaned[MAX_SCENES - 1:])]
    question = str(obj.get("next_question") or "").strip()
    if not question:
        raise ValueError("no next_question in reply")
    words = question.split()
    if len(words) > MAX_QUESTION_WORDS:
        question = " ".join(words[:MAX_QUESTION_WORDS])
    return cleaned, question


def _load_image(path: object) -> Optional[bytes]:
    try:
        p = Path(str(path))
        return load_jpeg(p) if p.is_file() else None
    except Exception:  # noqa: BLE001 - a missing thumbnail only removes that image from the prompt
        return None


def _cache_path(cache_dir: Path, payload: object) -> Path:
    key = json.dumps([PROMPT_VERSION, payload], ensure_ascii=False, sort_keys=True)
    return cache_dir / f"{hashlib.sha1(key.encode('utf-8')).hexdigest()}.json"


def _read_cache(path: Optional[Path]) -> Optional[ConversationResult]:
    if path is None:
        return None
    try:
        entry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict) or entry.get("prompt_version") != PROMPT_VERSION:
        return None
    scenes, question = entry.get("scenes"), entry.get("next_question")
    if not isinstance(scenes, list) or not scenes or not isinstance(question, str) or not question:
        return None
    return {"scenes": [str(s) for s in scenes], "next_question": question, "model": entry.get("model")}


def _write_cache(path: Optional[Path], result: ConversationResult) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"prompt_version": PROMPT_VERSION, **result}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    except OSError:
        pass


def update_from_dialogue(
    initial: str,
    turns: Sequence[Mapping],
    current_scenes: Sequence[str],
    top_videos: Sequence[Mapping] = (),
    *,
    llm: Optional[LLMClient] = None,
    cache_dir: Optional[Path] = CACHE_DIR,
) -> ConversationResult:
    """``{"scenes": [str], "next_question": str, "model": str|None}``; never raises.

    ``turns`` are ``{"role": "team"|"oracle", "text"}``; ``top_videos`` are ``{"video_id",
    "caption_hint"}`` plus an optional ``"image_path"`` of the candidate's best keyframe (at most
    :data:`MAX_CANDIDATES` are used). ``llm`` defaults to :func:`decompose.default_llm`;
    ``cache_dir=None`` disables the disk cache. Only model replies are cached.
    """
    try:
        clean_turns = _clean_turns(turns)
        scenes_in = [s.strip() for s in current_scenes if isinstance(s, str) and s.strip()]
        initial_text = (initial or "").strip()
        if not initial_text and not scenes_in and not clean_turns:
            return {"scenes": [], "next_question": FALLBACK_QUESTIONS[0][0], "model": None}
        candidates = []
        for video in list(top_videos)[:MAX_CANDIDATES]:
            if isinstance(video, Mapping) and video.get("video_id"):
                candidates.append({
                    "video_id": str(video["video_id"]),
                    "caption_hint": str(video.get("caption_hint") or "").strip()[:300],
                    "image_path": str(video["image_path"]) if video.get("image_path") else None,
                })
        payload = {"initial": initial_text, "turns": clean_turns, "scenes": scenes_in,
                   "candidates": candidates}
        path = _cache_path(cache_dir, payload) if cache_dir is not None else None
        cached = _read_cache(path)
        if cached is not None:
            return cached

        client = llm if llm is not None else decompose_module.default_llm()
        if client is not None:
            images: list[bytes] = []
            lines = []
            for i, cand in enumerate(candidates):
                image = _load_image(cand["image_path"]) if cand["image_path"] else None
                tag = ""
                if image is not None:
                    images.append(image)
                    tag = f" (image {len(images)} is its best-matching keyframe)"
                hint = cand["caption_hint"] or "no caption"
                lines.append(f"{i + 1}. {cand['video_id']}: {hint}{tag}")
            prompt = PROMPT.format(
                initial=initial_text[:MAX_INPUT_CHARS] or "(none)",
                dialogue="\n".join(f"{t['role'].upper()}: {t['text']}" for t in clean_turns) or "(none yet)",
                scenes="\n".join(f"{i + 1}. {s}" for i, s in enumerate(scenes_in)) or "(none yet)",
                image_note=" (attached images)" if images else "",
                candidates="\n".join(lines) or "(none)",
            )
            (scenes, question), model = client.generate(prompt, _parse, images)
            result: ConversationResult = {"scenes": scenes, "next_question": question, "model": model}
            _write_cache(path, result)
            return result
    except Exception:  # noqa: BLE001 - the contract: fall back to rules, never fail the chat
        pass
    try:
        return rule_update(initial, turns, current_scenes)
    except Exception:  # noqa: BLE001
        return {"scenes": [s for s in current_scenes if isinstance(s, str)], "next_question":
                FALLBACK_QUESTIONS[0][0], "model": None}
