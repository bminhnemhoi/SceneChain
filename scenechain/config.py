"""Runtime settings for the VBS tool, read from the environment and the repository ``.env``.

Precedence, highest first: explicit keyword overrides passed to :func:`load_settings`, the process
environment, the ``.env`` file at the repository root, built-in defaults. The ``.env`` file is parsed
here and never copied into ``os.environ``, so loading settings has no side effects.

Recognised variables:

``VBS_COLLECTION``  collection directory (default ``collections/aic2026``)
``VBS_HOST``        bind address (default ``127.0.0.1``)
``VBS_PORT``        port (default ``8787``)
``VBS_LOG_DIR``     interaction log directory (default ``logs``)
``DRES_URL``, ``DRES_USER``, ``DRES_PASSWORD``   DRES server and credentials
``DRES_COLLECTION``  collection name sent with DRES answers (default: not sent; set it only when the
                    organisers confirm the DRES collection name)
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``), ``OPENAI_API_KEY``   LLM keys
``OPENAI_BASE_URL``, ``OPENAI_MODEL``   optional OpenAI-compatible endpoint and model (default: OpenAI
                    itself and :data:`scenechain.llm.OPENAI_MODEL`)

Relative paths coming from the environment or ``.env`` are resolved against the repository root;
paths passed as overrides are resolved against the current working directory. Secrets are excluded
from ``repr(Settings)`` so a settings object can be logged safely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
VBS_ROOT = REPO_ROOT  # runtime folders (collections/, cache/, logs/) live at the repository root
DEFAULT_COLLECTION_DIR = VBS_ROOT / "collections" / "default"
DEFAULT_LOG_DIR = VBS_ROOT / "logs"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


@dataclass(frozen=True)
class Settings:
    """Everything the server needs to start. Secret fields are hidden from ``repr``."""

    collection_dir: Path = DEFAULT_COLLECTION_DIR
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    dres_url: Optional[str] = None
    dres_user: Optional[str] = None
    dres_password: Optional[str] = field(default=None, repr=False)
    dres_collection: Optional[str] = None
    gemini_key: Optional[str] = field(default=None, repr=False)
    openai_key: Optional[str] = field(default=None, repr=False)
    openai_base_url: Optional[str] = None
    openai_model: Optional[str] = None
    log_dir: Path = DEFAULT_LOG_DIR


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines of a dotenv file; a missing file yields an empty dict.

    Blank lines and ``#`` comments are ignored, an optional ``export`` prefix is accepted and one
    pair of matching surrounding quotes is removed from the value.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_settings(
    env_file: Optional[Path] = REPO_ROOT / ".env",
    environ: Optional[Mapping[str, str]] = None,
    **overrides: object,
) -> Settings:
    """Build :class:`Settings` from overrides, the environment and ``env_file``.

    ``environ`` defaults to ``os.environ``; pass a mapping to make the result independent of the
    calling process. ``overrides`` use the :class:`Settings` field names; ``None`` values are ignored
    so command-line options that were not given fall through to the environment.
    """
    unknown = set(overrides) - set(Settings.__dataclass_fields__)
    if unknown:
        raise TypeError(f"unknown settings: {sorted(unknown)}")
    file_values = read_env_file(env_file) if env_file is not None else {}
    env = os.environ if environ is None else environ

    def pick(*names: str) -> Optional[str]:
        for name in names:
            value = env.get(name)
            if value:
                return value
        for name in names:
            value = file_values.get(name)
            if value:
                return value
        return None

    def repo_path(value: Optional[str], default: Path) -> Path:
        if not value:
            return default
        p = Path(value).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()

    port_text = pick("VBS_PORT")
    try:
        port = int(port_text) if port_text else DEFAULT_PORT
    except ValueError as exc:
        raise ValueError(f"VBS_PORT must be an integer, got {port_text!r}") from exc

    values: dict[str, object] = {
        "collection_dir": repo_path(pick("VBS_COLLECTION"), DEFAULT_COLLECTION_DIR),
        "host": pick("VBS_HOST") or DEFAULT_HOST,
        "port": port,
        "dres_url": pick("DRES_URL"),
        "dres_user": pick("DRES_USER"),
        "dres_password": pick("DRES_PASSWORD"),
        "dres_collection": pick("DRES_COLLECTION"),
        "gemini_key": pick("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "openai_key": pick("OPENAI_API_KEY"),
        "openai_base_url": pick("OPENAI_BASE_URL"),
        "openai_model": pick("OPENAI_MODEL"),
        "log_dir": repo_path(pick("VBS_LOG_DIR"), DEFAULT_LOG_DIR),
    }
    for name, value in overrides.items():
        if value is None:
            continue
        if name in ("collection_dir", "log_dir"):
            value = Path(str(value)).expanduser().resolve()
        elif name == "port":
            value = int(value)  # type: ignore[call-overload]
        values[name] = value
    return Settings(**values)  # type: ignore[arg-type]
