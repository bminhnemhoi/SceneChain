"""Settings precedence and secret handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from scenechain.config import DEFAULT_COLLECTION_DIR, REPO_ROOT, load_settings, read_env_file


def test_env_file_parsing(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nexport A=1\nB = "two words"\nC=\'x\'\nnot a pair\n\nD=a=b\n',
                   encoding="utf-8")
    assert read_env_file(env) == {"A": "1", "B": "two words", "C": "x", "D": "a=b"}
    assert read_env_file(tmp_path / "missing") == {}


def test_precedence_and_defaults(tmp_path):
    env = tmp_path / ".env"
    env.write_text("VBS_PORT=9000\nGEMINI_API_KEY=file-key\nVBS_COLLECTION=collections/x\n",
                   encoding="utf-8")
    s = load_settings(env_file=env, environ={"VBS_PORT": "9100", "OPENAI_API_KEY": "env-key"})
    assert s.port == 9100  # environment beats .env
    assert s.gemini_key == "file-key" and s.openai_key == "env-key"
    assert s.collection_dir == (REPO_ROOT / "collections/x").resolve()
    assert s.host == "127.0.0.1" and s.dres_url is None

    o = load_settings(env_file=env, environ={}, port=1234, collection_dir=tmp_path, dres_url=None)
    assert o.port == 1234 and o.collection_dir == tmp_path.resolve()

    d = load_settings(env_file=None, environ={})
    assert d.collection_dir == DEFAULT_COLLECTION_DIR and d.port == 8787 and d.gemini_key is None


def test_google_api_key_fallback_and_secret_free_repr():
    s = load_settings(env_file=None, environ={"GOOGLE_API_KEY": "g-secret", "DRES_PASSWORD": "pw-secret"})
    assert s.gemini_key == "g-secret"
    assert "secret" not in repr(s)


def test_invalid_values_raise():
    with pytest.raises(ValueError):
        load_settings(env_file=None, environ={"VBS_PORT": "eighty"})
    with pytest.raises(TypeError):
        load_settings(env_file=None, environ={}, colection_dir=Path("."))
