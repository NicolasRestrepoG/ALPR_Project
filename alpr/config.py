"""Project paths and secrets.

Two small jobs that every phase needs and no phase owns: finding the repository
root, and reading credentials without ever putting them in the repository.

**Secrets live in the environment or in a gitignored `.env`, never in code and
never in `data/manifest.json`.** Phase 1 needs a free Roboflow API key; Sprint 5.3
will need a commercial OCR API key for the baseline comparison. Both arrive the
same way.

Deliberately not using `python-dotenv`: the file format is a dozen lines of
parsing, and Phase 1's property of running on the standard library alone is worth
more than the dependency. Precedence follows the same convention that library
uses — **a real environment variable always beats the `.env` file** — so
`ROBOFLOW_API_KEY=… acquire fetch …` overrides the file for one command without
editing anything.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_NAME = ".env"

#: Override the .env location, mostly for tests and CI.
ENV_FILE_VAR = "ALPR_ENV_FILE"


def repo_root() -> Path:
    """The directory containing `alpr/`, `tests/` and `pyproject.toml`."""
    return Path(__file__).resolve().parents[1]


def env_file() -> Path | None:
    """Locate the `.env`: `$ALPR_ENV_FILE`, then the repo root, then cwd upward.

    Walking up from the cwd means `acquire` works the same from a subdirectory as
    it does from the project root.
    """
    override = os.environ.get(ENV_FILE_VAR)
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None

    candidate = repo_root() / ENV_FILE_NAME
    if candidate.is_file():
        return candidate

    current = Path.cwd().resolve()
    for directory in (current, *current.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def _strip_inline_comment(value: str) -> str:
    """Drop a trailing ` # comment`, but only outside quotes."""
    for i, char in enumerate(value):
        if char == "#" and (i == 0 or value[i - 1].isspace()):
            return value[:i]
    return value


def parse_env(text: str) -> dict[str, str]:
    """Parse `.env` content into a mapping.

    Handles `KEY=value`, `export KEY=value`, `#` comments, blank lines, and single
    or double quoted values (inside which `#` is literal). Malformed lines are
    skipped rather than raising: a stray line in a credentials file should not stop
    the whole tool, and the missing key produces a clear message downstream anyway.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        else:
            value = _strip_inline_comment(value).strip()
        values[key] = value
    return values


def load_dotenv() -> dict[str, str]:
    """Read the `.env` file, or return `{}` if there isn't one.

    Does **not** mutate `os.environ`. Global mutation is exactly the kind of
    action-at-a-distance that makes a missing credential hard to trace, and
    nothing here needs the value to reach a subprocess.
    """
    path = env_file()
    if path is None:
        return {}
    try:
        return parse_env(path.read_text(encoding="utf-8"))
    except OSError:
        return {}


def get_secret(name: str, default: str | None = None) -> str | None:
    """Environment variable, else `.env`, else `default`. Empty values count as unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    value = load_dotenv().get(name, "").strip()
    return value or default


def secret_is_set(name: str) -> bool:
    """True if `name` resolves to a non-empty value. Never returns the value itself."""
    return get_secret(name) is not None


def secret_source(name: str) -> str:
    """Where a secret came from, for diagnostics. Never includes the value."""
    if os.environ.get(name, "").strip():
        return "environment"
    if load_dotenv().get(name, "").strip():
        path = env_file()
        return f"{path}" if path else ".env"
    return "not set"
