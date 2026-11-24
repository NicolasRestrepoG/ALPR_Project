"""`.env` parsing and secret resolution.

The property that matters most: a real environment variable beats the file, and a
secret's *value* never appears in any diagnostic output.
"""

from __future__ import annotations

import pytest

from alpr.config import (
    ENV_FILE_VAR,
    get_secret,
    load_dotenv,
    parse_env,
    repo_root,
    secret_is_set,
    secret_source,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Point every test at its own .env and clear the real variables."""
    env = tmp_path / ".env"
    env.write_text("", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_VAR, str(env))
    for name in ("ROBOFLOW_API_KEY", "ALPR_DATA_ROOT", "COMMERCIAL_OCR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return env


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("KEY=value", {"KEY": "value"}),
        ("export KEY=value", {"KEY": "value"}),
        ("KEY = value", {"KEY": "value"}),
        ("KEY='value'", {"KEY": "value"}),
        ('KEY="value"', {"KEY": "value"}),
        ("KEY=value # trailing comment", {"KEY": "value"}),
        ("# whole line comment", {}),
        ("", {}),
        ("\n\n  \n", {}),
        ("KEY=", {"KEY": ""}),
        ("NOEQUALS", {}),
        ("=novalue", {}),
        ("A=1\nB=2", {"A": "1", "B": "2"}),
    ],
)
def test_parse_env(text, expected):
    assert parse_env(text) == expected


def test_a_hash_inside_quotes_is_literal():
    """API keys can legitimately contain '#'."""
    assert parse_env('KEY="ab#cd"') == {"KEY": "ab#cd"}
    assert parse_env("KEY='ab#cd'") == {"KEY": "ab#cd"}


def test_a_hash_without_leading_space_is_kept():
    """`abc#def` is a value; `abc # def` is a value plus a comment."""
    assert parse_env("KEY=abc#def") == {"KEY": "abc#def"}
    assert parse_env("KEY=abc # def") == {"KEY": "abc"}


def test_values_containing_equals_survive():
    assert parse_env("KEY=a=b=c") == {"KEY": "a=b=c"}


def test_malformed_lines_are_skipped_not_raised():
    """A stray line in a credentials file must not break the whole tool."""
    assert parse_env("GOOD=1\nthis is nonsense\nALSO_GOOD=2") == {
        "GOOD": "1",
        "ALSO_GOOD": "2",
    }


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def test_reads_from_the_env_file(isolated_env):
    isolated_env.write_text("ROBOFLOW_API_KEY=from_file\n", encoding="utf-8")
    assert get_secret("ROBOFLOW_API_KEY") == "from_file"
    assert load_dotenv()["ROBOFLOW_API_KEY"] == "from_file"


def test_a_real_environment_variable_wins(isolated_env, monkeypatch):
    """So `ROBOFLOW_API_KEY=other acquire fetch …` overrides for one command."""
    isolated_env.write_text("ROBOFLOW_API_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("ROBOFLOW_API_KEY", "from_environment")
    assert get_secret("ROBOFLOW_API_KEY") == "from_environment"


def test_an_empty_value_counts_as_unset(isolated_env):
    """The shipped .env has `ROBOFLOW_API_KEY=` — that must not read as a key."""
    isolated_env.write_text("ROBOFLOW_API_KEY=\n", encoding="utf-8")
    assert get_secret("ROBOFLOW_API_KEY") is None
    assert secret_is_set("ROBOFLOW_API_KEY") is False


def test_whitespace_only_environment_value_counts_as_unset(isolated_env, monkeypatch):
    monkeypatch.setenv("ROBOFLOW_API_KEY", "   ")
    assert get_secret("ROBOFLOW_API_KEY") is None


def test_default_is_returned_when_unset(isolated_env):
    assert get_secret("NOTHING_HERE", "fallback") == "fallback"
    assert get_secret("NOTHING_HERE") is None


def test_missing_env_file_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_FILE_VAR, str(tmp_path / "does_not_exist"))
    assert load_dotenv() == {}
    assert get_secret("ROBOFLOW_API_KEY") is None


def test_no_mutation_of_os_environ(isolated_env, monkeypatch):
    """Loading must not leak values into the process environment."""
    import os

    isolated_env.write_text("ROBOFLOW_API_KEY=from_file\n", encoding="utf-8")
    load_dotenv()
    assert "ROBOFLOW_API_KEY" not in os.environ


# --------------------------------------------------------------------------- #
# Diagnostics must not leak the value
# --------------------------------------------------------------------------- #


def test_secret_source_names_the_origin_without_the_value(isolated_env, monkeypatch):
    assert secret_source("ROBOFLOW_API_KEY") == "not set"

    isolated_env.write_text("ROBOFLOW_API_KEY=supersecret\n", encoding="utf-8")
    from_file = secret_source("ROBOFLOW_API_KEY")
    assert "supersecret" not in from_file
    assert str(isolated_env) in from_file

    monkeypatch.setenv("ROBOFLOW_API_KEY", "othersecret")
    assert secret_source("ROBOFLOW_API_KEY") == "environment"


def test_keys_command_never_prints_a_value(isolated_env, capsys):
    from alpr.data_sources.acquire import main

    isolated_env.write_text("ROBOFLOW_API_KEY=supersecret\n", encoding="utf-8")
    assert main(["keys"]) == 0
    out = capsys.readouterr().out
    assert "supersecret" not in out
    assert "ROBOFLOW_API_KEY" in out
    assert "set" in out


def test_keys_check_reports_failure_without_leaking_the_key(isolated_env, capsys, monkeypatch):
    """`--check` calls the provider; a rejected key must not appear in the output."""
    from alpr.data_sources import roboflow
    from alpr.data_sources.acquire import main

    isolated_env.write_text("ROBOFLOW_API_KEY=supersecret\n", encoding="utf-8")
    monkeypatch.setattr(
        roboflow, "check_api_key", lambda key, **kw: (False, "rejected: key does not exist")
    )
    assert main(["keys", "--check"]) == 1
    out = capsys.readouterr().out
    assert "supersecret" not in out
    assert "FAILED" in out


def test_keys_check_without_a_key_is_an_error(isolated_env, capsys):
    from alpr.data_sources.acquire import main

    assert main(["keys", "--check"]) == 1
    assert "nothing to check" in capsys.readouterr().out


def test_a_rejected_key_explains_which_key_to_copy():
    """The 401 is almost always the publishable key instead of the private one."""
    from alpr.data_sources.roboflow import KEY_REJECTED_HELP

    assert "Private API Key" in KEY_REJECTED_HELP
    assert "Publishable Key" in KEY_REJECTED_HELP
    assert "acquire keys --check" in KEY_REJECTED_HELP


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_roboflow_reads_the_key_from_the_env_file(isolated_env):
    from alpr.data_sources.roboflow import api_key

    isolated_env.write_text("ROBOFLOW_API_KEY=from_file\n", encoding="utf-8")
    assert api_key() == "from_file"


def test_data_root_can_be_set_from_the_env_file(isolated_env, tmp_path):
    """.env.example documents ALPR_DATA_ROOT, so it must actually take effect."""
    from alpr.data_sources.storage import default_data_root

    target = tmp_path / "elsewhere"
    isolated_env.write_text(f"ALPR_DATA_ROOT={target}\n", encoding="utf-8")
    assert default_data_root() == target.resolve()


def test_repo_root_contains_the_project():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "alpr").is_dir()


def test_the_real_env_file_is_gitignored():
    """A key pasted into .env must be uncommittable."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", ".env"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, ".env is NOT gitignored — a key could be committed"


def test_the_example_file_is_committed_but_the_real_one_is_not():
    """`.env.*` in .gitignore also matches `.env.example`, so it needs a negation.

    Without it a fresh clone has no template documenting which keys are required.
    """
    import subprocess

    def ignored(name: str) -> bool:
        return (
            subprocess.run(
                ["git", "check-ignore", name],
                cwd=repo_root(),
                capture_output=True,
            ).returncode
            == 0
        )

    assert ignored(".env"), ".env must be ignored"
    assert not ignored(".env.example"), ".env.example must be committable"


def test_the_example_file_holds_no_value():
    """.env.example is committed, so it must never contain a real key."""
    example = repo_root() / ".env.example"
    if not example.exists():
        pytest.skip(".env.example not present")
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        assert value.strip() == "", f"{stripped!r} has a value in a committed file"
