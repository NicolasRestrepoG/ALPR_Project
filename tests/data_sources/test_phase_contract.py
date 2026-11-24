"""Properties of Phase 1 as a *phase*, not of any one module.

"One phase = one module + one command + its own tests. A phase is done when you
can run its tests and its command with the rest of the repo deleted." These tests
are what keep that true as later phases arrive.
"""

from __future__ import annotations

import subprocess
import sys

from alpr.data_sources.acquire import build_parser, main

HEAVY = {
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "PIL",
    "pandas",
    "scipy",
    "sklearn",
    "requests",
    "tqdm",
    "gdown",
    "yt_dlp",
    "ultralytics",
}


def test_acquisition_layer_imports_nothing_heavy():
    """The acquisition layer runs with no ML libraries — and no HTTP client — installed.

    `requests`, `tqdm` and `gdown` are real dependencies but are imported lazily,
    inside the function that downloads, so the test suite stays this fast and this
    portable. PyTorch arrives in Phase 4, not on day one.
    """
    code = (
        "import sys; import alpr.data_sources; "
        f"heavy = {sorted(HEAVY)!r}; "
        "found = sorted(set(heavy) & set(sys.modules)); "
        "print(found); sys.exit(1 if found else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"heavy modules imported at load time: {proc.stdout.strip()}"


def test_generated_files_are_group_readable(tmp_path):
    """Atomic writes go through mkstemp, which defaults to 0600. Assets are not secrets."""
    from alpr.data_sources.storage import atomic_write_text

    target = tmp_path / "municipios.txt"
    atomic_write_text(target, "MEDELLIN\n")
    assert target.stat().st_mode & 0o777 == 0o644


def test_jsonl_index_is_group_readable(tmp_path):
    from alpr.data_sources.base import Record, write_jsonl

    path = tmp_path / "index.jsonl"
    write_jsonl(path, [Record(source="s", image_path="a.jpg", license_key="mit")])
    assert path.stat().st_mode & 0o777 == 0o644


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


def test_cli_exposes_the_sprint_1_1_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "acquire must have subcommands"
    assert {"list", "paths", "fetch", "index", "audit"} <= set(actions[0].choices)


def test_cli_list_runs(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "ccpd" in out and "divipola" in out


def test_cli_audit_reports_the_commercial_verdict(capsys):
    assert main(["audit"]) == 0
    assert "COMMERCIAL DEPLOYMENT:" in capsys.readouterr().out


def test_cli_list_commercial_only_runs(capsys):
    assert main(["list", "--commercial-only"]) == 0
    assert "ccpd" in capsys.readouterr().out


def test_cli_fetch_without_a_target_is_an_error():
    import pytest

    with pytest.raises(SystemExit):
        main(["fetch"])
