"""The `generate` command, end to end on synthetic assets."""

from __future__ import annotations

import json
import subprocess
import sys

import cv2
import pytest

from alpr.data_sources.storage import DataPaths
from alpr.plates.labels import CHARSET
from alpr.synth.generate import main

from .conftest import _glyph


@pytest.fixture
def assets(tmp_path, monkeypatch):
    paths = DataPaths(tmp_path).ensure()
    out = paths.assets_dir / "glyphs"
    out.mkdir(parents=True)
    entries = {}
    for c in CHARSET:
        img, origin = _glyph(c)
        cv2.imwrite(str(out / f"{c}.png"), img)
        entries[c] = {"file": f"{c}.png", "origin_mm": list(origin)}
    (out / "glyphs.json").write_text(json.dumps({"glyphs": entries}))
    (paths.assets_dir / "municipios.txt").write_text("MEDELLIN\nCALI\n")
    monkeypatch.setenv("ALPR_DATA_ROOT", str(tmp_path))
    return paths


def test_demo_sheet(assets, capsys):
    out = assets.root / "demo.png"
    assert main(["demo", "--per-row", "2", "--out", str(out)]) == 0
    sheet = cv2.imread(str(out))
    assert sheet.shape[0] == 5 * (96 * 2 + 22) and sheet.shape[1] == 2 * 192 * 2
    assert "plain backgrounds" in capsys.readouterr().err  # no corpus built here


def test_compare_needs_the_frozen_test_set(assets, capsys):
    assert main(["compare"]) == 1
    assert "annotate freeze" in capsys.readouterr().err


def test_synth_imports_no_ml_framework():
    code = (
        "import sys, alpr.synth.generate, alpr.synth.pipeline; "
        "bad = sorted({'torch','torchvision','ultralytics'} & set(sys.modules)); "
        "sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
