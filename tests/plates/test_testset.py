"""Freezing the provisional test set: what goes in, what stays out, and that it stays put."""

from __future__ import annotations

import json
import os
import stat

import cv2
import numpy as np
import pytest

from alpr.data_sources.base import Box, PlateType, Record, write_jsonl
from alpr.data_sources.storage import DataPaths
from alpr.plates.annotate import main
from alpr.plates.labels import LabelStore, labels_path
from alpr.plates.rectify import rectify, rotate_image, rotate_points
from alpr.plates.testset import (
    MANIFEST_NAME,
    FrozenSetError,
    entry_image,
    freeze,
    frozen_dir,
    load_entries,
    verify,
)

from .conftest import SCENE_QUAD, SCENE_SIZE, place, quadrant_plate


def _scene(seed: int) -> np.ndarray:
    """A distinct scene per seed: the same plate over a different background."""
    rng = np.random.default_rng(seed)
    bg = rng.integers(0, 255, (SCENE_SIZE[1] // 50, SCENE_SIZE[0] // 50, 3), dtype=np.uint8)
    scene = cv2.resize(bg, SCENE_SIZE, interpolation=cv2.INTER_NEAREST)
    plate = quadrant_plate()
    h, w = plate.shape[:2]
    m = cv2.getPerspectiveTransform(
        np.array([(0, 0), (w, 0), (w, h), (0, h)], np.float32) - 0.5,
        np.array(SCENE_QUAD, np.float32) - 0.5,
    )
    return cv2.warpPerspective(plate, m, SCENE_SIZE, dst=scene, borderMode=cv2.BORDER_TRANSPARENT)


@pytest.fixture
def labelled(tmp_path):
    """One source, seven images:

    a1, a2   car AAA111 — a2 is tagged glyph_source, so both must be excluded
    b1, b2   car BBB222 — b2 is b1 re-encoded: a near-duplicate, dropped
    b3       car BBB222 from another photo — kept, same group as b1
    s1       car CCC333, photo stored sideways, labelled with rot90 — kept
    u1       unreadable (unknown) — excluded
    """
    paths = DataPaths(tmp_path).ensure()
    raw = paths.raw("rf")
    raw.mkdir(parents=True)
    scenes = {
        "a1": _scene(1),
        "a2": _scene(2),
        "b1": _scene(3),
        "b3": _scene(4),
        "u1": _scene(5),
    }
    for name, img in scenes.items():
        cv2.imwrite(str(raw / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        str(raw / "b2.jpg"),
        cv2.cvtColor(scenes["b1"], cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )
    cv2.imwrite(
        str(raw / "s1.png"),
        cv2.cvtColor(rotate_image(place(quadrant_plate()), 270), cv2.COLOR_RGB2BGR),
    )
    names = ["a1.png", "a2.png", "b1.png", "b2.jpg", "b3.png", "s1.png", "u1.png"]
    write_jsonl(
        paths.index("rf"),
        [
            Record(
                source="rf",
                image_path=f"raw/rf/{n}",
                license_key="cc-by-4.0",
                box=Box(100, 100, 1200, 700),
            )
            for n in names
        ],
    )
    store = LabelStore(labels_path(paths, "rf"))
    car = PlateType.PARTICULAR
    store.put("raw/rf/a1.png", plate="AAA111", plate_type=car, corners=SCENE_QUAD)
    store.put(
        "raw/rf/a2.png", plate="AAA111", plate_type=car, corners=SCENE_QUAD, tags=["glyph_source"]
    )
    store.put("raw/rf/b1.png", plate="BBB222", plate_type=car, corners=SCENE_QUAD)
    store.put("raw/rf/b2.jpg", plate="BBB222", plate_type=car, corners=SCENE_QUAD)
    store.put("raw/rf/b3.png", plate="BBB222", plate_type=car, corners=SCENE_QUAD)
    store.put("raw/rf/s1.png", plate="CCC333", plate_type=car, corners=SCENE_QUAD, tags=["rot90"])
    store.put("raw/rf/u1.png", plate=None, plate_type=PlateType.UNKNOWN, corners=SCENE_QUAD)
    return paths


def test_freeze_applies_the_rules(labelled):
    report = freeze(labelled, ["rf"])
    assert report.excluded_glyph_source == 2  # both photos of the glyph-source car
    assert report.excluded_unknown == 1
    assert report.near_duplicates == 1
    entries = load_entries(labelled)
    assert sorted(e.key for e in entries) == ["raw/rf/b1.png", "raw/rf/b3.png", "raw/rf/s1.png"]
    assert {e.group for e in entries} == {"plate:BBB222", "plate:CCC333"}
    assert report.cars == 2 and report.by_type == {"particular": 3}
    e = entries[0]
    assert (e.source, e.license_key, e.image_path) == ("rf", "cc-by-4.0", "raw/rf/b1.png")


def test_frozen_images_are_byte_identical_and_rotation_still_applies(labelled):
    freeze(labelled, ["rf"])
    root = frozen_dir(labelled, "provisional")
    for e in load_entries(labelled):
        original = labelled.root / e.image_path
        assert (root / e.file).read_bytes() == original.read_bytes()
    sideways = next(e for e in load_entries(labelled) if e.plate == "CCC333")
    assert sideways.tags == ["rot90"]
    flat = rectify(entry_image(labelled, sideways), sideways.corners, (192, 96))
    assert flat[20, 20].tolist() == pytest.approx([255, 0, 0], abs=12)  # upright


def test_frozen_set_is_read_only_checksummed_and_never_rewritten(labelled):
    freeze(labelled, ["rf"])
    root = frozen_dir(labelled, "provisional")
    assert verify(labelled) == []
    for p in root.rglob("*"):
        if p.is_file():
            assert not p.stat().st_mode & stat.S_IWUSR, p
    with pytest.raises(FrozenSetError, match="already frozen"):
        freeze(labelled, ["rf"])

    # Tampering is caught.
    manifest = root / MANIFEST_NAME
    os.chmod(manifest, stat.S_IRUSR | stat.S_IWUSR)
    lines = manifest.read_text().splitlines()
    first = json.loads(lines[0]) | {"plate": "XXX000"}
    manifest.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    (root / "stray.txt").write_text("x")
    problems = verify(labelled)
    assert any(p.startswith("changed: manifest.jsonl") for p in problems)
    assert any("stray.txt" in p for p in problems)


def test_a_failed_freeze_leaves_nothing_behind(labelled, monkeypatch):
    import alpr.plates.testset as ts

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ts, "_readme", boom)
    with pytest.raises(RuntimeError):
        freeze(labelled, ["rf"])
    assert not (labelled.root / "provisional_test").exists() or not any(
        (labelled.root / "provisional_test").iterdir()
    )


def test_cli_freeze_and_verify(labelled, monkeypatch, capsys):
    monkeypatch.setenv("ALPR_DATA_ROOT", str(labelled.root))
    assert main(["freeze", "rf"]) == 0
    out = capsys.readouterr().out
    assert "3 entries" in out and "2 photos of glyph-source cars" in out
    assert main(["verify"]) == 0
    assert "intact" in capsys.readouterr().out
    assert main(["freeze", "rf"]) == 1
    assert "already frozen" in capsys.readouterr().err


def test_missing_set_is_explained(tmp_path):
    with pytest.raises(FrozenSetError, match="annotate freeze"):
        verify(DataPaths(tmp_path))


def test_rotated_fixture_really_is_sideways():
    """Guard for the fixture above: without the rotation, its corners would not fit."""
    q = rotate_points(SCENE_QUAD, 270, SCENE_SIZE)
    assert max(p[1] for p in q) - min(p[1] for p in q) > max(p[0] for p in q) - min(p[0] for p in q)
