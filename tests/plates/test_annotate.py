"""The labelling app, its HTTP surface, the contact sheet and the layout measurement."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from alpr.data_sources.base import Box, Record, write_jsonl
from alpr.data_sources.own_photos import read_labels
from alpr.data_sources.storage import DataPaths
from alpr.plates.annotate import (
    AnnotateApp,
    contact_sheet,
    main,
    make_server,
    measure_layout,
)
from alpr.plates.geometry import PlateGeometry
from alpr.plates.labels import LabelError, labels_path
from alpr.plates.rectify import RectifyError, rotate_image, rotate_points

from .conftest import SCENE_QUAD, SCENE_SIZE, draw_plate, place, quadrant_plate


@pytest.fixture
def source(tmp_path):
    """A one-image indexed source whose box loosely surrounds the plate."""
    paths = DataPaths(tmp_path).ensure()
    img_path = paths.raw("rf") / "scene.png"
    img_path.parent.mkdir(parents=True)
    scene = place(quadrant_plate())
    cv2.imwrite(str(img_path), cv2.cvtColor(scene, cv2.COLOR_RGB2BGR))
    xs, ys = [p[0] for p in SCENE_QUAD], [p[1] for p in SCENE_QUAD]
    rec = Record(
        source="rf",
        image_path="raw/rf/scene.png",
        license_key="cc-by-4.0",
        box=Box(min(xs), min(ys), max(xs), max(ys)),
    )
    write_jsonl(paths.index("rf"), [rec])
    return paths


def test_display_and_image_coordinates_round_trip(source):
    app = AnnotateApp(source, "rf")
    shown = app.to_display_coords(0, SCENE_QUAD)
    back = app.to_image_coords(0, shown)
    assert np.allclose(back, SCENE_QUAD, atol=1e-6)
    body, meta = app.crop(0)
    assert body[:2] == b"\xff\xd8"  # JPEG
    xs, ys = [p[0] for p in SCENE_QUAD], [p[1] for p in SCENE_QUAD]
    assert meta["ox"] <= min(xs) and meta["oy"] <= min(ys)  # the whole plate is in view
    assert meta["ox"] + meta["width"] / meta["scale"] >= max(xs)


def test_save_stores_full_image_coordinates(source):
    app = AnnotateApp(source, "rf")
    shown = app.to_display_coords(0, SCENE_QUAD)
    state = app.save(
        0,
        {"points": shown, "plate": "abc123", "plate_type": "particular", "tags": ["glyph_source"]},
    )
    assert state["labelled"] == 1 and state["done"] == [True]
    lb = read_labels(labels_path(source, "rf"))["raw/rf/scene.png"]
    assert np.allclose(lb.corners, SCENE_QUAD, atol=0.01)
    view = app.view(0)
    assert view["label"]["plate"] == "ABC123"
    assert np.allclose(view["label"]["points"], shown, atol=0.05)


def test_save_requires_four_points(source):
    app = AnnotateApp(source, "rf")
    with pytest.raises(LabelError, match="four"):
        app.save(0, {"points": [[1, 1], [50, 1], [50, 30]], "plate": "ABC123"})


def test_preview_is_a_jpeg(source):
    app = AnnotateApp(source, "rf")
    jpg = app.preview(0, app.to_display_coords(0, SCENE_QUAD))
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (96 * 3, 192 * 3, 3)


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers["Content-Type"], r.read()


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, json.loads(r.read())


def test_http_surface(source):
    app = AnnotateApp(source, "rf")
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert server.server_address[0] == "127.0.0.1"  # loopback only
        status, ctype, body = _get(base + "/")
        assert status == 200 and "text/html" in ctype and b"Plate Labelling" in body
        status, _, body = _get(base + "/api/state")
        assert json.loads(body)["total"] == 1
        status, ctype, _ = _get(base + "/api/crop/0.jpg")
        assert ctype == "image/jpeg"

        pts = app.to_display_coords(0, SCENE_QUAD)
        q = ",".join(f"{v:.2f}" for p in pts for v in p)
        status, ctype, _ = _get(f"{base}/api/preview/0?p={q}")
        assert ctype == "image/jpeg"

        status, state = _post(base + "/api/label/0", {"points": pts, "plate": "FGO681"})
        assert state["labelled"] == 1

        with pytest.raises(urllib.error.HTTPError) as err:
            _post(base + "/api/label/0", {"points": pts, "plate": "NOTAPLATE"})
        assert err.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/item/99")
        assert err.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/crop/..%2F..%2Fetc")
        assert err.value.code == 404

        status, state = _post(base + "/api/delete/0", {})
        assert state["labelled"] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_contact_sheet(source):
    app = AnnotateApp(source, "rf")
    app.save(0, {"points": app.to_display_coords(0, SCENE_QUAD), "plate": "ABC123"})
    sheet = contact_sheet(source, "rf", n=4, cols=4)
    assert sheet.shape[1] == 4 * 192 * 2
    # The rectified tile is upright: red top-left, blue bottom-right.
    assert tuple(sheet[40, 40]) == pytest.approx((255, 0, 0), abs=12)
    assert tuple(sheet[150, 340]) == pytest.approx((0, 0, 255), abs=12)


def test_contact_sheet_needs_labels(source):
    with pytest.raises(LabelError, match="no labelled"):
        contact_sheet(source, "rf")


def test_measure_layout_recovers_a_known_geometry():
    """Draw a plate from a deliberately different geometry, warp it into a scene,
    and check the measurement recovers that geometry — not MT001's numbers."""
    truth = PlateGeometry(logo_gutter=30.0, char_top=33.0, muni_gap=12.0, char_gap=9.0)
    scene = place(draw_plate(truth))
    results, skipped = measure_layout([(scene, SCENE_QUAD)])
    assert skipped == 0
    med = {k: float(np.median(v)) for k, v in results.items()}
    assert med["char_top"] == pytest.approx(truth.char_top, abs=1.0)
    assert med["char_h"] == pytest.approx(truth.char_h, abs=1.0)
    assert med["char_gap"] == pytest.approx(truth.char_gap, abs=1.0)
    assert med["logo_gutter"] == pytest.approx(truth.logo_gutter, abs=1.0)
    assert med["muni_gap"] == pytest.approx(truth.muni_gap, abs=1.0)
    assert med["muni_h"] == pytest.approx(truth.muni_h, abs=1.0)
    assert med["side_margin"] == pytest.approx(truth.side_margin, abs=1.0)


def test_measure_layout_skips_plates_without_six_characters():
    blank = place(np.full((640, 1320, 3), (240, 200, 30), np.uint8))
    results, skipped = measure_layout([(blank, SCENE_QUAD)])
    assert skipped == 1 and results["char_top"] == []


def test_cli_status(source, monkeypatch, capsys):
    monkeypatch.setenv("ALPR_DATA_ROOT", str(source.root))
    assert main(["status", "rf"]) == 0
    out = capsys.readouterr().out
    assert "0 of 1 labelled" in out and "below 5" in out


def test_cli_reports_errors_without_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ALPR_DATA_ROOT", str(tmp_path))
    assert main(["status", "missing_source"]) == 1
    assert "acquire fetch" in capsys.readouterr().err


def test_plates_package_imports_no_ml_framework():
    import subprocess
    import sys

    code = (
        "import sys, alpr.plates.annotate, alpr.plates.rectify, alpr.plates.geometry, "
        "alpr.plates.glyphs, alpr.plates.render, alpr.plates.testset; "
        "bad = sorted({'torch','torchvision','ultralytics'} & set(sys.modules)); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.fixture
def sideways(tmp_path):
    """A photo stored turned 90 degrees anticlockwise (EXIF lost): it needs a 90-degree
    clockwise turn to stand upright. The source's box is in the stored frame."""
    paths = DataPaths(tmp_path).ensure()
    img_path = paths.raw("rf") / "side.png"
    img_path.parent.mkdir(parents=True)
    stored = rotate_image(place(quadrant_plate()), 270)
    cv2.imwrite(str(img_path), cv2.cvtColor(stored, cv2.COLOR_RGB2BGR))
    quad = rotate_points(SCENE_QUAD, 270, SCENE_SIZE)
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    rec = Record(
        source="rf",
        image_path="raw/rf/side.png",
        license_key="cc-by-4.0",
        box=Box(min(xs), min(ys), max(xs), max(ys)),
    )
    write_jsonl(paths.index("rf"), [rec])
    return paths, quad


def test_sideways_plate_is_refused_until_the_photo_is_turned(sideways):
    paths, stored_quad = sideways
    app = AnnotateApp(paths, "rf")
    with pytest.raises(RectifyError, match="aspect"):
        app.save(0, {"points": app.to_display_coords(0, stored_quad), "plate": "ABC123"})

    shown = app.to_display_coords(0, SCENE_QUAD, rotation=90)
    app.save(0, {"points": shown, "plate": "ABC123", "rotation": 90, "tags": ["worn"]})
    lb = read_labels(labels_path(paths, "rf"))["raw/rf/side.png"]
    assert lb.tags == ("rot90", "worn")
    assert np.allclose(lb.corners, SCENE_QUAD, atol=0.01)  # corners in the upright frame

    # The sheet honours the tag: upright tile, red top-left, blue bottom-right.
    sheet = contact_sheet(paths, "rf", n=1, cols=1)
    assert tuple(sheet[40, 40]) == pytest.approx((255, 0, 0), abs=12)
    assert tuple(sheet[150, 340]) == pytest.approx((0, 0, 255), abs=12)


def test_view_reopens_rotated_and_keeps_corners_when_turned_again(sideways):
    paths, stored_quad = sideways
    app = AnnotateApp(paths, "rf")
    app.save(
        0, {"points": app.to_display_coords(0, SCENE_QUAD, 90), "plate": "ABC123", "rotation": 90}
    )
    view = app.view(0)
    assert view["rotation"] == 90 and view["label"]["tags"] == []
    assert np.allclose(app.to_image_coords(0, view["label"]["points"], 90), SCENE_QUAD, atol=0.05)
    back = app.view(0, 0)  # shown unturned: the same corners, in the stored frame
    assert np.allclose(app.to_image_coords(0, back["label"]["points"], 0), stored_quad, atol=0.05)


def test_http_rotation_parameter(sideways):
    paths, _ = sideways
    app = AnnotateApp(paths, "rf")
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _, ctype, body = _get(base + "/api/crop/0.jpg?r=90")
        assert ctype == "image/jpeg"
        upright = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        _, _, body = _get(base + "/api/crop/0.jpg?r=0")
        stored = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        assert upright.shape[1] > upright.shape[0] and stored.shape[0] > stored.shape[1]
        _, _, body = _get(base + "/api/item/0?r=270")
        assert json.loads(body)["rotation"] == 270
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(base + "/api/crop/0.jpg?r=45")
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
