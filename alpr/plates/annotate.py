"""`annotate` — label plates by clicking four corners and typing the text.

    annotate serve <source>     open the labelling page in a browser (local only)
    annotate status <source>    progress, plate types, character coverage
    annotate sheet <source>     contact sheet of rectified crops — the Sprint 2.1 check
    annotate measure <source>   measure the plate layout from labelled, rectified plates

The page is served by the standard library on 127.0.0.1 and talks to nothing else.
It shows each plate zoomed to the source's box when there is one, a magnifier for
precise corner clicks, and — as soon as four corners are down — the rectified
192 x 96 crop with the MT-001 character cells drawn over it. A rotated, mirrored or
misaligned preview means a wrong click, and it is visible before the label is saved,
not three weeks later as a plateau in training.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import webbrowser
from collections import OrderedDict
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from alpr.data_sources.base import PlateType
from alpr.data_sources.storage import DataPaths
from alpr.plates.geometry import CANONICAL_SIZE, MT001, PROVENANCE, PlateGeometry
from alpr.plates.labels import (
    KNOWN_TAGS,
    Item,
    LabelError,
    LabelStore,
    coverage,
    labels_path,
    list_items,
)
from alpr.plates.rectify import RectifyError, load_image, rectify

DEFAULT_PORT = 8765

#: Display sizing for the crop shown in the page. Small plates are upscaled so a
#: click lands within a fraction of a source pixel; huge ones are shrunk to fit.
DISPLAY_MAX = 1600
DISPLAY_MIN_W = 1000
MAX_UPSCALE = 4.0

#: Context around a source box, as a fraction of its width / height. Generous on
#: purpose: Roboflow boxes are often tight or slightly off, and the corners must be
#: the plate's, not the box's.
PAD_X = 0.30
PAD_Y = 0.70

PREVIEW_SCALE = 3


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


class _LRU(OrderedDict):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def remember(self, key: Any, value: Any) -> Any:
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)
        return value


class AnnotateApp:
    """Everything the HTTP handler needs, and nothing about HTTP."""

    def __init__(self, paths: DataPaths, source: str, *, order: str = "size") -> None:
        self.paths = paths
        self.source = source
        self.items: list[Item] = list_items(paths, source, order=order)
        if not self.items:
            raise LabelError(f"{source!r} has nothing to label")
        self.store = LabelStore(labels_path(paths, source))
        self._lock = threading.Lock()
        self._images = _LRU(3)
        self._crops = _LRU(48)

    # -- images ------------------------------------------------------------- #

    def _image(self, i: int) -> np.ndarray:
        with self._lock:
            if i in self._images:
                self._images.move_to_end(i)
                return self._images[i]
        img = load_image(self.items[i].image)
        with self._lock:
            return self._images.remember(i, img)

    def crop(self, i: int) -> tuple[bytes, dict[str, float]]:
        """The display JPEG for item `i`, and how display pixels map to the image."""
        with self._lock:
            if i in self._crops:
                return self._crops[i]
        img = self._image(i)
        h, w = img.shape[:2]
        box = self.items[i].box
        if box is None:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            px, py = box.width * PAD_X, box.height * PAD_Y
            x0, y0 = max(0, int(box.x1 - px)), max(0, int(box.y1 - py))
            x1, y1 = min(w, int(box.x2 + px)), min(h, int(box.y2 + py))
        region = img[y0:y1, x0:x1]
        cw, ch = x1 - x0, y1 - y0
        scale = min(DISPLAY_MAX / max(cw, ch), MAX_UPSCALE)
        scale = max(scale, min(DISPLAY_MIN_W / cw, MAX_UPSCALE))
        dw, dh = max(1, round(cw * scale)), max(1, round(ch * scale))
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        shown = cv2.resize(region, (dw, dh), interpolation=interp)
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(shown, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        if not ok:
            raise RectifyError(f"could not encode crop for {self.items[i].key}")
        meta = {
            "ox": float(x0),
            "oy": float(y0),
            "scale": float(dw / cw),
            "width": dw,
            "height": dh,
            "image_w": w,
            "image_h": h,
        }
        with self._lock:
            return self._crops.remember(i, (buf.tobytes(), meta))

    def to_image_coords(
        self, i: int, points: Sequence[Sequence[float]]
    ) -> list[tuple[float, float]]:
        _, m = self.crop(i)
        return [(m["ox"] + x / m["scale"], m["oy"] + y / m["scale"]) for x, y in points]

    def to_display_coords(self, i: int, points: Sequence[Sequence[float]]) -> list[list[float]]:
        _, m = self.crop(i)
        return [[(x - m["ox"]) * m["scale"], (y - m["oy"]) * m["scale"]] for x, y in points]

    def preview(self, i: int, display_points: Sequence[Sequence[float]]) -> bytes:
        """Rectified crop at PREVIEW_SCALE, with the MT-001 cells drawn over it."""
        corners = self.to_image_coords(i, display_points)
        flat = rectify(self._image(i), corners, CANONICAL_SIZE)
        return encode_jpeg(overlay_geometry(flat, PREVIEW_SCALE))

    # -- labels ------------------------------------------------------------- #

    def view(self, i: int) -> dict[str, Any]:
        item = self.items[i]
        _, meta = self.crop(i)
        label = self.store.get(item.key)
        out: dict[str, Any] = {
            "index": i,
            "total": len(self.items),
            "key": item.key,
            "crop": meta,
            "label": None,
        }
        if label is not None:
            out["label"] = {
                "plate": label.plate or "",
                "plate_type": label.plate_type.value,
                "tags": list(label.tags),
                "points": self.to_display_coords(i, label.corners) if label.corners else [],
            }
        return out

    def save(self, i: int, payload: dict[str, Any]) -> dict[str, Any]:
        points = payload.get("points") or []
        try:
            plate_type = PlateType(payload.get("plate_type") or "particular")
        except ValueError:
            raise LabelError(f"unknown plate type {payload.get('plate_type')!r}") from None
        corners = self.to_image_coords(i, points) if points else None
        if corners is not None and len(corners) != 4:
            raise LabelError(f"click exactly four corners (got {len(corners)})")
        self.store.put(
            self.items[i].key,
            plate=payload.get("plate"),
            plate_type=plate_type,
            corners=corners,
            tags=payload.get("tags") or [],
        )
        return self.state()

    def delete(self, i: int) -> dict[str, Any]:
        self.store.delete(self.items[i].key)
        return self.state()

    def state(self) -> dict[str, Any]:
        done = [it.key in self.store for it in self.items]
        cov = coverage(self.store.all())
        first_open = next((i for i, d in enumerate(done) if not d), 0)
        plates = sorted({lb.plate for lb in self.store.all() if lb.plate})
        return {
            "source": self.source,
            "total": len(self.items),
            "labelled": sum(done),
            "done": done,
            "first_open": first_open,
            "tags": list(KNOWN_TAGS),
            "types": [t.value for t in PlateType],
            "plates": plates,
            "coverage": {
                "distinct_plates": cov.distinct_plates,
                "by_type": cov.by_type,
                "char_counts": cov.char_counts,
                "missing": cov.missing(),
            },
        }


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RectifyError("JPEG encoding failed")
    return buf.tobytes()


def overlay_geometry(flat: np.ndarray, scale: int, geometry: PlateGeometry = MT001) -> np.ndarray:
    """Upscale a rectified crop (nearest-neighbour: show the pixels the model sees)
    and outline where MT-001 says the characters and municipality band are."""
    h, w = flat.shape[:2]
    big = cv2.resize(flat, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    canvas = (w * scale, h * scale)
    layer = big.copy()
    rects = [*geometry.char_boxes(), geometry.muni_band]
    for k, rect in enumerate(rects):
        r = geometry.to_px(rect, canvas)
        color = (0, 200, 255) if k < 6 else (255, 80, 200)
        cv2.rectangle(layer, (round(r.x), round(r.y)), (round(r.right), round(r.bottom)), color, 1)
    return cv2.addWeighted(layer, 0.75, big, 0.25, 0)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _ui_html() -> bytes:
    return resources.files("alpr.plates").joinpath("annotate_ui.html").read_bytes()


def make_handler(app: AnnotateApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "annotate"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        # -- helpers -------------------------------------------------------- #

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
            self._json({"error": message}, status)

        def _index(self, part: str) -> int | None:
            stem = part.split(".")[0]
            if not stem.isdigit():
                return None
            i = int(stem)
            return i if 0 <= i < len(app.items) else None

        # -- routes --------------------------------------------------------- #

        def do_GET(self) -> None:  # noqa: N802 — http.server API
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if not parts:
                    self._send(HTTPStatus.OK, _ui_html(), "text/html; charset=utf-8")
                elif parts == ["api", "state"]:
                    self._json(app.state())
                elif len(parts) == 3 and parts[:2] == ["api", "item"]:
                    i = self._index(parts[2])
                    self._json(app.view(i)) if i is not None else self._error("no such item", 404)
                elif len(parts) == 3 and parts[:2] == ["api", "crop"]:
                    i = self._index(parts[2])
                    if i is None:
                        return self._error("no such item", 404)
                    body, _ = app.crop(i)
                    self._send(HTTPStatus.OK, body, "image/jpeg")
                elif len(parts) == 3 and parts[:2] == ["api", "preview"]:
                    i = self._index(parts[2])
                    if i is None:
                        return self._error("no such item", 404)
                    raw = (parse_qs(url.query).get("p") or [""])[0]
                    nums = [float(v) for v in raw.split(",") if v]
                    if len(nums) != 8:
                        return self._error("need four corners")
                    pts = [(nums[k], nums[k + 1]) for k in range(0, 8, 2)]
                    self._send(HTTPStatus.OK, app.preview(i, pts), "image/jpeg")
                else:
                    self._error("not found", 404)
            except (LabelError, RectifyError, ValueError) as exc:
                self._error(str(exc))

        def do_POST(self) -> None:  # noqa: N802 — http.server API
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._error("body is not JSON")
            if len(parts) != 3 or parts[0] != "api" or parts[1] not in ("label", "delete"):
                return self._error("not found", 404)
            i = self._index(parts[2])
            if i is None:
                return self._error("no such item", 404)
            try:
                result = app.save(i, payload) if parts[1] == "label" else app.delete(i)
                self._json(result)
            except (LabelError, RectifyError, ValueError) as exc:
                self._error(str(exc))

    return Handler


def make_server(app: AnnotateApp, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bound to loopback only: labels and photos never leave this machine."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))


# --------------------------------------------------------------------------- #
# Contact sheet and layout measurement
# --------------------------------------------------------------------------- #


def _labelled_quads(app_paths: DataPaths, source: str) -> list[tuple[Item, Any]]:
    store = LabelStore(labels_path(app_paths, source))
    out = []
    for item in list_items(app_paths, source, order="name"):
        lb = store.get(item.key)
        if lb is not None and lb.corners is not None:
            out.append((item, lb))
    return out


def contact_sheet(
    paths: DataPaths, source: str, *, n: int = 12, cols: int = 4, seed: int = 0
) -> np.ndarray:
    """Rectified crops of `n` labelled plates, captioned. The Sprint 2.1 eyeball check:
    every plate flat, upright, filling the frame. Any rotated or mirrored crop means
    a corner-order bug — fix it here, not in Phase 4."""
    pairs = _labelled_quads(paths, source)
    if not pairs:
        raise LabelError(f"no labelled plates with corners for {source!r} yet")
    rng = np.random.default_rng(seed)
    if len(pairs) > n:
        pairs = [pairs[k] for k in sorted(rng.choice(len(pairs), n, replace=False))]
    scale = 2
    tile_w, tile_h = CANONICAL_SIZE[0] * scale, CANONICAL_SIZE[1] * scale
    cap = 26
    rows = -(-len(pairs) // cols)
    sheet = np.full((rows * (tile_h + cap), cols * tile_w, 3), 255, np.uint8)
    for k, (item, lb) in enumerate(pairs):
        flat = rectify(load_image(item.image), lb.corners, CANONICAL_SIZE)
        tile = cv2.resize(flat, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        r, c = divmod(k, cols)
        y, x = r * (tile_h + cap), c * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = tile
        text = f"{lb.plate or '?'}  {lb.plate_type.value}"
        cv2.putText(
            sheet,
            text,
            (x + 6, y + tile_h + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def measure_layout(
    samples: Sequence[tuple[np.ndarray, Any]], geometry: PlateGeometry = MT001
) -> tuple[dict[str, list[float]], int]:
    """Measure character and municipality positions on rectified plates, in mm.

    Each plate is rectified at 4 px/mm, binarised, and its six character blobs found.
    Positions use blob *centres* and the shared top/bottom line, which do not depend
    on which glyph sits in a cell (a "1" is narrow, an "M" wide; their centres are
    still the cell's). Returns per-field measurement lists and how many plates were
    skipped because six clean character blobs could not be isolated.
    """
    px = 4.0
    size = geometry.render_size(px)
    out: dict[str, list[float]] = {
        k: []
        for k in (
            "char_top",
            "char_h",
            "char_gap",
            "logo_gutter",
            "muni_gap",
            "muni_h",
            "side_margin",
        )
    }
    skipped = 0
    for image, corners in samples:
        flat = rectify(image, corners, size)
        gray = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(
            cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        _, _, stats, _ = cv2.connectedComponentsWithStats(th)
        blobs = [
            s
            for s in stats[1:]
            if 55 * px < s[3] < 100 * px and 3 * px < s[2] < 60 * px and 5 * px < s[1] < 75 * px
        ]
        if len(blobs) < 6:
            skipped += 1
            continue
        hmed = statistics.median(s[3] for s in blobs)
        chars = sorted((s for s in blobs if abs(s[3] - hmed) < 0.08 * hmed), key=lambda s: s[0])
        if len(chars) != 6:
            skipped += 1
            continue
        tops = [s[1] / px for s in chars]
        bottoms = [(s[1] + s[3]) / px for s in chars]
        centres = [(s[0] + s[2] / 2) / px for s in chars]
        top, bottom = statistics.median(tops), statistics.median(bottoms)
        pitch = ((centres[2] - centres[0]) + (centres[5] - centres[3])) / 4.0
        out["char_top"].append(top)
        out["char_h"].append(bottom - top)
        out["char_gap"].append(pitch - geometry.char_w)
        out["logo_gutter"].append((centres[3] - centres[2]) - geometry.char_w)
        out["side_margin"].append(
            (
                (centres[0] - geometry.char_w / 2)
                + (geometry.plate_w - centres[5] - geometry.char_w / 2)
            )
            / 2
        )
        x_lo, x_hi = chars[0][0], chars[5][0] + chars[5][2]
        muni = [
            s
            for s in stats[1:]
            if bottom * px < s[1] < (bottom + 30) * px
            and 10 * px < s[3] < 32 * px
            and x_lo <= s[0] <= x_hi
        ]
        if len(muni) >= 3:
            m_top = statistics.median(s[1] / px for s in muni)
            m_bot = statistics.median((s[1] + s[3]) / px for s in muni)
            out["muni_gap"].append(m_top - bottom)
            out["muni_h"].append(m_bot - m_top)
    return out, skipped


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    app = AnnotateApp(DataPaths.default(), args.source, order=args.order)
    server = make_server(app, args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"labelling {args.source}: {len(app.items)} items, {len(app.store)} already labelled")
    print(f"labels file: {app.store.path}")
    print(f"open {url}   (Ctrl-C to stop; every save is written immediately)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    items = list_items(paths, args.source)
    store = LabelStore(labels_path(paths, args.source))
    cov = coverage(store.all())
    with_corners = sum(1 for lb in store.all() if lb.corners)
    print(f"{args.source}: {cov.labelled} of {len(items)} labelled ({with_corners} with corners)")
    print(f"labels file: {store.path}")
    if cov.by_type:
        print("by type: " + ", ".join(f"{k} {v}" for k, v in sorted(cov.by_type.items())))
    print(f"distinct particular plates: {cov.distinct_plates}")
    missing = cov.missing()
    print("\ncharacter coverage (distinct plates; gate: every character >= 5)")
    line = "  ".join(f"{c}:{n}" for c, n in cov.char_counts.items())
    print("  " + line)
    if missing:
        print(f"\nbelow 5: {' '.join(sorted(missing))}")
    else:
        print("\nevery character appears at least 5 times")
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    sheet = contact_sheet(paths, args.source, n=args.n, seed=args.seed)
    out = Path(args.out) if args.out else paths.root / "labels" / f"{args.source}_sheet.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"written to {out}")
    print("check: every plate flat, upright and filling the frame; none rotated or mirrored")
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    pairs = [
        (item, lb)
        for item, lb in _labelled_quads(paths, args.source)
        if lb.plate_type is PlateType.PARTICULAR
        and (not args.glyph_sources_only or "glyph_source" in lb.tags)
    ]
    if not pairs:
        print("no labelled particular plates with corners yet", file=sys.stderr)
        return 1
    samples = [(load_image(item.image), lb.corners) for item, lb in pairs]
    results, skipped = measure_layout(samples)
    print(
        f"measured {len(pairs) - skipped} plates ({skipped} skipped: six clean characters "
        "not isolated)\n"
    )
    print(f"{'field':12s} {'MT001':>7s} {'source':>9s} {'median':>7s} {'IQR':>13s} {'n':>4s}")
    for name, values in results.items():
        current = getattr(MT001, name)
        prov = PROVENANCE[name].value if name in PROVENANCE else "derived"
        if not values:
            print(f"{name:12s} {current:7.1f} {prov:>9s} {'-':>7s} {'-':>13s} {0:4d}")
            continue
        q1, q3 = np.percentile(values, [25, 75])
        print(
            f"{name:12s} {current:7.1f} {prov:>9s} {statistics.median(values):7.1f} "
            f"{q1:6.1f}-{q3:6.1f} {len(values):4d}"
        )
    print(
        "\nThe 'source' column says where geometry.py's value came from (spec, tape or "
        "photo).\nIn photos the characters' bottom edge reads ~5 mm low (shadow of the "
        "embossing), so\nchar_h reads high and muni_gap low; compare top edges instead."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="open the labelling page")
    p.add_argument(
        "source", help="own_photos, or any indexed source (e.g. roboflow_proyecto_placas)"
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument(
        "--order",
        choices=["size", "name"],
        default="size",
        help="largest plates first (default) or by file name",
    )
    p.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="labelling progress and character coverage")
    p.add_argument("source")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("sheet", help="contact sheet of rectified labelled crops")
    p.add_argument("source")
    p.add_argument("-n", type=int, default=12, help="how many plates (default 12)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="output JPEG (default data/labels/<source>_sheet.jpg)")
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("measure", help="measure plate layout from labelled plates")
    p.add_argument("source")
    p.add_argument(
        "--glyph-sources-only",
        action="store_true",
        help="only plates tagged glyph_source (sharpest, most frontal)",
    )
    p.set_defaults(func=cmd_measure)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LabelError, RectifyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
