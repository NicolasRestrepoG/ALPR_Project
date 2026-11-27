"""`generate` — the Phase 3 command.

generate backgrounds    build the vehicle-front background corpus (test set excluded)
generate demo           contact sheet: rows from severity 0 (clean) to 1 (barely readable)
generate compare        20 synthetic + 20 real crops, shuffled and numbered, with a key:
                        the Sprint 3.1 check — if you can sort them at a glance, the
                        gap is visible, and therefore learnable

generate corpus         the training corpus: spec → hash → data/corpus/<hash>/
generate ablation       four corpora differing only in stage order (Sprint 3.2)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from alpr.data_sources.storage import DataPaths
from alpr.plates.geometry import CANONICAL_SIZE
from alpr.plates.labels import LabelError
from alpr.plates.render import renderer_from_assets
from alpr.synth.backgrounds import BackgroundStore, build
from alpr.synth.pipeline import Synthesizer

DEFAULT_BG_SOURCES = ["ccpd", "roboflow_proyecto_placas", "roboflow_placas_colombia_itm"]


def load_municipalities(paths: DataPaths) -> list[str]:
    f = paths.assets_dir / "municipios.txt"
    if not f.exists():
        return ["BOGOTA D.C."]
    return [line.strip() for line in f.read_text().splitlines() if line.strip()]


def default_synthesizer(paths: DataPaths, *, quiet: bool = False) -> Synthesizer:
    try:
        backgrounds: BackgroundStore | None = BackgroundStore(paths)
    except LabelError as exc:
        if not quiet:
            print(f"note: {exc}; using plain backgrounds", file=sys.stderr)
        backgrounds = None
    return Synthesizer(renderer_from_assets(paths), backgrounds, load_municipalities(paths))


def _tile(crop: np.ndarray, label: str, scale: int = 2) -> np.ndarray:
    w, h = CANONICAL_SIZE
    t = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    cap = np.full((22, t.shape[1], 3), 255, np.uint8)
    cv2.putText(cap, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([t, cap])


def demo_sheet(syn: Synthesizer, per_row: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        tiles = [
            _tile(smp.crop, f"s={s:g} {smp.params.text}")
            for smp in (syn.sample(rng, s) for _ in range(per_row))
        ]
        rows.append(np.hstack(tiles))
    return np.vstack(rows)


def compare_sheet(
    paths: DataPaths, syn: Synthesizer, n: int = 20, seed: int = 0, cols: int = 8
) -> tuple[np.ndarray, list[str]]:
    """n synthetic and n real rectified crops, shuffled; returns the sheet and the key."""
    from alpr.plates.rectify import rectify
    from alpr.plates.testset import entry_image, load_entries

    rng = np.random.default_rng(seed)
    entries = [e for e in load_entries(paths) if e.plate_type == "particular" and e.corners]
    if len(entries) < n:
        raise LabelError(f"need {n} particular plates in the frozen test set, found {len(entries)}")
    pick = rng.choice(len(entries), n, replace=False)
    real = [
        rectify(entry_image(paths, entries[k]), entries[k].corners, CANONICAL_SIZE) for k in pick
    ]
    synth = [syn.sample(rng, float(rng.uniform(0.2, 1.0))).crop for _ in range(n)]
    items = [(c, "real") for c in real] + [(c, "synthetic") for c in synth]
    order = rng.permutation(len(items))
    tiles, key = [], []
    for i, k in enumerate(order, start=1):
        crop, kind = items[k]
        tiles.append(_tile(crop, f"#{i}"))
        key.append(f"#{i}: {kind}")
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    if rows[-1].shape[1] < rows[0].shape[1]:
        pad = np.full((rows[-1].shape[0], rows[0].shape[1] - rows[-1].shape[1], 3), 255, np.uint8)
        rows[-1] = np.hstack([rows[-1], pad])
    return np.vstack(rows), key


def _save(img: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def cmd_backgrounds(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    written, failed = build(
        paths,
        args.sources,
        ccpd_n=args.ccpd,
        seed=args.seed,
        workers=args.workers,
        force=args.force,
    )
    print(f"{written} background crops written to {paths.root / 'backgrounds'} ({failed} failed)")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    sheet = demo_sheet(default_synthesizer(paths), args.per_row, args.seed)
    out = Path(args.out) if args.out else paths.root / "synth" / "demo_sheet.png"
    _save(sheet, out)
    print(f"written to {out} — rows: severity 0, 0.25, 0.5, 0.75, 1")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    sheet, key = compare_sheet(paths, default_synthesizer(paths), args.n, args.seed)
    out = Path(args.out) if args.out else paths.root / "synth" / "compare.png"
    _save(sheet, out)
    key_path = out.with_name(out.stem + "_key.txt")
    key_path.write_text("\n".join(key) + "\n")
    print(f"written to {out}\nanswer key: {key_path} — guess first, then check")
    return 0


GATE_SECONDS = 15 * 60
GATE_SPREAD = 0.15


def _print_report(rep, label: str = "") -> None:
    state = "reused (already built)" if rep.reused else f"built in {rep.seconds:.0f} s"
    print(f"{label}corpus {rep.hash}: {rep.size} samples, {state}")
    print(f"  {rep.path}")
    if not rep.reused:
        print(f"  {rep.samples_per_s:.0f} samples/s on {rep.workers} workers")
    print("  classes: " + ", ".join(f"{k} {v}" for k, v in sorted(rep.class_counts.items())))
    print(f"  glyph spread: letters {rep.letter_spread:.1%}, digits {rep.digit_spread:.1%}")


def cmd_corpus(args: argparse.Namespace) -> int:
    from alpr.synth.corpus import CorpusSpec, build_corpus

    spec = CorpusSpec(size=args.size, seed=args.seed, order=args.order)
    rep = build_corpus(DataPaths.default(), spec, workers=args.workers, force=args.force)
    _print_report(rep)
    ok_time = rep.reused or rep.seconds < GATE_SECONDS * rep.size / 20_000
    ok_spread = max(rep.letter_spread, rep.digit_spread) <= GATE_SPREAD
    verdict = {True: "PASS", False: "FAIL"}
    print(f"gates: time {verdict[ok_time]}, glyph spread {verdict[ok_spread]}")
    return 0 if ok_time and ok_spread else 1


def cmd_ablation(args: argparse.Namespace) -> int:
    from alpr.synth.corpus import CorpusSpec, build_corpus
    from alpr.synth.pipeline import ORDERS

    paths = DataPaths.default()
    for order in ORDERS:
        spec = CorpusSpec(size=args.size, seed=args.seed, order=order)
        _print_report(build_corpus(paths, spec, workers=args.workers), f"[{order}] ")
    print("train each for 8 epochs in Phase 4; the correct order must win by >= 3 points")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backgrounds", help="build the background corpus")
    p.add_argument("sources", nargs="*", default=DEFAULT_BG_SOURCES)
    p.add_argument("--ccpd", type=int, default=19000, help="CCPD crops to draw (default 19000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true", help="rebuild an existing corpus")
    p.set_defaults(func=cmd_backgrounds)

    p = sub.add_parser("demo", help="severity contact sheet")
    p.add_argument("--per-row", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("corpus", help="build (or reuse) a training corpus")
    p.add_argument("--size", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order", default="correct", help="stage order (ablation only)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--force", action="store_true", help="rebuild even if it exists")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("ablation", help="four corpora, one per stage order")
    p.add_argument("--size", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=None)
    p.set_defaults(func=cmd_ablation)

    p = sub.add_parser("compare", help="shuffled synthetic vs real crops, with a key")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LabelError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
