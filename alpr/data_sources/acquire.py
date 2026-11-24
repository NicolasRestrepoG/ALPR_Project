"""`acquire` — the Phase 1 command.

One command speaks to eleven heterogeneous sources: a 13 GB Drive archive, three
API-key exports, two behind a signed agreement, one browsed by hand, one built from
YouTube, one shot on a phone, and two reference documents.

    acquire list [--commercial-only] [--purpose P] [--access A] [--skip-manual]
    acquire paths [--ensure]
    acquire fetch <name>... | --all [--force] [--commercial-only] [--skip-manual]
    acquire index <name>... | --all
    acquire manifest [--audit]
    acquire verify
    acquire audit
    acquire email <ufpr_alpr|rodosol_alpr> [--write]
"""

from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path

import alpr.data_sources  # noqa: F401 — importing the package registers every source
from alpr.data_sources.base import Status
from alpr.data_sources.manifest import (
    Manifest,
    ManifestError,
    build_manifest,
    verify_manifest,
)
from alpr.data_sources.registry import (
    AccessMode,
    Purpose,
    Source,
    SourceInfo,
    all_infos,
    get,
    select,
)
from alpr.data_sources.storage import DataPaths


def _fmt_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)).rstrip()
    sep = "  ".join("-" * w for w in widths)
    body = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in rows
    ]
    return "\n".join([line, sep, *body])


def _filters(args: argparse.Namespace) -> dict:
    return {
        "commercial_only": getattr(args, "commercial_only", False),
        "skip_manual": getattr(args, "skip_manual", False),
        "purpose": Purpose(args.purpose) if getattr(args, "purpose", None) else None,
        "access": AccessMode(args.access) if getattr(args, "access", None) else None,
    }


def cmd_list(args: argparse.Namespace) -> int:
    classes = select(**_filters(args))
    if not classes:
        print("no sources match those filters")
        return 0
    rows = []
    for cls in classes:
        info: SourceInfo = cls.info
        rows.append(
            [
                info.name,
                info.license.name,
                "yes" if info.commercial_ok else "NO",
                info.access.value,
                ",".join(sorted(p.value for p in info.purposes)),
                info.size_hint,
            ]
        )
    print(_fmt_table(rows, ["source", "licence", "commercial", "access", "purposes", "size"]))
    print(f"\n{len(classes)} of {len(all_infos())} registered sources")
    return 0


def cmd_paths(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    print(f"data root   {paths.root}")
    print(f"raw         {paths.raw_dir}")
    print(f"index       {paths.index_dir}")
    print(f"assets      {paths.assets_dir}")
    print(f"real_test   {paths.real_test_dir}")
    print(f"manifest    {paths.manifest_path}")
    if args.ensure:
        paths.ensure()
        print("\ndirectories created")
    return 0


def _resolve(args: argparse.Namespace) -> list[type[Source]]:
    if args.all:
        return select(**_filters(args))
    if not args.sources:
        raise SystemExit("name at least one source, or pass --all")
    return select(names=args.sources, **_filters(args))


def _extra_fetch_kwargs(source: Source, args: argparse.Namespace) -> dict:
    """Pass source-specific flags only to the sources that accept them.

    Keeps `Source.fetch(force=...)` as the universal signature while letting a
    13 GB archive expose the extraction controls it genuinely needs.
    """
    accepted = inspect.signature(type(source).fetch).parameters
    candidates = {
        "extract": not getattr(args, "no_extract", False),
        "subsets": getattr(args, "subsets", None),
        "keep_fraction": getattr(args, "keep_fraction", 1.0),
        "delete_archive": getattr(args, "delete_archive", False),
    }
    return {k: v for k, v in candidates.items() if k in accepted}


def cmd_fetch(args: argparse.Namespace) -> int:
    paths = DataPaths.default().ensure()
    failures = 0
    for cls in _resolve(args):
        source = cls(paths)
        print(f"\n=== {source.name} — {cls.info.title}")
        try:
            result = source.fetch(force=args.force, **_extra_fetch_kwargs(source, args))
        except Exception as exc:  # noqa: BLE001 — one bad source must not stop the rest
            failures += 1
            print(f"[{Status.FAILED.value}] {source.name}: {type(exc).__name__}: {exc}")
            continue
        print(result)
        if result.status is Status.FAILED:
            failures += 1
    print()
    return 1 if failures else 0


def cmd_index(args: argparse.Namespace) -> int:
    paths = DataPaths.default().ensure()
    failures = 0
    for cls in _resolve(args):
        source = cls(paths)
        try:
            count = source.write_index()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"{cls.info.name}: FAILED — {type(exc).__name__}: {exc}")
            continue
        errors = getattr(source, "last_index_errors", [])
        skipped = getattr(source, "last_index_skipped", 0)
        print(f"{source.name}: {count:,} records → {source.index_path}")
        if skipped:
            print(f"  {skipped} file(s) skipped or refused; first few:")
            for err in errors[:5]:
                print(f"    {err}")
    return 1 if failures else 0


def cmd_manifest(args: argparse.Namespace) -> int:
    paths = DataPaths.default().ensure()
    previous = None
    if paths.manifest_path.exists():
        try:
            previous = Manifest.load(paths.manifest_path)
        except ManifestError as exc:
            print(f"warning: existing manifest unreadable, rebuilding ({exc})")
    manifest = build_manifest(paths, previous=previous)
    manifest.save(paths.manifest_path)

    rows = [
        [
            e.name,
            e.status,
            f"{e.file_count:,}",
            f"{e.bytes_on_disk / 1e6:,.1f} MB" if e.bytes_on_disk else "-",
            f"{e.record_count:,}" if e.record_count else "-",
            "yes" if e.commercial_ok else "NO",
        ]
        for e in sorted(manifest.sources.values(), key=lambda e: e.name)
    ]
    print(_fmt_table(rows, ["source", "status", "files", "size", "records", "commercial"]))
    print(f"\nwritten to {paths.manifest_path}")
    if args.audit:
        print()
        print(manifest.audit_report())
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    paths = DataPaths.default()
    try:
        manifest = Manifest.load(paths.manifest_path)
    except ManifestError as exc:
        print(exc)
        return 1
    result = verify_manifest(manifest, paths)
    print(result)
    return 0 if result.ok else 1


def cmd_audit(args: argparse.Namespace) -> int:
    """Answer one question: can what is on disk be commercially deployed?"""
    paths = DataPaths.default()
    try:
        manifest = Manifest.load(paths.manifest_path)
    except ManifestError:
        manifest = build_manifest(paths)
    print(manifest.audit_report())
    return 0 if manifest.commercial_ok() else 2


def cmd_import_captures(args: argparse.Namespace) -> int:
    """Turn the capture app's captures.jsonl into labels.csv."""
    from alpr.data_sources.captures import format_report, import_captures
    from alpr.data_sources.own_photos import OwnPhotosSource

    paths = DataPaths.default().ensure()
    root = Path(args.root) if args.root else OwnPhotosSource(paths).raw_dir
    report = import_captures(root, dry_run=args.dry_run)
    print(format_report(report, root, dry_run=args.dry_run))
    return 0 if report.ok else 1


def cmd_keys(args: argparse.Namespace) -> int:
    """Report which credentials resolve, and from where. Never prints a value."""
    from alpr.config import ENV_FILE_NAME, env_file, secret_source

    path = env_file()
    print(f"{ENV_FILE_NAME} file: {path if path else '(none found)'}\n")

    known = [
        ("ROBOFLOW_API_KEY", "roboflow_placas_colombianas + 2 others", "Phase 1"),
        ("ALPR_DATA_ROOT", "where datasets live", "optional"),
        ("COMMERCIAL_OCR_API_KEY", "baseline comparison", "Sprint 5.3"),
        ("CONTACT_NAME", "licence-request emails", "gated sources"),
        ("CONTACT_EMAIL", "licence-request emails", "gated sources"),
    ]
    rows = []
    for name, used_by, when in known:
        source = secret_source(name)
        rows.append([name, "set" if source != "not set" else "-", source, used_by, when])
    print(_fmt_table(rows, ["variable", "state", "source", "used by", "needed"]))
    print("\nValues are never printed, logged, or written to data/manifest.json.")

    if args.check:
        from alpr.data_sources.roboflow import api_key, check_api_key

        print("\nValidating against the provider:")
        key = api_key()
        if key is None:
            print("  ROBOFLOW_API_KEY   not set — nothing to check")
            return 1
        ok, message = check_api_key(key)
        print(f"  ROBOFLOW_API_KEY   {'OK' if ok else 'FAILED'} — {message}")
        return 0 if ok else 1
    return 0


def cmd_email(args: argparse.Namespace) -> int:
    """Print (or write) the licence-request email for a gated source."""
    cls = get(args.source)
    source = cls(DataPaths.default())
    if not hasattr(source, "email_draft"):
        print(f"{args.source} is not a gated source — no licence request needed")
        return 1
    draft = source.email_draft()
    if args.write:
        path = Path(source.draft_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(draft, encoding="utf-8")
        print(f"written to {path}")
    else:
        print(draft)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acquire",
        description="Fetch, normalize and licence-audit every ALPR data source.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_filters(p: argparse.ArgumentParser, *, full: bool = False) -> None:
        p.add_argument(
            "--commercial-only",
            action="store_true",
            help="exclude non-commercial and unverified-licence sources",
        )
        p.add_argument("--skip-manual", action="store_true", help="only unattended sources")
        p.add_argument(
            "--purpose",
            choices=[x.value for x in Purpose],
            help="only sources serving this purpose",
        )
        if full:
            p.add_argument("--access", choices=[x.value for x in AccessMode])

    p_list = sub.add_parser("list", help="show registered sources with licence and access mode")
    add_filters(p_list, full=True)
    p_list.set_defaults(func=cmd_list)

    p_paths = sub.add_parser("paths", help="show the on-disk layout")
    p_paths.add_argument("--ensure", action="store_true", help="create the directories")
    p_paths.set_defaults(func=cmd_paths)

    p_fetch = sub.add_parser("fetch", help="download source files")
    p_fetch.add_argument("sources", nargs="*")
    p_fetch.add_argument("--all", action="store_true")
    p_fetch.add_argument("--force", action="store_true", help="re-download even if present")
    p_fetch.add_argument(
        "--no-extract", action="store_true", help="download archives but do not unpack them"
    )
    p_fetch.add_argument(
        "--subsets",
        nargs="+",
        metavar="NAME",
        help="extract only these subdirectories (e.g. ccpd_base ccpd_np)",
    )
    p_fetch.add_argument(
        "--keep-fraction",
        type=float,
        default=1.0,
        metavar="F",
        help="deterministically keep only this fraction of images (0 < F <= 1)",
    )
    p_fetch.add_argument(
        "--delete-archive",
        action="store_true",
        help="remove the archive after extraction succeeds, to reclaim disk",
    )
    add_filters(p_fetch)
    p_fetch.set_defaults(func=cmd_fetch)

    p_index = sub.add_parser("index", help="normalize what is on disk into JSON Lines records")
    p_index.add_argument("sources", nargs="*")
    p_index.add_argument("--all", action="store_true")
    add_filters(p_index)
    p_index.set_defaults(func=cmd_index)

    p_manifest = sub.add_parser("manifest", help="rebuild data/manifest.json from disk")
    p_manifest.add_argument("--audit", action="store_true", help="also print the lineage audit")
    p_manifest.set_defaults(func=cmd_manifest)

    p_verify = sub.add_parser("verify", help="check the manifest against what is on disk")
    p_verify.set_defaults(func=cmd_verify)

    p_audit = sub.add_parser("audit", help="licence lineage audit")
    p_audit.set_defaults(func=cmd_audit)

    p_import = sub.add_parser(
        "import-captures",
        help="validate the capture app's captures.jsonl and write labels.csv",
    )
    p_import.add_argument(
        "root", nargs="?", help="folder holding testset/, close_ups/ and captures.jsonl"
    )
    p_import.add_argument("--dry-run", action="store_true", help="validate only; write nothing")
    p_import.set_defaults(func=cmd_import_captures)

    p_keys = sub.add_parser("keys", help="show which credentials resolve, and from where")
    p_keys.add_argument(
        "--check", action="store_true", help="validate the key against the provider (network)"
    )
    p_keys.set_defaults(func=cmd_keys)

    p_email = sub.add_parser("email", help="print the licence-request email for a gated source")
    p_email.add_argument("source", choices=["ufpr_alpr", "rodosol_alpr"])
    p_email.add_argument("--write", action="store_true", help="save to data/licence_requests/")
    p_email.set_defaults(func=cmd_email)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
