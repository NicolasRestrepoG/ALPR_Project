"""Reference assets: the DANE municipality list and the MT-001 plate spec.

These are not image datasets — they produce no `Record`s. They are the ground truth
the Phase 2 renderer is built against: the municipality strip printed below the
characters, and the millimetre geometry of the plate itself.

The municipality list matters more than it looks. It is rendered into the strip on
every synthetic plate, the strip is inside the rectified crop, so the model sees it
every single sample — and a long name compresses the strip in a way the validity
head learns from (plan §23, R14).
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from alpr.data_sources.base import Status
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceInfo,
    register,
)
from alpr.data_sources.storage import atomic_write_text, download

DIVIPOLA_CSV_URL = "https://www.datos.gov.co/api/views/gdxc-w37w/rows.csv?accessType=DOWNLOAD"
DIVIPOLA_PAGE = "https://www.datos.gov.co/Mapas-Nacionales/DIVIPOLA-C-digos-municipios/gdxc-w37w"

MUNICIPIOS_FILE = "municipios.txt"

#: Row types in the DIVIPOLA export. "Área no municipalizada" entries are not
#: vehicle-registration municipalities and never appear on a plate strip.
DEFAULT_ROW_TYPES = ("Municipio", "Isla")

#: Characters that can survive onto a plate strip: the alphabet the renderer draws.
_ALLOWED_STRIP_CHARS = re.compile(r"[^A-Z0-9 ,.\-]")


def strip_accents(text: str) -> str:
    """NFKD-decompose and drop combining marks: MEDELLÍN → MEDELLIN, PEÑOL → PENOL."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def normalize_municipality(name: str) -> str:
    """Accent-stripped, uppercased, whitespace-collapsed — as it appears on a plate."""
    out = strip_accents(name).upper()
    out = _ALLOWED_STRIP_CHARS.sub(" ", out)
    return " ".join(out.split())


def _find_column(fieldnames: Iterable[str], *, want: str, avoid: tuple[str, ...] = ()) -> str:
    """Locate a column by normalized substring, so a renamed header doesn't break us."""
    for field in fieldnames:
        norm = normalize_municipality(field)
        if want in norm and not any(a in norm for a in avoid):
            return field
    raise ValueError(
        f"no column matching {want!r} in DIVIPOLA export; columns were {list(fieldnames)}"
    )


def parse_divipola_csv(text: str, *, row_types: tuple[str, ...] = DEFAULT_ROW_TYPES) -> list[str]:
    """Extract sorted unique normalized municipality names from the DANE CSV."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("DIVIPOLA export has no header row")

    name_col = _find_column(reader.fieldnames, want="MUNICIPIO", avoid=("CODIGO",))
    try:
        type_col = _find_column(reader.fieldnames, want="TIPO")
    except ValueError:
        type_col = None

    wanted = {normalize_municipality(t) for t in row_types}
    names: set[str] = set()
    for row in reader:
        if type_col is not None and wanted:
            row_type = normalize_municipality(row.get(type_col) or "")
            if row_type and row_type not in wanted:
                continue
        name = normalize_municipality(row.get(name_col) or "")
        if name:
            names.add(name)
    return sorted(names)


@register
class MunicipalitiesSource(Source):
    """DANE DIVIPOLA municipality list → `data/assets/municipios.txt`."""

    info = SourceInfo(
        name="divipola",
        title="DANE DIVIPOLA municipality list",
        license_key="public-gov-co",
        access=AccessMode.AUTOMATIC,
        purposes=frozenset({Purpose.ASSETS}),
        url=DIVIPOLA_PAGE,
        size_hint="~1,100 rows, ~70 KB",
        notes="Direct CSV export from datos.gov.co. Feeds the plate municipality strip.",
    )

    @property
    def csv_path(self) -> Path:
        return self.raw_dir / "divipola_municipios.csv"

    @property
    def output_path(self) -> Path:
        return self.paths.assets_dir / MUNICIPIOS_FILE

    def fetch(self, *, force: bool = False) -> FetchResult:
        self.paths.ensure()
        download(DIVIPOLA_CSV_URL, self.csv_path, force=force, progress=False)
        text = self.csv_path.read_text(encoding="utf-8", errors="replace")
        names = parse_divipola_csv(text)
        if len(names) < 900:
            return FetchResult(
                source=self.name,
                status=Status.FAILED,
                message=(
                    f"only {len(names)} municipality names parsed; expected ~1,000+. "
                    f"The DIVIPOLA export layout may have changed — check {self.csv_path}"
                ),
            )
        atomic_write_text(self.output_path, "\n".join(names) + "\n")
        return FetchResult(
            source=self.name,
            status=Status.DOWNLOADED,
            message=f"{len(names)} municipality names → {self.output_path}",
            paths=[str(self.csv_path), str(self.output_path)],
            bytes_written=self.output_path.stat().st_size,
        )

    def load(self) -> list[str]:
        """Read the generated list. Phase 2's renderer samples from this."""
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"{self.output_path} does not exist — run `acquire fetch divipola` first"
            )
        lines = self.output_path.read_text(encoding="utf-8").splitlines()
        return [ln.strip() for ln in lines if ln.strip()]


# --------------------------------------------------------------------------- #
# MT-001 plate specification
# --------------------------------------------------------------------------- #

#: Candidate locations for the Ficha Técnica MT-001 PDF, tried in order.
#: The Fiscalía copy cited in the plan (§25) returns 404 as of 2026-08-08; it is
#: kept first in case the host restores it.
MT001_URLS = (
    "https://www.fiscalia.gov.co/colombia//wp-content/uploads/policiajudicial/"
    "DOCPJFISCALIA/PLACA%20ANICA%20NACIONAL.pdf",
    "https://www.fiscalia.gov.co/colombia/wp-content/uploads/policiajudicial/"
    "DOCPJFISCALIA/PLACA%20ANICA%20NACIONAL.pdf",
)

MT001_INSTRUCTIONS = """\
The Ficha Técnica MT-001 PDF could not be downloaded automatically.

  The URL published in the project plan (§25, Fiscalía copy) now returns 404.

To obtain it by hand:
  1. Search for "Ficha Tecnica MT-001 placa unica nacional" or
     "Resolucion 3078 de 2007 MinTransporte placa unica".
  2. Save the PDF to:  {dest}
  3. Re-run `acquire fetch mt001` — it will be picked up and checksummed into the
     manifest (Sprint 1.2).

This is corroborating evidence, not a blocker. Every dimension Phase 2 needs is
already transcribed in the project plan, §8 (330 x 160 mm, characters
40 x 80 mm with 10 mm stroke and 8 mm gaps, municipality band 22 mm tall). The
PDF is worth having so those numbers can be re-checked against the source rather
than against a summary, and so the geometry module can cite a document hash.

Plain-language summary that agrees with §8:
  https://www.dupliplacas.com/requisitos-y-caracteristicas-placas-vehiculos-y-motocicletas-en-colombia.html
"""


@register
class PlateSpecSource(Source):
    """Ficha Técnica MT-001 — the plate geometry ground truth for Phase 2."""

    info = SourceInfo(
        name="mt001",
        title="Ficha Técnica MT-001 (plate specification)",
        license_key="public-gov-co",
        access=AccessMode.AUTOMATIC,
        purposes=frozenset({Purpose.ASSETS}),
        url=MT001_URLS[0],
        size_hint="1 PDF",
        notes="Plate geometry ground truth. Upstream URL is currently dead; see instructions.",
    )

    @property
    def output_path(self) -> Path:
        return self.paths.assets_dir / "spec" / "MT-001.pdf"

    def fetch(self, *, force: bool = False) -> FetchResult:
        self.paths.ensure()
        dest = self.output_path
        if dest.exists() and not force:
            return FetchResult(
                source=self.name,
                status=Status.READY,
                message=f"already present: {dest}",
                paths=[str(dest)],
            )

        errors = []
        for url in MT001_URLS:
            try:
                download(url, dest, force=force, progress=False)
            except Exception as exc:  # noqa: BLE001 — try the next candidate
                errors.append(f"{url} -> {type(exc).__name__}: {exc}")
                continue
            if self._looks_like_pdf(dest):
                return FetchResult(
                    source=self.name,
                    status=Status.DOWNLOADED,
                    message=f"downloaded from {url}",
                    paths=[str(dest)],
                    bytes_written=dest.stat().st_size,
                )
            dest.unlink(missing_ok=True)
            errors.append(f"{url} -> served HTML, not a PDF")

        return FetchResult(
            source=self.name,
            status=Status.MANUAL_REQUIRED,
            message=self.manual_instructions() + "\nAttempts:\n  " + "\n  ".join(errors),
        )

    def manual_instructions(self) -> str:
        return MT001_INSTRUCTIONS.format(dest=self.output_path)

    @staticmethod
    def _looks_like_pdf(path: Path) -> bool:
        try:
            with path.open("rb") as fh:
                return fh.read(5) == b"%PDF-"
        except OSError:
            return False
