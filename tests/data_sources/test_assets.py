"""Municipality normalization and the DIVIPOLA parser.

The municipality strip is rendered on every synthetic plate and sits inside the
rectified crop, so the model sees it in every single sample. Names must arrive in
exactly the form a plate shows them: accent-free and uppercase.
"""

from __future__ import annotations

import pytest

from alpr.data_sources.assets import (
    MunicipalitiesSource,
    PlateSpecSource,
    normalize_municipality,
    parse_divipola_csv,
    strip_accents,
)
from alpr.data_sources.storage import DataPaths

CSV = """\
Código Departamento,Nombre Departamento,Código Municipio,Nombre Municipio,\
Tipo: Municipio / Isla / Área no municipalizada,longitud,Latitud
05,ANTIOQUIA,05001,MEDELLÍN,Municipio,"-75,581775","6,246631"
05,ANTIOQUIA,05004,ABRIAQUÍ,Municipio,"-76,064304","6,632282"
11,BOGOTÁ,11001,"BOGOTÁ, D.C.",Municipio,"-74,1","4,6"
88,SAN ANDRÉS,88001,SAN ANDRÉS,Isla,"-81,7","12,5"
94,GUAINÍA,94883,SAN FELIPE,Área no municipalizada,"-67,0","1,9"
05,ANTIOQUIA,05002,  el  peñol  ,Municipio,"-75,2","6,2"
"""


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("MEDELLÍN", "MEDELLIN"),
        ("Abriaquí", "ABRIAQUI"),
        ("El Peñol", "EL PENOL"),
        ("BOGOTÁ, D.C.", "BOGOTA, D.C."),
        ("SAN ANDRÉS", "SAN ANDRES"),
        ("  villa   de   leyva  ", "VILLA DE LEYVA"),
        ("Sopó", "SOPO"),
        ("Ubaté", "UBATE"),
    ],
)
def test_normalize_municipality(raw, expected):
    assert normalize_municipality(raw) == expected


def test_normalized_names_use_only_renderable_characters():
    """The renderer draws A-Z, digits, space, comma, period and hyphen. Nothing else."""
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.-")
    for raw in ("MEDELLÍN", "El Peñol", "BOGOTÁ, D.C.", "Chocó", "Güicán de la Sierra"):
        assert set(normalize_municipality(raw)) <= allowed


def test_strip_accents_keeps_the_letter():
    assert strip_accents("ñ") == "n"
    assert strip_accents("Ü") == "U"


# --------------------------------------------------------------------------- #
# DIVIPOLA parsing
# --------------------------------------------------------------------------- #


def test_parse_divipola_extracts_normalized_names():
    names = parse_divipola_csv(CSV)
    assert names == [
        "ABRIAQUI",
        "BOGOTA, D.C.",
        "EL PENOL",
        "MEDELLIN",
        "SAN ANDRES",
    ]


def test_non_municipal_areas_are_excluded_by_default():
    """ "Área no municipalizada" entries are not vehicle-registration municipalities."""
    assert "SAN FELIPE" not in parse_divipola_csv(CSV)
    assert "SAN FELIPE" in parse_divipola_csv(CSV, row_types=())


def test_names_are_sorted_and_deduplicated():
    doubled = CSV + '05,ANTIOQUIA,05999,Medellin,Municipio,"-75,5","6,2"\n'
    names = parse_divipola_csv(doubled)
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert names.count("MEDELLIN") == 1


def test_a_renamed_header_still_resolves():
    renamed = CSV.replace("Nombre Municipio", "NOMBRE DEL MUNICIPIO")
    assert "MEDELLIN" in parse_divipola_csv(renamed)


def test_missing_municipality_column_raises():
    with pytest.raises(ValueError, match="no column matching"):
        parse_divipola_csv("a,b\n1,2\n")


def test_code_columns_are_not_mistaken_for_the_name_column():
    names = parse_divipola_csv(CSV)
    assert all(not n.isdigit() for n in names)


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


def test_municipalities_source_writes_the_asset_file(tmp_path, monkeypatch):
    from alpr.data_sources import assets

    paths = DataPaths(tmp_path).ensure()
    source = MunicipalitiesSource(paths)

    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(CSV, encoding="utf-8")
        return dest

    monkeypatch.setattr(assets, "download", fake_download)
    plausible = ["MEDELLIN"] + [f"M{i}" for i in range(1000)]
    monkeypatch.setattr(assets, "parse_divipola_csv", lambda text, **kw: plausible)

    result = source.fetch()
    assert result.ok
    assert source.output_path.exists()
    assert source.load()[0] == "MEDELLIN"


def test_municipalities_source_fails_loudly_on_a_short_list(tmp_path, monkeypatch):
    """A layout change upstream must not silently produce a 3-name list."""
    from alpr.data_sources import assets
    from alpr.data_sources.base import Status

    paths = DataPaths(tmp_path).ensure()
    source = MunicipalitiesSource(paths)

    def fake_download(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(CSV, encoding="utf-8")
        return dest

    monkeypatch.setattr(assets, "download", fake_download)
    result = source.fetch()
    assert result.status is Status.FAILED
    assert "expected ~1,000+" in result.message
    assert not source.output_path.exists()


def test_load_before_fetch_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="acquire fetch divipola"):
        MunicipalitiesSource(DataPaths(tmp_path)).load()


def test_plate_spec_reports_manual_instructions_when_unreachable(tmp_path, monkeypatch):
    """The plan's MT-001 URL is dead; the source must instruct, not crash."""
    from alpr.data_sources import assets
    from alpr.data_sources.base import Status

    def always_fails(url, dest, **kw):
        raise RuntimeError("404")

    monkeypatch.setattr(assets, "download", always_fails)
    result = PlateSpecSource(DataPaths(tmp_path).ensure()).fetch()

    assert result.status is Status.MANUAL_REQUIRED
    assert "Ficha Tecnica MT-001" in result.message
    assert "not a blocker" in result.message


def test_plate_spec_rejects_an_html_error_page(tmp_path, monkeypatch):
    """A 404 page served with a 200 status must not be saved as the spec PDF."""
    from alpr.data_sources import assets
    from alpr.data_sources.base import Status

    def serves_html(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"<!DOCTYPE html><html>Not found</html>")
        return dest

    monkeypatch.setattr(assets, "download", serves_html)
    source = PlateSpecSource(DataPaths(tmp_path).ensure())
    result = source.fetch()

    assert result.status is Status.MANUAL_REQUIRED
    assert not source.output_path.exists()


def test_plate_spec_accepts_a_real_pdf(tmp_path, monkeypatch):
    from alpr.data_sources import assets
    from alpr.data_sources.base import Status

    def serves_pdf(url, dest, **kw):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"%PDF-1.4\n...")
        return dest

    monkeypatch.setattr(assets, "download", serves_pdf)
    result = PlateSpecSource(DataPaths(tmp_path).ensure()).fetch()
    assert result.status is Status.DOWNLOADED
