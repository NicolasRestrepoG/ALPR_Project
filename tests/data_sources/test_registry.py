"""Source registration and filtering — especially `--commercial-only`."""

from __future__ import annotations

import pytest

import alpr.data_sources  # noqa: F401 — registers the real sources
from alpr.data_sources.base import Status
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceError,
    SourceInfo,
    all_infos,
    commercial_blockers,
    get,
    register,
    select,
)

#: Every source the plan's §8 table calls for, minus Mapillary (structurally
#: unusable: plates are auto-blurred at upload).
EXPECTED_SOURCES = {
    "ccpd",
    "dashcam",
    "divipola",
    "mt001",
    "own_photos",
    "platesmania",
    "roboflow_placas_colombia_itm",
    "roboflow_placas_colombianas",
    "roboflow_proyecto_placas",
    "rodosol_alpr",
    "ufpr_alpr",
}

#: Sources that block commercial deployment, and why. Asserting the exact set —
#: rather than "at least these" — means adding a source forces a deliberate
#: decision about its licence instead of letting it drift in unnoticed.
#:
#: The two Roboflow projects the plan (§8) listed as "unverified — check" are
#: absent because the check was done on 2026-08-09: the Roboflow API reports MIT
#: and CC BY 4.0 respectively, so they are on the clean path.
EXPECTED_BLOCKERS = {
    "dashcam": "youtube-tos",
    "platesmania": "platesmania-terms",
    "rodosol_alpr": "academic-nc",
    "ufpr_alpr": "academic-nc",
}


def test_every_planned_source_is_registered():
    assert {i.name for i in all_infos()} == EXPECTED_SOURCES


def test_ccpd_is_mit_and_commercially_clean():
    info = get("ccpd").info
    assert info.license_key == "mit"
    assert info.commercial_ok is True
    assert Purpose.DETECTOR_PRETRAIN in info.purposes


def test_unknown_source_names_the_alternatives():
    with pytest.raises(SourceError, match="Known sources"):
        get("ufpr")  # the real name is ufpr_alpr


def test_select_filters_are_anded():
    assert {c.info.name for c in select(purpose=Purpose.ASSETS)} == {"divipola", "mt001"}
    assert {c.info.name for c in select(purpose=Purpose.DETECTOR_PRETRAIN)} == {"ccpd"}
    assert select(purpose=Purpose.ASSETS, access=AccessMode.GATED) == []


def test_select_by_name_still_applies_the_filters():
    assert [c.info.name for c in select(names=["ccpd"], commercial_only=True)] == ["ccpd"]


def test_commercial_only_excludes_exactly_the_expected_sources():
    """Sprint 1.2 exit gate: --commercial-only excludes UFPR, RodoSol, Platesmania
    and every unverified-licence project."""
    blocked = {i.name: i.license_key for i in commercial_blockers()}
    assert blocked == EXPECTED_BLOCKERS

    kept = {c.info.name for c in select(commercial_only=True)}
    assert kept == EXPECTED_SOURCES - set(EXPECTED_BLOCKERS)
    assert "ufpr_alpr" not in kept
    assert "rodosol_alpr" not in kept
    assert "platesmania" not in kept


def test_the_commercially_clean_path_is_what_the_plan_says():
    """Plan §8: own synthetic + own captures + CCPD + Roboflow CC-BY."""
    kept = {c.info.name for c in select(commercial_only=True)}
    assert {"ccpd", "roboflow_placas_colombianas", "own_photos"} <= kept


def test_gated_sources_are_flagged_as_gated():
    for name in ("ufpr_alpr", "rodosol_alpr"):
        assert get(name).info.access is AccessMode.GATED
        assert get(name).info.automatic is False


def test_manual_sources_are_excluded_by_skip_manual():
    unattended = {c.info.name for c in select(skip_manual=True)}
    assert "own_photos" not in unattended
    assert "platesmania" not in unattended
    assert "ufpr_alpr" not in unattended
    assert "ccpd" in unattended
    assert "roboflow_placas_colombianas" in unattended  # API key counts as unattended


# --------------------------------------------------------------------------- #
# Registration mechanics, on throwaway sources
# --------------------------------------------------------------------------- #


def _make_source_class(name: str, license_key: str = "unknown", **kw) -> type[Source]:
    class _Fake(Source):
        info = SourceInfo(
            name=name,
            title=f"fake {name}",
            license_key=license_key,
            access=kw.pop("access", AccessMode.MANUAL),
            purposes=kw.pop("purposes", frozenset({Purpose.REAL_TEST})),
            **kw,
        )

        def fetch(self, *, force: bool = False) -> FetchResult:
            return FetchResult(source=name, status=Status.SKIPPED)

    _Fake.__name__ = f"Fake{name.title()}"
    return _Fake


@pytest.fixture
def temp_registry(monkeypatch):
    """Give each test its own registry so registrations don't leak."""
    from alpr.data_sources import registry

    monkeypatch.setattr(registry, "_REGISTRY", {})
    return registry


def test_register_returns_the_class(temp_registry):
    cls = _make_source_class("alpha")
    assert register(cls) is cls
    assert get("alpha") is cls


def test_duplicate_names_are_refused(temp_registry):
    register(_make_source_class("alpha"))
    with pytest.raises(SourceError, match="duplicate source name"):
        register(_make_source_class("alpha"))


def test_registering_the_same_class_twice_is_harmless(temp_registry):
    cls = _make_source_class("alpha")
    register(cls)
    register(cls)
    assert len(all_infos()) == 1


def test_a_class_without_info_is_refused(temp_registry):
    class NoInfo(Source):
        def fetch(self, *, force: bool = False):  # pragma: no cover
            raise NotImplementedError

    with pytest.raises(SourceError, match="must define a class-level"):
        register(NoInfo)


def test_commercial_only_excludes_unverified_licences(temp_registry):
    """The Roboflow-project mitigation: unknown licences are excluded by construction.

    A source whose terms were never read must fail closed, so that forgetting to
    check produces a smaller dataset rather than an unshippable model.
    """
    register(_make_source_class("clean", license_key="mit"))
    register(_make_source_class("unverified", license_key="unknown"))
    register(_make_source_class("research", license_key="academic-nc"))
    register(_make_source_class("scraped", license_key="platesmania-terms"))

    kept = {c.info.name for c in select(commercial_only=True)}
    assert kept == {"clean"}

    blocked = {i.name for i in commercial_blockers()}
    assert blocked == {"unverified", "research", "scraped"}


def test_skip_manual_keeps_only_unattended_sources(temp_registry):
    register(_make_source_class("auto", access=AccessMode.AUTOMATIC))
    register(_make_source_class("keyed", access=AccessMode.API_KEY))
    register(_make_source_class("byhand", access=AccessMode.MANUAL))
    register(_make_source_class("emailed", access=AccessMode.GATED))

    assert {c.info.name for c in select(skip_manual=True)} == {"auto", "keyed"}


def test_source_paths_derive_from_the_data_root(tmp_path, temp_registry):
    from alpr.data_sources.storage import DataPaths

    cls = register(_make_source_class("alpha"))
    source = cls(DataPaths(tmp_path))
    assert source.raw_dir == tmp_path / "raw" / "alpha"
    assert source.index_path == tmp_path / "index" / "alpha.jsonl"


def test_default_index_is_empty_and_writes_an_empty_file(tmp_path, temp_registry):
    from alpr.data_sources.storage import DataPaths

    cls = register(_make_source_class("alpha"))
    source = cls(DataPaths(tmp_path).ensure())
    assert source.write_index() == 0
    assert source.index_path.read_text() == ""
