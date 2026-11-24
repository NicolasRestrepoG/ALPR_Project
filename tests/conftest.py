"""Test-wide isolation.

The suite must not read the developer's real `.env`. Without this, whether a test
passes depends on whether someone has pasted an API key into a gitignored file —
the test that caught it asserted a source reports "no key configured", and it
started failing the moment a real key existed locally.

Every test therefore runs against a `.env` path that does not exist, unless it
deliberately opts in by setting `ALPR_ENV_FILE` itself.
"""

from __future__ import annotations

import pytest

from alpr.config import ENV_FILE_VAR

#: Credentials that must never leak from the developer's machine into a test.
MANAGED_VARS = (
    "ROBOFLOW_API_KEY",
    "ALPR_DATA_ROOT",
    "COMMERCIAL_OCR_API_KEY",
    "CONTACT_NAME",
    "CONTACT_EMAIL",
)


@pytest.fixture(autouse=True)
def _hermetic_credentials(tmp_path_factory, monkeypatch):
    absent = tmp_path_factory.mktemp("no-env") / ".env"
    monkeypatch.setenv(ENV_FILE_VAR, str(absent))
    for name in MANAGED_VARS:
        monkeypatch.delenv(name, raising=False)
