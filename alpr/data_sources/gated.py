"""UFPR-ALPR and RodoSol-ALPR — Brazilian datasets behind a signed agreement.

These are **first-class sources that happen to be manual**. They print the exact
procedure and exit cleanly rather than raising, because a source that raises is a
source you stop running, and these have a multi-day lead time: the request has to go
out early even though the plan expects to make little or no use of them.

Both are **academic non-commercial**. Any checkpoint whose weights were influenced
by them cannot be commercially deployed (plan §22 item 10), which is why they carry
the `academic-nc` licence key and are excluded from `--commercial-only`.

Risk R4 in the plan: the licence agreement asks for an institutional address, and a
request from a personal account is likely to be declined. That is an accepted
outcome — the plan does not depend on these.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from alpr.config import get_secret
from alpr.data_sources.base import PlateType, Record, Status
from alpr.data_sources.imagesize import safe_image_size
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceInfo,
    register,
)
from alpr.data_sources.storage import IMAGE_SUFFIXES, iter_files

CONTACT = "rblsantos@inf.ufpr.br"

#: The sender's details come from the environment or `.env`, never from code, so
#: the committed templates stay free of personal data.
SENDER_NAME_ENV = "CONTACT_NAME"
SENDER_EMAIL_ENV = "CONTACT_EMAIL"
SENDER_NAME_PLACEHOLDER = "<your name>"
SENDER_EMAIL_PLACEHOLDER = "<your email>"

EMAIL_TEMPLATE = """\
To: {contact}
Subject: {subject}

Dear Prof. Laroca and colleagues,

I am writing to request access to the {dataset} dataset for research use.

I am an independent researcher working on a license plate recognition system for
Colombian vehicle plates. The recognizer is being built from scratch and trained
primarily on synthetically generated plates; {dataset} would be used strictly as
an evaluation reference to compare cross-dataset generalisation, and would not be
used to train any model intended for commercial deployment.

I acknowledge and accept the terms of the licence agreement, in particular that
the dataset is provided for academic and non-commercial research purposes only,
that it will not be redistributed, and that any publication making use of it will
cite the corresponding paper.

Please let me know if you require the signed agreement in any particular form, or
any further detail about the intended use.

Thank you for making this dataset available.

Kind regards,
{name}
{email}
"""


class GatedSource(Source):
    """A dataset that arrives only after a human exchange of email."""

    dataset_label: str
    agreement_url: str
    paper: str
    citation_note: str = ""

    def fetch(self, *, force: bool = False) -> FetchResult:
        """Never raises. Reports what is on disk, or what the human must do."""
        present = sum(1 for _ in iter_files(self.raw_dir, IMAGE_SUFFIXES))
        if present:
            return FetchResult(
                source=self.name,
                status=Status.READY,
                message=f"{present} images already present at {self.raw_dir}",
                paths=[str(self.raw_dir)],
            )
        return FetchResult(
            source=self.name,
            status=Status.MANUAL_REQUIRED,
            message=self.manual_instructions(),
        )

    def manual_instructions(self) -> str:
        return f"""\
{self.dataset_label} requires a signed licence agreement, emailed to the authors.
Expect a 1-5 day turnaround, so send the request early.

  1. Read the agreement:  {self.agreement_url}
  2. Email {CONTACT} from your most institutional-looking address.
     Set {SENDER_NAME_ENV} and {SENDER_EMAIL_ENV} in .env, then run
       acquire email {self.name} --write
     for a ready-to-send draft, saved to {self.draft_path}.
  3. On approval you receive a download link. Unpack it into:
       {self.raw_dir}
  4. Re-run `acquire fetch {self.name}` — it will detect the files and index them.

LICENCE: academic, non-commercial. Any model whose weights were influenced by this
dataset CANNOT be commercially deployed. `acquire audit` will report NO while it is
present, and `--commercial-only` excludes it.

The plan marks this "evaluation only, if at all" (risk R4): a request from a
non-institutional address is likely to be declined, and nothing depends on it.

Paper: {self.paper}"""

    def email_draft(self, name: str | None = None, email: str | None = None) -> str:
        """The request email, signed from `CONTACT_NAME` / `CONTACT_EMAIL`.

        Unset values render as visible placeholders rather than failing, so the
        draft can always be previewed.
        """
        return EMAIL_TEMPLATE.format(
            contact=CONTACT,
            subject=f"{self.dataset_label} dataset access request",
            dataset=self.dataset_label,
            name=name or get_secret(SENDER_NAME_ENV, SENDER_NAME_PLACEHOLDER),
            email=email or get_secret(SENDER_EMAIL_ENV, SENDER_EMAIL_PLACEHOLDER),
        )

    def index(self) -> Iterator[Record]:
        """Index whatever the human unpacked. Annotations are not parsed here.

        Both datasets ship per-image text annotations in their own formats. Parsing
        them is deliberately deferred: until a licence is actually granted, writing
        a parser for a file layout nobody has seen is speculative work. The images
        are indexed so the manifest and the lineage audit can see them.
        """
        for image_path in iter_files(self.raw_dir, IMAGE_SUFFIXES):
            size = safe_image_size(image_path)
            yield Record(
                source=self.name,
                image_path=self.paths.relative(image_path),
                license_key=self.info.license_key,
                plate=None,
                plate_type=PlateType.FOREIGN,  # Brazilian layouts, never read as Colombian
                image_size=size,
                tags=(self.name, "non_commercial"),
            )

    @property
    def draft_path(self) -> Path:
        """Filled-in drafts hold personal data, so they go under the gitignored
        data root."""
        return self.paths.root / "licence_requests" / f"{self.name}.txt"


@register
class UfprAlprSource(GatedSource):
    """UFPR-ALPR — 4,500 Brazilian images, academic non-commercial."""

    dataset_label = "UFPR-ALPR"
    agreement_url = "https://web.inf.ufpr.br/vri/databases/ufpr-alpr/license-agreement/"
    paper = "Laroca et al., 'A Robust Real-Time Automatic License Plate Recognition', IJCNN 2018"

    info = SourceInfo(
        name="ufpr_alpr",
        title="UFPR-ALPR (Brazil)",
        license_key="academic-nc",
        access=AccessMode.GATED,
        purposes=frozenset({Purpose.RECOGNIZER_EVAL}),
        url="https://github.com/raysonlaroca/ufpr-alpr-dataset",
        size_hint="4,500 images",
        notes="Signed agreement by email, 1-5 days. Evaluation only, if at all.",
    )


@register
class RodosolAlprSource(GatedSource):
    """RodoSol-ALPR — 20,000 Brazilian images, academic non-commercial."""

    dataset_label = "RodoSol-ALPR"
    agreement_url = "https://github.com/raysonlaroca/rodosol-alpr-dataset"
    paper = (
        "Laroca et al., 'On the Cross-dataset Generalization in "
        "License Plate Recognition', VISAPP 2022"
    )

    info = SourceInfo(
        name="rodosol_alpr",
        title="RodoSol-ALPR (Brazil)",
        license_key="academic-nc",
        access=AccessMode.GATED,
        purposes=frozenset({Purpose.RECOGNIZER_EVAL}),
        url="https://github.com/raysonlaroca/rodosol-alpr-dataset",
        size_hint="20,000 images",
        notes="Same process and same contact as UFPR-ALPR. Evaluation only, if at all.",
    )
