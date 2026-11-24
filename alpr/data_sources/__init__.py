"""Phase 1 — data acquisition.

Fetch every dataset, normalize the annotations into one record type, and track
licence lineage. No ML dependencies: this package is fully testable without
PyTorch, OpenCV or numpy installed.

Importing this package registers every source, so `registry.select(...)` sees them
all. Add a new source by writing one module and decorating its class with
`@register` — then import it below.
"""

from alpr.data_sources import (  # noqa: F401 — imported for registration
    assets,
    ccpd,
    dashcam,
    gated,
    own_photos,
    platesmania,
    roboflow,
)
from alpr.data_sources.base import (
    AcquisitionError,
    Box,
    CornerOrderError,
    License,
    PlateType,
    Record,
    RecordValidationError,
    Status,
    Winding,
    assert_corner_order,
    corners_to_box,
    order_corners,
    read_jsonl,
    winding,
    write_jsonl,
)
from alpr.data_sources.manifest import (
    Manifest,
    SourceEntry,
    build_manifest,
    verify_manifest,
)
from alpr.data_sources.registry import (
    AccessMode,
    FetchResult,
    Purpose,
    Source,
    SourceInfo,
    all_infos,
    commercial_blockers,
    get,
    register,
    select,
)
from alpr.data_sources.storage import DataPaths

__all__ = [
    "AccessMode",
    "AcquisitionError",
    "Box",
    "CornerOrderError",
    "DataPaths",
    "FetchResult",
    "License",
    "Manifest",
    "PlateType",
    "Purpose",
    "Record",
    "RecordValidationError",
    "Source",
    "SourceEntry",
    "SourceInfo",
    "Status",
    "Winding",
    "all_infos",
    "assert_corner_order",
    "build_manifest",
    "commercial_blockers",
    "corners_to_box",
    "get",
    "order_corners",
    "read_jsonl",
    "register",
    "select",
    "verify_manifest",
    "winding",
    "write_jsonl",
]
