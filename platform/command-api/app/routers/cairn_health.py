"""CAIRN engine health endpoint for the operator UI."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)


_ensure_repo_root_on_path()

from cairn_engine import CairnEngine  # noqa: E402
from cairn_engine import __version__ as CAIRN_PACKAGE_VERSION  # noqa: E402


EXPECTED_CAIRN_VERSION = "2.0.0-dev"

router = APIRouter(prefix="/cairn", tags=["cairn"])
_engine = CairnEngine()


def _major_minor(version: str) -> tuple[int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)", version.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _compatible(actual: str) -> bool:
    return _major_minor(actual) == _major_minor(EXPECTED_CAIRN_VERSION)


def health_payload(engine: CairnEngine = _engine) -> Dict[str, Any]:
    meta = engine.metadata()
    started_ts = float(meta.get("started_ts", time.time()))
    runtime_s = max(0.0, time.time() - started_ts)
    engine_version = str(meta.get("engine_version", CAIRN_PACKAGE_VERSION))
    return {
        "status": "ok",
        "engine": meta.get("engine", "CAIRN"),
        "engine_version": engine_version,
        "package_version": CAIRN_PACKAGE_VERSION,
        "expected_adapter_version": EXPECTED_CAIRN_VERSION,
        "compatible": _compatible(engine_version),
        "frames_processed": int(getattr(engine, "frames_processed", 0)),
        "runtime_s": round(runtime_s, 3),
        "started_ts": started_ts,
        "risk_config": meta.get("risk_config", {}),
    }


@router.get("/health")
def cairn_health():
    return health_payload()
