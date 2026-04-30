#!/usr/bin/env python3
"""Quick Jetson readiness check for Cairn-Edge.

This script is safe to run on non-Jetson systems; missing Jetson files are
reported as warnings so developers can still use laptops for early work.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict


def read_text(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5).strip()
    except Exception:
        return None


def main() -> None:
    report: Dict[str, Any] = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "is_jetson_hint": Path("/etc/nv_tegra_release").exists(),
        "nv_tegra_release": read_text("/etc/nv_tegra_release"),
        "jetson_clocks_available": bool(run(["which", "jetson_clocks"])),
        "nvpmodel_available": bool(run(["which", "nvpmodel"])),
        "tegrastats_available": bool(run(["which", "tegrastats"])),
        "nvpmodel_query": run(["nvpmodel", "-q"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "notes": [],
    }

    if not report["is_jetson_hint"]:
        report["notes"].append("Not running on Jetson or /etc/nv_tegra_release missing.")
    if not report["jetson_clocks_available"]:
        report["notes"].append("jetson_clocks not found; production benchmarking should lock clocks.")
    if not report["nvpmodel_available"]:
        report["notes"].append("nvpmodel not found; cannot verify 15W/10W/7W power profile.")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
