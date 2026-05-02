#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


class SignedUpdateLoader:
    """Verify and atomically install signed model update packages."""

    def __init__(self, package_dir: str | Path, public_key_path: str | Path, engine_path: str | Path, version_path: str | Path) -> None:
        self.package_dir = Path(package_dir)
        self.public_key_path = Path(public_key_path)
        self.engine_path = Path(engine_path)
        self.version_path = Path(version_path)

    def current_version(self) -> str:
        if not self.version_path.exists():
            return "0.0.0"
        return self.version_path.read_text(encoding="utf-8").strip()

    @staticmethod
    def version_gt(candidate: str, current: str) -> bool:
        def parse(v: str) -> tuple[int, ...]:
            parts = []
            for token in v.replace("-", ".").split("."):
                if token.isdigit():
                    parts.append(int(token))
                else:
                    parts.append(0)
            return tuple(parts)
        return parse(candidate) > parse(current)

    def load_public_key(self) -> Ed25519PublicKey:
        key = serialization.load_pem_public_key(self.public_key_path.read_bytes())
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("deployment public key must be Ed25519")
        return key

    def verify_package(self, package: Dict[str, Any]) -> bytes:
        required = {"version", "engine_b64", "signature", "drift_report"}
        missing = required - set(package)
        if missing:
            raise ValueError(f"package missing fields: {sorted(missing)}")
        engine = base64.b64decode(package["engine_b64"])
        payload = json.dumps({"version": package["version"], "drift_report": package["drift_report"]}, sort_keys=True, separators=(",", ":")).encode("utf-8") + engine
        signature = base64.b64decode(package["signature"])
        self.load_public_key().verify(signature, payload)
        return engine

    def install_package(self, package_path: str | Path) -> bool:
        path = Path(package_path)
        package = json.loads(path.read_text(encoding="utf-8"))
        version = str(package["version"])
        if not self.version_gt(version, self.current_version()):
            return False
        engine = self.verify_package(package)
        self.engine_path.parent.mkdir(parents=True, exist_ok=True)
        self.version_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="engine_", suffix=".tmp", dir=str(self.engine_path.parent))
        with os.fdopen(fd, "wb") as fh:
            fh.write(engine)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, self.engine_path)
        self.version_path.write_text(version, encoding="utf-8")
        return True

    def check_for_updates(self) -> Optional[Path]:
        if not self.package_dir.exists():
            return None
        packages = sorted(self.package_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for package in packages:
            try:
                if self.install_package(package):
                    return package
            except Exception as exc:
                print(f"rejected update package {package}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify and install signed Cairn-Edge model update packages.")
    parser.add_argument("--package-dir", default="updates/incoming")
    parser.add_argument("--public-key", required=True)
    parser.add_argument("--engine-path", default="models/yolov8n_current.engine")
    parser.add_argument("--version-path", default="models/yolov8n_current.version")
    args = parser.parse_args()
    loader = SignedUpdateLoader(args.package_dir, args.public_key, args.engine_path, args.version_path)
    installed = loader.check_for_updates()
    if installed:
        print(f"installed model update: {installed}")
        return 0
    print("no valid newer update found")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
