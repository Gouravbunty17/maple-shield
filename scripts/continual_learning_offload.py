#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key


MIN_NEW_EXAMPLES = 100
MAX_MAP_DROP = 0.05


def read_examples(metadata_jsonl: Path) -> List[Dict[str, Any]]:
    if not metadata_jsonl.exists():
        return []
    rows = []
    for line in metadata_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def drift_check(baseline_map: float, candidate_map: float) -> Dict[str, Any]:
    drop = baseline_map - candidate_map
    accepted = drop <= MAX_MAP_DROP
    return {"baseline_map": baseline_map, "candidate_map": candidate_map, "absolute_drop": drop, "accepted": accepted, "max_allowed_drop": MAX_MAP_DROP}


def load_private_key(path: Path) -> Ed25519PrivateKey:
    key = load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("deployment key must be Ed25519")
    return key


def package_update(engine_path: Path, signing_key_path: Path, version: str, drift_report: Dict[str, Any], output_path: Path) -> Path:
    engine_bytes = engine_path.read_bytes()
    key = load_private_key(signing_key_path)
    signed_payload = json.dumps({"version": version, "drift_report": drift_report}, sort_keys=True, separators=(",", ":")).encode("utf-8") + engine_bytes
    signature = base64.b64encode(key.sign(signed_payload)).decode("ascii")
    package = {
        "version": version,
        "engine_b64": base64.b64encode(engine_bytes).decode("ascii"),
        "signature": signature,
        "drift_report": drift_report,
        "created_at": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Off-device continual learning update packager.")
    parser.add_argument("--metadata-jsonl", required=True)
    parser.add_argument("--candidate-engine", required=True, help="Fine-tuned TensorRT engine to package")
    parser.add_argument("--signing-key", required=True, help="Deployment Ed25519 private key PEM")
    parser.add_argument("--version", required=True)
    parser.add_argument("--baseline-map", type=float, required=True)
    parser.add_argument("--candidate-map", type=float, required=True)
    parser.add_argument("--output", default="artifacts/model_update_package.json")
    args = parser.parse_args()

    examples = read_examples(Path(args.metadata_jsonl))
    new_ground_truth = [ex for ex in examples if ex.get("ground_truth")]
    if len(new_ground_truth) <= MIN_NEW_EXAMPLES:
        raise SystemExit(f"not enough new ground-truth examples: {len(new_ground_truth)} <= {MIN_NEW_EXAMPLES}")

    report = drift_check(args.baseline_map, args.candidate_map)
    report["new_ground_truth_examples"] = len(new_ground_truth)
    if not report["accepted"]:
        raise SystemExit(f"rejected model update due to drift: {report}")

    package = package_update(Path(args.candidate_engine), Path(args.signing_key), args.version, report, Path(args.output))
    print(f"wrote signed update package: {package}")
    print("Fine-tuning note: run YOLOv8n training off-device before this packaging step, using exported hard examples and a held-out clean validation set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
