"""
sparse_model_prep.py — Maple Shield 2:4 Sparse Model Preparation

Takes a YOLOv8 .pt model, applies 2:4 semi-structured pruning to all
eligible weight matrices, validates error stays within the 0.03–0.07%
max error budget, and exports a pruned ONNX model for edge deployment.

Usage:
    python sparse_model_prep.py                          # default yolov8n.pt
    python sparse_model_prep.py --model models/yolo.pt  # custom path
    python sparse_model_prep.py --validate-only          # skip export
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 2:4 pruning helpers
# ---------------------------------------------------------------------------

def prune_2_4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Apply 2:4 semi-structured pruning: keep the 2 largest-magnitude weights
    in every group of 4 consecutive elements along the last dimension.
    Returns a dense tensor with zeros in pruned positions.
    """
    orig_shape = tensor.shape
    t = tensor.reshape(-1, 4)
    # Zero out the 2 smallest-magnitude elements in each group of 4
    _, indices = torch.topk(t.abs(), k=2, dim=1, largest=False)
    mask = torch.ones_like(t, dtype=torch.bool)
    mask.scatter_(1, indices, False)
    pruned = t * mask.float()
    return pruned.reshape(orig_shape)


def is_eligible(name: str, tensor: torch.Tensor) -> bool:
    """
    Determine if a weight is eligible for 2:4 pruning.
    Eligible: 2-D weight matrices (Linear / Conv1x1) with last dim % 4 == 0.
    Skip: biases, batch norm params, small embeddings.
    """
    if tensor.dim() < 2:
        return False
    if "bn" in name or "bias" in name or "norm" in name:
        return False
    if tensor.shape[-1] % 4 != 0:
        return False
    # Minimum size — tiny layers don't benefit
    if tensor.numel() < 1024:
        return False
    return True


def measure_error(original: torch.Tensor, pruned: torch.Tensor) -> float:
    """Relative L2 error between original and pruned weight (%)."""
    diff = (original - pruned).norm().item()
    base = original.norm().item()
    return 100.0 * diff / (base + 1e-12)


# ---------------------------------------------------------------------------
# Main prep pipeline
# ---------------------------------------------------------------------------

class SparseModelPrep:
    """
    Applies 2:4 sparse pruning to a YOLOv8 model and validates correctness.
    """

    # Max allowed relative L2 error per layer (%)
    MAX_ERROR_PCT = 0.07
    # Warn if error exceeds this (drone vs bird misidentification risk)
    WARN_ERROR_PCT = 0.03

    def __init__(self, model_path: str, output_dir: str = "models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report: dict = {}

    def load(self) -> nn.Module:
        print(f"Loading model: {self.model_path}")
        # Ultralytics YOLO
        try:
            from ultralytics import YOLO
            yolo = YOLO(str(self.model_path))
            model = yolo.model
            model.eval()
            print(f"  Loaded via ultralytics — {sum(p.numel() for p in model.parameters()):,} params")
            return model
        except ImportError:
            pass
        # Fallback: raw torch load
        ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
        model = ckpt.get("model") or ckpt
        if hasattr(model, "float"):
            model = model.float()
        model.eval()
        print(f"  Loaded via torch — {sum(p.numel() for p in model.parameters()):,} params")
        return model

    def prune(self, model: nn.Module) -> dict:
        """
        Apply 2:4 pruning in-place. Returns a per-layer report.
        Raises ValueError if any layer exceeds MAX_ERROR_PCT.
        """
        layer_report = {}
        pruned_count = 0
        skipped_count = 0

        for name, param in model.named_parameters():
            if not is_eligible(name, param.data):
                skipped_count += 1
                continue

            original = param.data.clone()
            pruned = prune_2_4(original)
            error_pct = measure_error(original, pruned)

            if error_pct > self.MAX_ERROR_PCT:
                raise ValueError(
                    f"Layer '{name}' error {error_pct:.4f}% exceeds budget "
                    f"{self.MAX_ERROR_PCT}% — aborting to prevent drone/bird misidentification."
                )

            param.data.copy_(pruned)
            sparsity = (pruned == 0).float().mean().item()
            layer_report[name] = {
                "shape": list(param.shape),
                "params": param.numel(),
                "error_pct": round(error_pct, 5),
                "sparsity": round(sparsity, 4),
                "warn": error_pct > self.WARN_ERROR_PCT,
            }
            pruned_count += 1

        print(f"  Pruned {pruned_count} layers, skipped {skipped_count}")
        warnings = [n for n, v in layer_report.items() if v["warn"]]
        if warnings:
            print(f"  [WARN] {len(warnings)} layers above {self.WARN_ERROR_PCT}% error threshold:")
            for w in warnings[:5]:
                print(f"    {w}: {layer_report[w]['error_pct']:.4f}%")

        return layer_report

    def export_onnx(self, model: nn.Module, imgsz: int = 640) -> Path:
        """Export pruned model to ONNX."""
        stem = self.model_path.stem
        out_path = self.output_dir / f"{stem}_sparse24.onnx"
        dummy = torch.zeros(1, 3, imgsz, imgsz)
        print(f"  Exporting ONNX → {out_path}")
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
            opset_version=17,
        )
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  ONNX saved ({size_mb:.1f} MB)")
        return out_path

    def run(self, validate_only: bool = False) -> dict:
        t0 = time.time()
        model = self.load()

        print("Applying 2:4 sparse pruning...")
        layer_report = self.prune(model)

        total_params = sum(v["params"] for v in layer_report.values())
        avg_error = np.mean([v["error_pct"] for v in layer_report.values()])
        avg_sparsity = np.mean([v["sparsity"] for v in layer_report.values()])

        print(f"\n  Pruned layers : {len(layer_report)}")
        print(f"  Total params  : {total_params:,}")
        print(f"  Avg sparsity  : {avg_sparsity:.1%}")
        print(f"  Avg L2 error  : {avg_error:.5f}%  (budget: {self.MAX_ERROR_PCT}%)")

        onnx_path = None
        if not validate_only:
            onnx_path = self.export_onnx(model)

        elapsed = time.time() - t0
        self.report = {
            "model": str(self.model_path),
            "onnx_output": str(onnx_path) if onnx_path else None,
            "pruned_layers": len(layer_report),
            "total_pruned_params": total_params,
            "avg_sparsity_pct": round(avg_sparsity * 100, 2),
            "avg_error_pct": round(avg_error, 5),
            "max_error_pct": round(max(v["error_pct"] for v in layer_report.values()), 5),
            "within_budget": True,
            "elapsed_s": round(elapsed, 2),
            "layers": layer_report,
        }

        report_path = self.output_dir / "sparse_prep_report.json"
        report_path.write_text(json.dumps(self.report, indent=2))
        print(f"\nReport saved → {report_path}")
        print(f"Done in {elapsed:.1f}s")
        return self.report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maple Shield 2:4 Sparse Model Prep")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to .pt model")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="ONNX export image size")
    parser.add_argument("--validate-only", action="store_true",
                        help="Run pruning validation without ONNX export")
    args = parser.parse_args()

    prep = SparseModelPrep(args.model, args.output_dir)
    report = prep.run(validate_only=args.validate_only)

    status = "PASS" if report["within_budget"] else "FAIL"
    print(f"\n=== Sparse Prep: {status} ===")
    print(f"  Layers pruned : {report['pruned_layers']}")
    print(f"  Avg sparsity  : {report['avg_sparsity_pct']}%")
    print(f"  Max L2 error  : {report['max_error_pct']}%  (budget: 0.07%)")
    if report["onnx_output"]:
        print(f"  ONNX model    : {report['onnx_output']}")
