from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn


class AdversarialPatchCNN(nn.Module):
    """Small binary CNN for 64x64 RGB ROI review."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def export_onnx(model: nn.Module, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["roi"],
        output_names=["logits"],
        opset_version=13,
        dynamic_axes=None,
    )
    return output


def build_int8_engine_command(onnx_path: str | Path, engine_path: str | Path, calibration_cache: Optional[str | Path] = None) -> list[str]:
    cmd = [
        "trtexec",
        f"--onnx={Path(onnx_path)}",
        f"--saveEngine={Path(engine_path)}",
        "--int8",
        "--workspace=512",
        "--shapes=roi:1x3x64x64",
    ]
    if calibration_cache:
        cmd.append(f"--calib={Path(calibration_cache)}")
    return cmd
