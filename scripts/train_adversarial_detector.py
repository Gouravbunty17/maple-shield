#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.adversarial_export_int8 import AdversarialPatchCNN, build_int8_engine_command, export_onnx


class RoiDataset(Dataset):
    def __init__(self, manifest: str | Path) -> None:
        self.rows = []
        with Path(manifest).open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        image = cv2.imread(row["path"])
        if image is None:
            raise FileNotFoundError(row["path"])
        image = cv2.cvtColor(cv2.resize(image, (64, 64)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return tensor, label


def train(manifest: Path, output_dir: Path, epochs: int, batch_size: int, lr: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = RoiDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdversarialPatchCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"epoch={epoch + 1} loss={running / max(1, len(loader)):.4f}")
    weights = output_dir / "adversarial_patch_cnn.pt"
    torch.save(model.state_dict(), weights)
    export_onnx(model.cpu(), output_dir / "adversarial_patch_cnn.onnx")
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 64x64 ROI binary review model and prepare INT8 export command.")
    parser.add_argument("--manifest", required=True, help="CSV with columns path,label where label 0=real, 1=adversarial")
    parser.add_argument("--output-dir", default="artifacts/adversarial_detector")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--build-engine", action="store_true", help="Run trtexec locally after ONNX export")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train(Path(args.manifest), output_dir, args.epochs, args.batch_size, args.lr)
    cmd = build_int8_engine_command(output_dir / "adversarial_patch_cnn.onnx", output_dir / "adversarial_patch_cnn.engine")
    print("TensorRT INT8 build command:")
    print(" ".join(cmd))
    if args.build_engine:
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
