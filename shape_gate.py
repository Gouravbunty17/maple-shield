"""
shape_gate.py — Maple Shield SparseFlow Shape Gate

Decides at runtime whether to execute a sparse 2:4 kernel or fall back to
dense, based on the effective M dimension (M_eff) of the matrix multiply.

Rules derived from SparseFlow Phase-3 benchmarks (RTX 4090, SM89):
  - M_eff >= 512  → sparse wins (ffn_down hero shape, ~1.684x peak)
  - M_eff 256–511 → sparse viable (fused gate/up blocks, ~1.36–1.44x)
  - M_eff < 256   → dense wins (overhead exceeds gain, skip sparse)

On CPU (no CUDA) the gate always returns DENSE — sparse GEMM kernels
require cuSPARSELt and are no-ops without a compatible GPU.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto

import torch


class KernelMode(Enum):
    DENSE  = auto()
    SPARSE = auto()


@dataclass(frozen=True)
class GateConfig:
    # Minimum M for sparse to be beneficial (ffn_down / large projections)
    sparse_threshold_high: int = 512
    # Minimum M where sparse is sometimes viable (fused gate_up blocks)
    sparse_threshold_low: int = 256
    # Force dense regardless of M (useful for correctness testing)
    force_dense: bool = False
    # Force sparse regardless of M (useful for benchmarking)
    force_sparse: bool = False


_DEFAULT_CONFIG = GateConfig()

# Detect once at import time — avoids repeated cuda checks in hot paths
_CUDA_AVAILABLE: bool = torch.cuda.is_available()
_SM_VERSION: int = 0
if _CUDA_AVAILABLE:
    props = torch.cuda.get_device_properties(0)
    _SM_VERSION = props.major * 10 + props.minor  # e.g. SM89 → 89


def cuda_capable() -> bool:
    """True if the current device can run cuSPARSELt 2:4 sparse kernels."""
    return _CUDA_AVAILABLE and _SM_VERSION >= 80  # Ampere+ required


def gate(M: int, config: GateConfig = _DEFAULT_CONFIG) -> KernelMode:
    """
    Return the kernel mode for a matmul with effective batch dimension M.

    Args:
        M:      Effective M dimension (batch * seq_len, or feature rows).
        config: Gate thresholds and override flags.

    Returns:
        KernelMode.SPARSE or KernelMode.DENSE
    """
    if config.force_dense:
        return KernelMode.DENSE
    if not cuda_capable():
        return KernelMode.DENSE
    if config.force_sparse and M >= config.sparse_threshold_low:
        return KernelMode.SPARSE
    if M >= config.sparse_threshold_high:
        return KernelMode.SPARSE
    if M >= config.sparse_threshold_low:
        # Mid-range: sparse viable but marginal; default to sparse
        return KernelMode.SPARSE
    return KernelMode.DENSE


def gate_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    config: GateConfig = _DEFAULT_CONFIG,
) -> tuple[torch.Tensor, KernelMode]:
    """
    Shape-gated matmul: runs sparse or dense based on A's M dimension.

    Args:
        A: Input tensor  [..., M, K]
        B: Weight tensor [K, N]  (dense; caller should pre-sparsify if needed)

    Returns:
        (result tensor, KernelMode used)
    """
    M_eff = A.shape[-2] if A.dim() >= 2 else 1
    mode = gate(M_eff, config)

    if mode == KernelMode.SPARSE and hasattr(torch, "sparse_semi_structured"):
        try:
            B_sparse = torch.sparse_semi_structured.from_dense(B.T)
            result = torch.mm(A.reshape(-1, A.shape[-1]), B_sparse.T)
            result = result.reshape(*A.shape[:-1], B.shape[1])
            return result, KernelMode.SPARSE
        except Exception:
            pass  # Fall through to dense

    result = torch.matmul(A, B)
    return result, KernelMode.DENSE


def report() -> dict:
    """Return a status dict useful for logging and dashboard display."""
    return {
        "cuda_available": _CUDA_AVAILABLE,
        "sm_version": _SM_VERSION,
        "sparse_capable": cuda_capable(),
        "sparse_threshold_high": _DEFAULT_CONFIG.sparse_threshold_high,
        "sparse_threshold_low": _DEFAULT_CONFIG.sparse_threshold_low,
        "expected_speedup_ffn_down": "~1.684x (M>=512, SM89)",
        "expected_speedup_fused": "~1.36–1.44x (M 256–511)",
    }


if __name__ == "__main__":
    import json
    print("=== Shape Gate Report ===")
    print(json.dumps(report(), indent=2))

    test_cases = [64, 128, 256, 384, 512, 768, 1024, 2048]
    print("\nM_eff -> KernelMode:")
    for m in test_cases:
        mode = gate(m)
        marker = " (hero shape)" if m >= 512 else (" (fused gate/up)" if m >= 256 else " (dense - overhead)")
        print(f"  M={m:5d}  ->  {mode.name}{marker}")
