# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Thin Torch adapter for CatCCOS QuantMatmulAllToAll."""

from __future__ import annotations

import importlib
from typing import Optional

import torch

__all__ = ["destroy_qmma_group", "init_qmma_group", "quant_matmul_alltoall"]


class QmmaConfig:
    """Process-local QMMA domain; the device kernel and tiling stay in CatCCOS."""

    def __init__(self, rank: int, world_size: int, lanes: int, heap_size: int):
        self.rank = rank
        self.world_size = world_size
        self.lanes = lanes
        self.heap_size = heap_size


_config: Optional[QmmaConfig] = None
_anchor: Optional[torch.Tensor] = None


def _require_native():
    from deep_ep._native import require_native

    return require_native("EP_GMM_FUSED")


def _control_device(group, default_device):
    backend = str(torch.distributed.get_backend(group)).lower()
    if backend == "hccl":
        return default_device
    if backend == "gloo":
        return "cpu"
    raise ValueError(f"QMMA group bootstrap supports HCCL or Gloo, got {backend}")


def _require_native_core():
    """Load the component module before looking up its registered operators."""
    core = _require_native()
    importlib.import_module(f"{__package__}._C")
    return core


def init_qmma_group(
    *,
    group=None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    local_rank: Optional[int] = None,
    init_method: Optional[str] = None,
    lanes: int = 1,
    heap_size: int = 1 << 30,
) -> QmmaConfig:
    """Create the CatCCOS QMMA SHMEM domain collectively on its EP ranks."""
    global _anchor, _config
    if _config is not None:
        raise RuntimeError("QMMA group is already initialized")
    if lanes < 1 or lanes > 4:
        raise ValueError("lanes must be in [1, 4]")
    if heap_size <= 0:
        raise ValueError("heap_size must be positive")

    import torch_npu

    if group is not None:
        if any(value is not None for value in (rank, world_size, init_method)):
            raise ValueError("group cannot be combined with rank, world_size or init_method")
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before using group")
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        local_rank = torch_npu.npu.current_device() if local_rank is None else local_rank
        device = f"npu:{local_rank}"
        if init_method is None:
            init_method = ""
    else:
        if rank is None or world_size is None:
            raise ValueError("rank and world_size are required without a process group")
        if local_rank is None:
            local_rank = rank

    if not 0 <= rank < world_size <= 32 or world_size < 2:
        raise ValueError("QMMA requires 0 <= rank < world_size and 2 <= world_size <= 32")
    if local_rank < 0:
        raise ValueError("local_rank must be non-negative")
    if init_method is None:
        raise ValueError("init_method is required without a distributed process group")
    torch_npu.npu.set_device(local_rank)
    anchor = torch.empty(0, dtype=torch.uint8, device=f"npu:{local_rank}")

    core = _require_native_core()
    native = torch.ops.deep_ep_ep_gmm_fused
    if group is not None:
        control_device = _control_device(group, f"npu:{local_rank}")
        uid_size = native._qmma_unique_id_size()
        members = tuple(torch.distributed.get_process_group_ranks(group))
        root = members[0]
        uid = torch.empty(uid_size, dtype=torch.int32, device=control_device)
        status = torch.ones(1, dtype=torch.int32, device=control_device)
        if rank == 0:
            try:
                uid.copy_(torch.tensor(native._qmma_unique_id(), dtype=torch.int32, device=control_device))
            except Exception:
                status.zero_()
        torch.distributed.broadcast(status, src=root, group=group)
        if not bool(status.item()):
            raise RuntimeError("QMMA root failed to create the SHMEM unique ID")
        torch.distributed.broadcast(uid, src=root, group=group)
        unique_id = uid.cpu().tolist()
        anchor = torch.empty(0, dtype=torch.uint8, device=device)
    else:
        unique_id = []
    if len(init_method.encode("utf-8")) >= 256:
        raise ValueError("init_method is too long")
    if core.runtime_busy():
        raise RuntimeError("A SHMEM runtime already exists in this process")
    torch_npu.npu.synchronize()
    try:
        native._initialize_qmma(anchor, rank, world_size, lanes, heap_size, init_method, unique_id)
    except Exception:
        try:
            native._poison_qmma()
        except Exception:
            pass
        raise
    _anchor = anchor
    _config = QmmaConfig(rank, world_size, lanes, heap_size)
    return _config


def destroy_qmma_group() -> None:
    """Collectively release the QMMA SHMEM domain."""
    global _anchor, _config
    if _config is None:
        return
    importlib.import_module(f"{__package__}._C")
    torch.ops.deep_ep_ep_gmm_fused._destroy_qmma(_anchor)
    _anchor = None
    _config = None


def _get_config() -> QmmaConfig:
    if _config is None:
        raise RuntimeError("QMMA is not initialized; call init_qmma_group() first")
    return _config


def _problem_shape(a: torch.Tensor, b: torch.Tensor, format: int) -> tuple[int, int, int]:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be two-dimensional")
    if format == 1:
        if a.shape[1] == 0 or b.shape[1] == 0:
            raise ValueError("packed MXFP4 tensors must have a non-empty packed width")
        m, k, n = int(a.shape[0]), int(a.shape[1]) * 2, int(b.shape[1]) * 2
    else:
        m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    if int(b.shape[0]) != k:
        raise ValueError("b's first dimension must equal k")
    return m, n, k


def _check_scale(
    scale: Optional[torch.Tensor], name: str, expected_shape: tuple[int, ...], device: torch.device
) -> None:
    if scale is None:
        raise ValueError(f"{name} is required for MX inputs")
    if tuple(scale.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(scale.shape)}")
    if scale.device != device:
        raise ValueError(f"{name} and the inputs must be on the same NPU")
    if not scale.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if scale.element_size() != 1:
        raise ValueError(f"{name} must use one-byte E8M0-compatible physical storage")


def quant_matmul_alltoall(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    *,
    format: int,
    comm_interval: int = 0,
    lanes: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply local matmul and exchange its N slices with UDMA AllToAll.

    ``format=0`` is MXFP8 with A ``[m,k]`` and B ``[k,n]``. ``format=1`` is
    packed MXFP4 with A ``[m,k/2]`` and B ``[k,n/2]``. ``format=2`` is BF16.
    The result is BF16 ``[world_size*m, n/world_size]`` on every rank.
    """
    if format not in (0, 1, 2):
        raise ValueError("format must be 0 (MXFP8), 1 (MXFP4), or 2 (BF16)")
    if comm_interval < 0:
        raise ValueError("comm_interval must be non-negative")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous")
    if a.device != b.device:
        raise ValueError("a and b must be on the same NPU")

    config = _get_config()
    if lanes is not None and lanes != config.lanes:
        raise ValueError(f"lanes={lanes} differs from the initialized lanes={config.lanes}")

    m, n, k = _problem_shape(a, b, format)
    ep_size = config.world_size
    if k <= 0 or k % 32:
        raise ValueError("k must be positive and divisible by 32")
    if n <= 0 or n % ep_size:
        raise ValueError("n must be positive and divisible by ep_size")
    q = n // ep_size
    if q % 256:
        raise ValueError("n/ep_size must be divisible by 256")

    if format == 2:
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("BF16 format requires BF16 a and b")
        if a_scale is not None or b_scale is not None:
            raise ValueError("BF16 format does not take MX scales")
    else:
        if a.element_size() != 1 or b.element_size() != 1:
            raise ValueError("MXFP8 and packed MXFP4 inputs must use one-byte physical storage")
        padded_groups = ((k + 31) // 32 + 1) // 2 * 2
        _check_scale(a_scale, "a_scale", (m, padded_groups), a.device)
        _check_scale(b_scale, "b_scale", (padded_groups // 2, n, 2), a.device)

    expected_shape = (ep_size * m, q)
    if out is None:
        return torch.ops.deep_ep_ep_gmm_fused.quant_matmul_alltoall(
            a, b, a_scale, b_scale, format, comm_interval, ep_size
        )
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != torch.bfloat16 or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 on the same NPU as the inputs")
    return torch.ops.deep_ep_ep_gmm_fused.quant_matmul_alltoall_out(
        a, b, a_scale, b_scale, format, comm_interval, ep_size, out
    )
