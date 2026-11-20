# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Modified by zhu-mingzhe71 2026
"""Stable Python facade for CatCCOS UDMA operators."""

import struct
from numbers import Integral
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch

from .runtime import _get_qmma_config, get_udma_config
from ._tiling import TILING_FIELDS, validate_tiling


RouteInput = Union[torch.Tensor, Sequence[int]]


def _tiling_arguments(operator, tiling, x, weight, ep_size, output_all_to_allv=False):
    if tiling is None:
        return {}
    profile = validate_tiling(
        operator,
        tiling,
        output_all_to_allv=output_all_to_allv,
        m=x.shape[0],
        n=weight.shape[2],
        k=x.shape[1],
        world_size=ep_size,
    )
    return {"tiling": [profile[name] for name in TILING_FIELDS[operator]]}


_MOE_ROUTE_MAX_RANKS = 8
_MOE_ROUTE_MAX_LOCAL_EXPERTS = 64
_MOE_ROUTE_MAX_SEGMENTS = _MOE_ROUTE_MAX_RANKS * _MOE_ROUTE_MAX_LOCAL_EXPERTS
_MOE_ROUTE_HEADER_SIZE = 4 * 4
_MOE_ROUTE_PEER_ROWS_OFFSET = _MOE_ROUTE_HEADER_SIZE
_MOE_ROUTE_EXPERT_ROWS_OFFSET = _MOE_ROUTE_PEER_ROWS_OFFSET + _MOE_ROUTE_MAX_RANKS * 8
_MOE_ROUTE_SEGMENTS_OFFSET = _MOE_ROUTE_EXPERT_ROWS_OFFSET + _MOE_ROUTE_MAX_LOCAL_EXPERTS * 8
_MOE_ROUTE_SEGMENT_SIZE = 24
_MOE_ROUTE_METADATA_SIZE = _MOE_ROUTE_SEGMENTS_OFFSET + _MOE_ROUTE_MAX_SEGMENTS * _MOE_ROUTE_SEGMENT_SIZE


def _host_route_values(route: RouteInput, name: str):
    if torch.is_tensor(route):
        if route.device.type != "cpu":
            return None
        if route.dtype != torch.int64:
            raise TypeError("{} CPU tensor must have dtype torch.int64".format(name))
        values = route.reshape(-1).tolist()
    else:
        if isinstance(route, (str, bytes)):
            raise TypeError("{} must be a Tensor or a sequence of integers".format(name))
        try:
            values = list(route)
        except TypeError as error:
            raise TypeError("{} must be a Tensor or a sequence of integers".format(name)) from error

    normalized = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("{}[{}] must be an integer".format(name, index))
        value = int(value)
        if value < 0:
            raise ValueError("{}[{}] must be non-negative".format(name, index))
        normalized.append(value)
    return normalized


def _normalize_route(
    route: RouteInput,
    name: str,
    x: torch.Tensor,
    expected_elements: int,
):
    host_values = _host_route_values(route, name)
    if host_values is not None:
        if len(host_values) != expected_elements:
            raise ValueError("{} must contain {} elements".format(name, expected_elements))
        return (
            torch.tensor(host_values, dtype=torch.int64, device=x.device),
            host_values,
        )

    if route.dtype != torch.int64:
        raise TypeError("{} NPU tensor must have dtype torch.int64".format(name))
    if route.device != x.device:
        raise ValueError("{} and x must be on the same NPU".format(name))
    route = route.reshape(-1).contiguous()
    if route.numel() != expected_elements:
        raise ValueError("{} must contain {} elements".format(name, expected_elements))
    return route, None


def _normalize_received_rows(
    received_rows: Optional[int],
    global_route: torch.Tensor,
    global_host_values,
    max_rows: int,
) -> int:
    host_sum = sum(global_host_values) if global_host_values is not None else None
    if received_rows is None:
        received_rows = host_sum if host_sum is not None else int(global_route.sum().item())
    elif isinstance(received_rows, bool) or not isinstance(received_rows, Integral):
        raise TypeError("received_rows must be an integer")
    else:
        received_rows = int(received_rows)

    if received_rows < 0 or received_rows > max_rows:
        raise ValueError("received_rows must be in [0, ep_size * x.shape[0]]")
    if host_sum is not None and received_rows != host_sum:
        raise ValueError("received_rows does not match sum(global_tokens_per_local_expert)")
    return received_rows


def _build_route_metadata(
    global_host_values,
    ep_size: int,
    expert_num: int,
    device,
) -> torch.Tensor:
    """Pack CatCCOS route prefixes on the host and copy them to the NPU."""
    local_expert_num = expert_num // ep_size
    if ep_size > _MOE_ROUTE_MAX_RANKS:
        raise ValueError("route metadata supports at most 8 EP ranks")
    if local_expert_num > _MOE_ROUTE_MAX_LOCAL_EXPERTS:
        raise ValueError("route metadata supports at most 64 local experts")

    segment_count = ep_size * local_expert_num
    metadata = bytearray(_MOE_ROUTE_METADATA_SIZE)
    struct.pack_into(
        "<IIII",
        metadata,
        0,
        ep_size,
        local_expert_num,
        segment_count,
        0,
    )
    source_offsets = [0] * ep_size
    peer_rows = [0] * ep_size
    expert_rows = [0] * local_expert_num
    output_offset = 0
    segment_index = 0
    for local_expert in range(local_expert_num):
        for source_rank in range(ep_size):
            rows = global_host_values[source_rank * local_expert_num + local_expert]
            if rows > 0xFFFFFFFF:
                raise ValueError("global_tokens_per_local_expert values must fit uint32")
            struct.pack_into(
                "<QQIHH",
                metadata,
                _MOE_ROUTE_SEGMENTS_OFFSET + segment_index * _MOE_ROUTE_SEGMENT_SIZE,
                source_offsets[source_rank],
                output_offset,
                rows,
                source_rank,
                local_expert,
            )
            source_offsets[source_rank] += rows
            peer_rows[source_rank] += rows
            expert_rows[local_expert] += rows
            output_offset += rows
            segment_index += 1

    for source_rank, rows in enumerate(peer_rows):
        struct.pack_into("<Q", metadata, _MOE_ROUTE_PEER_ROWS_OFFSET + source_rank * 8, rows)
    for local_expert, rows in enumerate(expert_rows):
        struct.pack_into(
            "<Q",
            metadata,
            _MOE_ROUTE_EXPERT_ROWS_OFFSET + local_expert * 8,
            rows,
        )

    # frombuffer avoids converting the 12.6-KiB descriptor into Python ints.
    # The following .to() is the one additional H2D accepted by the host-route
    # convenience path.
    return torch.frombuffer(metadata, dtype=torch.uint8).to(device=device)


def alltoallv_gmm(
    x: torch.Tensor,
    weight: torch.Tensor,
    local_tokens_per_expert: RouteInput,
    global_tokens_per_local_expert: RouteInput,
    *,
    ep_size: int,
    expert_num: int,
    received_rows: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    output_all_to_allv: bool = False,
    all_to_allv_out: Optional[torch.Tensor] = None,
    tiling: Optional[Mapping[str, int]] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch tokens with UDMA and apply the destination rank's grouped matmuls.

    ``x`` is ordered by global expert according to ``local_tokens_per_expert``.
    Route inputs may be flat int64 tensors or host integer sequences with
    ``expert_num`` elements. The global table is laid out by source rank and
    then local expert.
    It is a regular contiguous BF16 NPU tensor; UDMA reads it directly as the
    local source of remote PUT operations, without symmetric-input staging.
    The output has exact shape ``[received_rows, n]``, where ``received_rows``
    equals the sum of the global route table. Pass the host-known value to avoid
    a device-to-host synchronization when the global route is an NPU tensor.
    Host routes are a convenience path that performs a small host-to-device copy
    on every call. When exporting AllToAllV, a host global route additionally
    builds and copies CatCCOS route-prefix metadata so the kernel can select its
    optimized metadata path. Reuse device route tensors for performance-critical
    loops; their export path avoids D2H and uses the compatible kernel fallback.

    When ``output_all_to_allv=True``, the call returns ``(output,
    all_to_allv_output)``. The second tensor is the exact BF16 activation consumed
    by grouped matmul, ordered by local expert first and source rank second, with
    shape ``[sum(global_tokens_per_local_expert), k]``. The default path selects
    the original kernel variant and performs no export copy.

    ``tiling`` overrides the built-in shape profile with a measured search
    result. Every rank must pass the same configuration for this collective.
    Load it once with :func:`load_tiling_config` outside the hot loop.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    if expert_num <= 0 or expert_num % ep_size:
        raise ValueError("expert_num must be positive and divisible by ep_size")
    tiling_arguments = _tiling_arguments(
        "all_to_allv_grouped_mat_mul",
        tiling,
        x,
        weight,
        ep_size,
        output_all_to_allv,
    )
    local_tokens_per_expert, local_host_values = _normalize_route(
        local_tokens_per_expert,
        "local_tokens_per_expert",
        x,
        expert_num,
    )
    global_tokens_per_local_expert, global_host_values = _normalize_route(
        global_tokens_per_local_expert,
        "global_tokens_per_local_expert",
        x,
        expert_num,
    )
    if local_host_values is not None and sum(local_host_values) != x.shape[0]:
        raise ValueError("sum(local_tokens_per_expert) must equal x.shape[0]")
    received_rows = _normalize_received_rows(
        received_rows,
        global_tokens_per_local_expert,
        global_host_values,
        ep_size * x.shape[0],
    )
    arguments = (
        x,
        weight,
        local_tokens_per_expert,
        global_tokens_per_local_expert,
        ep_size,
        expert_num,
        received_rows,
    )
    if all_to_allv_out is not None and not output_all_to_allv:
        raise ValueError("all_to_allv_out requires output_all_to_allv=True")
    if not output_all_to_allv:
        if out is None:
            return torch.ops.deep_ep_ep_gmm_fused.all_to_allv_grouped_mat_mul_udma(*arguments, **tiling_arguments)
        return torch.ops.deep_ep_ep_gmm_fused.all_to_allv_grouped_mat_mul_udma_out(*arguments, out, **tiling_arguments)

    if all_to_allv_out is None:
        all_to_allv_out = torch.empty((received_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    elif tuple(all_to_allv_out.shape) != (received_rows, x.shape[1]):
        raise ValueError("all_to_allv_out must have shape [sum(global_tokens_per_local_expert), x.shape[1]]")
    route_metadata = (
        _build_route_metadata(global_host_values, ep_size, expert_num, x.device)
        if global_host_values is not None
        else None
    )
    if out is None:
        return torch.ops.deep_ep_ep_gmm_fused.all_to_allv_grouped_mat_mul_udma_export(
            *arguments, route_metadata, all_to_allv_out, **tiling_arguments
        )
    return torch.ops.deep_ep_ep_gmm_fused.all_to_allv_grouped_mat_mul_udma_out_export(
        *arguments, route_metadata, out, all_to_allv_out, **tiling_arguments
    )


def gmm_alltoallv(
    x: torch.Tensor,
    weight: torch.Tensor,
    local_tokens_per_expert: RouteInput,
    global_tokens_per_local_expert: RouteInput,
    *,
    ep_size: int,
    expert_num: int,
    out: Optional[torch.Tensor] = None,
    tiling: Optional[Mapping[str, int]] = None,
) -> torch.Tensor:
    """Apply local grouped matmuls and return tokens with UDMA AllToAllV.

    ``x`` is a regular contiguous BF16 NPU tensor grouped by this rank's local
    experts and has ``sum(global_tokens_per_local_expert)`` rows. The result has
    shape ``[sum(local_tokens_per_expert), n]`` and resides in regular NPU memory.
    Route inputs may be flat int64 tensors or host integer sequences with
    ``expert_num`` elements. The global table is laid out by source rank and
    then local expert. Host routes are copied to the same NPU as ``x`` on every
    call. When ``local_tokens_per_expert`` is host-resident, its known sum is
    used to allocate the exact output without a route-count D2H synchronization.
    UDMA GET writes directly to that local allocation, so a reusable ``out`` can
    be created with :func:`torch.empty` and no symmetric-output staging or D2D
    copy is required.

    ``tiling`` accepts a search result loaded with :func:`load_tiling_config`.
    Meta/FakeTensor and ``torch.compile`` require ``out`` with shape
    ``[sum(local_tokens_per_expert), weight.shape[2]]``; that row count cannot
    be inferred from the input shape under imbalanced routing.
    All ranks in the collective must use the same configuration.
    """
    config = get_udma_config()
    if ep_size != config.world_size:
        raise ValueError("TP=1 requires ep_size == UDMA world_size")
    if expert_num <= 0 or expert_num % ep_size:
        raise ValueError("expert_num must be positive and divisible by ep_size")
    tiling_arguments = _tiling_arguments(
        "grouped_mat_mul_all_to_allv",
        tiling,
        x,
        weight,
        ep_size,
    )
    local_tokens_per_expert, local_host_values = _normalize_route(
        local_tokens_per_expert,
        "local_tokens_per_expert",
        x,
        expert_num,
    )
    global_tokens_per_local_expert, global_host_values = _normalize_route(
        global_tokens_per_local_expert,
        "global_tokens_per_local_expert",
        x,
        expert_num,
    )
    if global_host_values is not None and sum(global_host_values) != x.shape[0]:
        raise ValueError("sum(global_tokens_per_local_expert) must equal x.shape[0]")
    if local_host_values is not None:
        output_rows = sum(local_host_values)
        expected_output_shape = (output_rows, weight.shape[2])
        if out is None:
            out = torch.empty(expected_output_shape, dtype=x.dtype, device=x.device)
        elif tuple(out.shape) != expected_output_shape:
            raise ValueError("out must have shape [sum(local_tokens_per_expert), n]")
    arguments = (
        x,
        weight,
        local_tokens_per_expert,
        global_tokens_per_local_expert,
        ep_size,
        expert_num,
    )
    if out is None:
        return torch.ops.deep_ep_ep_gmm_fused.grouped_mat_mul_all_to_allv_udma(*arguments, **tiling_arguments)
    return torch.ops.deep_ep_ep_gmm_fused.grouped_mat_mul_all_to_allv_udma_out(*arguments, out, **tiling_arguments)


# Descriptive aliases share the callable, signature and return contract.
all_to_allv_grouped_matmul = alltoallv_gmm
grouped_matmul_all_to_allv = gmm_alltoallv


def _qmma_problem_shape(a: torch.Tensor, b: torch.Tensor, format: int) -> tuple[int, int, int]:
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


def _qmma_check_scale(
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

    config = _get_qmma_config()
    if lanes is not None and lanes != config.lanes:
        raise ValueError(f"lanes={lanes} differs from the initialized lanes={config.lanes}")

    m, n, k = _qmma_problem_shape(a, b, format)
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
        _qmma_check_scale(a_scale, "a_scale", (m, padded_groups), a.device)
        _qmma_check_scale(b_scale, "b_scale", (padded_groups // 2, n, 2), a.device)

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
