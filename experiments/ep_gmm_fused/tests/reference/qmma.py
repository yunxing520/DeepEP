# Copyright (c) 2026, Lu Lu

"""Structured inputs and exact BF16 references for QuantMatmulAllToAll."""

from __future__ import annotations

import numpy as np

from ..framework.qmma_cases import QmmaCase

ROW_PERIOD, COL_PERIOD = 17, 19


def _make_patterns(case: QmmaCase):
    groups = case.k // 32
    rng = np.random.default_rng([case.format, case.m, case.n, case.k, case.seed])
    a_index = rng.integers(4, size=(ROW_PERIOD, groups, 2))
    b_index = rng.integers(4, size=(groups, 2, COL_PERIOD))
    values = np.array([0.5, -1.0, 1.5, -2.0], dtype=np.float32)
    a_values, b_values = values[a_index], values[b_index]
    a_scale = b_scale = None
    if case.format == 2:
        a_data, b_data = a_values, b_values
    else:
        codes = np.array([0x30, 0xB8, 0x3C, 0xC0] if case.format == 0 else [0x1, 0xA, 0x3, 0xC], dtype=np.uint8)
        a_data, b_data = codes[a_index], codes[b_index]
        a_exp = rng.integers(-2, 1, size=(ROW_PERIOD, groups))
        b_exp = rng.integers(-2, 1, size=(groups, COL_PERIOD))
        a_values = np.ldexp(a_values, a_exp[:, :, None].astype(np.int32))
        b_values = np.ldexp(b_values, b_exp[:, None, :].astype(np.int32))
        # E8M0 stores exponent + 127. Pad to pairs of scale groups.
        a_scale = np.pad((a_exp + 127).astype(np.uint8), ((0, 0), (0, groups % 2)), constant_values=127)
        b_scale = np.pad((b_exp + 127).astype(np.uint8), ((0, groups % 2), (0, 0)), constant_values=127)

    # Each group alternates two values across its 32 K positions.
    reference = (a_values.reshape(ROW_PERIOD, -1) @ b_values.reshape(-1, COL_PERIOD)) * 16
    a_rows = np.tile(a_data, (1, 1, 16)).reshape(ROW_PERIOD, case.k)
    b_rows = np.tile(b_data, (1, 16, 1)).reshape(case.k, COL_PERIOD)
    return a_rows, b_rows, a_scale, b_scale, reference


def _row_indices(first: int, stop: int, source_rank: int):
    return (np.arange(first, stop) + source_rank) % ROW_PERIOD


def _pack_fp4(values):
    return values[:, 0::2] | (values[:, 1::2] << 4)


def make_qmma_inputs(torch, case: QmmaCase, device: str, rank: int, *, block_rows: int):
    a_rows, b_rows, a_scales, b_scales, reference = _make_patterns(case)
    n_indices = np.arange(case.n) % COL_PERIOD
    packed = case.format == 1
    dtype = torch.bfloat16 if case.format == 2 else torch.uint8
    a = torch.empty((case.m, case.k // (2 if packed else 1)), dtype=dtype, device=device)
    b = torch.empty((case.k, case.n // (2 if packed else 1)), dtype=dtype, device=device)
    if packed:
        a_rows = _pack_fp4(a_rows)

    def copy_block(destination, block):
        destination.copy_(torch.from_numpy(np.ascontiguousarray(block)).to(destination.dtype))

    for start in range(0, case.m, block_rows):
        stop = min(start + block_rows, case.m)
        copy_block(a[start:stop], a_rows[_row_indices(start, stop, rank)])
    for start in range(0, case.k, block_rows):
        stop = min(start + block_rows, case.k)
        block = b_rows[start:stop, n_indices]
        copy_block(b[start:stop], _pack_fp4(block) if packed else block)

    a_scale = b_scale = None
    if a_scales is not None:
        a_scale = torch.from_numpy(a_scales[_row_indices(0, case.m, rank)]).to(device)
        b_scale = torch.from_numpy(
            np.ascontiguousarray(b_scales[:, n_indices].reshape(-1, 2, case.n).transpose(0, 2, 1))
        ).to(device)
    reference = torch.from_numpy(reference).to(torch.bfloat16).float().numpy()
    return (a, b, a_scale, b_scale), reference


def check_qmma_output(torch, output, reference, case: QmmaCase, rank: int, *, block_rows: int) -> None:
    chunk_n = case.n // case.ep_size
    expected_shape = (case.m * case.ep_size, chunk_n)
    if tuple(output.shape) != expected_shape or output.dtype != torch.bfloat16:
        raise AssertionError(
            f"{case.case_id}: expected BF16 {expected_shape}, got {output.dtype} {tuple(output.shape)}"
        )
    columns = np.arange(rank * chunk_n, (rank + 1) * chunk_n) % COL_PERIOD
    for source in range(case.ep_size):
        base = source * case.m
        for first in range(0, case.m, block_rows):
            stop = min(first + block_rows, case.m)
            expected = reference[np.ix_(_row_indices(first, stop, source), columns)]
            actual = output[base + first : base + stop].cpu().float().numpy()
            bad = actual != expected
            if bad.any():
                row, col = np.unravel_index(bad.argmax(), bad.shape)
                raise AssertionError(
                    f"{case.case_id}: destination rank {rank}, source rank {source}, "
                    f"local GEMM coordinate ({first + row}, {rank * chunk_n + col}): "
                    f"expected {expected[row, col]}, got {actual[row, col]}"
                )
