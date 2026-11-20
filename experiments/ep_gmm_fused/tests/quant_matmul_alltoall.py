# Copyright (c) 2026, Lu Lu

"""Check Ascend950 QuantMatmulAllToAll over 50 per-rank shapes."""

import argparse
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import subprocess
import sys

from .framework.installed import load_installed_ep_gmm_fused
from .quant_matmul_alltoall_cases import CASES, FORMAT_NAMES, QmmaCase, required_symmetric_bytes
from .run import execute

EP_SIZES = (2, 4, 8, 16)
ROW_PERIOD, COL_PERIOD = 17, 19
BLOCK_ROWS = 1024


def _make_patterns(case: QmmaCase):
    import numpy as np

    groups = case.k // 32
    rng = np.random.default_rng([case.format, case.m, case.n, case.k])
    a_index = rng.integers(4, size=(ROW_PERIOD, groups, 2))
    b_index = rng.integers(4, size=(groups, 2, COL_PERIOD))
    values = np.array([0.5, -1.0, 1.5, -2.0], dtype=np.float32)
    a_values, b_values = values[a_index], values[b_index]
    a_scale = b_scale = None
    if case.format == 2:
        a_data, b_data = a_values, b_values
    else:
        # E4M3 bytes and E2M1 nibbles for the four values above.
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


def _row_indices(np, first: int, stop: int, source_rank: int):
    return (np.arange(first, stop) + source_rank) % ROW_PERIOD


def _pack_fp4(values):
    return values[:, 0::2] | (values[:, 1::2] << 4)


def _make_inputs(torch, case: QmmaCase, device: str, rank: int):
    import numpy as np

    a_rows, b_rows, a_scales, b_scales, reference = _make_patterns(case)
    n_indices = np.arange(case.n) % COL_PERIOD
    packed = case.format == 1
    dtype = torch.bfloat16 if case.format == 2 else torch.uint8
    a = torch.empty((case.m, case.k // (2 if packed else 1)), dtype=dtype, device=device)
    b = torch.empty((case.k, case.n // (2 if packed else 1)), dtype=dtype, device=device)
    if packed:
        a_rows = _pack_fp4(a_rows)

    def copy_block(dest, block):
        dest.copy_(torch.from_numpy(np.ascontiguousarray(block)).to(dest.dtype))

    for start in range(0, case.m, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, case.m)
        copy_block(a[start:stop], a_rows[_row_indices(np, start, stop, rank)])
    for start in range(0, case.k, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, case.k)
        block = b_rows[start:stop, n_indices]
        copy_block(b[start:stop], _pack_fp4(block) if packed else block)

    a_scale = b_scale = None
    if a_scales is not None:
        a_scale = torch.from_numpy(a_scales[_row_indices(np, 0, case.m, rank)]).to(device)
        b_scale = torch.from_numpy(
            np.ascontiguousarray(b_scales[:, n_indices].reshape(-1, 2, case.n).transpose(0, 2, 1))
        ).to(device)
    reference = torch.from_numpy(reference).to(torch.bfloat16).float().numpy()
    return (a, b, a_scale, b_scale), reference


def _check_output(torch, output, reference, case: QmmaCase, rank: int, ep_size: int) -> None:
    import numpy as np

    chunk_n = case.n // ep_size
    expected_shape = (case.m * ep_size, chunk_n)
    if tuple(output.shape) != expected_shape or output.dtype != torch.bfloat16:
        raise AssertionError(
            f"{case.case_id}: expected BF16 {expected_shape}, got {output.dtype} {tuple(output.shape)}"
        )
    columns = np.arange(rank * chunk_n, (rank + 1) * chunk_n) % COL_PERIOD
    for source in range(ep_size):
        base = source * case.m
        for first in range(0, case.m, BLOCK_ROWS):
            stop = min(first + BLOCK_ROWS, case.m)
            expected = reference[np.ix_(_row_indices(np, first, stop, source), columns)]
            actual = output[base + first : base + stop].cpu().float().numpy()
            bad = actual != expected
            if bad.any():
                row, col = np.unravel_index(bad.argmax(), bad.shape)
                raise AssertionError(
                    f"{case.case_id}: destination rank {rank}, source rank {source}, "
                    f"local GEMM coordinate ({first + row}, {rank * chunk_n + col}): "
                    f"expected {expected[row, col]}, got {actual[row, col]}"
                )


def _run_case(case: QmmaCase, args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    import torch_npu

    rank, local_rank, ep_size, local_size = (
        int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    )
    if ep_size != local_size or ep_size not in EP_SIZES:
        raise ValueError(f"QMMA tests require a single node with EP in {EP_SIZES}")
    package = load_installed_ep_gmm_fused()
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout))
    heap_size = args.heap_size or required_symmetric_bytes(case, ep_size)
    package.init_qmma_group(group=dist.group.WORLD, local_rank=local_rank, heap_size=heap_size)
    inputs, reference = _make_inputs(torch, case, f"npu:{local_rank}", rank)
    output = package.quant_matmul_alltoall(*inputs, format=case.format)
    _check_output(torch, output, reference, case, rank, ep_size)
    # Only enter collective cleanup after the check succeeds on this rank.
    dist.barrier()
    package.destroy_qmma_group()
    dist.destroy_process_group()
    if rank == 0:
        print(f"PASS {case.case_id} EP{ep_size} ({package.__file__})", flush=True)


def _run_suite(cases: list[QmmaCase], args: argparse.Namespace) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = Path("benchmark-runs/ep_gmm_fused") / f"qmma-{stamp}"
    log_dir.mkdir(parents=True)
    failures = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={args.ep_size}",
            "-m",
            f"{__package__}.quant_matmul_alltoall",
            "--case",
            case.case_id,
            "--timeout",
            str(args.timeout),
        ]
        if args.heap_size is not None:
            command += ["--heap-size", str(args.heap_size)]
        # A fresh SHMEM runtime per case avoids reusing the first case's fixed workspace.
        log = log_dir / f"{case.case_id}.log"
        try:
            execute(command, log, args.timeout)
        except (RuntimeError, subprocess.TimeoutExpired):
            failures.append(case.case_id)
            print(f"FAIL {case.case_id}; see {log}", flush=True)
        else:
            print(f"PASS {case.case_id}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)}/{len(cases)} QMMA cases failed; logs: {log_dir}")
    print(f"All {len(cases)} QMMA cases passed; logs: {log_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="list all 50 shapes and workspace estimates")
    selection.add_argument("--all", action="store_true", help="launch the selected suite (fresh torchrun job per case)")
    selection.add_argument("--case", metavar="ID", help="run a single case from --list")
    parser.add_argument("--format", choices=FORMAT_NAMES, help="filter by input format")
    parser.add_argument("--ep-size", type=int, choices=EP_SIZES, default=8, help="rank count for --list and --all")
    parser.add_argument("--heap-size", type=int, help="override the estimated SHMEM heap size in bytes")
    parser.add_argument("--timeout", type=int, default=600, help="per-case timeout in seconds for --all")
    args = parser.parse_args()
    if args.heap_size is not None and args.heap_size <= 0:
        parser.error("--heap-size must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    cases = [
        case
        for case in CASES
        if (args.format is None or FORMAT_NAMES[case.format] == args.format)
        and (args.case is None or case.case_id == args.case)
    ]
    if not cases:
        parser.error("no matching case; use --list to see valid IDs")
    if args.list:
        print("case_id\tformat\tM\tN\tK\tworkspace_GiB")
        for case in cases:
            workspace = required_symmetric_bytes(case, args.ep_size) / (1 << 30)
            print(f"{case.case_id}\t{FORMAT_NAMES[case.format]}\t{case.m}\t{case.n}\t{case.k}\t{workspace:.3f}")
    elif args.all:
        if "LOCAL_RANK" in os.environ:
            parser.error("run --all with python -m, outside torchrun")
        _run_suite(cases, args)
    else:
        if "LOCAL_RANK" not in os.environ:
            parser.error("launch --case with torchrun")
        _run_case(cases[0], args)


if __name__ == "__main__":
    main()
