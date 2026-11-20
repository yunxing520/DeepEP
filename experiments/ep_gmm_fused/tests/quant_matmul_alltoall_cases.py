# Copyright (c) 2026, Lu Lu

"""Curated single-rank shape coverage for Ascend950 QuantMatmulAllToAll."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QmmaCase:
    case_id: str
    format: int
    m: int
    n: int
    k: int


_MX_M = (10_000, 16_383, 16_384, 16_385, 32_768, 65_536, 131_072, 262_144, 299_999, 300_000)
_BF16_K3328_M = (3_667, 4_095, 4_096, 4_097, 8_192, 12_288, 16_384, 24_576, 29_999, 30_000)
_BF16_K4096_LARGE_M = (4_518, 4_608, 8_192, 12_288, 16_383, 16_384, 16_385, 30_000, 35_999, 36_000)
_BF16_K4096_SMALL_M = (17, 31, 32, 33, 127, 128, 129, 256, 511, 512)

FORMAT_NAMES = ("mxfp8", "mxfp4", "bf16")
_GROUPS = (
    ("mxfp8", 0, 8192, 8192, _MX_M),
    ("mxfp4", 1, 8192, 8192, _MX_M),
    ("bf16-k3328", 2, 12_288, 3328, _BF16_K3328_M),
    ("bf16-k4096-large", 2, 12_288, 4096, _BF16_K4096_LARGE_M),
    ("bf16-k4096-small", 2, 12_288, 4096, _BF16_K4096_SMALL_M),
)

CASES = tuple(QmmaCase(f"{prefix}-m{m}", fmt, m, n, k) for prefix, fmt, n, k, m_values in _GROUPS for m in m_values)


def required_symmetric_bytes(case: QmmaCase, ep_size: int) -> int:
    """Size the heap for the default communication interval and up to 64 AICs."""
    # CatCCOS allocates BF16 TX (padded to full rounds), BF16 Y, and 64-byte control flags.
    rows_per_round = 128 * min(32, (case.m + 255) // 256)
    rounds = (case.m + rows_per_round - 1) // rows_per_round
    data_bytes = (rounds * rows_per_round + case.m) * case.n * 2
    control_bytes = (2 * ep_size + 2 + rounds * ep_size * 64) * 64
    return (data_bytes + control_bytes + 4095) // 4096 * 4096
