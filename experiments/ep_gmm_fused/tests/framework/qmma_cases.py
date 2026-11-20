# Copyright (c) 2026, Lu Lu

"""Curated single-rank shape coverage for Ascend950 QuantMatmulAllToAll."""

from dataclasses import asdict, dataclass

QMMA = "quant_matmul_alltoall"
QMMA_EP_SIZES = (2, 4, 8, 16)
FORMAT_NAMES = ("mxfp8", "mxfp4", "bf16")


@dataclass(frozen=True)
class QmmaCase:
    format: int
    m: int
    n: int
    k: int
    ep_size: int = 8
    seed: int = 20260920
    operator: str = QMMA
    scenario: dict | None = None

    def __post_init__(self):
        if self.operator != QMMA or self.format not in range(len(FORMAT_NAMES)):
            raise ValueError("invalid QMMA operator or input format")
        if self.ep_size not in QMMA_EP_SIZES:
            raise ValueError(f"QMMA case requires EP in {QMMA_EP_SIZES}")
        if min(self.m, self.n, self.k) < 1 or self.k % 32 or self.n % (256 * self.ep_size):
            raise ValueError("QMMA requires positive M/N/K, K divisible by 32 and N/EP divisible by 256")
        if self.seed < 0:
            raise ValueError("QMMA seed must be nonnegative")

    @property
    def case_id(self):
        if self.format == 0:
            prefix = "mxfp8"
        elif self.format == 1:
            prefix = "mxfp4"
        elif self.k == 3328:
            prefix = "bf16-k3328"
        else:
            prefix = "bf16-k4096-small" if self.m <= 512 else "bf16-k4096-large"
        return f"{prefix}-m{self.m}"

    def record(self):
        return {"case_id": self.case_id, **asdict(self)}


_MX_M = (10_000, 16_383, 16_384, 16_385, 32_768, 65_536, 131_072, 262_144, 299_999, 300_000)
_BF16_K3328_M = (3_667, 4_095, 4_096, 4_097, 8_192, 12_288, 16_384, 24_576, 29_999, 30_000)
_BF16_K4096_LARGE_M = (4_518, 4_608, 8_192, 12_288, 16_383, 16_384, 16_385, 30_000, 35_999, 36_000)
_BF16_K4096_SMALL_M = (17, 31, 32, 33, 127, 128, 129, 256, 511, 512)

_GROUPS = (
    ("qmma_mxfp8", 0, 8192, 8192, _MX_M),
    ("qmma_mxfp4", 1, 8192, 8192, _MX_M),
    ("qmma_bf16_k3328", 2, 12_288, 3328, _BF16_K3328_M),
    ("qmma_bf16_large_m", 2, 12_288, 4096, _BF16_K4096_LARGE_M),
    ("qmma_bf16_small_m", 2, 12_288, 4096, _BF16_K4096_SMALL_M),
)


def qmma_cases(seed: int, ep_size: int) -> list[QmmaCase]:
    return [
        QmmaCase(format=fmt, m=m, n=n, k=k, ep_size=ep_size, seed=seed, scenario={"category": category})
        for category, fmt, n, k, m_values in _GROUPS
        for m in m_values
    ]


def required_symmetric_bytes(case: QmmaCase) -> int:
    """Size the heap for the default communication interval and up to 64 AICs."""
    # Match the kernel layout: BF16 TX/Y, cache-line-aligned READY flags, then producer tickets.
    rows_per_round = 128 * min(32, (case.m + 255) // 256)
    rounds = (case.m + rows_per_round - 1) // rows_per_round
    data_bytes = (rounds * rows_per_round + case.m) * case.n * 2
    ready_bytes = (2 * case.ep_size + 2) * 64
    producer_bytes = rounds * case.ep_size * 64 * 64
    layout_bytes = data_bytes + ready_bytes + producer_bytes
    return (layout_bytes + 4095) // 4096 * 4096
