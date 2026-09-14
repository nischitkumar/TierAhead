"""Memory-tier hardware parameters -- Main.md section 4.5's tier table,
structured. Every number here carries a `source` field naming exactly where
it came from: a vendor datasheet, a cited paper, or "measured_here" once a
real calibration file (tiermoe.calibrate) supplies it. None of these are
independently re-measured by this framework on CXL silicon (nobody has any,
per Main.md section 8.4's own framing) -- they are the same literature-cited
constants Experiments.md's E3/E6 sections specify importing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TierSpec:
    tag: str
    label: str
    capacity_gb_min: float
    capacity_gb_max: float
    bw_gbps: float
    added_latency_ns: float
    source: str


TIER_SPECS: dict[str, TierSpec] = {
    "hbm3": TierSpec(
        tag="hbm3", label="HBM3 (H100 SXM)",
        capacity_gb_min=24, capacity_gb_max=96, bw_gbps=3350.0, added_latency_ns=0.0,
        source="NVIDIA H100 datasheet (reference tier, ~0 added latency by definition)",
    ),
    "ddr5_local": TierSpec(
        tag="ddr5_local", label="Local DDR5 (8-channel)",
        capacity_gb_min=512, capacity_gb_max=512, bw_gbps=307.0, added_latency_ns=100.0,
        source="JEDEC DDR5 spec / vendor",
    ),
    "cxl2_x8": TierSpec(
        tag="cxl2_x8", label="CXL 2.0 x8 (Leo-class)",
        capacity_gb_min=512, capacity_gb_max=2048, bw_gbps=32.0, added_latency_ns=210.0,
        source="Sun et al., Demystifying CXL Memory with True CXL-Ready Systems, ISCA 2023; "
               "Astera Labs Leo product brief (170-250ns added latency, midpoint used as default)",
    ),
    "cxl2_x16": TierSpec(
        tag="cxl2_x16", label="CXL 2.0 x16 / CXL 3.x x8",
        capacity_gb_min=512, capacity_gb_max=2048, bw_gbps=64.0, added_latency_ns=210.0,
        source="Sun et al. ISCA 2023 (bandwidth scaled with lane count); same latency citation as cxl2_x8",
    ),
    "cxl3_multilink": TierSpec(
        tag="cxl3_multilink", label="CXL 3.x multi-link",
        capacity_gb_min=1024, capacity_gb_max=4096, bw_gbps=128.0, added_latency_ns=210.0,
        source="CXL Consortium CXL 3.x specification (multi-link aggregate bandwidth, projected)",
    ),
    "pcie4_x16": TierSpec(
        tag="pcie4_x16", label="PCIe 4.0 x16 (GPU link)",
        capacity_gb_min=0, capacity_gb_max=0, bw_gbps=25.0, added_latency_ns=1000.0,
        source="FloE (arXiv:2505.05950) measurements; used as the measured-physical-layer analogue "
               "for CXL in this repo's E3/E10 calibration ladder (same PCIe wire, DMA-driven transfers)",
    ),
}

# The four discrete link tiers E1/E6 sweep over.
LINK_BW_GBPS_GRID = [16, 32, 64, 128]
LINK_BW_LABELS = {
    16: "sub-CXL2.0x8", 32: "CXL2.0 x8", 64: "CXL2.0 x16 / CXL3.x x8", 128: "CXL3.x multi-link",
}

# CXL added-latency sensitivity sweep (Main.md section 8.4's own pre-emption:
# "we sweep 150 to 400 ns and the regime boundary barely moves, because
# bandwidth, not latency, is what decides it" -- tiermoe.sim.validation
# includes a test asserting exactly this).
CXL_LATENCY_NS_GRID = [150, 250, 400]
