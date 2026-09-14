"""Hardware capability probe -- the framework's single source of truth for
"what can actually run on this machine right now."

Every entry point that touches a GPU-only code path (tierahead.collect,
tierahead.calibrate.compute/transfer/e2e_offload/nf4_fidelity,
tierahead.sim.backends.cxlmemsim) calls this module first and either (a) runs
the real thing when the capability is present, (b) falls back to the
documented CPU-only equivalent (flop-estimate compute, analytical/DES
simulation), or (c) refuses with an actionable message naming exactly which
command to run on a machine that does have the capability -- never a silent
wrong answer.

This machine (as of the framework's own `doctor` run) is a no-CUDA Apple
Silicon Mac: every ``requires_cuda`` capability below is False here, which is
the expected and fully-supported case -- Main.md's own Phase-1 findings
(E1, E2, E4, E5, E8's real numbers) were *produced* on a machine in exactly
this state. See PREDICTED.md's "Verified" section for which numbers that
already covers, and Main.md section 8 ("Hardware Feasibility") for the
original no-CUDA strategy this module operationalizes.
"""
from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_available() -> bool:
    if not _has_module("torch"):
        return False
    import torch

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    if not _has_module("torch"):
        return False
    import torch

    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _perf_event_open_available() -> bool:
    """CXLMemSim (SlugLab/CXLMemSim, arXiv:2303.06153) instruments memory
    accesses via kernel probes + hardware performance counters
    (`perf_event_open`), which is a Linux-only syscall. Never available on
    Darwin, and on Linux typically needs elevated privilege
    (`CAP_PERFMON`/root or `perf_event_paranoid<=1`)."""
    if platform.system() != "Linux":
        return False
    try:
        subprocess.run(["perf", "--version"], capture_output=True, timeout=3, check=False)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _cxlmemsim_binary() -> str | None:
    return shutil.which("cxlmemsim") or shutil.which("CXLMemSim")


@dataclass(frozen=True)
class Capabilities:
    platform: str
    machine: str
    python_version: str
    cuda_available: bool
    mps_available: bool
    torch_installed: bool
    linux_perf_available: bool
    cxlmemsim_binary: str | None
    notes: list[str] = field(default_factory=list)

    @property
    def can_collect_traces(self) -> bool:
        """tierahead.collect (real forward-hook trace collection on live MoE
        weights) needs CUDA -- MPS technically runs OLMoE fp16 forward
        passes (Main.md section 8.1) but is absent here by design scope: the
        framework treats trace collection as a GPU-server task and ships
        with the pilot's already-collected real traces for everything else
        to run against on a laptop."""
        return self.cuda_available

    @property
    def can_run_real_gpu_calibration(self) -> bool:
        return self.cuda_available

    @property
    def can_run_cxlmemsim_backend(self) -> bool:
        return self.linux_perf_available and self.cxlmemsim_binary is not None

    @property
    def can_train_mlp_predictor(self) -> bool:
        """CPU/MPS both work -- Experiments.md E4/E5 verified full sweeps
        under a minute on CPU alone; this is NOT gated on CUDA."""
        return self.torch_installed

    def describe(self) -> str:
        lines = [
            f"platform:            {self.platform} ({self.machine})",
            f"python:              {self.python_version}",
            f"CUDA available:      {self.cuda_available}",
            f"MPS available:       {self.mps_available}",
            f"torch installed:     {self.torch_installed}",
            f"Linux perf available:{self.linux_perf_available}",
            f"CXLMemSim binary:    {self.cxlmemsim_binary or 'not found on PATH'}",
            "",
            "capability            -> strategy on this machine",
            "-" * 60,
            f"trace collection      -> {'REAL (CUDA)' if self.can_collect_traces else 'UNAVAILABLE here -- use the pilot committed traces under results/, or run tierahead collect on a CUDA box'}",
            f"GPU compute calib     -> {'REAL (CUDA)' if self.can_run_real_gpu_calibration else 'flop_estimate fallback (roofline/sim tag every number compute_provenance=flop_estimate)'}",
            f"MLP predictor         -> {'CPU/MPS -- runs here, no CUDA needed' if self.can_train_mlp_predictor else 'torch not installed -- pip install tierahead[predictor]'}",
            f"DES / analytical sim  -> ALWAYS available (pure Python, this machine included)",
            f"CXLMemSim backend     -> {'available' if self.can_run_cxlmemsim_backend else 'UNAVAILABLE here (needs Linux + perf + CXLMemSim built from source) -- sim falls back to the des/analytical backend, tagged accordingly'}",
        ]
        return "\n".join(lines)


def probe() -> Capabilities:
    torch_installed = _has_module("torch")
    notes = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        notes.append(
            "Apple Silicon detected: no CUDA, ever. This is a fully-supported mode -- "
            "roofline, analyze, policy, sim, tco, and dashboard all run natively here. "
            "Only tierahead.collect and tierahead.calibrate's GPU submodules require a CUDA box."
        )
    return Capabilities(
        platform=platform.system(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        cuda_available=_cuda_available(),
        mps_available=_mps_available(),
        torch_installed=torch_installed,
        linux_perf_available=_perf_event_open_available(),
        cxlmemsim_binary=_cxlmemsim_binary(),
        notes=notes,
    )
