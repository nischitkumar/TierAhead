"""[1] Trace Collector (Main.md section 4.1) -- CUDA-required.

PyTorch forward hooks on every MoE gate module; logs per token per layer:
router logits (top-k + entropy), selected expert IDs, gate weights, token
position, request ID. Zero model modification. Ported from
elp_probe/src/collect.py (the pilot's own collector, which produced every
real trace this framework's other components run against) into an
importable, CLI-wrapped module.

Every entry point here checks tiermoe.hw.probe().can_collect_traces first
and refuses with an actionable message rather than failing deep inside a
torch call when CUDA isn't available -- this machine (no CUDA) always hits
that refusal; see README.md's "Trace collection" section for the exact
command to run on a GPU box instead.
"""
from .collector import Collector, TraceWriter, run_batch_erosion, run_request_single

__all__ = ["Collector", "TraceWriter", "run_batch_erosion", "run_request_single"]
