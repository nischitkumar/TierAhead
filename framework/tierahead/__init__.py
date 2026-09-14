"""TierAhead: router-guided memory tiering for cost-efficient MoE inference on
HBM + CXL memory systems.

See ``README.md`` for the architecture (the eight components below map
directly onto Main.md's own "Framework Architecture" section):

  [0] tierahead.roofline    -- regime model (rho = t_transfer / t_compute)
  [1] tierahead.collect     -- router-decision trace collector (GPU-required)
  [2] tierahead.analyze     -- workload characterization (Coverage/Gini/Recall/HTB)
  [3] tierahead.policy.residency     -- fixed-residency placement
  [4] tierahead.policy.predictor_*   -- prefetch engine (frequency table + MLP)
  [4b] tierahead.policy.precision_gate -- confidence-gated precision fallback
  [5] tierahead.sim         -- tiered-memory discrete-event simulator
  [6] tierahead.tco         -- pooling & TCO model
  [7] tierahead.dashboard   -- Streamlit + Plotly interactive explorer
  [8] tierahead.kv          -- KV-cache co-tenancy traffic class

``tierahead.baseline.compare`` is the flagship end-to-end entry point: it runs
the "normal" (HBM-only, no tiering) baseline and the CXL-tiered configuration
side by side on the same trace-driven workload and reports the delta.
"""
__version__ = "0.1.0"
