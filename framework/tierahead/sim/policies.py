"""Per-layer, trace-driven policy bake-off (Main.md section 4.4's policy
table / Experiments.md E6's "policies to compare, all on identical traces").

Walks REAL held-out (test-split) decode traces layer by layer, request by
request, and for each layer's actual selected experts asks: was this expert
resident (no cost), successfully prefetched (no TPOT cost, bytes still
spent), or a genuine demand miss (TPOT stall = bytes/BW)? Aggregates into
TPOT mean/p50/p99, demand-stall rate, bytes/token, and prefetch
precision/recall -- the same metrics Main.md's Metrics Dictionary defines,
plus the "prefetch accuracy/coverage" pair (Main.md section 4.4's own
"report accuracy, coverage, and timeliness separately" design constraint;
"precision" here is that accuracy leg).

Policies (mentor-review additions marked NEW):
  none                      -- demand fetch, r=0 forced. Lower bound.
  static-c                  -- fixed residency, popularity-ranked (train split).
  popularity-prefetch [NEW] -- prefetch the globally most-popular non-resident
                               experts each layer, UNCONDITIONED on current
                               router state. Isolates the value of
                               conditioning: prefetch-freq should beat this
                               by a real margin, or conditioning isn't
                               earning its keep (Main.md section 4.4's own
                               "critical baseline" framing).
  lru                       -- classic recency-based cache, dynamic.
  hybrid-lru-prefetch [NEW] -- keep an LRU-tracked hot set resident (adapts
                               to drift, unlike static-c) AND prefetch
                               predicted misses (unlike plain lru). This is
                               the literal "keep hot experts resident, and
                               prefetch cold ones as needed" hybrid a mentor
                               review asked this framework to demonstrate.
  prefetch-freq             -- conditional frequency-table predictor.
  prefetch-v2               -- MLP predictor (torch; skipped, not faked, if absent).
  prefetch-v2+precision-gate -- + confidence-gated FP16/NF4/skip byte reduction.
  oracle                    -- perfect foreknowledge. Upper bound.

Concurrency [NEW]: `concurrency` models N decode sessions active AT ONCE,
self-throttling and sharing one physical link (a session cannot request
layer l+1's experts before layer l's fetch has arrived, so it never "gets
ahead" of its own link usage) -- a closed queueing network, for which
`bw_gbps / concurrency` is the standard egalitarian-processor-sharing result
under persistent contention, not a hand-wave. See
`tierahead.sim.concurrency`'s module docstring for the full reasoning
(including a documented mistake an earlier draft made trying to "validate"
this against the WRONG kind of queueing model, and the correction). Two
honest caveats, both favoring under- rather than over-confidence:
  - bw/N is most ACCURATE exactly when contention is persistent (every
    session is often simultaneously link-bound) and becomes increasingly
    CONSERVATIVE (predicts worse TPOT than reality) the more intermittent
    real contention is -- it never understates the problem.
  - `tierahead.sim.concurrency.max_sustainable_concurrency` computes the point
    at which contention becomes persistent for THIS operating point;
    `simulate_policy` tags the result's `extra` dict
    (`concurrency_stable`, `aggregate_rho_at_concurrency`,
    `max_sustainable_concurrency`) so a caller can see directly whether the
    reported `tpot_ms_mean` sits in the "bw/N is conservative" regime or the
    "bw/N is the theoretically expected answer" regime.

Fidelity note (read before comparing this to a real GPU-timed run): within
one request's own walk, this module does not model fine-grained contention
between that request's own concurrently in-flight prefetch jobs (it checks
window feasibility, as experiments/e5_lookahead does); cross-request
contention is what `concurrency` adds. Full N-class event-driven contention
(arbitrary, non-identical traffic) is `tierahead.sim.engine`, used by
`tierahead.kv.cotenancy`. See PREDICTED.md's Methodology section.

Requires no GPU. prefetch-v2 (the MLP predictor) requires torch (the
`predictor` extra) and is skipped -- not silently substituted -- when
unavailable.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from tierahead.policy.precision_gate import PrecisionGatePolicy
from tierahead.policy.predictor_freq import FrequencyTablePredictor
from tierahead.policy.residency import FixedResidencyPolicy
from tierahead.roofline.model import get_t_compute_ms
from tierahead.specs import MODEL_SPECS
from tierahead.traces.io import expert_counts, explode_topk, load_traces, train_test_split_reqs

ALL_POLICIES = [
    "none", "static-c", "popularity-prefetch", "lru", "hybrid-lru-prefetch",
    "prefetch-freq", "prefetch-v2", "prefetch-v2+precision-gate", "oracle",
]
PREFETCH_POLICIES = {"popularity-prefetch", "hybrid-lru-prefetch", "prefetch-freq",
                     "prefetch-v2", "prefetch-v2+precision-gate"}


@dataclass
class PolicyResult:
    policy: str
    model: str
    residency_pct: float
    bw_gbps: float
    precision: str
    batch: int
    concurrency: int
    n_test_reqs: int
    n_decode_tokens: int
    tpot_ms_mean: float
    tpot_ms_p50: float
    tpot_ms_p99: float
    demand_stall_rate: float
    bytes_per_token_mean: float
    compute_provenance: str
    oracle_gap_closed_pct: float | None = None
    prefetch_precision: float | None = None
    prefetch_recall: float | None = None
    extra: dict = field(default_factory=dict)


class _LRUCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, capacity)
        self._od: OrderedDict = OrderedDict()

    def peek(self, expert: int) -> bool:
        """Read-only membership check -- does NOT update recency. Used to
        check whether a PREFETCHED expert would already have been resident
        (so hybrid-lru-prefetch doesn't double-count a cache hit as a
        'prefetch hit')."""
        return expert in self._od

    def access(self, expert: int) -> bool:
        """Returns True if this was a hit. Always records the access
        (inserting on miss, evicting LRU if over capacity) -- this is what a
        real cache does whether the byte arrived via demand-fetch or a
        successful prefetch, so callers should call this for EVERY access
        that resolves an expert into the cache, hit or miss."""
        if expert in self._od:
            self._od.move_to_end(expert)
            return True
        if self.capacity > 0:
            self._od[expert] = True
            if len(self._od) > self.capacity:
                self._od.popitem(last=False)
        return False


def _bandwidth_bytes_per_ms(bw_gbps: float, concurrency: int = 1) -> float:
    """Fair-share approximation for N concurrent decode streams sharing one
    link -- see module docstring for why this is exact in the saturating
    regime and conservative otherwise."""
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    return (bw_gbps * 1e6) / concurrency


def _popularity_rank_by_layer(train_counts: pd.DataFrame, n_experts: int) -> dict[int, list[int]]:
    """Global (unconditioned) popularity ranking per layer, from TRAIN split
    counts -- the ranking `popularity-prefetch` uses, and the same ranking
    `static-c`/residency uses for placement (this function is the shared
    primitive; residency.py has its own copy scoped to shared-expert pinning,
    kept separate because that one also handles the shared-expert carve-out)."""
    ranking = {}
    for layer, sub in train_counts.groupby("layer"):
        cnt = sub.set_index("expert")["count"].reindex(range(n_experts), fill_value=0)
        ranking[int(layer)] = cnt.sort_values(ascending=False).index.tolist()
    return ranking


def simulate_policy(*, decode_train: pd.DataFrame, decode_test: pd.DataFrame, spec, policy: str,
                     residency_pct: float, bw_gbps: float, precision: str, batch: int = 1,
                     calib: dict | None = None, depth: int = 1, m_mult: int = 2,
                     tau_lo: float = 0.2, tau_hi: float = 0.6, seed: int = 0,
                     mlp_available: bool = True, added_latency_ns: float = 0.0,
                     concurrency: int = 1) -> PolicyResult | None:
    """Returns None (caller should skip, not crash) if `policy` needs torch
    and it isn't installed."""
    n_layers, n_experts, k = spec.n_layers, spec.n_experts, spec.k
    t_layer_ms, compute_prov = get_t_compute_ms(spec, batch, calib)
    expert_bytes_fp16 = spec.expert_bytes("fp16")
    expert_bytes_nf4 = spec.expert_bytes("nf4")
    expert_bytes = spec.expert_bytes(precision)
    bw_bytes_per_ms = _bandwidth_bytes_per_ms(bw_gbps, concurrency)
    added_latency_ms = added_latency_ns / 1e6  # fixed per-transfer tier latency (e.g. CXL2.0 ~210ns), paid once per demand fetch

    train_counts = expert_counts(explode_topk(decode_train))
    effective_residency = 0.0 if policy == "none" else residency_pct
    residency = FixedResidencyPolicy(residency_pct=effective_residency, n_experts=n_experts,
                                      n_shared_experts=spec.n_shared_experts).fit(train_counts)

    predictor = None
    if policy in ("prefetch-freq", "hybrid-lru-prefetch"):
        predictor = FrequencyTablePredictor(n_experts=n_experts, n_layers=n_layers, d_values=(depth,)).fit(
            decode_train, set(decode_train["req"].unique()))
    pop_rank = None
    if policy == "popularity-prefetch":
        pop_rank = _popularity_rank_by_layer(train_counts, n_experts)
    mlp_bundle = None
    if policy in ("prefetch-v2", "prefetch-v2+precision-gate"):
        if not mlp_available:
            return None
        mlp_bundle = _fit_mlp_predictor(decode_train, n_experts, n_layers, depth, seed)
        if mlp_bundle is None:
            return None

    m = min(m_mult * k, n_experts)
    gate = PrecisionGatePolicy(tau_lo=tau_lo, tau_hi=tau_hi) if policy == "prefetch-v2+precision-gate" else None

    lru_caches: dict[int, _LRUCache] = {}
    if policy in ("lru", "hybrid-lru-prefetch"):
        cap = max(1, int(round(n_experts * residency_pct / 100.0)))
        lru_caches = {l: _LRUCache(cap) for l in range(n_layers)}

    pivot = decode_test.pivot_table(index=["req", "tok"], columns="layer", values="topk", aggfunc="first")
    tpot_samples, stall_events, total_events, bytes_samples = [], 0, 0, []
    nf4_bytes_spent, fp16_bytes_spent = 0.0, 0.0
    n_fp16_hits, n_nf4_hits, n_gated_skip = 0, 0, 0
    n_predicted_total, n_predicted_correct = 0, 0  # precision/recall accumulators (prefetch-eligible policies only)

    for req, tok in pivot.index:
        row = pivot.loc[(req, tok)]
        tpot_ms = 0.0
        token_bytes = 0.0
        for l in range(n_layers):
            actual = row.get(l)
            if actual is None or (isinstance(actual, float) and np.isnan(actual)):
                continue
            tpot_ms += t_layer_ms

            if policy == "oracle":
                miss = set(actual) - residency.resident_set(l)
                token_bytes += len(miss) * expert_bytes
                continue

            if policy == "lru":
                # No separate static-resident set here -- the LRU cache IS
                # the entire resident set (capacity == the configured
                # residency budget). An earlier version of this policy
                # checked `actual - static_resident` against a SEPARATE,
                # same-sized LRU cache underneath the static set, which
                # silently gave 'lru' double the memory budget of every
                # other policy at the same --residency-pct -- caught while
                # implementing hybrid-lru-prefetch and fixed here; see
                # PREDICTED.md's Methodology section for the corrected numbers.
                for e in actual:
                    total_events += 1
                    hit = lru_caches[l].access(e)
                    if hit:
                        continue
                    stall_events += 1
                    tpot_ms += expert_bytes / bw_bytes_per_ms + added_latency_ms
                    token_bytes += expert_bytes
                continue

            resident = residency.resident_set(l)
            miss = set(actual) - resident

            if not miss:
                continue

            # ---- what was predicted `depth` layers earlier? ----
            src_layer = l - depth
            src_actual = row.get(src_layer) if src_layer >= 0 else None
            has_lookahead = src_actual is not None and not (isinstance(src_actual, float) and np.isnan(src_actual))

            predicted: set[int] = set()
            gated_precision: dict[int, str] = {}
            if policy in PREFETCH_POLICIES and (has_lookahead or policy == "popularity-prefetch"):
                m_affordable = min(m, int((depth * t_layer_ms * bw_bytes_per_ms) // expert_bytes)) if expert_bytes > 0 else 0
                if policy == "popularity-prefetch":
                    # Unconditioned on router state -- always the same
                    # globally-popular candidates for this target layer.
                    # Still gated by the same lookahead-window byte budget as
                    # every other prefetch policy (a real system needs SOME
                    # advance layers to issue the fetch at all, even one that
                    # doesn't depend on router content).
                    if l >= depth and m_affordable > 0:
                        predicted = set(pop_rank.get(l, [])[:m_affordable])
                elif policy in ("prefetch-freq", "hybrid-lru-prefetch") and has_lookahead:
                    if m_affordable > 0:
                        predicted = set(predictor.topm(list(src_actual), src_layer, depth, m_affordable))
                elif policy == "prefetch-v2" and has_lookahead:
                    if m_affordable > 0:
                        predicted = set(_mlp_topm(mlp_bundle, src_layer, list(src_actual), n_experts, m_affordable))
                elif policy == "prefetch-v2+precision-gate" and has_lookahead:
                    probs = _mlp_probs(mlp_bundle, src_layer, list(src_actual), n_experts)
                    if probs is not None:
                        order = np.argsort(-probs)
                        budget_bytes = depth * t_layer_ms * bw_bytes_per_ms
                        spent = 0.0
                        for e in order.tolist():
                            p = float(probs[e])
                            dec = gate.decide_one(int(e), p)
                            if dec.decision == "skip":
                                continue
                            cost = expert_bytes_fp16 if dec.decision == "fp16" else expert_bytes_nf4
                            if spent + cost > budget_bytes:
                                break
                            spent += cost
                            predicted.add(int(e))
                            gated_precision[int(e)] = dec.decision

            if policy in PREFETCH_POLICIES:
                n_predicted_total += len(predicted)
                n_predicted_correct += len(predicted & miss)

            for e in miss:
                total_events += 1
                if policy == "hybrid-lru-prefetch":
                    # Dynamic hot set (LRU) is checked FIRST -- a real system
                    # would never re-fetch something already cache-resident
                    # just because it wasn't in this step's prediction.
                    if lru_caches[l].peek(e):
                        lru_caches[l].access(e)  # refresh recency
                        continue
                if e in predicted:
                    if policy == "prefetch-v2+precision-gate":
                        prec = gated_precision.get(e, "fp16")
                        cost = expert_bytes_fp16 if prec == "fp16" else expert_bytes_nf4
                        token_bytes += cost
                        if prec == "nf4":
                            nf4_bytes_spent += cost
                            n_nf4_hits += 1
                        else:
                            fp16_bytes_spent += cost
                            n_fp16_hits += 1
                    else:
                        token_bytes += expert_bytes
                    if policy == "hybrid-lru-prefetch":
                        lru_caches[l].access(e)  # prefetched byte now occupies a cache slot
                    continue  # hidden behind compute -- no stall
                # true miss: demand-fetch, always at full precision
                stall_events += 1
                tpot_ms += expert_bytes_fp16 / bw_bytes_per_ms + added_latency_ms
                token_bytes += expert_bytes_fp16
                if policy == "hybrid-lru-prefetch":
                    lru_caches[l].access(e)  # demand-fetched byte also enters the cache
                if policy == "prefetch-v2+precision-gate" and e not in predicted:
                    n_gated_skip += 1

        tpot_samples.append(tpot_ms)
        bytes_samples.append(token_bytes)

    if not tpot_samples:
        return None

    tpot_arr = np.array(tpot_samples)
    extra = {}
    if policy == "prefetch-v2+precision-gate":
        total_gate_bytes = nf4_bytes_spent + fp16_bytes_spent
        n_gated_hits = n_fp16_hits + n_nf4_hits
        bytes_if_all_fp16 = n_gated_hits * expert_bytes_fp16
        extra = {
            "nf4_bytes_spent": nf4_bytes_spent, "fp16_bytes_spent": fp16_bytes_spent,
            "n_fp16_hits": n_fp16_hits, "n_nf4_hits": n_nf4_hits, "n_gated_skip": n_gated_skip,
            "gate_bytes_saved_vs_all_fp16_pct": (
                100.0 * (1 - total_gate_bytes / bytes_if_all_fp16) if bytes_if_all_fp16 > 0 else 0.0
            ),
            "tau_lo": tau_lo, "tau_hi": tau_hi,
        }

    prefetch_precision = (n_predicted_correct / n_predicted_total) if policy in PREFETCH_POLICIES and n_predicted_total else None
    prefetch_recall = (n_predicted_correct / total_events) if policy in PREFETCH_POLICIES and total_events else None

    if concurrency > 1:
        from tierahead.sim.concurrency import aggregate_rho, max_sustainable_concurrency
        # Worst-case per-stream demand rate: every non-resident slot among
        # the k routed experts needing a fetch, every layer -- the same
        # conservative reference `none` itself experiences at this
        # residency. Using the worst case (not this policy's own achieved
        # bytes/token) means the flag is policy-independent and never gives
        # a falsely-reassuring "not yet link-bound" tag just because a good
        # predictor happened to reduce this run's realized bytes.
        #
        # concurrency_stable=True: streams are not yet ALWAYS simultaneously
        # link-bound at this concurrency -- the bw/N tpot_ms_mean above is a
        # CONSERVATIVE estimate (real TPOT is plausibly better).
        # concurrency_stable=False: the link is now the persistent
        # bottleneck for every stream -- this is exactly the regime where
        # bw/N's classical closed-network fair-share result is ACCURATE, not
        # just a proxy (see tierahead.sim.concurrency's module docstring for
        # the full reasoning, including a documented wrong turn this
        # framework took and corrected while testing this).
        bytes_per_stream_per_ms = ((1 - residency_pct / 100.0) * k * expert_bytes_fp16) / t_layer_ms
        agg_rho = aggregate_rho(bytes_per_stream_per_ms, bw_gbps, concurrency)
        extra["concurrency_stable"] = agg_rho < 1.0
        extra["aggregate_rho_at_concurrency"] = agg_rho
        extra["max_sustainable_concurrency"] = max_sustainable_concurrency(bytes_per_stream_per_ms, bw_gbps)

    return PolicyResult(
        policy=policy, model=spec.tag, residency_pct=residency_pct, bw_gbps=bw_gbps, precision=precision,
        batch=batch, concurrency=concurrency, n_test_reqs=int(pivot.index.get_level_values("req").nunique()),
        n_decode_tokens=len(tpot_samples),
        tpot_ms_mean=float(tpot_arr.mean()), tpot_ms_p50=float(np.percentile(tpot_arr, 50)),
        tpot_ms_p99=float(np.percentile(tpot_arr, 99)),
        demand_stall_rate=(stall_events / total_events) if total_events else 0.0,
        bytes_per_token_mean=float(np.mean(bytes_samples)) if bytes_samples else 0.0,
        compute_provenance=compute_prov, prefetch_precision=prefetch_precision, prefetch_recall=prefetch_recall,
        extra=extra,
    )


def _fit_mlp_predictor(decode_train: pd.DataFrame, n_experts: int, n_layers: int, depth: int, seed: int):
    """Trains on the 'ids' feature variant (selected expert IDs only, one-hot)
    -- deliberately NOT 'ids_gate_entropy' (E4's overall best-recall variant),
    because the per-token trace walk in simulate_policy only pivots the
    `topk` column (see its `pivot = decode_test.pivot_table(...)` call) and
    therefore never has gate weights/entropy available for `src_layer` at
    prediction time; training on a richer variant than the inference path
    can actually supply would be a real, if seemingly-passing-until-graded
    feature-shape mismatch. Depth/offline ablation studies that DO want the
    richer variant should use tierahead.policy.lookahead.direct_recall_for_depth
    directly (it has full feature access via features.build_layer_dataset)."""
    try:
        import tierahead.policy.features as feat
        import tierahead.policy.predictor_mlp as mlpmod
    except ImportError:
        return None
    train_reqs = set(decode_train["req"].unique())
    models = {}
    for layer in range(n_layers - depth):
        ds = feat.build_layer_dataset(decode_train, layer, depth, "ids", n_experts, reqs_filter=train_reqs)
        if ds is None or len(ds.X) < 50:
            continue
        mean = ds.X.mean(axis=0, keepdims=True)
        std = ds.X.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X_std = ((ds.X - mean) / std).astype(np.float32)
        model = mlpmod.train_predictor(X_std, ds.Y, n_experts, hidden=128, epochs=40, seed=seed)
        models[layer] = {"model": model, "mean": mean, "std": std}
    return {"models": models, "n_experts": n_experts} if models else None


def _mlp_features_for_layer(mlp_bundle, src_layer: int, src_actual: list[int], n_experts: int):
    """Builds the SAME shape 'ids' one-hot the model at this layer was
    trained on (see _fit_mlp_predictor's docstring for why not the richer
    variant)."""
    entry = mlp_bundle["models"].get(src_layer)
    if entry is None:
        return None, None
    x = np.zeros((1, n_experts), dtype=np.float32)
    x[0, src_actual] = 1.0
    x_std = ((x - entry["mean"]) / entry["std"]).astype(np.float32)
    return entry, x_std


def _mlp_topm(mlp_bundle, src_layer: int, src_actual: list[int], n_experts: int, m: int) -> list[int]:
    import tierahead.policy.predictor_mlp as mlpmod
    entry, x_std = _mlp_features_for_layer(mlp_bundle, src_layer, src_actual, n_experts)
    if entry is None:
        return []
    logits = mlpmod.predict_logits(entry["model"], x_std)[0]
    return np.argsort(-logits)[:m].tolist()


def _mlp_probs(mlp_bundle, src_layer: int, src_actual: list[int], n_experts: int):
    import tierahead.policy.predictor_mlp as mlpmod
    entry, x_std = _mlp_features_for_layer(mlp_bundle, src_layer, src_actual, n_experts)
    if entry is None:
        return None
    logits = mlpmod.predict_logits(entry["model"], x_std)[0]
    return mlpmod.sigmoid_probs(logits)


@lru_cache(maxsize=32)
def _load_and_split_decode_cached(trace_dir_str: str, seed: int):
    """Loading + decompressing `traces.jsonl.zst` and computing the
    request-level train/test split is pure and deterministic in
    (trace_dir, seed) alone -- but `run_policy_bakeoff` used to redo both on
    EVERY call, including every single grid point inside
    `run_concurrency_sweep` and every gate in `tierahead.sim.validation`. A
    4-level concurrency sweep across 4 concurrency-aware validation gates was
    reloading and re-splitting the same Mixtral trace file 20+ times over,
    which is most of why that test file went from ~1-2 minutes to ~14 minutes
    once the concurrency gates were added -- not because the underlying
    computation is inherently that expensive. Memoized here; callers get
    their own `.copy()` (see `_load_and_split_decode`) so nothing downstream
    can mutate the cached frames for anyone else."""
    df = load_traces(Path(trace_dir_str) / "traces.jsonl.zst")
    decode = df[df.phase == "decode"].reset_index(drop=True)
    train_reqs, test_reqs = train_test_split_reqs(df, seed=seed)
    decode_train = decode[decode.req.isin(train_reqs)].reset_index(drop=True)
    decode_test = decode[decode.req.isin(test_reqs)].reset_index(drop=True)
    return decode_train, decode_test


def _load_and_split_decode(trace_dir: Path, seed: int):
    decode_train, decode_test = _load_and_split_decode_cached(str(trace_dir), seed)
    return decode_train.copy(), decode_test.copy()


def run_policy_bakeoff(model_tag: str, trace_dir, policies: list[str] | None = None, residency_pct: float = 25.0,
                        bw_gbps: float = 64.0, precision: str = "fp16", batch: int = 1, calib: dict | None = None,
                        depth: int = 1, m_mult: int = 2, seed: int = 0, added_latency_ns: float = 0.0,
                        concurrency: int = 1) -> pd.DataFrame:
    """The E6 headline entry point: fits residency+predictors on the TRAIN
    split, evaluates every requested policy on the SAME held-out TEST split,
    and reports TPOT/demand-stall/bytes/precision/recall with
    oracle-gap-closed. This is what `tierahead sim` and the baseline-vs-CXL
    comparison call. `concurrency > 1` rescales the effective link bandwidth
    (fair-share approximation, see module docstring) to model N requests
    decoding simultaneously and contending for the same physical link."""
    from tierahead.hw import probe

    policies = policies or ALL_POLICIES
    spec = MODEL_SPECS[model_tag]
    decode_train, decode_test = _load_and_split_decode(Path(trace_dir), seed)

    mlp_available = probe().can_train_mlp_predictor
    rows = []
    for policy in policies:
        result = simulate_policy(
            decode_train=decode_train, decode_test=decode_test, spec=spec, policy=policy,
            residency_pct=residency_pct, bw_gbps=bw_gbps, precision=precision, batch=batch,
            calib=calib, depth=depth, m_mult=m_mult, seed=seed, mlp_available=mlp_available,
            added_latency_ns=added_latency_ns, concurrency=concurrency,
        )
        if result is None:
            continue
        rows.append(result)

    none_row = next((r for r in rows if r.policy == "none"), None)
    oracle_row = next((r for r in rows if r.policy == "oracle"), None)
    if none_row and oracle_row and oracle_row.tpot_ms_mean != none_row.tpot_ms_mean:
        denom = none_row.tpot_ms_mean - oracle_row.tpot_ms_mean
        for r in rows:
            r.oracle_gap_closed_pct = 100.0 * (none_row.tpot_ms_mean - r.tpot_ms_mean) / denom

    return pd.DataFrame([r.__dict__ for r in rows])


def run_concurrency_sweep(model_tag: str, trace_dir, concurrency_grid: list[int], residency_pct: float = 25.0,
                           bw_gbps: float = 128.0, precision: str = "fp16", depth: int = 1,
                           policies: list[str] | None = None, calib: dict | None = None, seed: int = 0) -> pd.DataFrame:
    """Answers the mentor-review question directly: does prefetch keep
    hiding latency as concurrent load increases, or does it collapse? Runs
    the full bake-off once per concurrency level and stacks the results so
    `oracle_gap_closed_pct` vs `concurrency` can be plotted per policy."""
    frames = []
    for c in concurrency_grid:
        df = run_policy_bakeoff(model_tag, trace_dir, policies=policies, residency_pct=residency_pct,
                                 bw_gbps=bw_gbps, precision=precision, depth=depth, calib=calib, seed=seed,
                                 concurrency=c)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
