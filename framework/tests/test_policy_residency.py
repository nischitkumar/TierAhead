import pandas as pd

from tiermoe.policy.residency import FixedResidencyPolicy, ilp_upper_bound


def _counts(rows):
    return pd.DataFrame(rows, columns=["layer", "expert", "count"])


def test_fixed_residency_picks_top_by_train_count():
    train = _counts([(0, 0, 100), (0, 1, 50), (0, 2, 10), (0, 3, 1)])
    policy = FixedResidencyPolicy(residency_pct=50.0, n_experts=4).fit(train)
    assert policy.resident_set(0) == {0, 1}


def test_shared_experts_always_resident():
    train = _counts([(0, 0, 1), (0, 1, 1), (0, 2, 1000), (0, 3, 1000)])
    # experts 0,1 are "shared" (IDs [0, n_shared)) -- must be resident even
    # though they're the LEAST popular in this synthetic train data.
    policy = FixedResidencyPolicy(residency_pct=0.0, n_experts=4, n_shared_experts=2).fit(train)
    assert {0, 1}.issubset(policy.resident_set(0))


def test_coverage_is_one_at_full_residency():
    train = _counts([(0, 0, 5), (0, 1, 3)])
    test = _counts([(0, 0, 20), (0, 1, 10)])
    policy = FixedResidencyPolicy(residency_pct=100.0, n_experts=2).fit(train)
    assert policy.coverage(test) == 1.0


def test_ilp_upper_bound_is_never_worse_than_deployable():
    train = _counts([(0, 0, 100), (0, 1, 90), (0, 2, 5), (0, 3, 5)])
    test = _counts([(0, 0, 50), (0, 1, 5), (0, 2, 40), (0, 3, 5)])
    result = ilp_upper_bound(train, test, n_experts=4, residency_pct=25.0)
    assert result["oracle_coverage"] >= result["deployable_coverage"] - 1e-9
    assert result["headroom_pp"] >= -1e-6
