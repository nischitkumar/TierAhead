from tierahead.eunion import analytic_e_union_uniform, empirical_e_union


def test_analytic_e_union_monotone_in_batch():
    vals = [analytic_e_union_uniform(n_experts=64, k=8, batch=b) for b in [1, 2, 4, 8, 16, 32]]
    assert all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))


def test_analytic_e_union_bounded_by_n_experts():
    assert analytic_e_union_uniform(n_experts=8, k=2, batch=1000) <= 8.0 + 1e-9


def test_empirical_e_union_batch1_equals_mean_k():
    # batch=1: union of one row's k experts is exactly k, every trial.
    sets = [frozenset([0, 1]), frozenset([2, 3]), frozenset([0, 2])]
    est = empirical_e_union(sets, batch=1, n_trials=50, seed=0)
    assert est.mean == 2.0


def test_empirical_e_union_full_batch_is_full_union():
    sets = [frozenset([0, 1]), frozenset([2, 3])]
    est = empirical_e_union(sets, batch=2, n_trials=10, seed=0)
    assert est.mean == 4.0  # only one way to pick both rows without replacement
