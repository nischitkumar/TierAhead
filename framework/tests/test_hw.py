from tiermoe.hw import probe
from tiermoe.sim.backends import select_backend


def test_probe_returns_capabilities():
    caps = probe()
    assert caps.platform in ("Darwin", "Linux", "Windows")
    assert isinstance(caps.cuda_available, bool)


def test_select_backend_auto_never_picks_cxlmemsim_without_support():
    name, _note = select_backend("auto")
    caps = probe()
    if not caps.can_run_cxlmemsim_backend:
        assert name in ("des", "analytical")


def test_select_backend_explicit_cxlmemsim_raises_when_unavailable():
    caps = probe()
    if caps.can_run_cxlmemsim_backend:
        return  # this machine actually has it -- nothing to assert here
    try:
        select_backend("cxlmemsim")
        assert False
    except RuntimeError:
        pass


def test_select_backend_unknown_name_raises():
    try:
        select_backend("not-a-real-backend")
        assert False
    except ValueError:
        pass
