from types import SimpleNamespace
from orchestrator.reorder_logic import _build


def inv(sku="X-1", available=132, static_rop=60):
    return SimpleNamespace(sku_id=sku, item_name="Item", total_available=available,
                           reorder_point=static_rop, status="OK")


def test_computed_rop_uses_lead_time():
    # 150/month = 5/day, std 30/month = 1/day, 47-day lead time
    res = _build(inv(), {"X-1": (150.0, 30.0)}, {"X-1": 47.0})
    expected = 5 * 47 + 1.64 * 1 * (47 ** 0.5)
    assert res.source == "computed"
    assert abs(res.rop - expected) < 1e-6
    assert res.needs_reorder          # 132 < ~246


def test_falls_back_to_static_without_demand_history():
    res = _build(inv(available=50, static_rop=60), {}, {})
    assert res.source == "static" and res.rop == 60 and res.needs_reorder


def test_assumed_lead_time_is_flagged():
    res = _build(inv(), {"X-1": (150.0, 30.0)}, {})
    assert res.lead_time == 30.0 and res.lead_time_assumed