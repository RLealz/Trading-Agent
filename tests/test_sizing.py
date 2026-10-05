import pytest

from trading_agent.sizing import QUARTER_KELLY_CAP, kelly, position_fraction


def test_kelly_formula():
    assert kelly(0.6, 1.0) == pytest.approx(0.2)
    assert kelly(0.5, 1.0) == 0.0
    assert kelly(0.4, 1.0) == 0.0  # never negative
    assert kelly(0.5, 2.0) == pytest.approx(0.25)


def test_quarter_kelly_hard_cap():
    assert QUARTER_KELLY_CAP == 0.25
    assert position_fraction(0.9, 1.0, multiplier=1.0) == pytest.approx(0.25 * 0.8)
    assert position_fraction(0.9, 1.0, multiplier=0.1) == pytest.approx(0.1 * 0.8)


@pytest.mark.parametrize("p,b", [(float("nan"), 1.0), (0.9, 0.0), (0.9, -1.0), (1.5, 1.0)])
def test_bad_inputs_size_zero(p, b):
    assert position_fraction(p, b, 0.25) == 0.0
