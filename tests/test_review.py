import numpy as np
import pytest

from trading_agent.config import AppConfig, ReviewConfig
from trading_agent.journal import Journal
from trading_agent.reflex import ReflexParams
from trading_agent.review import NightlyReview, brier, label_records
from trading_agent.schema_store import SchemaStore

SYM = "BTC/USDT"
DAY_MS = 86_400_000


def test_brier():
    assert brier(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert brier(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == 0.25


def test_label_records_uses_future_mid_per_symbol():
    recs = [{"symbol": SYM, "block": i, "snapshot": {"mid": float(m)}} for i, m in enumerate([1, 2, 1, 3])]
    labelled = label_records(recs, horizon=1)
    assert [y for _, y in labelled] == [1.0, 0.0, 1.0]


def _write_tape(journal, n, scale, seed, day0=0):
    """Overconfident model: raw score = scale * true logit."""
    rng = np.random.default_rng(seed)
    mid = 100.0
    for i in range(n):
        true_logit = rng.normal(0, 1)
        score = scale * true_logit
        p = 1 / (1 + np.exp(-score))
        up = rng.random() < 1 / (1 + np.exp(-true_logit))
        journal.write({
            "ts": day0 + i * 60_000, "symbol": SYM, "block": i, "score": score, "p_up": p,
            "source": "reflex", "fired": False, "regime": "trending", "direction": "long",
            "snapshot": {"mid": mid},
        })
        # next mid encodes the outcome of *this* record at horizon 1
        mid = mid * (1.001 if up else 0.999)


def test_review_promotes_better_calibration(tmp_path):
    cfg = AppConfig(state_dir=tmp_path, symbols=(SYM,),
                    review=ReviewConfig(label_horizon_blocks=1, min_samples=200))
    store = SchemaStore(tmp_path)
    store.save(ReflexParams(symbol=SYM, weights=[0.0] * 7))
    j = Journal(tmp_path)
    _write_tape(j, 3000, scale=4.0, seed=0)
    rep = NightlyReview(cfg, store, j).run("1970-01-01")
    r = rep.symbols[SYM]
    assert r.promoted
    new = store.load_current(SYM)
    assert new.version == 2
    assert 0.15 < new.platt_a < 0.4
    assert r.brier_oos_new < r.brier_oos_old
    assert rep.report_path.exists()
    assert "Brier" in rep.report_path.read_text()


def test_review_does_not_promote_when_calibrated(tmp_path):
    cfg = AppConfig(state_dir=tmp_path, symbols=(SYM,),
                    review=ReviewConfig(label_horizon_blocks=1, min_samples=200))
    store = SchemaStore(tmp_path)
    store.save(ReflexParams(symbol=SYM, weights=[0.0] * 7))
    j = Journal(tmp_path)
    _write_tape(j, 3000, scale=1.0, seed=1)
    r = NightlyReview(cfg, store, j).run("1970-01-01").symbols[SYM]
    assert not r.promoted
    assert store.load_current(SYM).version == 1


def test_review_skips_thin_tape(tmp_path):
    cfg = AppConfig(state_dir=tmp_path, symbols=(SYM,), review=ReviewConfig(min_samples=500))
    store = SchemaStore(tmp_path)
    store.save(ReflexParams(symbol=SYM))
    j = Journal(tmp_path)
    _write_tape(j, 100, scale=4.0, seed=0)
    assert not NightlyReview(cfg, store, j).run("1970-01-01").symbols[SYM].promoted


def test_schema_store_atomic_versions(tmp_path):
    store = SchemaStore(tmp_path)
    assert store.load_current(SYM) is None
    assert store.save(ReflexParams(symbol=SYM)).version == 1
    assert store.save(ReflexParams(symbol=SYM, bias=0.3)).version == 2
    assert store.load_current(SYM).bias == 0.3
    assert store.load_version(SYM, 1).bias == 0.0
