"""Nightly review: read the whole tape, grade calibration, ship a better schema before the next open.

Promotion rule: a new schema is written only if it is measurably better out-of-sample
(chronological split) or strictly more conservative (regime disabled). Hard risk limits
are not reachable from here.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from datetime import date as Date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from .config import AppConfig
from .journal import Journal, utc_date
from .reflex import ReflexParams, fit_platt
from .schema import Regime
from .schema_store import SchemaStore

log = logging.getLogger(__name__)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[tuple[float, float, int, float, float]]:
    out = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append((float(lo), float(hi), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return out


def _labelled(records: list[dict], horizon: int, block_ms: int | None = None) -> list[tuple[dict, float, float]]:
    """(record, y_up, forward log return in bps) using the same symbol's mid `horizon` records later.

    With block_ms, pairs spanning a gap (restart, outage) are dropped instead of mislabelled."""
    by_sym: dict[str, list[dict]] = {}
    for r in records:
        by_sym.setdefault(r["symbol"], []).append(r)
    out = []
    for recs in by_sym.values():
        recs.sort(key=lambda r: (r.get("ts", 0), r["block"]))
        for i in range(len(recs) - horizon):
            if block_ms and recs[i + horizon]["ts"] - recs[i]["ts"] > horizon * block_ms * 1.5:
                continue
            m0, m1 = recs[i]["snapshot"]["mid"], recs[i + horizon]["snapshot"]["mid"]
            out.append((recs[i], 1.0 if m1 > m0 else 0.0, math.log(m1 / m0) * 1e4))
    return out


def label_records(records: list[dict], horizon: int, block_ms: int | None = None) -> list[tuple[dict, float]]:
    return [(r, y) for r, y, _ in _labelled(records, horizon, block_ms)]


PROMOTION_T = 2.33  # one-sided ~99%: a refit must beat the incumbent beyond noise


def paired_improvement_t(p_old: np.ndarray, p_new: np.ndarray, y: np.ndarray) -> float:
    """t-statistic of the per-candle reduction in squared error (positive = new is better)."""
    d = (p_old - y) ** 2 - (p_new - y) ** 2
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    if sd == 0:
        return 0.0 if d.mean() <= 0 else float("inf")
    return float(d.mean() / (sd / math.sqrt(len(d))))


def _sig(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class SymbolReview:
    symbol: str
    n: int
    promoted: bool = False
    note: str = ""
    old_version: int | None = None
    new_version: int | None = None
    brier_all: float = float("nan")
    brier_fired: float = float("nan")
    n_fired: int = 0
    brier_climatology: float = float("nan")
    brier_oos_old: float = float("nan")
    brier_oos_new: float = float("nan")
    improvement_t: float = float("nan")
    log_loss_all: float = float("nan")
    platt: tuple[float, float] | None = None
    payoff_ratio_est: float | None = None
    disabled_regimes: list[str] = field(default_factory=list)
    tier1_n: int = 0
    tier1_shrink: float | None = None
    missed_right: int = 0
    missed_blocked_confident: int = 0
    reliability: list = field(default_factory=list)


@dataclass
class ReviewReport:
    date: str
    symbols: dict[str, SymbolReview]
    report_path: Path


class NightlyReview:
    def __init__(self, cfg: AppConfig, store: SchemaStore, journal: Journal,
                 narrator: Callable[[str], str | None] | None = None) -> None:
        self.cfg, self.store, self.journal, self.narrator = cfg, store, journal, narrator

    def _load(self, day: str) -> list[dict]:
        d = Date.fromisoformat(day)
        lo = (d - timedelta(days=self.cfg.review.history_days - 1)).isoformat()
        hi = (d + timedelta(days=1)).isoformat()  # next day only to label the session's tail
        return [r for ds in self.journal.dates() if lo <= ds <= hi for r in self.journal.read(ds)]

    def run(self, day: str) -> ReviewReport:
        rc = self.cfg.review
        labelled = [(r, y, f) for r, y, f in _labelled(self._load(day), rc.label_horizon_blocks, self.cfg.block_ms)
                    if utc_date(int(r["ts"])) <= day and r.get("snapshot", {}).get("warm", True)]
        results = {}
        for sym in self.cfg.symbols:
            rows = [x for x in labelled if x[0]["symbol"] == sym]
            results[sym] = self._review_symbol(sym, rows)
        path = self._write_report(day, results)
        return ReviewReport(day, results, path)

    def _review_symbol(self, sym: str, rows: list[tuple[dict, float, float]]) -> SymbolReview:
        rc, fee = self.cfg.review, self.cfg.execution.fee_bps
        cur = self.store.load_current(sym) or ReflexParams(symbol=sym)
        rv = SymbolReview(sym, len(rows), old_version=cur.version)
        if not rows:
            rv.note = "no labelled records"
            return rv

        recs = [r for r, _, _ in rows]
        y = np.array([v for _, v, _ in rows])
        fwd = np.array([f for _, _, f in rows])
        p = np.array([r["p_up"] for r in recs])
        s = np.array([r["score"] for r in recs])
        fired = np.array([bool(r.get("fired")) for r in recs])

        rv.brier_all, rv.log_loss_all = brier(p, y), log_loss(p, y)
        rv.n_fired = int(fired.sum())
        rv.brier_fired = brier(p[fired], y[fired]) if fired.any() else float("nan")
        rv.reliability = reliability(p, y)

        # every miss: directional calls we did not take, and how many of them were right
        dirs = np.array([{"long": 1, "short": -1}.get(r.get("direction"), 0) for r in recs])
        right = ((dirs == 1) & (y == 1)) | ((dirs == -1) & (y == 0))
        missed = (~fired) & (dirs != 0)
        rv.missed_right = int((missed & right).sum())
        conf = np.maximum(p, 1 - p)
        rv.missed_blocked_confident = int((missed & (conf > self.cfg.policy.fire_confidence)).sum())

        if len(rows) < rc.min_samples:
            rv.note = f"thin tape ({len(rows)} < {rc.min_samples}); schema unchanged"
            return rv

        cut = int(len(rows) * (1 - rc.oos_fraction))
        base_rate = float(y[:cut].mean())
        rv.brier_climatology = brier(np.full(len(y) - cut, base_rate), y[cut:])
        a, b = fit_platt(s[:cut], y[:cut])
        p_old = _sig(cur.platt_a * s[cut:] + cur.platt_b)
        p_new = _sig(a * s[cut:] + b)
        rv.brier_oos_old, rv.brier_oos_new = brier(p_old, y[cut:]), brier(p_new, y[cut:])
        rv.improvement_t = paired_improvement_t(p_old, p_new, y[cut:])
        improved = rv.improvement_t > PROMOTION_T

        cand = dataclasses.replace(cur, disabled_regimes=list(cur.disabled_regimes))
        p_used = p_new if improved else p_old
        if improved:
            cand.platt_a, cand.platt_b = a, b
            rv.platt = (a, b)

        # regime gating: a regime whose calibrated Brier is worse than climatology does not trade
        regimes = np.array([r.get("regime") for r in recs[cut:]])
        min_n = max(30, rc.min_samples // 10)
        for reg in Regime:
            m = regimes == reg.value
            if m.sum() < min_n:
                continue
            b_reg = brier(p_used[m], y[cut:][m])
            clim = brier(np.full(m.sum(), base_rate), y[cut:][m])
            if b_reg > clim + 2e-3 and reg.value not in cand.disabled_regimes:
                cand.disabled_regimes.append(reg.value)
            elif b_reg < clim and reg.value in cand.disabled_regimes:
                cand.disabled_regimes.remove(reg.value)
        rv.disabled_regimes = list(cand.disabled_regimes)

        # payoff ratio: realised forward move in the called direction, net of round-trip fees (estimate)
        d_ret = (fwd * dirs)[dirs != 0] - 2 * fee
        wins, losses = d_ret[d_ret > 0], -d_ret[d_ret < 0]
        if len(wins) >= 20 and len(losses) >= 20:
            rv.payoff_ratio_est = float(np.clip(wins.mean() / losses.mean(), 0.5, 2.0))

        # Tier 1 calibration: shrink unproven confidence toward 0.5
        t1 = [(r["tier1_raw_confidence"], float(rt)) for r, rt in zip(recs, right)
              if r.get("source") == "tier1" and r.get("tier1_raw_confidence") is not None and r.get("direction") != "neutral"]
        rv.tier1_n = len(t1)
        if len(t1) >= 30:
            c = np.array([x for x, _ in t1])
            yt = np.array([v for _, v in t1])
            grid = np.linspace(0, 1, 21)
            best = float(grid[int(np.argmin([brier(0.5 + g * (c - 0.5), yt) for g in grid]))])
            if abs(best - cand.tier1_shrink) > 1e-9:
                cand.tier1_shrink = best
                rv.tier1_shrink = best

        changed = improved or set(cand.disabled_regimes) != set(cur.disabled_regimes) or rv.tier1_shrink is not None
        if changed:
            if rv.payoff_ratio_est is not None:
                cand.payoff_ratio = rv.payoff_ratio_est
            saved = self.store.save(cand)
            rv.promoted, rv.new_version = True, saved.version
            rv.note = "promoted"
        else:
            rv.note = "no measurable improvement; schema unchanged"
        return rv

    def _write_report(self, day: str, results: dict[str, SymbolReview]) -> Path:
        def f(x):
            return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

        lines = [f"# Nightly review — {day}", ""]
        for rv in results.values():
            lines += [
                f"## {rv.symbol}", "",
                f"- labelled candles: {rv.n} · fired: {rv.n_fired} · outcome: **{rv.note}**",
                f"- schema: v{rv.old_version} → " + (f"v{rv.new_version}" if rv.promoted else "unchanged"),
                f"- Brier (all candles): {f(rv.brier_all)} · Brier (fills only): {f(rv.brier_fired)} · log loss: {f(rv.log_loss_all)}",
                f"- OOS Brier: current {f(rv.brier_oos_old)} · refit {f(rv.brier_oos_new)} · climatology {f(rv.brier_climatology)} "
                f"· paired t {f(rv.improvement_t)} (promote > {PROMOTION_T})",
                f"- misses: {rv.missed_right} directional calls not taken were right; "
                f"{rv.missed_blocked_confident} were above fire confidence but blocked by quality/risk/toxicity",
                f"- payoff ratio (estimate, net of fees): {f(rv.payoff_ratio_est)} · disabled regimes: {rv.disabled_regimes or 'none'}",
                f"- Tier 1 verdicts graded: {rv.tier1_n} · shrink → {f(rv.tier1_shrink)}",
                "", "| p bin | n | mean p | realised |", "|---|---|---|---|",
            ]
            lines += [f"| {lo:.1f}–{hi:.1f} | {n} | {mp:.3f} | {fr:.3f} |" for lo, hi, n, mp, fr in rv.reliability]
            lines.append("")
        text = "\n".join(lines)
        if self.narrator:
            try:
                narrative = self.narrator(text)
                if narrative:
                    text += "\n## Tier 1 commentary (advisory only; changes nothing)\n\n" + narrative + "\n"
            except Exception:
                log.exception("narrator failed; report written without commentary")
        out = Path(self.cfg.state_dir) / "reports"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{day}.md"
        path.write_text(text)
        return path
