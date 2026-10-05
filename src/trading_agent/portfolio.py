"""Cash, positions and the equity series the risk gate reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .execution import Fill

_PERSISTED = ("cash", "positions", "marks", "peak_equity", "day_start_equity", "session_date", "realized_fees")


class Portfolio:
    def __init__(self, starting_cash: float) -> None:
        self.cash = float(starting_cash)
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.peak_equity = float(starting_cash)
        self.day_start_equity = float(starting_cash)
        self.session_date: str | None = None
        self.realized_fees = 0.0

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def notional(self, symbol: str) -> float:
        return self.position(symbol) * self.marks.get(symbol, 0.0)

    def gross_notional(self, exclude: str | None = None) -> float:
        return sum(abs(q * self.marks.get(s, 0.0)) for s, q in self.positions.items() if s != exclude)

    def equity(self) -> float:
        return self.cash + sum(q * self.marks.get(s, 0.0) for s, q in self.positions.items())

    def _update_peak(self) -> None:
        self.peak_equity = max(self.peak_equity, self.equity())

    def mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price
        self._update_peak()

    def apply_fill(self, fill: Fill) -> None:
        self.cash -= fill.qty * fill.price + fill.fee
        self.realized_fees += fill.fee
        q = self.position(fill.symbol) + fill.qty
        if abs(q) < 1e-12:
            self.positions.pop(fill.symbol, None)
        else:
            self.positions[fill.symbol] = q
        self.marks.setdefault(fill.symbol, fill.price)
        self._update_peak()

    def drawdown(self) -> float:
        return max(0.0, 1.0 - self.equity() / self.peak_equity) if self.peak_equity > 0 else 1.0

    def daily_pnl(self) -> float:
        return self.equity() / self.day_start_equity - 1.0 if self.day_start_equity > 0 else -1.0

    def daily_loss(self) -> float:
        return max(0.0, -self.daily_pnl())

    def roll_session(self, date: str) -> bool:
        """Start a new UTC session. Returns True when the date actually changed."""
        if self.session_date is None:
            self.session_date = date
            return False
        if date == self.session_date:
            return False
        self.session_date = date
        self.day_start_equity = self.equity()
        return True

    # Persisted so a restart cannot reset the drawdown or daily-loss clock.
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: getattr(self, k) for k in _PERSISTED}))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, starting_cash: float) -> "Portfolio":
        pf = cls(starting_cash)
        path = Path(path)
        if path.exists():
            for k, v in json.loads(path.read_text()).items():
                if k in _PERSISTED:
                    setattr(pf, k, v)
        return pf
