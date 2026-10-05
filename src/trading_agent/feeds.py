"""Market data feeds. Every event is stamped with the time the information became available."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .state_engine import BookUpdate, Trade


@dataclass(frozen=True)
class BlockClose:
    ts: int


Event = BookUpdate | Trade | BlockClose

_DEFAULT_PX = {"BTC": 60_000.0, "ETH": 3_000.0, "SOL": 150.0}


class SyntheticFeed:
    """SYNTHETIC data for tests and paper dry-runs. Not a model of any real market.

    Latent per-symbol drift follows an AR(1); order flow and book imbalance leak a noisy
    view of it, so a classifier has something to learn. Volatility switches regimes.
    """

    def __init__(self, symbols: list[str], block_ms: int, seed: int = 0, n_blocks: int | None = None,
                 start_ms: int = 1_760_000_000_000, updates_per_block: int = 4, trades_per_block: int = 8) -> None:
        self.symbols = list(symbols)
        self.block_ms = block_ms
        self.seed = seed
        self.n_blocks = n_blocks
        self.start_ms = start_ms - start_ms % block_ms
        self.k = updates_per_block
        self.m = trades_per_block

    def __iter__(self) -> Iterator[Event]:
        rng = np.random.default_rng(self.seed)
        px = {s: _DEFAULT_PX.get(s.split("/")[0], 100.0) for s in self.symbols}
        mu = {s: 0.0 for s in self.symbols}
        hv = {s: False for s in self.symbols}
        i = 0
        while self.n_blocks is None or i < self.n_blocks:
            t0 = self.start_ms + i * self.block_ms
            events: list[tuple[int, int, Event]] = []
            for si, s in enumerate(self.symbols):
                hv[s] = (rng.random() < 0.95) if hv[s] else (rng.random() < 0.01)
                vol = 25.0 if hv[s] else 8.0
                mu[s] = 0.95 * mu[s] + rng.normal(0, 0.6)
                r_bps = mu[s] + vol * rng.normal()
                tilt = math.tanh(mu[s] / 2)
                for j in range(self.k):
                    px[s] *= math.exp(r_bps / self.k / 1e4)
                    spread = px[s] * rng.uniform(0.5, 2.0) * (3 if hv[s] else 1) / 1e4
                    bid, ask = px[s] - spread / 2, px[s] + spread / 2
                    bsz = math.exp(rng.normal(0, 0.3)) * (1 + 0.5 * tilt)
                    asz = math.exp(rng.normal(0, 0.3)) * (1 - 0.5 * tilt)
                    tick = spread / 2
                    bids = tuple((bid - n * tick, bsz * (1 + 0.5 * n)) for n in range(5))
                    asks = tuple((ask + n * tick, asz * (1 + 0.5 * n)) for n in range(5))
                    ts = t0 + int((j + 0.5) * self.block_ms / (self.k + 1))
                    events.append((ts, si, BookUpdate(ts, s, bids, asks)))
                for j in range(self.m):
                    ts = t0 + int((j + 1) * self.block_ms / (self.m + 2))
                    side = "buy" if rng.random() < 0.5 + 0.35 * tilt else "sell"
                    events.append((ts, si, Trade(ts, s, px[s], float(rng.exponential(1.0)), side)))
            events.sort(key=lambda e: (e[0], e[1]))
            for _, _, ev in events:
                yield ev
            yield BlockClose(t0 + self.block_ms - 1)
            i += 1


def with_block_closes(events: Iterable[BookUpdate | Trade], block_ms: int) -> Iterator[Event]:
    """Insert BlockClose markers whenever event time crosses a block boundary."""
    end: int | None = None
    for ev in events:
        if end is None:
            end = ev.ts - ev.ts % block_ms + block_ms
        while ev.ts >= end:
            yield BlockClose(end - 1)
            end += block_ms
        yield ev
    if end is not None:
        yield BlockClose(end - 1)


class CsvReplayFeed:
    """Replay CSV: ts,symbol,kind,bid_px,bid_sz,ask_px,ask_sz,price,size,side (kind = book|trade)."""

    def __init__(self, path: str | Path, block_ms: int) -> None:
        self.path = Path(path)
        self.block_ms = block_ms

    def _events(self) -> Iterator[BookUpdate | Trade]:
        with self.path.open() as f:
            for row in csv.DictReader(f):
                ts = int(row["ts"])
                if row["kind"] == "book":
                    yield BookUpdate(ts, row["symbol"], ((float(row["bid_px"]), float(row["bid_sz"])),),
                                     ((float(row["ask_px"]), float(row["ask_sz"])),))
                else:
                    yield Trade(ts, row["symbol"], float(row["price"]), float(row["size"]), row["side"])

    def __iter__(self) -> Iterator[Event]:
        return with_block_closes(self._events(), self.block_ms)


class CcxtPollingFeed:
    """Polls a ccxt client. Events are stamped with local receive time, which is when the
    information became available to us; venue timestamps are not trusted for causality."""

    def __init__(self, client: Any, symbols: list[str], block_ms: int, poll_s: float = 2.0, depth: int = 5) -> None:
        self.client, self.symbols, self.block_ms, self.poll_s, self.depth = client, symbols, block_ms, poll_s, depth

    def __iter__(self) -> Iterator[Event]:
        seen: dict[str, int] = {}
        last = 0
        end = None
        while True:
            now = max(int(time.time() * 1000), last)
            if end is None:
                end = now - now % self.block_ms + self.block_ms
            if now >= end:
                yield BlockClose(end - 1)
                end += self.block_ms
                continue
            for s in self.symbols:
                try:
                    ob = self.client.fetch_order_book(s, limit=self.depth)
                    trades = self.client.fetch_trades(s, since=seen.get(s))
                except Exception:
                    continue  # a missed poll leaves the snapshot stale; the risk gate catches that
                stamp = max(int(time.time() * 1000), last)
                if stamp >= end:
                    break
                last = stamp
                yield BookUpdate(stamp, s, tuple((float(p), float(q)) for p, q, *_ in ob["bids"][: self.depth]),
                                 tuple((float(p), float(q)) for p, q, *_ in ob["asks"][: self.depth]))
                for t in trades:
                    if t["timestamp"] > seen.get(s, 0):
                        yield Trade(stamp, s, float(t["price"]), float(t["amount"]), t["side"])
                if trades:
                    seen[s] = max(t["timestamp"] for t in trades)
            time.sleep(self.poll_s)
