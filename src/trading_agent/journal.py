"""Append-only tape: one JSON line per symbol per candle, fills and misses alike."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


class Journal:
    def __init__(self, state_dir: str | Path, fsync: bool = False) -> None:
        self.dir = Path(state_dir) / "journal"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync

    def path(self, date: str) -> Path:
        return self.dir / f"{date}.jsonl"

    def write(self, record: dict) -> None:
        with self.path(utc_date(int(record["ts"]))).open("a") as f:
            f.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
            if self.fsync:
                f.flush()
                os.fsync(f.fileno())

    def read(self, date: str) -> list[dict]:
        p = self.path(date)
        if not p.exists():
            return []
        out = []
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn last line after a crash
        return out

    def dates(self) -> list[str]:
        return sorted(p.stem for p in self.dir.glob("*.jsonl"))

    def read_all(self) -> list[dict]:
        return [r for d in self.dates() for r in self.read(d)]
