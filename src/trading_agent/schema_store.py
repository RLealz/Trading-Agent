"""Versioned reflex parameters per symbol with an atomic `current` pointer."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from .reflex import ReflexParams


class SchemaStore:
    def __init__(self, state_dir: str | Path) -> None:
        self.root = Path(state_dir) / "schemas"

    def _dir(self, symbol: str) -> Path:
        return self.root / symbol.replace("/", "_")

    def versions(self, symbol: str) -> list[int]:
        d = self._dir(symbol)
        return sorted(int(p.stem[1:]) for p in d.glob("v*.json")) if d.exists() else []

    def save(self, params: ReflexParams) -> ReflexParams:
        d = self._dir(params.symbol)
        d.mkdir(parents=True, exist_ok=True)
        version = (self.versions(params.symbol) or [0])[-1] + 1
        saved = dataclasses.replace(params, version=version,
                                    weights=list(params.weights), mean=list(params.mean),
                                    std=list(params.std), disabled_regimes=list(params.disabled_regimes))
        (d / f"v{version}.json").write_text(json.dumps(saved.to_dict(), indent=2))
        tmp = d / "current.json.tmp"
        tmp.write_text(json.dumps({"version": version}))
        os.replace(tmp, d / "current.json")
        return saved

    def load_version(self, symbol: str, version: int) -> ReflexParams:
        return ReflexParams.from_dict(json.loads((self._dir(symbol) / f"v{version}.json").read_text()))

    def current_version(self, symbol: str) -> int | None:
        ptr = self._dir(symbol) / "current.json"
        return json.loads(ptr.read_text())["version"] if ptr.exists() else None

    def load_current(self, symbol: str) -> ReflexParams | None:
        v = self.current_version(symbol)
        return None if v is None else self.load_version(symbol, v)
