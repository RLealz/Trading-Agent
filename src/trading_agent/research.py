"""Tier 1 discovery: regime, candidates, theses, bear cases, invalidation triggers.

Two calls: (1) research with live web search, (2) compile the findings into a typed report.
Then code validates sourcing: no un-sourced metric survives unless labelled an estimate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict

from .config import Tier1Config
from .deliberation import FALLBACK_BETA
from .schema import Regime

log = logging.getLogger(__name__)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(_Strict):
    url: str
    title: str
    published: str  # YYYY-MM-DD


class Metric(_Strict):
    name: str
    value: str
    is_estimate: bool
    source: SourceRef | None = None


class Catalyst(_Strict):
    description: str
    date: str | None = None
    confirmed: bool
    source: SourceRef | None = None


class Thesis(_Strict):
    symbol: str
    thesis: str
    value_accrual: Literal["accrues", "partial", "does_not_accrue", "unknown"]
    value_accrual_reasoning: str
    metrics: list[Metric]
    catalysts: list[Catalyst]
    bear_case: str
    invalidation: str


class ResearchReport(_Strict):
    as_of: str
    regime: Regime
    regime_reasoning: str
    regime_sources: list[SourceRef]
    candidates: list[Thesis]


def validate_report(r: ResearchReport) -> list[str]:
    errs = []
    if not 5 <= len(r.candidates) <= 10:
        errs.append(f"need 5-10 candidates, got {len(r.candidates)}")
    if not r.regime_sources:
        errs.append("regime call has no sources")
    for t in r.candidates:
        if not t.bear_case.strip():
            errs.append(f"{t.symbol}: missing bear case")
        if not t.invalidation.strip():
            errs.append(f"{t.symbol}: missing invalidation trigger")
        if not t.value_accrual_reasoning.strip():
            errs.append(f"{t.symbol}: value accrual not argued")
        for m in t.metrics:
            if not m.is_estimate and (m.source is None or not m.source.published.strip()):
                errs.append(f"{t.symbol}: metric '{m.name}' has no dated source and is not labelled an estimate")
        for c in t.catalysts:
            if c.confirmed and (c.source is None or not c.source.published.strip()):
                errs.append(f"{t.symbol}: confirmed catalyst '{c.description}' has no dated source")
    return errs


def _src(s: SourceRef | None) -> str:
    return f"[{s.title}]({s.url}), {s.published}" if s else "no source"


def render_markdown(r: ResearchReport) -> str:
    out = [f"# Research — as of {r.as_of}", "", f"**Regime:** {r.regime.value}. {r.regime_reasoning}", ""]
    out += [f"- {_src(s)}" for s in r.regime_sources] + [""]
    for t in r.candidates:
        out += [f"## {t.symbol}", "", t.thesis, "",
                f"**Value accrual to the token holder:** {t.value_accrual} — {t.value_accrual_reasoning}", "",
                "| Metric | Value | Source |", "|---|---|---|"]
        out += [f"| {m.name} | {m.value}{' (estimate)' if m.is_estimate else ''} | {_src(m.source)} |" for m in t.metrics]
        out += ["", "**Catalysts**", ""]
        out += [f"- {'CONFIRMED' if c.confirmed else 'speculative'}: {c.description}"
                f"{f' ({c.date})' if c.date else ''} — {_src(c.source)}" for c in t.catalysts]
        out += ["", f"**Bear case:** {t.bear_case}", "", f"**Invalidation:** {t.invalidation}", ""]
    return "\n".join(out)


RESEARCH_PROMPT = """Today is {today}. Establish the current crypto market regime from BTC and ETH structure, dominance,
stablecoin supply, funding, open interest and the narrative currently rotating. Then surface 5-10 candidates where price
has detached from fundamentals, adoption, or a dated catalyst.

Per candidate: emission schedule, unlock calendar, revenue, fee capture, TVL, real active users, holder concentration,
and the question most people skip: does any of that value reach the token holder, or does it stop at the protocol?

Separate confirmed catalysts from speculation. Write the strongest bear case you can for every thesis, then state the
single observation that would kill it.

Cite live primary sources with publication dates. Never invent a metric. Label every estimate as an estimate."""


class ClaudeResearcher:
    def __init__(self, client: Any | None = None, cfg: Tier1Config | None = None, max_searches: int = 30) -> None:
        self.cfg = cfg or Tier1Config()
        self.client = client or anthropic.Anthropic(timeout=self.cfg.timeout_s)
        self.max_searches = max_searches

    def _research(self, today: str) -> str:
        messages: list[dict] = [{"role": "user", "content": RESEARCH_PROMPT.format(today=today)}]
        tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": self.max_searches}]
        for _ in range(8):
            with self.client.messages.stream(
                model=self.cfg.model, max_tokens=64000, messages=messages, tools=tools,
                thinking={"type": "adaptive"}, output_config={"effort": self.cfg.effort},
            ) as stream:
                resp = stream.get_final_message()
            if resp.stop_reason == "refusal":
                raise RuntimeError("research request refused")
            if resp.stop_reason != "pause_turn":
                return "".join(b.text for b in resp.content if b.type == "text")
            messages.append({"role": "assistant", "content": resp.content})
        raise RuntimeError("research did not finish")

    def run(self) -> tuple[ResearchReport, list[str]]:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        notes = self._research(today)
        resp = self.client.beta.messages.parse(
            model=self.cfg.model, max_tokens=16000,
            system="Compile research notes into the typed report. Copy sources and dates exactly as given. "
                   "Any number without a dated source in the notes must be marked is_estimate=true. Do not add facts.",
            messages=[{"role": "user", "content": f"as_of={today}\n\n{notes}"}],
            output_format=ResearchReport,
            thinking={"type": "adaptive"}, output_config={"effort": "medium"},
            betas=[FALLBACK_BETA], fallbacks="default",
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError("research compile step returned no report")
        report = resp.parsed_output
        return report, validate_report(report)
