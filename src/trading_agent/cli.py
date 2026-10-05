"""Command line entry point: trading-agent <command>."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import AppConfig, load_config
from .execution import CcxtExchange, PaperExchange
from .feeds import BlockClose, CcxtPollingFeed, CsvReplayFeed, SyntheticFeed
from .journal import Journal
from .loop import TradingLoop
from .portfolio import Portfolio
from .reflex import ReflexModel, ReflexParams, build_dataset, train_reflex
from .review import NightlyReview
from .risk import CapitalGate, KillSwitch, RiskGate
from .schema_store import SchemaStore
from .state_engine import AccountState, StateEngine

log = logging.getLogger("trading_agent")


def _ccxt_client(name: str, with_keys: bool):
    import ccxt  # optional dependency: pip install .[live]

    opts = {"enableRateLimit": True}
    if with_keys:
        opts.update(apiKey=os.environ.get("EXCHANGE_API_KEY"), secret=os.environ.get("EXCHANGE_API_SECRET"))
    return getattr(ccxt, name)(opts)


def _feed(cfg: AppConfig, args):
    if getattr(args, "live_data", None):
        return CcxtPollingFeed(_ccxt_client(args.live_data, with_keys=False), list(cfg.symbols), cfg.block_ms)
    if getattr(args, "csv", None):
        return CsvReplayFeed(args.csv, cfg.block_ms)
    return SyntheticFeed(list(cfg.symbols), cfg.block_ms, seed=args.seed, n_blocks=args.blocks)


def _load_theses(cfg: AppConfig) -> dict[str, str]:
    p = cfg.state_dir / "research" / "latest.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out = {}
    for t in data.get("candidates", []):
        for sym in cfg.symbols:
            if sym.split("/")[0].upper() == t["symbol"].split("/")[0].upper():
                out[sym] = f"{t['thesis']} | bear case: {t['bear_case']} | invalidation: {t['invalidation']}"
    return out


def cmd_train(cfg: AppConfig, args) -> int:
    store = SchemaStore(cfg.state_dir)
    engines = {s: StateEngine(s, cfg.observation) for s in cfg.symbols}
    snaps: dict[str, list] = {s: [] for s in cfg.symbols}
    for ev in _feed(cfg, args):
        if isinstance(ev, BlockClose):
            for s, e in engines.items():
                if e._book is not None:
                    snaps[s].append(e.close_block(ev.ts, AccountState()))
        elif ev.symbol in engines:
            engines[ev.symbol].on_event(ev)
    for s in cfg.symbols:
        X, y = build_dataset(snaps[s], cfg.review.label_horizon_blocks)
        params = store.save(train_reflex(s, X, y, base=store.load_current(s)))
        print(f"{s}: trained on {len(X)} samples -> schema v{params.version}")
    return 0


def _build_loop(cfg: AppConfig, args, exchange) -> TradingLoop:
    store = SchemaStore(cfg.state_dir)
    models = {}
    for s in cfg.symbols:
        params = store.load_current(s)
        if params is None:
            log.warning("%s has no trained schema; reflex stays neutral (run `trading-agent train`)", s)
            params = ReflexParams(symbol=s)
        models[s] = ReflexModel(params)
    kill = KillSwitch(cfg.state_dir)
    kill.arm()
    gate = RiskGate(cfg.risk, kill, CapitalGate(cfg.state_dir, cfg.mode))
    deliberator = None
    if args.tier1:
        from .deliberation import ClaudeDeliberator
        deliberator = ClaudeDeliberator(cfg=cfg.tier1)
    pf_path = cfg.state_dir / "portfolio.json"
    if getattr(args, "fresh", False) and pf_path.exists():
        pf_path.unlink()
    portfolio = Portfolio.load(pf_path, cfg.starting_equity)
    return TradingLoop(cfg, exchange, portfolio, gate, models,
                       Journal(cfg.state_dir, fsync=cfg.mode == "live"), deliberator=deliberator,
                       store=store, theses=_load_theses(cfg), portfolio_path=str(pf_path),
                       clock_ms=(lambda: int(datetime.now(timezone.utc).timestamp() * 1000))
                       if cfg.mode == "live" or getattr(args, "live_data", None) else None)


def _install_signal_kill(loop: TradingLoop) -> None:
    def handler(signum, _frame):
        log.warning("signal %s: stopping loop", signum)
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, handler)


def cmd_paper(cfg: AppConfig, args) -> int:
    if cfg.mode != "paper":
        print("config mode is not paper", file=sys.stderr)
        return 2
    loop = _build_loop(cfg, args, PaperExchange(cfg.execution))
    _install_signal_kill(loop)
    try:
        n = loop.run(_feed(cfg, args), max_blocks=None if args.live_data else args.blocks)
    except KeyboardInterrupt:
        n = -1
    finally:
        loop.shutdown()
    pf = loop.portfolio
    print(f"blocks={n} equity={pf.equity():.2f} drawdown={pf.drawdown():.4f} fees={pf.realized_fees:.2f} "
          f"positions={ {k: round(v, 6) for k, v in pf.positions.items()} }")
    return 0


def cmd_live(cfg: AppConfig, args) -> int:
    if cfg.mode != "live":
        print("refusing: config mode is not live", file=sys.stderr)
        return 2
    if CapitalGate(cfg.state_dir, "live").approved_capital() is None:
        print("refusing: no valid human capital approval (trading-agent approve-capital)", file=sys.stderr)
        return 2
    client = _ccxt_client(args.exchange, with_keys=True)
    loop = _build_loop(cfg, args, CcxtExchange(client))
    _install_signal_kill(loop)
    try:
        loop.run(CcxtPollingFeed(client, list(cfg.symbols), cfg.block_ms))
    except KeyboardInterrupt:
        pass
    finally:
        loop.shutdown()
    return 0


def cmd_review(cfg: AppConfig, args) -> int:
    day = args.date or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    narrator = None
    if args.narrate:
        from .deliberation import ClaudeDeliberator
        narrator = ClaudeDeliberator(cfg=cfg.tier1).narrate_review
    rep = NightlyReview(cfg, SchemaStore(cfg.state_dir), Journal(cfg.state_dir), narrator).run(day)
    for rv in rep.symbols.values():
        print(f"{rv.symbol}: n={rv.n} brier={rv.brier_all:.4f} {rv.note}")
    print(f"report: {rep.report_path}")
    return 0


def cmd_research(cfg: AppConfig, args) -> int:
    from .research import ClaudeResearcher, render_markdown

    report, errors = ClaudeResearcher(cfg=cfg.tier1).run()
    out = cfg.state_dir / "research"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{report.as_of}.md").write_text(render_markdown(report))
    (out / f"{report.as_of}.json").write_text(report.model_dump_json(indent=2))
    if errors:
        print("report REJECTED by sourcing validation; not promoted to latest:", *errors, sep="\n  ")
        return 1
    (out / "latest.json").write_text(report.model_dump_json(indent=2))
    print(f"research written: {out / (report.as_of + '.md')}")
    return 0


def cmd_kill(cfg: AppConfig, args) -> int:
    KillSwitch(cfg.state_dir).trip(args.reason)
    print("kill switch TRIPPED; loop flattens on next block")
    return 0


def cmd_reset_kill(cfg: AppConfig, args) -> int:
    KillSwitch(cfg.state_dir).reset(args.operator)
    print(f"kill switch reset by {args.operator}")
    return 0


def cmd_approve(cfg: AppConfig, args) -> int:
    p = CapitalGate.approve(cfg.state_dir, args.amount, args.operator, args.days)
    print(f"capital approval written: {p}")
    return 0


def cmd_status(cfg: AppConfig, args) -> int:
    ks = KillSwitch(cfg.state_dir)
    store = SchemaStore(cfg.state_dir)
    print(json.dumps({
        "mode": cfg.mode,
        "kill_tripped": ks.tripped(), "kill_reason": ks.reason(),
        "approved_capital": CapitalGate(cfg.state_dir, cfg.mode).approved_capital(),
        "schemas": {s: (p.version if (p := store.load_current(s)) else None) for s in cfg.symbols},
    }, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="trading-agent")
    ap.add_argument("--config", default="config/default.toml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("paper", "train"):
        p = sub.add_parser(name)
        p.add_argument("--blocks", type=int, default=3000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--csv")
        p.add_argument("--tier1", action="store_true", help="enable Claude deliberation on escalation")
        p.add_argument("--live-data", metavar="EXCHANGE", help="paper-trade on live public data from a ccxt exchange")
        p.add_argument("--fresh", action="store_true", help="discard persisted portfolio state")
    p = sub.add_parser("live")
    p.add_argument("--exchange", default="binance")
    p.add_argument("--tier1", action="store_true")
    p = sub.add_parser("review")
    p.add_argument("--date")
    p.add_argument("--narrate", action="store_true")
    sub.add_parser("research")
    p = sub.add_parser("kill")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("reset-kill")
    p.add_argument("--operator", required=True)
    p = sub.add_parser("approve-capital")
    p.add_argument("--amount", type=float, required=True)
    p.add_argument("--operator", required=True)
    p.add_argument("--days", type=float, default=1.0)
    sub.add_parser("status")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config) if Path(args.config).exists() else AppConfig()
    return {
        "paper": cmd_paper, "train": cmd_train, "live": cmd_live, "review": cmd_review,
        "research": cmd_research, "kill": cmd_kill, "reset-kill": cmd_reset_kill,
        "approve-capital": cmd_approve, "status": cmd_status,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
