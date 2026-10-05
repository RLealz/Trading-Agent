from trading_agent.research import Catalyst, Metric, ResearchReport, SourceRef, Thesis, render_markdown, validate_report

SRC = SourceRef(url="https://example.org/doc", title="Primary doc", published="2026-10-01")


def thesis(sym, **kw):
    base = dict(
        symbol=sym, thesis="t", value_accrual="partial", value_accrual_reasoning="fee switch off",
        metrics=[Metric(name="fees_30d", value="1.0M USD", is_estimate=False, source=SRC)],
        catalysts=[Catalyst(description="unlock", date="2026-11-01", confirmed=True, source=SRC)],
        bear_case="b", invalidation="i",
    )
    base.update(kw)
    return Thesis(**base)


def report(n=5, **kw):
    return ResearchReport(as_of="2026-10-05", regime="trending", regime_reasoning="r",
                          regime_sources=[SRC], candidates=[thesis(f"T{i}", **kw) for i in range(n)])


def test_valid_report():
    assert validate_report(report()) == []


def test_candidate_count_bounds():
    assert validate_report(report(n=4))
    assert validate_report(report(n=11))


def test_unsourced_metric_rejected_unless_estimate():
    bad = report(metrics=[Metric(name="tvl", value="5B", is_estimate=False, source=None)])
    assert any("tvl" in e for e in validate_report(bad))
    ok = report(metrics=[Metric(name="tvl", value="~5B", is_estimate=True, source=None)])
    assert validate_report(ok) == []


def test_confirmed_catalyst_needs_source():
    bad = report(catalysts=[Catalyst(description="listing", date=None, confirmed=True, source=None)])
    assert validate_report(bad)


def test_falsifiable_thesis_required():
    assert validate_report(report(invalidation=" "))
    assert validate_report(report(bear_case=""))


def test_markdown_labels_estimates():
    md = render_markdown(report(metrics=[Metric(name="tvl", value="~5B", is_estimate=True, source=None)]))
    assert "(estimate)" in md and "Invalidation" in md
