from types import SimpleNamespace

import anthropic
import httpx2 as httpx

from trading_agent.deliberation import ClaudeDeliberator
from trading_agent.schema import Direction


class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None

    def parse(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp


def client(messages):
    return SimpleNamespace(beta=SimpleNamespace(messages=messages))


def snap():
    from tests.test_reflex import snapshots
    return snapshots(0, 40)


def test_valid_structured_verdict(decision_factory):
    d = decision_factory(direction=Direction.SHORT)
    msgs = FakeMessages(SimpleNamespace(stop_reason="end_turn", parsed_output=d))
    s = snap()
    out = ClaudeDeliberator(client=client(msgs)).deliberate(s[-1], s[-10:], decision_factory())
    assert out == d
    assert msgs.kwargs["model"] == "claude-opus-5-5"
    assert msgs.kwargs["thinking"] == {"type": "adaptive"}
    assert msgs.kwargs["fallbacks"] == "default"
    assert s[-1].render() in msgs.kwargs["messages"][0]["content"]


def test_refusal_means_no_verdict(decision_factory):
    msgs = FakeMessages(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    s = snap()
    assert ClaudeDeliberator(client=client(msgs)).deliberate(s[-1], s, decision_factory()) is None


def test_api_error_means_no_verdict(decision_factory):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    msgs = FakeMessages(exc=anthropic.APIConnectionError(request=req))
    s = snap()
    assert ClaudeDeliberator(client=client(msgs)).deliberate(s[-1], s, decision_factory()) is None
