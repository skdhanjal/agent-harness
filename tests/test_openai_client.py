"""Verification test for the OpenAI adapter's usage/cost capture.

No real API call: `OpenAI.chat.completions.create` is monkeypatched with a
fake response shaped like the SDK's own, so this proves `create_action_turn`
reads `response.usage` and prices it correctly -- not that the SDK works.
"""

from types import SimpleNamespace

import httpx2
import pytest
from openai import AuthenticationError, RateLimitError

from agent_harness.persistence.retry import FatalError
from agent_harness.schemas.openai_client import OpenAIChatClient


def _openai_error(error_cls: type[Exception], message: str, status_code: int) -> Exception:
    response = httpx2.Response(
        status_code=status_code,
        request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"),
    )
    return error_cls(message, response=response, body=None)


def _fake_response(prompt_tokens: int, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_create_action_turn_captures_tokens_and_computes_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIChatClient(model="gpt-4o-mini")
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kwargs: _fake_response(prompt_tokens=1000, completion_tokens=1000),
    )

    turn = client.create_action_turn(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert turn.tokens_in == 1000
    assert turn.tokens_out == 1000
    # gpt-4o-mini: $0.15/1M in, $0.60/1M out -> 1000 tokens of each = $0.00075
    assert turn.cost_usd == pytest.approx(0.00075)


def test_create_action_turn_falls_back_to_default_pricing_for_unknown_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIChatClient(model="some-future-model")
    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        lambda **kwargs: _fake_response(prompt_tokens=1_000_000, completion_tokens=0),
    )

    turn = client.create_action_turn(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert turn.cost_usd == pytest.approx(0.15)  # gpt-4o-mini's input rate, as the fallback


def test_create_action_turn_handles_missing_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIChatClient(model="gpt-4o-mini")
    response = _fake_response(prompt_tokens=0, completion_tokens=0)
    response.usage = None
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kwargs: response)

    turn = client.create_action_turn(messages=[{"role": "user", "content": "hi"}], tools=[])

    assert turn.tokens_in == 0
    assert turn.tokens_out == 0
    assert turn.cost_usd == 0.0


def test_create_action_turn_reraises_auth_error_as_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIChatClient(model="gpt-4o-mini")

    def _raise(**kwargs: object) -> None:
        raise _openai_error(AuthenticationError, "invalid api key", 401)

    monkeypatch.setattr(client._client.chat.completions, "create", _raise)

    with pytest.raises(FatalError):
        client.create_action_turn(messages=[{"role": "user", "content": "hi"}], tools=[])


def test_create_action_turn_still_retries_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """RateLimitError must NOT become FatalError -- it's exactly what backoff is for."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIChatClient(model="gpt-4o-mini")

    def _raise(**kwargs: object) -> None:
        raise _openai_error(RateLimitError, "rate limited", 429)

    monkeypatch.setattr(client._client.chat.completions, "create", _raise)

    with pytest.raises(RateLimitError):
        client.create_action_turn(messages=[{"role": "user", "content": "hi"}], tools=[])
