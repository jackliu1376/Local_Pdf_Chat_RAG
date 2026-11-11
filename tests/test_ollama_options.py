from types import SimpleNamespace

import core.generator as generator


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "ok"}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


def test_query_answer_limits_ollama_output_and_uses_configured_timeout(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(generator, "get_session", lambda: session)
    monkeypatch.setattr(generator, "vector_store", SimpleNamespace(is_ready=True))
    monkeypatch.setattr(
        generator,
        "recursive_retrieval",
        lambda **kwargs: (["context"], ["doc-1"], [{"source": "local.pdf"}]),
    )

    answer = generator.query_answer("问题", model_choice="ollama")

    assert answer == "ok"
    _, kwargs = session.calls[0]
    assert kwargs["json"]["options"]["num_predict"] == 1024
    assert kwargs["timeout"] == 300


def test_call_llm_simple_limits_query_rewrite_output(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(generator, "get_session", lambda: session)

    result = generator.call_llm_simple("prompt", "ollama")

    assert result == "ok"
    _, kwargs = session.calls[0]
    assert kwargs["json"]["options"]["num_predict"] == 64
    assert kwargs["timeout"] == 300


class ThinkingResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "<think>reasoning</think>查询"}


class ThinkingSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return ThinkingResponse()


def test_call_llm_simple_disables_thinking_and_strips_think(monkeypatch):
    session = ThinkingSession()
    monkeypatch.setattr(generator, "get_session", lambda: session)

    result = generator.call_llm_simple("prompt", "ollama")

    assert result == "查询"
    _, kwargs = session.calls[0]
    assert kwargs["json"]["think"] is False
