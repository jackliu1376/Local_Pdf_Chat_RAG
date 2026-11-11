import core.reranker as reranker


class FakeCrossEncoder:
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device


def test_get_cross_encoder_loads_the_configured_reranker_model(monkeypatch):
    monkeypatch.setattr("sentence_transformers.CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(reranker, "_cross_encoder", None)

    encoder = reranker.get_cross_encoder()

    assert encoder is not None
    assert encoder.model_name == "BAAI/bge-reranker-base"


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"response": "7"}


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse()


def test_llm_relevance_score_disables_thinking(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("utils.network.get_session", lambda: session)
    reranker.get_llm_relevance_score.cache_clear()

    score = reranker.get_llm_relevance_score("query", "doc")

    assert score == 7.0
    _, kwargs = session.calls[0]
    assert kwargs["json"]["think"] is False
    reranker.get_llm_relevance_score.cache_clear()
