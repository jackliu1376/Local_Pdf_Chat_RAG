import core.embeddings as embeddings


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, show_progress_bar=False):
        self.calls.append(texts)
        return [[0.0, 0.0] for _ in texts]


def test_encode_query_applies_the_configured_prefix(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "get_embed_model", lambda: fake)
    monkeypatch.setattr(
        embeddings,
        "EMBED_QUERY_PREFIX",
        "为这个句子生成表示以用于检索相关文章：",
        raising=False,
    )

    embeddings.encode_query("什么是总线？")

    assert fake.calls == [["为这个句子生成表示以用于检索相关文章：什么是总线？"]]


def test_encode_texts_does_not_apply_the_query_prefix(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(embeddings, "get_embed_model", lambda: fake)
    monkeypatch.setattr(
        embeddings,
        "EMBED_QUERY_PREFIX",
        "为这个句子生成表示以用于检索相关文章：",
        raising=False,
    )

    embeddings.encode_texts(["文档一", "文档二"])

    assert fake.calls == [["文档一", "文档二"]]
