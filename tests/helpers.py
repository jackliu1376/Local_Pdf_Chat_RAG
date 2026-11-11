import numpy as np

from core.knowledge_base import KnowledgeBase


class FakeFile:
    def __init__(self, path):
        self.name = str(path)


def fake_extractor(path):
    return open(path, encoding="utf-8").read()


def fake_splitter(text):
    return [text[i:i + 10] for i in range(0, len(text), 10)]


def fake_embedder(texts, show_progress=False):
    return np.array([[float(len(t)), 1.0] for t in texts], dtype="float32")


def make_kb(tmp_path, **kwargs):
    kwargs.setdefault("extractor", fake_extractor)
    kwargs.setdefault("splitter", fake_splitter)
    kwargs.setdefault("embedder", fake_embedder)
    return KnowledgeBase(tmp_path / "rag.db", **kwargs)
