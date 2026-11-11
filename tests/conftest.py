import pytest

from core.bm25_index import bm25_manager
from core.vector_store import vector_store


@pytest.fixture(autouse=True)
def clear_indexes():
    vector_store.clear()
    bm25_manager.clear()
    yield
    vector_store.clear()
    bm25_manager.clear()
