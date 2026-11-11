from fastapi.testclient import TestClient

import api_router
from tests.helpers import FakeFile, make_kb


def test_documents_endpoint_lists_persisted_documents(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["filename"] == "notes.txt"


def test_upload_persists_document(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("notes.txt", b"abcdefghij", "text/plain")},
        )

    assert response.status_code == 200
    assert response.json()["file_info"]["chunks"] == 1
    assert kb.stats().documents == 1


def test_delete_document_endpoint(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.delete("/api/documents/notes.txt")

    assert response.status_code == 200
    assert kb.stats().documents == 0


def test_clear_endpoint(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb, raising=False)

    with TestClient(api_router.app) as client:
        response = client.post("/api/clear")

    assert response.status_code == 200
    assert kb.list_documents() == []
