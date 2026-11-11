# Persistence and Incremental Indexing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the knowledge base survive restarts and append new documents without re-parsing unchanged files.

**Architecture:** SQLite is the source of truth (`documents`, `chunks`, `meta`); FAISS and BM25 are derived indexes rebuilt from SQLite on startup and after writes. Uploads deduplicate by SHA-256, replace changed same-name files, and embed only new chunks.

**Tech Stack:** Python 3.10+, stdlib `sqlite3`, existing FAISS/BM25 wrappers, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-persistence-design.md`

## Global Constraints

- No new runtime dependencies; use stdlib `sqlite3`.
- Data dir defaults to `data/` and is gitignored; `RAG_DATA_DIR` overrides it.
- Single global knowledge base; no multi-user/session isolation.
- Tests must be credential-free and network-free; use a temporary DB and injected fake extractor/splitter/embedder.
- Preserve source metadata (`source`, `doc_id`, `chunk_index`).
- UI and API run as separate processes; each reloads after its own writes, so the other may serve a stale in-memory index until restart. Accepted for now.
- Use `threading.RLock` for the service lock: `add_files`/`delete_document`/`clear` hold it while calling `load()`. Queries do not take the lock and may briefly see a partially rebuilt index.
- Do not create git commits unless the human partner explicitly asks.

## Review Focus

- Re-uploading an unchanged file must not duplicate chunks.
- Re-uploading a changed file with the same name must replace old chunks, not append stale content.
- Switching `EMBED_MODEL_NAME` must re-embed existing chunks before rebuilding indexes.
- Empty or failed extraction must not leave a document row or partial chunks.
- Concurrent uploads must not lose documents or corrupt SQLite/index state.

---

### Task 1: Data directory, schema, listing, and stats

**Files:**
- Modify: `config.py` (add `DATA_DIR`, `DB_PATH` near the runtime config)
- Create: `core/knowledge_base.py`
- Create: `tests/test_knowledge_base.py`

**Interfaces:**
- Consumes: `extract_text`, `split_text`, `encode_texts`, `vector_store`, `bm25_manager`, `EMBED_MODEL_NAME`
- Produces: `DocumentInfo(filename, chunks, created_at)`, `FileResult(filename, status, chunks, message)`, `KnowledgeBaseStats(documents, chunks, embedding_model)`, `KnowledgeBase.__init__(db_path, *, extractor, splitter, embedder, embed_model_name)`, `list_documents()`, `stats()`

- [ ] **Step 1: Write the failing test**

```python
def test_new_knowledge_base_creates_empty_schema(tmp_path):
    kb = make_kb(tmp_path)
    assert kb.list_documents() == []
    stats = kb.stats()
    assert stats.documents == 0
    assert stats.chunks == 0
```

Test helpers (same file):

```python
import numpy as np
from core.bm25_index import bm25_manager
from core.knowledge_base import KnowledgeBase
from core.vector_store import vector_store


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py::test_new_knowledge_base_creates_empty_schema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'core.knowledge_base'`

- [ ] **Step 3: Implement the skeleton**

In `config.py`:

```python
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(Path(__file__).parent / "data")))
DB_PATH = DATA_DIR / "rag.db"
```

In `core/knowledge_base.py`, create the dataclasses and:

```python
class KnowledgeBase:
    def __init__(self, db_path: Path, *, extractor=extract_text,
                 splitter=split_text, embedder=encode_texts,
                 embed_model_name: str = EMBED_MODEL_NAME) -> None: ...

    def list_documents(self) -> list[DocumentInfo]: ...
    def stats(self) -> KnowledgeBaseStats: ...
```

`__init__` creates the parent directory and the three tables (`documents`, `chunks`, `meta`) plus `idx_chunks_document`. Each connection sets `PRAGMA foreign_keys=ON` and `journal_mode=WAL`. `stats()` returns counts plus `meta.embedding_model` (falling back to `embed_model_name`). Add `data/` to `.gitignore` in this task so the runtime DB never enters git.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py::test_new_knowledge_base_creates_empty_schema -v`
Expected: PASS

---

### Task 2: Add files, deduplicate, replace, and rebuild indexes

**Files:**
- Modify: `core/knowledge_base.py`
- Modify: `tests/test_knowledge_base.py`
- Create: `tests/helpers.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `add_files(files) -> list[FileResult]`, `load() -> KnowledgeBaseStats`
- `files` items expose `.name` (path-like) or are paths.

- [ ] **Step 1: Write the failing tests**

```python
def test_add_files_persists_document_and_chunks(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "added"
    assert results[0].chunks == 3
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 3
    assert vector_store.total_chunks == 3
    assert bm25_manager.bm25_index is not None


def test_add_same_file_twice_is_unchanged(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "unchanged"
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 3


def test_changed_file_replaces_old_chunks(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])
    path.write_text("0123456789", encoding="utf-8")

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "replaced"
    assert kb.stats().documents == 1
    assert kb.stats().chunks == 1


def test_empty_document_is_reported_and_not_persisted(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    kb = make_kb(tmp_path)

    results = kb.add_files([FakeFile(path)])

    assert results[0].status == "error"
    assert kb.stats().documents == 0
    assert kb.stats().chunks == 0


def test_concurrent_adds_do_not_lose_documents(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("aaaaaaaaaa", encoding="utf-8")
    second.write_text("bbbbbbbbbb", encoding="utf-8")
    kb = make_kb(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda p: kb.add_files([FakeFile(p)]), [first, second]))

    assert kb.stats().documents == 2
    assert kb.stats().chunks == 2


def test_load_rebuilds_indexes_from_sqlite(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    make_kb(tmp_path).add_files([FakeFile(path)])

    stats = make_kb(tmp_path).load()

    assert stats.documents == 1
    assert stats.chunks == 3
    assert vector_store.total_chunks == 3
    assert bm25_manager.bm25_index is not None
```

Add helpers:

```python
from concurrent.futures import ThreadPoolExecutor

class FakeFile:
    def __init__(self, path):
        self.name = str(path)
```

Move `FakeFile`, `make_kb`, and the fake extractor/splitter/embedder to `tests/helpers.py`; put the autouse fixture that clears `vector_store`/`bm25_manager` in `tests/conftest.py`. `tests/test_knowledge_base.py` and the later API test import `from tests.helpers import FakeFile, make_kb`. `tests/` works as a namespace package because `pytest.ini` puts the repo root on `pythonpath`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py -v`
Expected: FAIL with `AttributeError: 'KnowledgeBase' object has no attribute 'add_files'` (or `load`)

- [ ] **Step 3: Implement `add_files` and `load`**

`add_files`:

1. Acquire `self._lock` (use `threading.RLock` because `load()` is called while the lock is held).
2. For each file: resolve path, read bytes, compute SHA-256.
3. If `file_hash` exists: append `FileResult(filename, "unchanged", 0, "相同内容已存在")` and continue.
4. Extract and split **before** deleting an existing same-name document; empty text -> `error`.
5. If the filename exists with a different hash, delete the old document in the transaction.
6. Insert the new document row and chunk rows; embed only the new chunks.
7. Commit per file; roll back on exception and return `error`.
8. After the loop, call `self.load()` once.

`load`:

1. Read all chunk rows joined to documents, ordered by `documents.created_at, chunks.chunk_index`.
2. Rebuild embeddings with `np.vstack([np.frombuffer(row["embedding"], dtype="float32") ...])`.
3. Clear `vector_store`/`bm25_manager` and call `build_index` on both.
4. Return `stats()`.

Store metadata as JSON with `source`, `doc_id`, `chunk_index`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py -v`
Expected: PASS

---

### Task 3: Delete and clear

**Files:**
- Modify: `core/knowledge_base.py`
- Modify: `tests/test_knowledge_base.py`

**Interfaces:**
- Produces: `delete_document(filename) -> bool`, `clear() -> None`

- [ ] **Step 1: Write the failing tests**

```python
def test_delete_document_removes_chunks_and_updates_indexes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    assert kb.delete_document("a.txt") is True
    assert kb.stats().documents == 0
    assert kb.stats().chunks == 0
    assert vector_store.total_chunks == 0


def test_clear_removes_all_documents_and_indexes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb = make_kb(tmp_path)
    kb.add_files([FakeFile(path)])

    kb.clear()

    assert kb.list_documents() == []
    assert vector_store.total_chunks == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py -k "delete or clear" -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement**

`delete_document` deletes the document row (chunks cascade), commits, calls `load()`, and returns whether a row was deleted. `clear()` deletes all chunks and documents, commits, and calls `load()`. Keep `meta` so the embedding model choice survives a clear.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py -v`
Expected: PASS

---

### Task 4: Re-embed when the embedding model changes

**Files:**
- Modify: `core/knowledge_base.py`
- Modify: `tests/test_knowledge_base.py`

**Interfaces:**
- `meta.embedding_model` / `meta.embedding_dim` are authoritative.
- `_ensure_embedding_model(conn)` re-embeds all chunks when the configured model differs.

- [ ] **Step 1: Write the failing test**

```python
def test_load_reembeds_when_embedding_model_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    first = make_kb(
        tmp_path,
        embed_model_name="model-a",
        embedder=lambda texts, show_progress=False: np.array(
            [[1.0, 0.0] for _ in texts], dtype="float32"
        ),
    )
    first.add_files([FakeFile(path)])

    second = make_kb(
        tmp_path,
        embed_model_name="model-b",
        embedder=lambda texts, show_progress=False: np.array(
            [[2.0, 0.0] for _ in texts], dtype="float32"
        ),
    )
    stats = second.load()

    assert stats.embedding_model == "model-b"
    with sqlite3.connect(tmp_path / "rag.db") as conn:
        blob = conn.execute("SELECT embedding FROM chunks LIMIT 1").fetchone()[0]
    assert np.frombuffer(blob, dtype="float32")[0] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py::test_load_reembeds_when_embedding_model_changes -v`
Expected: FAIL because the stored embedding remains `1.0`

- [ ] **Step 3: Implement**

In `load()` and at the start of `add_files`, call `_ensure_embedding_model(conn)`. It compares `meta.embedding_model` with `self.embed_model_name`; on mismatch, reads all chunk contents, calls `self.embedder(texts)` once, updates each `embedding` BLOB, and writes `embedding_model`/`embedding_dim`. When there are no chunks, just write the current model.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py -v`
Expected: PASS

---

### Task 5: Gradio integration

**Files:**
- Modify: `rag_demo.py`

**Interfaces:**
- Consumes: `knowledge_base`, `FileResult`
- Produces: `format_document_list() -> list[str]`; `process_multiple_files` now delegates and returns the same `(status_text, file_list)` shape.

- [ ] **Step 1: Implement the integration**

- Import `knowledge_base` from `core.knowledge_base`.
- Call `knowledge_base.load()` at module import, before the Gradio blocks are built, so `vector_store`, `bm25_manager`, and the file list reflect the persisted DB.
- Add `format_document_list()` returning `[f"📄 {doc.filename} ({doc.chunks} 块)" for doc in knowledge_base.list_documents()]`.
- Initialize the `file_list` textbox with `"\n".join(format_document_list())`.
- Replace the body of `process_multiple_files`: call `knowledge_base.add_files(files)`, format each `FileResult` (`added`, `replaced`, `unchanged`, `error`), append `知识库总计 {stats.chunks} 个文本块`, and return `("\n".join(lines), "\n".join(format_document_list()))`.
- Keep the `progress` calls coarse (start/end); the service does not take a Gradio callback.

- [ ] **Step 2: Run the app and verify manually**

Run: `python rag_demo.py`
Expected:

1. Existing persisted documents appear in “已处理文件” before any upload.
2. Uploading a new file appends it.
3. Uploading the same file again reports “已存在且未变化，跳过”.
4. Uploading a changed file reports “内容已更新”.

---

### Task 6: FastAPI integration and document endpoints

**Files:**
- Modify: `api_router.py`
- Create: `tests/test_api_documents.py`

**Interfaces:**
- `GET /api/documents` -> `{"documents": [{"filename", "chunks", "created_at"}]}`
- `DELETE /api/documents/{filename}` -> `{"deleted": true, "filename": ...}` or 404
- `POST /api/clear` -> `{"status": "cleared"}`
- `POST /api/upload` -> `file_info: {"filename", "chunks", "total_chunks"}`
- `GET /api/status` adds `documents` and `embedding_model`.

- [ ] **Step 1: Write the failing API tests**

Use a temporary `KnowledgeBase` with the same fake extractor/splitter/embedder and monkeypatch `api_router.knowledge_base` before creating `TestClient(api_router.app)`. Import `FakeFile` and `make_kb` from `tests.helpers`.

```python
def test_documents_endpoint_lists_persisted_documents(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb)

    with TestClient(api_router.app) as client:
        response = client.get("/api/documents")

    assert response.status_code == 200
    assert response.json()["documents"][0]["filename"] == "notes.txt"


def test_upload_persists_document(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    monkeypatch.setattr(api_router, "knowledge_base", kb)

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
    monkeypatch.setattr(api_router, "knowledge_base", kb)

    with TestClient(api_router.app) as client:
        response = client.delete("/api/documents/notes.txt")

    assert response.status_code == 200
    assert kb.stats().documents == 0


def test_clear_endpoint(tmp_path, monkeypatch):
    kb = make_kb(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_text("abcdefghij", encoding="utf-8")
    kb.add_files([FakeFile(path)])
    monkeypatch.setattr(api_router, "knowledge_base", kb)

    with TestClient(api_router.app) as client:
        response = client.post("/api/clear")

    assert response.status_code == 200
    assert kb.list_documents() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_api_documents.py -v`
Expected: FAIL with 404 for the new endpoints

- [ ] **Step 3: Implement**

- Import `knowledge_base`; call `knowledge_base.load()` in `lifespan`.
- Replace the `rag_demo.process_multiple_files` call in `/api/upload` with `knowledge_base.add_files`. Write the upload to `TemporaryDirectory()/Path(file.filename).name`, sanitize with `os.path.basename`, and return `file_info` with the per-file chunk count and total count.
- Add the three endpoints; return 404 when deleting a missing filename.
- Add `documents` and `embedding_model` to `/api/status`.
- Remove the regex parsing of the ingest message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_api_documents.py -v`
Expected: PASS

---

### Task 7: Docs, gitignore, and end-to-end verification

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`, `README.zh-CN.md`, `CLAUDE.md`, `example.env`

- [ ] **Step 1: Update `.gitignore` and docs**

- Add `data/` to `.gitignore`.
- README/README.zh-CN:
  - Replace the “index is in process memory / rebuild after restart” limitation with SQLite persistence and startup index rebuild.
  - Document `data/rag.db`, `RAG_DATA_DIR`, the new API endpoints, and how to reset (`POST /api/clear` or delete `data/`).
- `CLAUDE.md`: update the project overview (no longer in-memory only), ingestion architecture (service), and add the new test/API commands.
- `example.env`: add commented `RAG_DATA_DIR`.
- Note in the README that `eval/run_retrieval_eval.py` still builds its own in-memory evaluation index from `eval/data/`; persistence does not change the evaluation harness.

- [ ] **Step 2: Run the full test suite**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Verify persistence end to end**

1. Stop the UI; delete `data/` for a clean run.
2. Start `python rag_demo.py`; upload the sample PDF; ask one question.
3. Stop and restart the UI.
4. Confirm the document appears in “已处理文件” without re-uploading and a question still works.
5. Start `python api_router.py`; check `GET /api/documents` and `GET /api/status`.
6. `DELETE /api/documents/<filename>` and `POST /api/clear`; confirm the DB and indexes update.

- [ ] **Step 4: Report results**

Report the test count, the persisted document/chunk counts, and any limitation found. Do not commit unless the human partner asks.
