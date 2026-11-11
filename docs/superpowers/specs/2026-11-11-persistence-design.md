# Persistence and Incremental Indexing — Design

Date: 2026-11-11
Status: proposed

## Goal

Make the knowledge base survive application restarts and let new uploads append to the existing corpus instead of clearing it. Unchanged files should not be re-parsed or re-embedded.

## Non-goals

- Multi-user or per-session knowledge bases
- OCR, new file parsers, or chunking changes
- Distributed storage or an external vector database
- UI redesign

## Architecture

SQLite is the source of truth. FAISS and BM25 are derived indexes rebuilt from SQLite at startup and after each write. This avoids serializing the FAISS wrapper and keeps the index format free to change.

```text
Upload -> SHA-256 hash -> dedup / replace decision
       -> extract -> split -> embed only new chunks
       -> SQLite (documents, chunks, meta)
       -> rebuild FAISS + BM25 from SQLite

Startup -> read SQLite -> rebuild FAISS + BM25
```

Data lives in `data/` by default (`RAG_DATA_DIR` overrides it) and is gitignored.

### Schema

```sql
CREATE TABLE documents (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL UNIQUE,
  file_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  chunk_count INTEGER NOT NULL
);

CREATE TABLE chunks (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  embedding BLOB NOT NULL
);

CREATE INDEX idx_chunks_document ON chunks(document_id, chunk_index);

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
```

`meta` stores `embedding_model` and `embedding_dim`. If the configured embedding model differs from the stored model, all chunks are re-embedded before indexes are rebuilt.

### Identity and incremental rules

- `file_hash = sha256(file bytes)`
- `doc_id = sha256(f"{filename}:{file_hash}")[:16]`
- `chunk_id = f"{doc_id}:{chunk_index}"`
- Same filename and same hash: skip (`unchanged`)
- Same filename and different hash: delete old document/chunks, insert new (`replaced`)
- Same hash under a different filename: skip as duplicate content (`unchanged`)
- New filename: insert (`added`)

Only new chunks are embedded. After the batch, indexes are rebuilt once from SQLite.

### Service interface

`core/knowledge_base.py`:

```python
@dataclass(frozen=True)
class DocumentInfo:
    filename: str
    chunks: int
    created_at: str

@dataclass(frozen=True)
class FileResult:
    filename: str
    status: str  # added | unchanged | replaced | error
    chunks: int
    message: str

@dataclass(frozen=True)
class KnowledgeBaseStats:
    documents: int
    chunks: int
    embedding_model: str

class KnowledgeBase:
    def __init__(self, db_path: Path, *, extractor=extract_text,
                 splitter=split_text, embedder=encode_texts,
                 embed_model_name: str = EMBED_MODEL_NAME) -> None: ...

    def load(self) -> KnowledgeBaseStats: ...
    def add_files(self, files) -> list[FileResult]: ...
    def delete_document(self, filename: str) -> bool: ...
    def clear(self) -> None: ...
    def list_documents(self) -> list[DocumentInfo]: ...
    def stats(self) -> KnowledgeBaseStats: ...
```

A module-level singleton `knowledge_base = KnowledgeBase(DB_PATH)` is shared by the UI and API. Per-file transactions; a process-local `threading.Lock` serializes writes and index rebuilds.

## Integration

### Gradio

- `rag_demo` loads `knowledge_base` before building the UI, so the vector store, BM25 index, and document list are populated.
- `process_multiple_files` delegates to `knowledge_base.add_files` and formats `FileResult` values; it no longer clears indexes.
- The file list shows all persisted documents, not only the current upload.

### FastAPI

- `lifespan` calls `knowledge_base.load()` at startup.
- `/api/upload` saves the upload under its original (sanitized) filename, calls `knowledge_base.add_files`, and returns structured counts instead of regex-parsing a message.
- `/api/status` reports document count and embedding model.
- New endpoints: `GET /api/documents`, `DELETE /api/documents/{filename}`, `POST /api/clear`.
- The API no longer imports `rag_demo` for upload handling.

## Failure handling

- Extraction failure or empty text: `FileResult(status="error")`, no document row is committed.
- Each file is written in a transaction; a partial write is rolled back.
- Index rebuild failure leaves SQLite consistent; the next `load()` retries.
- WAL mode and a write lock handle concurrent uploads within one process. UI and API are separate processes; each reloads after its own writes, so the other process may serve a stale in-memory index until restart. This is an accepted limitation for the current single-user scope.
- Queries do not take the write lock. A query that runs during an upload may briefly observe a partially rebuilt index. Acceptable for single-user use; a later version can build into a temporary index and swap atomically.

## Testing

Unit tests use a temporary SQLite file and injected fake extractor/splitter/embedder, so they are fast and network-free:

- Empty schema and stats
- Add, duplicate skip, changed-file replace
- Empty document rejected without rows
- Concurrent adds
- Startup load rebuilds FAISS/BM25
- Delete and clear
- Embedding model change re-embeds
- API document list/delete/clear/upload
