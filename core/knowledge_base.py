"""Persistent knowledge base backed by SQLite.

SQLite is the source of truth. FAISS and BM25 are rebuilt from it, so the
derived indexes never need to be serialized.
"""

import hashlib
import json
import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import DB_PATH, EMBED_MODEL_NAME
from core.bm25_index import bm25_manager
from core.document_loader import extract_text
from core.embeddings import encode_texts
from core.text_splitter import split_text
from core.vector_store import vector_store


@dataclass(frozen=True)
class DocumentInfo:
    filename: str
    chunks: int
    created_at: str


@dataclass(frozen=True)
class FileResult:
    filename: str
    status: str
    chunks: int
    message: str


@dataclass(frozen=True)
class KnowledgeBaseStats:
    documents: int
    chunks: int
    embedding_model: str


CHUNK_QUERY = """
    SELECT c.id AS chunk_id, c.content, c.metadata_json, c.embedding
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY d.created_at, d.filename, c.chunk_index
"""


class KnowledgeBase:
    def __init__(
        self,
        db_path,
        *,
        extractor=extract_text,
        splitter=split_text,
        embedder=encode_texts,
        embed_model_name: str = EMBED_MODEL_NAME,
    ) -> None:
        self.db_path = Path(db_path)
        self.extractor = extractor
        self.splitter = splitter
        self.embedder = embedder
        self.embed_model_name = embed_model_name
        self._lock = threading.RLock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_hash
                    ON documents(file_hash);

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document
                    ON chunks(document_id, chunk_index);

                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def _get_meta(self, conn, key):
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, conn, key, value):
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, str(value)),
        )

    def _reembed_all_in_conn(self, conn):
        rows = conn.execute(
            "SELECT id, content FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            self._set_meta(conn, "embedding_model", self.embed_model_name)
            return

        texts = [row["content"] for row in rows]
        embeddings = np.asarray(self.embedder(texts), dtype="float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(rows):
            raise ValueError("embedder returned an unexpected shape")

        for row, vector in zip(rows, embeddings):
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (vector.tobytes(), row["id"]),
            )
        self._set_meta(conn, "embedding_model", self.embed_model_name)
        self._set_meta(conn, "embedding_dim", embeddings.shape[1])

    def _ensure_embedding_model(self, conn):
        stored_model = self._get_meta(conn, "embedding_model")
        if stored_model == self.embed_model_name:
            return
        self._reembed_all_in_conn(conn)

    def list_documents(self):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT filename, chunk_count, created_at FROM documents "
                "ORDER BY created_at, filename"
            ).fetchall()
        return [
            DocumentInfo(
                filename=row["filename"],
                chunks=row["chunk_count"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def stats(self):
        with closing(self._connect()) as conn:
            documents = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            stored_model = self._get_meta(conn, "embedding_model")
        return KnowledgeBaseStats(
            documents=documents,
            chunks=chunks,
            embedding_model=stored_model or self.embed_model_name,
        )

    @staticmethod
    def _hash_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def _document_id(filename, file_hash):
        return hashlib.sha256(
            f"{filename}:{file_hash}".encode("utf-8")
        ).hexdigest()[:16]

    def add_files(self, files):
        results = []
        with self._lock:
            with closing(self._connect()) as conn, conn:
                self._ensure_embedding_model(conn)

            for upload in files:
                filename = None
                try:
                    if isinstance(upload, (str, os.PathLike)):
                        path = Path(upload)
                    else:
                        path = Path(upload.name)
                    filename = path.name
                    file_hash = self._hash_file(path)
                    with closing(self._connect()) as conn, conn:
                        same_name = conn.execute(
                            "SELECT id, file_hash FROM documents WHERE filename = ?",
                            (filename,),
                        ).fetchone()
                        if same_name and same_name["file_hash"] == file_hash:
                            results.append(
                                FileResult(filename, "unchanged", 0, "已存在且未变化")
                            )
                            continue
                        if same_name is None:
                            duplicate = conn.execute(
                                "SELECT filename FROM documents WHERE file_hash = ?",
                                (file_hash,),
                            ).fetchone()
                            if duplicate:
                                results.append(
                                    FileResult(filename, "unchanged", 0, "相同内容已存在")
                                )
                                continue

                        text = self.extractor(str(path))
                        if not text or not text.strip():
                            results.append(
                                FileResult(
                                    filename,
                                    "error",
                                    0,
                                    "文档内容为空或无法提取文本",
                                )
                            )
                            continue

                        chunks = self.splitter(text)
                        if not chunks:
                            results.append(
                                FileResult(filename, "error", 0, "文档没有可索引的文本块")
                            )
                            continue

                        embeddings = np.asarray(self.embedder(chunks), dtype="float32")
                        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
                            raise ValueError("embedder returned an unexpected shape")

                        status = "replaced" if same_name else "added"
                        if same_name:
                            conn.execute(
                                "DELETE FROM documents WHERE id = ?",
                                (same_name["id"],),
                            )

                        doc_id = self._document_id(filename, file_hash)
                        created_at = datetime.now(timezone.utc).isoformat()
                        conn.execute(
                            "INSERT INTO documents "
                            "(id, filename, file_hash, created_at, chunk_count) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (doc_id, filename, file_hash, created_at, len(chunks)),
                        )
                        for chunk_index, (chunk, vector) in enumerate(
                            zip(chunks, embeddings)
                        ):
                            metadata = {
                                "source": filename,
                                "doc_id": doc_id,
                                "chunk_index": chunk_index,
                            }
                            conn.execute(
                                "INSERT INTO chunks "
                                "(id, document_id, chunk_index, content, "
                                "metadata_json, embedding) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (
                                    f"{doc_id}:{chunk_index}",
                                    doc_id,
                                    chunk_index,
                                    chunk,
                                    json.dumps(metadata, ensure_ascii=False),
                                    vector.tobytes(),
                                ),
                            )
                        self._set_meta(conn, "embedding_model", self.embed_model_name)
                        self._set_meta(conn, "embedding_dim", embeddings.shape[1])

                    message = (
                        f"成功处理 {len(chunks)} 个文本块"
                        if status == "added"
                        else f"内容已更新，处理 {len(chunks)} 个文本块"
                    )
                    results.append(FileResult(filename, status, len(chunks), message))
                except Exception as exc:
                    results.append(
                        FileResult(filename or str(upload), "error", 0, str(exc))
                    )

            self.load()
        return results

    def delete_document(self, filename):
        with self._lock:
            with closing(self._connect()) as conn, conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE filename = ?", (filename,)
                )
                deleted = cursor.rowcount > 0
            self.load()
            return deleted

    def clear(self):
        with self._lock:
            with closing(self._connect()) as conn, conn:
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM documents")
            self.load()

    def load(self):
        with self._lock:
            with closing(self._connect()) as conn, conn:
                self._ensure_embedding_model(conn)
                rows = conn.execute(CHUNK_QUERY).fetchall()
                stored_dim = self._get_meta(conn, "embedding_dim")

            dimensions = {
                np.frombuffer(row["embedding"], dtype="float32").size
                for row in rows
            }
            expected_dim = int(stored_dim) if stored_dim else None
            if len(dimensions) > 1 or (
                expected_dim is not None
                and dimensions
                and dimensions != {expected_dim}
            ):
                # A cross-process model switch can leave mixed or stale
                # embedding dimensions. Repair the corpus before stacking.
                with closing(self._connect()) as conn, conn:
                    self._reembed_all_in_conn(conn)
                    rows = conn.execute(CHUNK_QUERY).fetchall()

            chunks = [row["content"] for row in rows]
            chunk_ids = [row["chunk_id"] for row in rows]
            metadatas = [json.loads(row["metadata_json"]) for row in rows]

            vector_store.clear()
            bm25_manager.clear()
            if chunks:
                embeddings = np.vstack(
                    [
                        np.frombuffer(row["embedding"], dtype="float32")
                        for row in rows
                    ]
                )
                vector_store.build_index(
                    chunks, chunk_ids, metadatas, embeddings
                )
                bm25_manager.build_index(chunks, chunk_ids)
            return self.stats()


# Module-level singleton shared by the Gradio UI and the FastAPI app.
knowledge_base = KnowledgeBase(DB_PATH)
