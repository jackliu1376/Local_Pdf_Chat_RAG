# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Local PDF Chat RAG is an educational/reference RAG implementation (Python 3.10+) with two entry points: a Gradio web UI (`rag_demo.py`) and a FastAPI REST API (`api_router.py`). It supports local Ollama, SiliconFlow, and OpenAI-compatible Magick backends, plus optional SerpAPI web search. It is explicitly not a production service: no auth or tenancy; persistence is single-user SQLite.

The knowledge base is persisted in SQLite at `data/rag.db` (gitignored; `RAG_DATA_DIR` overrides). `vector_store` and `bm25_manager` are derived in-memory indexes rebuilt from SQLite at startup and after writes. `chunk_data_cache` and model caches remain in memory.

## Commands

```bash
# Setup (requirements-dev.txt includes requirements.txt)
python3.10 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
cp example.env .env              # then configure at least one backend

# Run
python rag_demo.py               # Gradio UI (opens browser)
python api_router.py             # FastAPI on 0.0.0.0
```

Both servers try ports 17995–17999 in order. API: `GET /api/status`, `POST /api/upload`, `POST /api/ask`, `GET /api/documents`, `DELETE /api/documents/{filename}`, `POST /api/clear`.

```bash
# Tests (the env var matters: langchain auto-installs the LangSmith pytest plugin)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_retrieval.py -v
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_retrieval.py::test_hybrid_merge_combines_semantic_and_sparse_scores -v

# CI parity (same steps as .github/workflows/ci.yml)
python -m compileall -q api_router.py config.py rag_demo.py version.py core features utils tests
python -m pytest -p pytest_cov.plugin --cov=core --cov=config --cov-report=term-missing
```

`pytest.ini` sets `testpaths = tests` and `pythonpath = .`. Persistence and API tests use a temporary SQLite DB with injected fake extractor/splitter/embedder:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_knowledge_base.py tests/test_api_documents.py -v
```

```bash
# Retrieval evaluation (local embedding/reranker only; no Ollama calls)
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 --rerank
PYTHONIOENCODING=utf-8 python -m eval.run_retrieval_eval --k 5 --json-out eval/results/baseline.json
```

`eval/golden_qa.jsonl` holds the golden questions; a chunk is relevant when its whitespace-stripped text contains any `relevant_snippets` entry. Metrics live in `eval/retrieval_metrics.py`.

Tests must remain credential-free and network-free. Note that `tests/test_api_status.py` asserts API keys are **not** configured, so a real `.env` with live keys can make the suite fail locally.

## Request architecture

### Document ingestion

`core/knowledge_base.py` owns ingestion and persistence through the module singleton `knowledge_base`:

```
SHA-256 file hash -> dedup/replace decision
  -> extract_text -> split_text -> encode_texts(new chunks)
  -> SQLite documents/chunks/meta
  -> load(): rebuild FAISS + BM25 from SQLite
```

- Same-name identity wins: a changed file with the same name replaces its old document even if the new content duplicates another document. A new filename whose content matches an existing document is skipped as a duplicate.
- Chunk IDs are `doc_id:chunk_index`; metadata carries `source`, `doc_id`, and `chunk_index`.
- `meta.embedding_model` is authoritative; switching `EMBED_MODEL_NAME` re-embeds all existing chunks. `load()` also validates stored dimensions and re-embeds if rows are mixed or stale after a cross-process model switch.
- Ollama query rewriting and LLM reranking send `think: false`, so reasoning models do not spend their small token budget in the thinking phase.
- `rag_demo.process_multiple_files()` and `api_router.py`'s `/api/upload` both delegate to `knowledge_base.add_files`; the API no longer imports `rag_demo` or regex-parses ingest messages.
- The UI loads the persisted KB at import so the document list and indexes are ready before the Gradio blocks are built. The API loads it in `lifespan`.

### Query pipeline

Both interfaces call `core.generator.query_answer()` (or `stream_answer()`), which runs:

1. `recursive_retrieval()` (`core/retriever.py`) — up to `MAX_RETRIEVAL_ITERATIONS` rounds. Each round optionally appends SerpAPI web snippets (deduplicated, not indexed), queries FAISS, queries BM25, merges with `hybrid_merge()` (weighted rank scores: `alpha` semantic + `1-alpha` BM25), reranks, then asks the LLM to rewrite the query. Returns parallel lists `(contexts, doc_ids, metadata)`.
2. `_build_context()` — formats sources as `[本地文档: ...]` or `[网络来源: ...] (URL: ...)` and builds source dicts.
3. `detect_conflicts()` (`features/conflict_detector.py`) and a keyword check for time-sensitive questions.
4. `_build_prompt()` — the prompt instructs the model to use only retrieved content and explicitly treats retrieved text as untrusted data. Preserve that instruction when editing prompts.
5. Backend call (`siliconflow`/`magick` via shared OpenAI-compatible helper, `ollama` via `/api/generate`), then `process_thinking_content()` converts `<think>...</think>` into a collapsible `<details>` block and escapes other HTML.

Cloud API helpers return error text instead of raising; callers generally surface strings as answers. Keep new failure paths actionable (README/CONTRIBUTING requirement) and avoid HTTP calls when a key is missing — `tests/test_no_key_fallback.py` enforces this.

## Configuration and model selection

`config.py` loads `.env`, falling back to `example.env` with a warning; it sets `HF_ENDPOINT=https://hf-mirror.com`. Keys beginning with `Your` are placeholders (`is_configured_api_key`). `DEFAULT_MODEL_CHOICE` is computed **at import time** with priority SiliconFlow → Magick → local Ollama probe (`localhost:11434`, 3s timeout) → SiliconFlow fallback. Persistent data lives at `DATA_DIR`/`DB_PATH` (`RAG_DATA_DIR`, default `data/`).

RAG hyperparameters are module constants, not environment variables: `CHUNK_SIZE=400`, `CHUNK_OVERLAP=40`, `HYBRID_ALPHA=0.7`, `RETRIEVAL_TOP_K=10`, `RERANK_TOP_K=5`, `MAX_RETRIEVAL_ITERATIONS=3`. `EMBED_MODEL_NAME` selects the embedding model and `EMBED_QUERY_PREFIX` optionally prepends a BGE query instruction. `RERANK_METHOD` (`cross_encoder` default, or `llm`) selects `core/reranker.py` behavior; `RERANK_MODEL_NAME` selects the CrossEncoder (default `BAAI/bge-reranker-base`). Ollama generation is bounded by `OLLAMA_MAX_OUTPUT_TOKENS` (1024), `OLLAMA_QUERY_MAX_TOKENS` (64), and `OLLAMA_TIMEOUT_SECONDS` (300); the LLM reranker is capped at 8 output tokens.

Model/network details:

- Embeddings: `EMBED_MODEL_NAME` (default `sentence-transformers/all-MiniLM-L6-v2`, 384-dim), lazy `lru_cache` singleton (`core/embeddings.py`). `EMBED_QUERY_PREFIX` is prepended only to queries, which is required for best BGE Chinese retrieval results.
- FAISS: `AutoFaissIndex` picks FlatL2 ≤10k vectors, IVFFlat ≤100k, IVFPQ above.
- CrossEncoder reranker (`BAAI/bge-reranker-base` by default) is lazily loaded behind a double-checked lock; LLM reranking uses Ollama scores cached by `lru_cache`.
- Ollama calls go through `utils/network.get_session()` (3 retries); cloud APIs use direct `requests.post`.
- Optional deps (python-docx, pandas, python-pptx) are imported inside `extract_text`; missing deps log and return `""`, which ingestion treats as an empty document.

## Conventions and constraints

- Keep pipeline boundaries visible and modules replaceable (`core/` order is documented in `core/__init__.py`). Prefer explicit wiring over hidden global state.
- Comments, docstrings, log messages, prompts, and UI text are predominantly Chinese; match existing style.
- Preserve source metadata (`source`, `doc_id`, `chunk_index`, web `title`/`url`/`timestamp`) when changing retrieval.
- `/data/` is runtime state and gitignored; tests must use `tmp_path` databases rather than the real one. `eval/data/` is input corpus and is not ignored.
- Public usage changes require updating both `README.md` and `README.zh-CN.md`.
- Version lives in `version.py` (currently 2.1.0); changelog in `CHANGELOG.md`; semantic versioning.
- Avoid unrelated formatting rewrites, credentials, model files, private test documents, and generated dependency folders in PRs.
- CI (`.github/workflows/ci.yml`) triggers only on push/PR to `main`, compiles the sources, and runs tests with coverage.
- `openwiki/` is generated reference documentation for the architecture; consult it for deeper component detail rather than treating it as source.
