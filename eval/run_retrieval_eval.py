"""Run a retrieval-only evaluation over the project's hybrid pipeline."""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"}

from config import RERANK_TOP_K, RETRIEVAL_TOP_K
from core.bm25_index import bm25_manager
from core.document_loader import extract_text
from core.embeddings import encode_query, encode_texts
from core.reranker import rerank_results
from core.retriever import hybrid_merge
from core.text_splitter import split_text
from core.vector_store import vector_store
from eval.retrieval_metrics import mrr, ndcg_at_k, recall_at_k


def normalize(text):
    """Remove whitespace so snippets match despite PDF line breaks."""
    return re.sub(r"\s+", "", text)


def load_dataset(path):
    entries = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return entries


def build_index(corpus_paths):
    all_chunks = []
    all_ids = []
    all_metadatas = []
    documents = []

    for doc_index, corpus_path in enumerate(corpus_paths):
        text = extract_text(str(corpus_path))
        if not text:
            print(f"WARNING: skipped {corpus_path} (no text extracted)", file=sys.stderr)
            continue

        chunks = split_text(text)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"doc_{doc_index:03d}_chunk_{chunk_index:03d}")
            all_metadatas.append(
                {
                    "source": corpus_path.name,
                    "doc_index": doc_index,
                    "chunk_index": chunk_index,
                }
            )
        documents.append(
            {"source": corpus_path.name, "chars": len(text), "chunks": len(chunks)}
        )

    if not all_chunks:
        raise SystemExit("no chunks built from the corpus")

    embeddings = encode_texts(all_chunks)
    vector_store.clear()
    bm25_manager.clear()
    vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)
    bm25_manager.build_index(all_chunks, all_ids)
    return all_chunks, all_ids, documents


def relevant_ids_for(snippets, chunks, ids):
    normalized_snippets = [normalize(snippet) for snippet in snippets]
    return {
        doc_id
        for doc_id, chunk in zip(ids, chunks)
        if any(snippet in normalize(chunk) for snippet in normalized_snippets)
    }


def retrieve(question, retrieval_top_k, rerank, k):
    query_embedding = encode_query(question)
    semantic_docs, semantic_ids, semantic_metadatas = vector_store.search(
        query_embedding, k=retrieval_top_k
    )
    prepared = {
        "ids": [semantic_ids],
        "documents": [semantic_docs],
        "metadatas": [semantic_metadatas],
    }
    bm25_results = (
        bm25_manager.search(question, top_k=retrieval_top_k)
        if bm25_manager.bm25_index
        else []
    )
    hybrid = hybrid_merge(prepared, bm25_results)

    if rerank:
        candidates = hybrid[:retrieval_top_k]
        docs = [item["content"] for _, item in candidates]
        doc_ids = [doc_id for doc_id, _ in candidates]
        metadatas = [item["metadata"] for _, item in candidates]
        reranked = rerank_results(
            question, docs, doc_ids, metadatas, top_k=max(k, RERANK_TOP_K)
        )
        return [doc_id for doc_id, _ in reranked]

    return [doc_id for doc_id, _ in hybrid]


def collect_corpus_files(explicit_files):
    if explicit_files:
        return [Path(path) for path in explicit_files]

    files = sorted(REPO_ROOT.glob("*.pdf"))
    data_dir = REPO_ROOT / "eval" / "data"
    if data_dir.is_dir():
        files.extend(
            sorted(
                path
                for path in data_dir.iterdir()
                if path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        )
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid retrieval against eval/golden_qa.jsonl"
    )
    parser.add_argument(
        "--corpus",
        nargs="*",
        default=None,
        help="corpus files; defaults to repository PDFs plus eval/data",
    )
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "eval" / "golden_qa.jsonl"
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retrieval-top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    corpus_files = collect_corpus_files(args.corpus)
    if not corpus_files:
        raise SystemExit("no corpus files found; pass --corpus")
    entries = load_dataset(args.dataset)
    chunks, ids, documents = build_index(corpus_files)

    rankings = []
    relevant_sets = []
    rows = []
    for entry in entries:
        relevant = relevant_ids_for(entry["relevant_snippets"], chunks, ids)
        if not relevant:
            raise SystemExit(
                f"{entry['id']}: no chunk contains any of {entry['relevant_snippets']}"
            )

        ranking = retrieve(entry["question"], args.retrieval_top_k, args.rerank, args.k)
        rankings.append(ranking)
        relevant_sets.append(relevant)
        first_rank = next(
            (
                rank
                for rank, doc_id in enumerate(ranking[: args.k], start=1)
                if doc_id in relevant
            ),
            None,
        )
        rows.append(
            {
                "id": entry["id"],
                "question": entry["question"],
                "relevant_ids": sorted(relevant),
                "top_ids": ranking[: args.k],
                "first_relevant_rank": first_rank,
            }
        )

    metrics = {
        f"recall@{args.k}": recall_at_k(rankings, relevant_sets, args.k),
        f"mrr@{args.k}": mrr(rankings, relevant_sets, args.k),
        f"ndcg@{args.k}": ndcg_at_k(rankings, relevant_sets, args.k),
    }

    print("Documents:")
    for document in documents:
        print(
            f"- {document['source']}: {document['chunks']} chunks, "
            f"{document['chars']} chars"
        )
    print(
        f"Corpus files: {len(corpus_files)} | Chunks: {len(chunks)} | "
        f"Queries: {len(entries)} | k={args.k} | rerank={args.rerank}"
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("\nPer-query:")
    for row in rows:
        hit = "hit" if row["first_relevant_rank"] else "miss"
        print(
            f"- {row['id']}: {hit} "
            f"(rank={row['first_relevant_rank']}, "
            f"relevant={','.join(row['relevant_ids'])})"
        )

    report = {
        "corpus_files": [str(path) for path in corpus_files],
        "documents": documents,
        "dataset": str(args.dataset),
        "k": args.k,
        "retrieval_top_k": args.retrieval_top_k,
        "rerank": args.rerank,
        "metrics": metrics,
        "per_query": rows,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON report: {args.json_out}")


if __name__ == "__main__":
    main()
