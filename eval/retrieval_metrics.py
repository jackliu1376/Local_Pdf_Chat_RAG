"""Binary-relevance retrieval metrics used by the evaluation harness."""

import math


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _validate(rankings, relevant_sets):
    if len(rankings) != len(relevant_sets):
        raise ValueError("rankings and relevant_sets must have the same length")


def recall_at_k(rankings, relevant_sets, k):
    """Return the mean fraction of relevant documents found in the top k."""
    _validate(rankings, relevant_sets)
    scores = []
    for ranking, relevant in zip(rankings, relevant_sets):
        if not relevant:
            continue
        retrieved = set(ranking[:k])
        scores.append(len(retrieved & relevant) / len(relevant))
    return _mean(scores)


def mrr(rankings, relevant_sets, k=None):
    """Return the mean reciprocal rank of the first relevant document."""
    _validate(rankings, relevant_sets)
    scores = []
    for ranking, relevant in zip(rankings, relevant_sets):
        if not relevant:
            continue
        candidates = ranking if k is None else ranking[:k]
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(candidates, start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return _mean(scores)


def ndcg_at_k(rankings, relevant_sets, k):
    """Return mean nDCG@k with binary relevance."""
    _validate(rankings, relevant_sets)
    scores = []
    for ranking, relevant in zip(rankings, relevant_sets):
        if not relevant:
            continue
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, doc_id in enumerate(ranking[:k], start=1)
            if doc_id in relevant
        )
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        scores.append(dcg / idcg if idcg else 0.0)
    return _mean(scores)
