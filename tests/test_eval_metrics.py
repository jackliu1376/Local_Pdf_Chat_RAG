import math

import pytest

from eval.retrieval_metrics import mrr, ndcg_at_k, recall_at_k


def test_recall_at_k_measures_fraction_of_relevant_documents_retrieved():
    rankings = [
        ["doc-a", "doc-b", "doc-c"],
        ["doc-x", "doc-z", "doc-y"],
    ]
    relevant = [{"doc-a", "doc-c"}, {"doc-z"}]

    assert recall_at_k(rankings, relevant, k=2) == pytest.approx(0.75)
    assert recall_at_k(rankings, relevant, k=3) == pytest.approx(1.0)


def test_mrr_uses_the_rank_of_the_first_relevant_result():
    rankings = [["doc-x", "doc-a"], ["doc-x", "doc-y"]]
    relevant = [{"doc-a"}, {"doc-y"}]

    assert mrr(rankings, relevant) == pytest.approx(0.5)


def test_mrr_at_k_returns_zero_when_no_relevant_result_is_in_top_k():
    assert mrr([["doc-x", "doc-y"]], [{"doc-a"}], k=2) == 0.0


def test_ndcg_at_k_rewards_relevant_results_at_higher_ranks():
    best = ndcg_at_k([["doc-a", "doc-x"]], [{"doc-a"}], k=2)
    worse = ndcg_at_k([["doc-x", "doc-a"]], [{"doc-a"}], k=2)

    assert best == pytest.approx(1.0)
    assert worse == pytest.approx(1 / math.log2(3))
    assert best > worse
