"""Tests for evaluation metrics."""

import pytest
from dartboard.evaluation.metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    average_precision,
    evaluate_retrieval,
    mean_average_precision,
    evaluate_batch,
)


def test_mrr_basic():
    """Test basic MRR calculation."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc2"}

    mrr = mean_reciprocal_rank(results, relevant, k=10)
    assert mrr == 0.5  # First relevant at rank 2, so 1/2


def test_mrr_first_position():
    """Test MRR when relevant doc is first."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1"}

    mrr = mean_reciprocal_rank(results, relevant, k=10)
    assert mrr == 1.0


def test_mrr_not_found():
    """Test MRR when no relevant docs found."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc5"}

    mrr = mean_reciprocal_rank(results, relevant, k=10)
    assert mrr == 0.0


def test_mrr_with_cutoff():
    """Test MRR with cutoff k."""
    results = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc4"}

    mrr_k3 = mean_reciprocal_rank(results, relevant, k=3)
    mrr_k10 = mean_reciprocal_rank(results, relevant, k=10)

    assert mrr_k3 == 0.0  # Not in top 3
    assert mrr_k10 == 0.25  # Found at position 4


def test_recall_at_k_basic():
    """Test basic Recall@K calculation."""
    results = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc2", "doc4", "doc6"}

    recall = recall_at_k(results, relevant, k=4)
    assert recall == pytest.approx(2 / 3)  # 2 out of 3 relevant docs found


def test_recall_at_k_all_found():
    """Test Recall@K when all relevant docs found."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc3"}

    recall = recall_at_k(results, relevant, k=10)
    assert recall == 1.0


def test_recall_at_k_none_found():
    """Test Recall@K when no relevant docs found."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc5", "doc6"}

    recall = recall_at_k(results, relevant, k=10)
    assert recall == 0.0


def test_precision_at_k_basic():
    """Test basic Precision@K calculation."""
    results = ["doc1", "doc2", "doc3", "doc4"]
    relevant = {"doc2", "doc4"}

    precision = precision_at_k(results, relevant, k=4)
    assert precision == 0.5  # 2 out of 4 are relevant


def test_precision_at_k_all_relevant():
    """Test Precision@K when all results are relevant."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3", "doc4"}

    precision = precision_at_k(results, relevant, k=3)
    assert precision == 1.0


def test_precision_at_k_none_relevant():
    """Test Precision@K when no results are relevant."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc5", "doc6"}

    precision = precision_at_k(results, relevant, k=3)
    assert precision == 0.0


def test_ndcg_perfect_ranking():
    """Test NDCG with perfect ranking."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3"}

    ndcg = ndcg_at_k(results, relevant, k=3)
    assert ndcg == 1.0


def test_ndcg_worst_ranking():
    """Test NDCG with worst possible ranking."""
    results = ["doc4", "doc5", "doc6", "doc1", "doc2"]
    relevant = {"doc1", "doc2"}

    ndcg = ndcg_at_k(results, relevant, k=3)
    assert ndcg == 0.0  # No relevant docs in top 3


def test_ndcg_partial_ranking():
    """Test NDCG with partial ranking."""
    results = ["doc1", "doc4", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3"}

    ndcg = ndcg_at_k(results, relevant, k=4)
    assert 0 < ndcg < 1  # Should be between 0 and 1


def test_average_precision_basic():
    """Test Average Precision calculation."""
    results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}

    ap = average_precision(results, relevant)

    # Precision at each relevant doc position:
    # doc1 at pos 1: 1/1 = 1.0
    # doc3 at pos 3: 2/3 = 0.667
    # doc5 at pos 5: 3/5 = 0.6
    # AP = (1.0 + 0.667 + 0.6) / 3 = 0.756
    assert ap == pytest.approx(0.756, abs=0.01)


def test_average_precision_no_relevant():
    """Test AP when no relevant docs."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc5", "doc6"}

    ap = average_precision(results, relevant)
    assert ap == 0.0


def test_average_precision_with_cutoff():
    """Test AP with cutoff k."""
    results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc5"}

    ap_k3 = average_precision(results, relevant, k=3)
    ap_all = average_precision(results, relevant, k=None)

    # With k=3, only doc1 is found
    assert ap_k3 == 0.5  # (1/1) / 2 relevant docs

    # Without cutoff, both are found
    assert ap_all > ap_k3


def test_evaluate_retrieval_comprehensive():
    """Test comprehensive evaluation with multiple metrics."""
    results = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5"}

    metrics = evaluate_retrieval(results, relevant, k_values=[1, 3, 5])

    # Should have MRR, Recall, Precision, NDCG for each k, plus AP
    assert "MRR@5" in metrics
    assert "Recall@1" in metrics
    assert "Recall@3" in metrics
    assert "Recall@5" in metrics
    assert "Precision@1" in metrics
    assert "Precision@3" in metrics
    assert "Precision@5" in metrics
    assert "NDCG@1" in metrics
    assert "NDCG@3" in metrics
    assert "NDCG@5" in metrics
    assert "AP" in metrics


def test_evaluate_retrieval_values():
    """Test that evaluation produces reasonable values."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc3"}

    metrics = evaluate_retrieval(results, relevant, k_values=[1, 3])

    # All metrics should be between 0 and 1
    for metric, value in metrics.items():
        assert 0 <= value <= 1


def test_mean_average_precision():
    """Test MAP across multiple queries."""
    all_results = [
        ["doc1", "doc2", "doc3"],
        ["doc4", "doc5", "doc6"],
    ]
    all_relevant = [
        {"doc1", "doc3"},
        {"doc4"},
    ]

    map_score = mean_average_precision(all_results, all_relevant)

    # Should average the AP scores
    assert 0 <= map_score <= 1


def test_map_perfect_results():
    """Test MAP with perfect results."""
    all_results = [
        ["doc1", "doc2"],
        ["doc3", "doc4"],
    ]
    all_relevant = [
        {"doc1", "doc2"},
        {"doc3", "doc4"},
    ]

    map_score = mean_average_precision(all_results, all_relevant)
    assert map_score == 1.0


def test_map_no_relevant():
    """Test MAP when no relevant docs found."""
    all_results = [
        ["doc1", "doc2"],
        ["doc3", "doc4"],
    ]
    all_relevant = [
        {"doc5"},
        {"doc6"},
    ]

    map_score = mean_average_precision(all_results, all_relevant)
    assert map_score == 0.0


def test_evaluate_batch():
    """Test batch evaluation."""
    all_results = [
        ["doc1", "doc2", "doc3"],
        ["doc4", "doc5", "doc6"],
    ]
    all_relevant = [
        {"doc1", "doc3"},
        {"doc4", "doc6"},
    ]

    metrics = evaluate_batch(all_results, all_relevant, k_values=[1, 3])

    # Should have averaged metrics
    assert "MRR@3" in metrics
    assert "Recall@1" in metrics
    assert "Precision@3" in metrics
    assert "NDCG@3" in metrics
    assert "MAP@3" in metrics


def test_evaluate_batch_consistency():
    """Test that batch evaluation produces consistent results."""
    all_results = [
        ["doc1", "doc2"],
        ["doc1", "doc2"],
    ]
    all_relevant = [
        {"doc1"},
        {"doc1"},
    ]

    metrics = evaluate_batch(all_results, all_relevant, k_values=[1, 2])

    # Since both queries are identical, averaged metrics should equal individual
    assert metrics["Recall@1"] == 1.0
    assert metrics["Precision@1"] == 1.0


def test_ndcg_with_graded_relevance():
    """Test NDCG with graded relevance scores."""
    results = ["doc1", "doc2", "doc3"]
    relevant = {"doc1", "doc2", "doc3"}
    gains = {"doc1": 3.0, "doc2": 2.0, "doc3": 1.0}

    ndcg = ndcg_at_k(results, relevant, k=3, gains=gains)
    assert ndcg == 1.0  # Perfect ranking with graded relevance


def test_empty_inputs():
    """Test metrics with empty inputs."""
    assert mean_reciprocal_rank([], {"doc1"}, k=10) == 0.0
    assert recall_at_k([], {"doc1"}, k=10) == 0.0
    assert precision_at_k([], {"doc1"}, k=10) == 0.0
    assert ndcg_at_k([], {"doc1"}, k=10) == 0.0
    assert average_precision([], {"doc1"}) == 0.0
