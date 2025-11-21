import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from dartboard.utils import (
    log_gaussian_kernel,
    cosine_similarity,
    cosine_distance,
    logsumexp_stable,
    normalize_embeddings,
    batch_cosine_similarity,
    clamp_similarity,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_embeddings():
    """Sample embedding vectors for testing (10 vectors of 384 dimensions)."""
    np.random.seed(42)
    return np.random.randn(10, 384)


@pytest.fixture
def identical_vectors():
    """Two identical unit vectors for testing."""
    vec = np.array([1.0, 0.0, 0.0])
    return vec.copy(), vec.copy()


@pytest.fixture
def orthogonal_vectors():
    """Two orthogonal unit vectors."""
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


@pytest.fixture
def opposite_vectors():
    """Two opposite unit vectors."""
    return np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])


@pytest.fixture
def zero_vector():
    """A zero vector for edge case testing."""
    return np.zeros(3)


# ============================================================================
# Tests for log_gaussian_kernel
# ============================================================================


class TestLogGaussianKernel:
    """Test suite for log_gaussian_kernel function."""

    def test_identical_vectors_gives_high_log_prob(self, identical_vectors):
        """Identical vectors should give the highest log-probability."""
        a, b = identical_vectors
        log_prob = log_gaussian_kernel(a, b, sigma=1.0)

        # Log probability should be negative (property of log probabilities)
        assert log_prob <= 0.0
        # For identical vectors, distance is 0, so we get close to max value
        assert log_prob > -5.0  # Should be relatively high

    def test_different_vectors_gives_lower_log_prob(self, orthogonal_vectors):
        """Different vectors should give lower log-probability than identical."""
        a, b = orthogonal_vectors
        log_prob = log_gaussian_kernel(a, b, sigma=1.0)

        # Should be significantly lower than for identical vectors
        assert log_prob < -0.5

    def test_log_probability_is_negative(self, sample_embeddings):
        """All log probabilities should be negative or zero."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]
        log_prob = log_gaussian_kernel(a, b, sigma=1.0)

        assert log_prob <= 0.0
        assert np.isfinite(log_prob)

    def test_smaller_sigma_increases_penalty(self, sample_embeddings):
        """Smaller sigma should give lower log-probability for different vectors."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        log_prob_large_sigma = log_gaussian_kernel(a, b, sigma=10.0)
        log_prob_small_sigma = log_gaussian_kernel(a, b, sigma=0.1)

        # Smaller sigma means tighter distribution, more penalty for distance
        assert log_prob_small_sigma < log_prob_large_sigma

    def test_dimension_mismatch_raises_error(self):
        """Mismatched dimensions should raise ValueError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            log_gaussian_kernel(a, b, sigma=1.0)

    def test_numerical_stability_with_extreme_values(self):
        """Test with very large and very small values."""
        # Very large values
        a = np.array([1e10, 1e10, 1e10])
        b = np.array([1e10, 1e10, 1e10])
        log_prob = log_gaussian_kernel(a, b, sigma=1.0)
        assert np.isfinite(log_prob)

        # Very small values
        a = np.array([1e-10, 1e-10, 1e-10])
        b = np.array([1e-10, 1e-10, 1e-10])
        log_prob = log_gaussian_kernel(a, b, sigma=1.0)
        assert np.isfinite(log_prob)

    def test_very_small_sigma_uses_eps(self):
        """Sigma smaller than eps should be clamped to prevent division by zero."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])

        # Should not raise error or return inf/nan
        log_prob = log_gaussian_kernel(a, b, sigma=1e-20, eps=1e-10)
        assert np.isfinite(log_prob)

    def test_symmetric_property(self, sample_embeddings):
        """log_gaussian_kernel(a, b) should equal log_gaussian_kernel(b, a)."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        log_prob_ab = log_gaussian_kernel(a, b, sigma=1.0)
        log_prob_ba = log_gaussian_kernel(b, a, sigma=1.0)

        assert_allclose(log_prob_ab, log_prob_ba)

    def test_zero_vectors(self, zero_vector):
        """Test with zero vectors."""
        a = zero_vector
        b = zero_vector.copy()

        log_prob = log_gaussian_kernel(a, b, sigma=1.0)
        assert np.isfinite(log_prob)
        assert log_prob <= 0.0


# ============================================================================
# Tests for cosine_similarity
# ============================================================================


class TestCosineSimilarity:
    """Test suite for cosine_similarity function."""

    def test_identical_vectors_returns_one(self, identical_vectors):
        """Cosine similarity of identical vectors should be 1.0."""
        a, b = identical_vectors
        similarity = cosine_similarity(a, b)

        assert_allclose(similarity, 1.0)

    def test_orthogonal_vectors_returns_zero(self, orthogonal_vectors):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        a, b = orthogonal_vectors
        similarity = cosine_similarity(a, b)

        assert_allclose(similarity, 0.0, atol=1e-10)

    def test_opposite_vectors_returns_negative_one(self, opposite_vectors):
        """Cosine similarity of opposite vectors should be -1.0."""
        a, b = opposite_vectors
        similarity = cosine_similarity(a, b)

        assert_allclose(similarity, -1.0)

    def test_result_in_valid_range(self, sample_embeddings):
        """Cosine similarity should always be in [-1, 1]."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]
        similarity = cosine_similarity(a, b)

        assert -1.0 <= similarity <= 1.0

    def test_zero_vector_returns_zero(self, zero_vector):
        """Cosine similarity with zero vector should return 0.0."""
        non_zero = np.array([1.0, 2.0, 3.0])

        similarity1 = cosine_similarity(zero_vector, non_zero)
        similarity2 = cosine_similarity(non_zero, zero_vector)
        similarity3 = cosine_similarity(zero_vector, zero_vector)

        assert similarity1 == 0.0
        assert similarity2 == 0.0
        assert similarity3 == 0.0

    def test_symmetric_property(self, sample_embeddings):
        """cosine_similarity(a, b) should equal cosine_similarity(b, a)."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        sim_ab = cosine_similarity(a, b)
        sim_ba = cosine_similarity(b, a)

        assert_allclose(sim_ab, sim_ba)

    def test_scaled_vectors_same_similarity(self):
        """Scaling vectors should not change cosine similarity."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])  # 2x scaled

        # Should be very similar (both point in same direction)
        similarity = cosine_similarity(a, b)
        assert_allclose(similarity, 1.0)

    def test_numerical_stability_with_large_values(self):
        """Test with very large magnitude vectors."""
        a = np.array([1e10, 1e10, 1e10])
        b = np.array([1e10, 1e10, 1e10])

        similarity = cosine_similarity(a, b)
        assert_allclose(similarity, 1.0, rtol=1e-5)
        assert np.isfinite(similarity)


# ============================================================================
# Tests for cosine_distance
# ============================================================================


class TestCosineDistance:
    """Test suite for cosine_distance function."""

    def test_identical_vectors_returns_zero(self, identical_vectors):
        """Cosine distance of identical vectors should be 0.0."""
        a, b = identical_vectors
        distance = cosine_distance(a, b)

        assert_allclose(distance, 0.0)

    def test_orthogonal_vectors_returns_one(self, orthogonal_vectors):
        """Cosine distance of orthogonal vectors should be 1.0."""
        a, b = orthogonal_vectors
        distance = cosine_distance(a, b)

        assert_allclose(distance, 1.0, atol=1e-10)

    def test_opposite_vectors_returns_two(self, opposite_vectors):
        """Cosine distance of opposite vectors should be 2.0."""
        a, b = opposite_vectors
        distance = cosine_distance(a, b)

        assert_allclose(distance, 2.0)

    def test_result_in_valid_range(self, sample_embeddings):
        """Cosine distance should always be in [0, 2]."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]
        distance = cosine_distance(a, b)

        assert 0.0 <= distance <= 2.0

    def test_complement_of_similarity(self, sample_embeddings):
        """Distance should equal 1 - similarity."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        similarity = cosine_similarity(a, b)
        distance = cosine_distance(a, b)

        assert_allclose(distance, 1.0 - similarity)

    def test_symmetric_property(self, sample_embeddings):
        """cosine_distance(a, b) should equal cosine_distance(b, a)."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        dist_ab = cosine_distance(a, b)
        dist_ba = cosine_distance(b, a)

        assert_allclose(dist_ab, dist_ba)


# ============================================================================
# Tests for logsumexp_stable
# ============================================================================


class TestLogsumexpStable:
    """Test suite for logsumexp_stable function."""

    def test_numerical_stability_with_large_negative_values(self):
        """Should handle very large negative values without underflow."""
        log_vals = np.array([-1000.0, -1000.0, -1000.0])
        result = logsumexp_stable(log_vals)

        assert np.isfinite(result)
        # log(3 * exp(-1000)) H -1000 + log(3)
        expected = -1000.0 + np.log(3.0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_numerical_stability_with_large_positive_values(self):
        """Should handle very large positive values without overflow."""
        log_vals = np.array([1000.0, 1000.0, 1000.0])
        result = logsumexp_stable(log_vals)

        assert np.isfinite(result)
        # log(3 * exp(1000)) H 1000 + log(3)
        expected = 1000.0 + np.log(3.0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_mixed_scale_values(self):
        """Should handle mixed-scale values correctly."""
        log_vals = np.array([0.0, -100.0, 100.0])
        result = logsumexp_stable(log_vals)

        assert np.isfinite(result)
        # Should be dominated by the largest value (100.0)
        assert result > 99.0

    def test_single_value(self):
        """LogSumExp of a single value should return that value."""
        log_vals = np.array([5.0])
        result = logsumexp_stable(log_vals)

        assert_allclose(result, 5.0)

    def test_axis_parameter_2d_array(self):
        """Test axis parameter with 2D array."""
        log_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Along axis 0 (across rows)
        result_axis0 = logsumexp_stable(log_vals, axis=0)
        assert result_axis0.shape == (3,)

        # Along axis 1 (across columns)
        result_axis1 = logsumexp_stable(log_vals, axis=1)
        assert result_axis1.shape == (2,)

        # No axis (all elements)
        result_none = logsumexp_stable(log_vals, axis=None)
        assert isinstance(result_none, (float, np.floating))

    def test_empty_array(self):
        """Test behavior with empty array."""
        log_vals = np.array([])
        result = logsumexp_stable(log_vals)

        # Should return -inf for empty array
        assert result == -np.inf

    def test_correctness_against_naive_computation(self):
        """Verify correctness for moderate values."""
        log_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = logsumexp_stable(log_vals)
        expected = np.log(np.sum(np.exp(log_vals)))

        assert_allclose(result, expected, rtol=1e-10)


# ============================================================================
# Tests for normalize_embeddings
# ============================================================================


class TestNormalizeEmbeddings:
    """Test suite for normalize_embeddings function."""

    def test_1d_vector_normalization(self):
        """Test normalization of 1D vector to unit length."""
        vec = np.array([3.0, 4.0])
        normalized = normalize_embeddings(vec)

        norm = np.linalg.norm(normalized)
        assert_allclose(norm, 1.0)

        # Direction should be preserved
        expected = np.array([0.6, 0.8])
        assert_allclose(normalized, expected)

    def test_2d_array_normalization(self):
        """Test normalization of 2D array (batch of vectors)."""
        embeddings = np.array([[3.0, 4.0], [5.0, 12.0], [1.0, 0.0]])
        normalized = normalize_embeddings(embeddings)

        # Each row should have unit norm
        norms = np.linalg.norm(normalized, axis=1)
        assert_allclose(norms, np.ones(3))

        # Check specific values
        assert_allclose(normalized[0], [0.6, 0.8])
        assert_allclose(normalized[1], [5.0 / 13.0, 12.0 / 13.0])
        assert_allclose(normalized[2], [1.0, 0.0])

    def test_already_normalized_vector(self):
        """Already normalized vector should remain unchanged."""
        vec = np.array([1.0, 0.0, 0.0])
        normalized = normalize_embeddings(vec)

        assert_allclose(normalized, vec)

    def test_zero_vector_1d(self, zero_vector):
        """Zero vector should be returned as-is (1D case)."""
        normalized = normalize_embeddings(zero_vector)

        assert_array_equal(normalized, zero_vector)

    def test_zero_vector_2d(self):
        """Zero vectors in 2D array should be returned as-is."""
        embeddings = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])  # Zero vector
        normalized = normalize_embeddings(embeddings)

        # First and third should be normalized
        assert_allclose(np.linalg.norm(normalized[0]), 1.0)
        assert_allclose(np.linalg.norm(normalized[2]), 1.0)

        # Second (zero vector) should remain zero
        assert_array_equal(normalized[1], [0.0, 0.0])

    def test_large_magnitude_vectors(self):
        """Test with very large magnitude vectors."""
        vec = np.array([1e10, 1e10, 1e10])
        normalized = normalize_embeddings(vec)

        norm = np.linalg.norm(normalized)
        assert_allclose(norm, 1.0, rtol=1e-5)

    def test_small_magnitude_vectors(self):
        """Test with very small magnitude vectors."""
        vec = np.array([1e-10, 1e-10, 1e-10])
        normalized = normalize_embeddings(vec)

        norm = np.linalg.norm(normalized)
        assert_allclose(norm, 1.0, rtol=1e-5)

    def test_shape_preservation(self, sample_embeddings):
        """Output shape should match input shape."""
        normalized = normalize_embeddings(sample_embeddings)

        assert normalized.shape == sample_embeddings.shape

    def test_negative_values(self):
        """Should handle negative values correctly."""
        vec = np.array([-3.0, -4.0])
        normalized = normalize_embeddings(vec)

        norm = np.linalg.norm(normalized)
        assert_allclose(norm, 1.0)
        assert_allclose(normalized, [-0.6, -0.8])


# ============================================================================
# Tests for batch_cosine_similarity
# ============================================================================


class TestBatchCosineSimilarity:
    """Test suite for batch_cosine_similarity function."""

    def test_single_document_identical_to_query(self):
        """Identical query and document should give similarity of 1.0."""
        query = np.array([1.0, 0.0, 0.0])
        docs = np.array([[1.0, 0.0, 0.0]])

        sims = batch_cosine_similarity(query, docs)

        assert sims.shape == (1,)
        assert_allclose(sims[0], 1.0)

    def test_multiple_documents(self):
        """Test with multiple documents."""
        query = np.array([1.0, 0.0, 0.0])
        docs = np.array(
            [
                [1.0, 0.0, 0.0],  # Identical
                [0.0, 1.0, 0.0],  # Orthogonal
                [-1.0, 0.0, 0.0],  # Opposite
                [0.5, 0.5, 0.0],  # 45 degrees
            ]
        )

        sims = batch_cosine_similarity(query, docs)

        assert sims.shape == (4,)
        assert_allclose(sims[0], 1.0, atol=1e-6)
        assert_allclose(sims[1], 0.0, atol=1e-6)
        assert_allclose(sims[2], -1.0, atol=1e-6)
        # 45 degrees: cos(45 deg) ~ 0.707, but vector [0.5, 0.5, 0.0] normalized
        assert 0.6 < sims[3] < 0.8

    def test_output_shape(self, sample_embeddings):
        """Output should have shape (n_documents,)."""
        query = sample_embeddings[0]
        docs = sample_embeddings[1:6]  # 5 documents

        sims = batch_cosine_similarity(query, docs)

        assert sims.shape == (5,)

    def test_all_similarities_in_valid_range(self, sample_embeddings):
        """All similarities should be in [-1, 1]."""
        query = sample_embeddings[0]
        docs = sample_embeddings[1:]

        sims = batch_cosine_similarity(query, docs)

        assert np.all(sims >= -1.0)
        assert np.all(sims <= 1.0)

    def test_zero_query_vector(self, sample_embeddings):
        """Zero query vector should be handled gracefully."""
        query = np.zeros(sample_embeddings.shape[1])
        docs = sample_embeddings

        # Should not raise error
        sims = batch_cosine_similarity(query, docs)
        assert sims.shape == (len(docs),)
        # Results may be close to zero due to eps in normalization

    def test_zero_document_vectors(self):
        """Zero document vectors should be handled gracefully."""
        query = np.array([1.0, 2.0, 3.0])
        docs = np.array(
            [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [3.0, 2.0, 1.0]]  # Zero vector
        )

        sims = batch_cosine_similarity(query, docs)
        assert sims.shape == (3,)
        # Zero vector should give near-zero similarity
        assert abs(sims[1]) < 0.1

    def test_consistency_with_pairwise_cosine_similarity(self, sample_embeddings):
        """Batch result should match individual cosine_similarity calls."""
        query = sample_embeddings[0]
        docs = sample_embeddings[1:4]

        batch_sims = batch_cosine_similarity(query, docs)

        for i, doc in enumerate(docs):
            individual_sim = cosine_similarity(query, doc)
            assert_allclose(batch_sims[i], individual_sim, rtol=1e-5)

    def test_large_batch(self):
        """Test with a large batch of documents."""
        np.random.seed(123)
        query = np.random.randn(384)
        docs = np.random.randn(1000, 384)

        sims = batch_cosine_similarity(query, docs)

        assert sims.shape == (1000,)
        assert np.all(np.isfinite(sims))
        assert np.all(sims >= -1.0)
        assert np.all(sims <= 1.0)


# ============================================================================
# Tests for clamp_similarity
# ============================================================================


class TestClampSimilarity:
    """Test suite for clamp_similarity function."""

    def test_value_within_range_unchanged(self):
        """Value within range should not be changed."""
        assert clamp_similarity(0.5) == 0.5
        assert clamp_similarity(0.0) == 0.0
        assert clamp_similarity(-0.5) == -0.5

    def test_value_above_max_clamped(self):
        """Value above max should be clamped to max."""
        assert clamp_similarity(1.5) == 1.0
        assert clamp_similarity(2.0) == 1.0
        assert clamp_similarity(100.0) == 1.0

    def test_value_below_min_clamped(self):
        """Value below min should be clamped to min."""
        assert clamp_similarity(-1.5) == -1.0
        assert clamp_similarity(-2.0) == -1.0
        assert clamp_similarity(-100.0) == -1.0

    def test_boundary_values(self):
        """Boundary values should be preserved."""
        assert clamp_similarity(1.0) == 1.0
        assert clamp_similarity(-1.0) == -1.0

    def test_custom_range(self):
        """Should work with custom min/max values."""
        assert clamp_similarity(5.0, min_val=0.0, max_val=10.0) == 5.0
        assert clamp_similarity(15.0, min_val=0.0, max_val=10.0) == 10.0
        assert clamp_similarity(-5.0, min_val=0.0, max_val=10.0) == 0.0

    def test_inverted_range(self):
        """Should work even with inverted min/max (though unusual)."""
        # When min_val > max_val, max() is applied first, then min()
        result = clamp_similarity(0.5, min_val=1.0, max_val=-1.0)
        # max(1.0, min(-1.0, 0.5)) = max(1.0, -1.0) = 1.0
        assert result == 1.0

    def test_floating_point_precision(self):
        """Should handle floating point precision correctly."""
        # Slightly above 1.0 due to floating point error
        assert clamp_similarity(1.0000000001) == 1.0
        # Slightly below -1.0 due to floating point error
        assert clamp_similarity(-1.0000000001) == -1.0

    def test_extreme_values(self):
        """Should handle extreme values."""
        assert clamp_similarity(np.inf) == 1.0
        assert clamp_similarity(-np.inf) == -1.0

    def test_return_type(self):
        """Should return float type."""
        result = clamp_similarity(0.5)
        assert isinstance(result, float)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests combining multiple utility functions."""

    def test_normalized_embeddings_have_unit_cosine_with_themselves(self):
        """Normalized embeddings should have cosine similarity of 1.0 with themselves."""
        np.random.seed(456)
        embeddings = np.random.randn(5, 10)
        normalized = normalize_embeddings(embeddings)

        for vec in normalized:
            sim = cosine_similarity(vec, vec)
            assert_allclose(sim, 1.0)

    def test_batch_similarity_with_normalized_embeddings(self, sample_embeddings):
        """Batch similarity should work correctly with pre-normalized embeddings."""
        normalized = normalize_embeddings(sample_embeddings)

        query = normalized[0]
        docs = normalized[1:]

        sims = batch_cosine_similarity(query, docs)

        # All similarities should be in valid range
        assert np.all(sims >= -1.0)
        assert np.all(sims <= 1.0)

    def test_distance_is_complement_of_similarity(self, sample_embeddings):
        """Cosine distance should equal 1 - cosine similarity."""
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        sim = cosine_similarity(a, b)
        dist = cosine_distance(a, b)

        assert_allclose(dist, 1.0 - sim)
        assert_allclose(sim + dist, 1.0)

    def test_gaussian_kernel_symmetry_and_finiteness(self, sample_embeddings):
        """Gaussian kernel should be symmetric and always finite."""
        for i in range(5):
            for j in range(i + 1, 5):
                a = sample_embeddings[i]
                b = sample_embeddings[j]

                log_prob_ab = log_gaussian_kernel(a, b, sigma=1.0)
                log_prob_ba = log_gaussian_kernel(b, a, sigma=1.0)

                assert_allclose(log_prob_ab, log_prob_ba)
                assert np.isfinite(log_prob_ab)
                assert log_prob_ab <= 0.0

    def test_clamping_cosine_similarities(self, sample_embeddings):
        """Clamping should handle edge cases in cosine similarity."""
        # Create a scenario that might produce values slightly outside [-1, 1]
        # due to floating point errors
        a = sample_embeddings[0]
        b = sample_embeddings[1]

        sim = cosine_similarity(a, b)
        clamped = clamp_similarity(sim)

        assert -1.0 <= clamped <= 1.0
        # Should be very close to original if already in range
        assert_allclose(sim, clamped, atol=1e-10)

    def test_logsumexp_with_log_gaussian_kernels(self, sample_embeddings):
        """Test logsumexp with log Gaussian kernel values."""
        query = sample_embeddings[0]
        docs = sample_embeddings[1:6]

        log_probs = np.array(
            [log_gaussian_kernel(query, doc, sigma=1.0) for doc in docs]
        )

        # Compute log of sum of probabilities
        log_sum = logsumexp_stable(log_probs)

        assert np.isfinite(log_sum)
        assert log_sum <= 0.0  # Log of probability sum should be negative

    def test_full_similarity_pipeline(self):
        """Test complete similarity computation pipeline."""
        np.random.seed(789)

        # Generate raw embeddings
        query_raw = np.random.randn(128)
        docs_raw = np.random.randn(10, 128)

        # Normalize
        query_norm = normalize_embeddings(query_raw)
        docs_norm = normalize_embeddings(docs_raw)

        # Compute similarities
        sims = batch_cosine_similarity(query_norm, docs_norm)

        # Clamp (should be no-op for valid similarities)
        sims_clamped = np.array([clamp_similarity(s) for s in sims])

        # Verify
        assert sims.shape == (10,)
        assert np.all(np.isfinite(sims))
        assert np.all(sims >= -1.0)
        assert np.all(sims <= 1.0)
        assert_allclose(sims, sims_clamped)

        # Verify normalized query has unit norm
        assert_allclose(np.linalg.norm(query_norm), 1.0)

        # Verify all docs have unit norm
        doc_norms = np.linalg.norm(docs_norm, axis=1)
        assert_allclose(doc_norms, np.ones(10))
