# Dartboard RAG Algorithm - Technical Specification

**Version:** 1.0
**Date:** 2025-11-20
**Status:** Draft
**Author:** AI-Assisted Specification

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background & Motivation](#background--motivation)
3. [Mathematical Foundation](#mathematical-foundation)
4. [System Architecture](#system-architecture)
5. [Component Specifications](#component-specifications)
6. [Data Models](#data-models)
7. [API Specification](#api-specification)
8. [Dataset Requirements](#dataset-requirements)
9. [Evaluation Framework](#evaluation-framework)
10. [Implementation Phases](#implementation-phases)
11. [Testing Strategy](#testing-strategy)
12. [Performance Requirements](#performance-requirements)
13. [Security Considerations](#security-considerations)
14. [Open Questions](#open-questions)

---

## 1. Executive Summary

### Purpose
Implement the Dartboard algorithm (from arxiv 2407.12101) - a principled retrieval-augmented generation (RAG) system that optimizes for relevant information gain rather than explicit diversity-relevance tradeoffs.

### Key Innovation
Dartboard achieves diversity **organically** through an information-theoretic framework where redundant passages add minimal information gain, eliminating the need for manual diversity parameter tuning (unlike MMR's λ parameter).

### Scope
- **Core Algorithm**: Probabilistic retrieval using Gaussian kernel scoring in log-space
- **Dataset Generation**: Synthetic and real dataset loading for evaluation
- **Evaluation Framework**: Metrics, baselines, and comparative benchmarking
- **API Integration**: FastAPI endpoints for production deployment

### Success Criteria
- Beats cosine similarity baseline by ≥10% on diversity metrics
- Maintains competitive relevance (within 5% of cosine similarity)
- Achieves <100ms query latency for 1000 candidates
- ≥80% test coverage with comprehensive edge case handling

---

## 2. Background & Motivation

### Problem Statement
Traditional RAG systems face a fundamental challenge: retrieved passages must be both **relevant** to the query and **diverse** enough to avoid redundancy in the limited context window. Existing approaches like Maximal Marginal Relevance (MMR) use explicit λ parameters to balance these objectives, requiring manual tuning for each use case.

### Dartboard Solution
Dartboard frames retrieval as a probabilistic optimization problem:
- **Assumption**: One passage is the "correct" answer
- **Objective**: Make k guesses to maximize the probability of including the correct passage
- **Result**: Diversity emerges naturally because redundant guesses don't improve coverage

### Key Advantages
1. **No manual parameter tuning** (σ controls uncertainty, not diversity)
2. **Theoretically principled** (information-theoretic foundation)
3. **Generalizes existing methods** (reduces to KNN when σ→0)
4. **State-of-the-art performance** on RGB benchmark

### Reference
- **Paper**: "Better RAG using Relevant Information Gain" (arxiv 2407.12101)
- **Authors**: Marc Pickett, Jeremy Hartman, et al.
- **Published**: July 2024, updated February 2025
- **GitHub**: https://github.com/EmergenceAI/dartboard

---

## 3. Mathematical Foundation

### 3.1 Core Scoring Function

The Dartboard score for a set of retrieved passages G given query q is:

```
s(G, q, A, σ) = Σ_{t∈A} P(T=t|q,σ) × min_{g∈G} D(t|g)
```

Where:
- **G**: Set of k retrieved passages (what we're building)
- **q**: Query embedding
- **A**: Candidate passage pool
- **σ**: Temperature parameter (controls distribution sharpness)
- **P(T=t|q,σ)**: Probability that passage t is the true target
- **D(t|g)**: Distance between target t and guess g

### 3.2 Gaussian Kernel Formulation

Using Gaussian similarity kernels, the score becomes:

```
s(G, q, A, σ) ∝ −Σ_{t∈A} N(q, t, σ) × max_{g∈G} N(t, g, σ)
```

Where **N(a, b, σ)** is the Gaussian kernel:

```
N(a, b, σ) = exp(-||a - b||² / (2σ²))
```

### 3.3 Log-Space Implementation

**Critical**: All computations MUST be in log-space to prevent numerical underflow/overflow.

Log-probability of Gaussian kernel:
```python
log_N(a, b, σ) = -log(σ) - 0.5 * log(2π) - ||a - b||² / (2σ²)
```

Score computation using logsumexp:
```python
score = logsumexp([
    log_P(t|q, σ) + log(max_{g∈G} N(t, g, σ))
    for t in candidates
])
```

### 3.4 Greedy Selection Algorithm

**Pseudocode**:
```
Initialize: G = ∅
For i = 1 to k:
    For each candidate c in (A \ G):
        score_c = Σ_{t∈A} P(t|q,σ) × max(max_{g∈G} N(t,g,σ), N(t,c,σ))

    Select c* = argmax_c(score_c)
    G = G ∪ {c*}

Return G
```

**Key Properties**:
- Greedy (O(k × n) selections, not O(n^k) exhaustive search)
- Deterministic (given fixed random seed)
- Monotonic (adding passages can only increase total information)

### 3.5 Temperature Parameter σ

**Role**: Controls query uncertainty, NOT a diversity-relevance tradeoff

- **Low σ (0.1 - 0.5)**: Sharp distribution, high confidence → more relevance-focused
- **Medium σ (0.5 - 2.0)**: Balanced uncertainty → natural diversity
- **High σ (2.0 - 10.0)**: Broad distribution, high uncertainty → more diversity

**Important**: Unlike MMR's λ, σ has a probabilistic interpretation and is more robust to different datasets.

### 3.6 Theoretical Properties

**Proven in paper (Appendix A.4)**:

1. **Generalizes KNN**: As σ → 0, Dartboard → cosine similarity ranking
2. **Generalizes MMR**: With specific distance weighting, equivalent to MMR
3. **No exact duplicates**: The max operation prevents selecting identical passages
4. **Submodular**: Information gain exhibits diminishing returns (greedy is near-optimal)

---

## 4. System Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Dartboard Retriever                       │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Triage (KNN)  │→ │ Reranking    │→ │ Result Ranking  │  │
│  │ Top-K=100     │  │ (Dartboard)  │  │ Top-k=5         │  │
│  └───────────────┘  └──────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐  ┌─────────────────────┐  ┌──────────────┐
│  Embedding   │  │  Distance           │  │  Token       │
│  Model       │  │  Computation        │  │  Vocabulary  │
│ (Sentence    │  │ (Cosine/Cross-      │  │  (T)         │
│  Transformer)│  │  Encoder)           │  │              │
└──────────────┘  └─────────────────────┘  └──────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Vector Storage / Index                    │
│         (FAISS, Annoy, or In-Memory for <100K docs)         │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Component Responsibilities

**Dartboard Retriever**:
- Orchestrates the full retrieval pipeline
- Manages configuration (σ, k, triage_K)
- Handles three reranker variants (cosine, cross-encoder, hybrid)

**Triage Module**:
- Fast KNN retrieval using vector similarity
- Reduces candidate pool from N documents to K candidates (e.g., 100)
- Uses FAISS or similar for efficiency

**Reranking Module**:
- Implements core Dartboard algorithm
- Computes log-space Gaussian scores
- Performs greedy selection with k iterations

**Embedding Model**:
- Generates dense vector representations
- Supports multiple backends (sentence-transformers, OpenAI, etc.)
- Caches embeddings for performance

**Distance Computation**:
- Calculates query-passage distances
- Calculates passage-passage distances (K×K matrix)
- Supports cosine similarity and cross-encoder scoring

**Token Vocabulary (T)**:
- Represents the "space" of possible passages
- Used for normalized probability distributions
- Can be word tokens, subword tokens, or cluster centroids

---

## 5. Component Specifications

### 5.1 Core Module: `dartboard/core.py`

#### Class: `DartboardConfig`
```python
@dataclass
class DartboardConfig:
    """Configuration for Dartboard retrieval."""

    # Temperature parameter for Gaussian kernel
    sigma: float = 1.0  # Range: [0.1, 10.0], typically 0.5-2.0

    # Number of passages to retrieve
    top_k: int = 5  # Range: [1, 20]

    # Number of candidates from triage phase
    triage_k: int = 100  # Range: [top_k, 1000]

    # Reranker variant
    reranker_type: Literal["cosine", "crossencoder", "hybrid"] = "hybrid"

    # Token vocabulary size for normalized similarity
    token_vocab_size: int = 5000  # Range: [1000, 50000]

    # Numerical stability epsilon
    log_eps: float = 1e-10

    # Minimum sigma to prevent division by zero
    sigma_min: float = 1e-5
```

#### Class: `DartboardRetriever`
```python
class DartboardRetriever:
    """Main retrieval class implementing Dartboard algorithm."""

    def __init__(
        self,
        config: DartboardConfig,
        embedding_model: EmbeddingModel,
        token_embeddings: np.ndarray,  # Shape: (vocab_size, embed_dim)
        cross_encoder: Optional[CrossEncoder] = None
    ):
        """Initialize retriever with configuration and models."""

    def retrieve(
        self,
        query: str,
        corpus: List[Chunk],
        return_scores: bool = False
    ) -> Union[List[Chunk], Tuple[List[Chunk], List[float]]]:
        """
        Retrieve top-k chunks using Dartboard algorithm.

        Args:
            query: User query string
            corpus: List of document chunks to search
            return_scores: If True, also return Dartboard scores

        Returns:
            List of top-k chunks (optionally with scores)

        Raises:
            ValueError: If query is empty or corpus is invalid
        """

    def _triage(
        self,
        query_embedding: np.ndarray,
        corpus: List[Chunk]
    ) -> List[Chunk]:
        """
        Fast KNN-based candidate selection.

        Returns: Top triage_k candidates by cosine similarity
        """

    def _compute_distances(
        self,
        query_embedding: np.ndarray,
        candidates: List[Chunk]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distance matrices.

        Returns:
            query_dists: (K,) array of query-to-candidate distances
            pairwise_dists: (K, K) array of candidate-to-candidate distances
        """

    def _greedy_selection(
        self,
        query_log_probs: np.ndarray,
        pairwise_log_probs: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Greedy selection using Dartboard scoring.

        Args:
            query_log_probs: Log P(candidate|query) for each candidate
            pairwise_log_probs: Log P(candidate_i|candidate_j) matrix
            k: Number of passages to select

        Returns:
            List of selected indices in order of selection
        """

    def _compute_dartboard_score(
        self,
        selected_indices: List[int],
        candidate_idx: int,
        query_log_probs: np.ndarray,
        pairwise_log_probs: np.ndarray
    ) -> float:
        """
        Compute Dartboard score for adding candidate to selected set.

        Implementation:
            For each potential target t:
                score += P(t|query) * max(max_{s in selected} N(t|s), N(t|candidate))

        Uses logsumexp for numerical stability.
        """
```

### 5.2 Utility Module: `dartboard/utils.py`

```python
def log_gaussian_kernel(
    a: np.ndarray,
    b: np.ndarray,
    sigma: float,
    eps: float = 1e-10
) -> float:
    """
    Compute log of Gaussian kernel N(a, b, σ).

    Formula: -log(σ) - 0.5*log(2π) - ||a-b||²/(2σ²)

    Args:
        a, b: Embedding vectors (must have same dimension)
        sigma: Temperature parameter
        eps: Small constant for numerical stability

    Returns:
        Log-probability value

    Raises:
        ValueError: If embeddings have different dimensions
    """

def logsumexp_stable(
    log_values: np.ndarray,
    axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Numerically stable log-sum-exp.

    Computes: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))

    Uses scipy.special.logsumexp internally.
    """

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """Compute cosine similarity between two vectors."""

def cosine_distance(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Compute cosine distance: 1 - cosine_similarity(a, b).

    Range: [0, 2], where 0 = identical, 2 = opposite
    """
```

### 5.3 Embedding Module: `dartboard/embeddings.py`

```python
class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        """Encode text(s) to dense vectors."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return embedding dimensionality."""

class SentenceTransformerModel(EmbeddingModel):
    """Wrapper for sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, batch_size=batch_size)

    @property
    def embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

class CrossEncoder:
    """Wrapper for cross-encoder reranking models."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        device: str = "cpu"
    ):
        from sentence_transformers import CrossEncoder as CE
        self.model = CE(model_name, device=device)

    def score(
        self,
        query: str,
        passages: List[str]
    ) -> np.ndarray:
        """
        Score query-passage pairs.

        Returns: Array of relevance scores (higher = more relevant)
        """
        pairs = [(query, passage) for passage in passages]
        return self.model.predict(pairs)

    def score_matrix(
        self,
        passages: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise passage similarity scores.

        Returns: (N, N) symmetric matrix of scores
        """
        n = len(passages)
        scores = np.zeros((n, n))

        # Compute upper triangle
        for i in range(n):
            for j in range(i+1, n):
                score = self.model.predict([(passages[i], passages[j])])[0]
                scores[i, j] = score
                scores[j, i] = score  # Symmetric

        return scores
```

### 5.4 Token Vocabulary Module: `dartboard/vocab.py`

```python
def build_token_vocabulary(
    embedding_model: EmbeddingModel,
    vocab_size: int = 5000,
    method: Literal["words", "subwords", "clusters"] = "words"
) -> np.ndarray:
    """
    Build token embedding vocabulary T.

    Methods:
        - "words": Most common words from pre-trained model
        - "subwords": Subword tokens from tokenizer
        - "clusters": K-means clusters of word embeddings

    Returns:
        Token embeddings array of shape (vocab_size, embed_dim)
    """

def build_from_words(
    embedding_model: EmbeddingModel,
    vocab_size: int
) -> np.ndarray:
    """
    Build vocabulary from most common words.

    Uses NLTK word frequency lists or similar.
    """

def build_from_clusters(
    embedding_model: EmbeddingModel,
    vocab_size: int,
    sample_size: int = 100000
) -> np.ndarray:
    """
    Build vocabulary from k-means clustering.

    1. Sample many words/phrases
    2. Encode to embeddings
    3. Run k-means with k=vocab_size
    4. Return cluster centroids as token embeddings
    """
```

---

## 6. Data Models

### 6.1 Core Data Structures

```python
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

@dataclass
class Chunk:
    """Represents a document chunk with metadata."""

    id: str  # Unique identifier
    text: str  # Full text content
    embedding: np.ndarray  # Dense vector representation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate chunk data."""
        if not self.id:
            raise ValueError("Chunk ID cannot be empty")
        if not self.text:
            raise ValueError("Chunk text cannot be empty")
        if self.embedding is None or len(self.embedding) == 0:
            raise ValueError("Chunk must have valid embedding")

@dataclass
class RetrievalResult:
    """Result of a Dartboard retrieval operation."""

    query: str
    chunks: List[Chunk]
    scores: List[float]
    metadata: Dict[str, Any]

    @property
    def top_chunk(self) -> Chunk:
        """Return highest-scoring chunk."""
        return self.chunks[0] if self.chunks else None

@dataclass
class Dataset:
    """Container for evaluation datasets."""

    name: str
    chunks: List[Chunk]
    queries: List[str]
    ground_truth: Dict[str, List[str]]  # query_id -> relevant_chunk_ids
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.queries)
```

### 6.2 Configuration Models

```python
@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation."""

    num_clusters: int = 5
    docs_per_cluster: int = 20
    cluster_separation: float = 0.8
    intra_cluster_similarity: float = 0.9
    embed_dim: int = 384
    random_seed: int = 42

@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    datasets: List[str]  # Dataset names to evaluate on
    retrievers: List[str]  # Retriever names to compare
    metrics: List[str]  # Metric names to compute
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    output_dir: str = "results/"
```

---

## 7. API Specification

### 7.1 REST API Endpoints

**Base URL**: `/api/dartboard`

#### POST `/retrieve`
Retrieve passages using Dartboard algorithm.

**Request Body**:
```json
{
  "query": "How to implement RAG?",
  "top_k": 5,
  "sigma": 1.2,
  "reranker_type": "hybrid",
  "corpus_ids": ["doc1", "doc2", ...],  // Optional: filter corpus
  "return_scores": true
}
```

**Response**:
```json
{
  "query": "How to implement RAG?",
  "results": [
    {
      "chunk_id": "chunk_123",
      "text": "RAG (Retrieval-Augmented Generation) combines...",
      "score": 0.8734,
      "rank": 1,
      "metadata": {"source": "doc1", "page": 3}
    },
    ...
  ],
  "execution_time_ms": 87.3,
  "config": {
    "sigma": 1.2,
    "top_k": 5,
    "reranker_type": "hybrid"
  }
}
```

#### POST `/evaluate`
Run evaluation on a dataset.

**Request Body**:
```json
{
  "dataset_name": "msmarco_dev",
  "config": {
    "sigma": 1.0,
    "top_k": 5
  },
  "metrics": ["ndcg", "precision", "diversity"]
}
```

**Response**:
```json
{
  "dataset": "msmarco_dev",
  "metrics": {
    "ndcg@5": 0.763,
    "precision@5": 0.842,
    "diversity": 0.671
  },
  "num_queries": 100,
  "execution_time_ms": 12300
}
```

#### GET `/health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "embedding_model": "all-MiniLM-L6-v2",
  "corpus_size": 10000
}
```

#### GET `/datasets`
List available datasets.

**Response**:
```json
{
  "datasets": [
    {
      "name": "synthetic_clustered",
      "num_chunks": 100,
      "num_queries": 20,
      "type": "synthetic"
    },
    {
      "name": "msmarco_dev",
      "num_chunks": 8841823,
      "num_queries": 6980,
      "type": "real"
    }
  ]
}
```

---

## 8. Dataset Requirements

### 8.1 Synthetic Datasets

**Purpose**: Controlled evaluation with known ground truth.

#### Clustered Dataset
- **Structure**: Documents grouped into semantic clusters
- **Properties**:
  - High intra-cluster similarity (0.8-0.9 cosine)
  - Low inter-cluster similarity (0.2-0.4 cosine)
  - Optimal retrieval: One document per cluster
- **Use Case**: Test diversity emergence

#### Query-Focused Dataset
- **Structure**: Queries with labeled relevant/irrelevant documents
- **Properties**:
  - Clear relevance boundaries
  - Some relevant docs are near-duplicates (test diversity)
  - Multiple relevant docs per query
- **Use Case**: Test relevance-diversity tradeoff

#### Adversarial Dataset
- **Structure**: Edge cases and failure modes
- **Properties**:
  - Near-duplicate documents (differ by 1-2 words)
  - Semantically identical but lexically different
  - Contradictory information
- **Use Case**: Stress testing and robustness

### 8.2 Real Datasets

#### MS MARCO
- **Source**: Microsoft Machine Reading Comprehension
- **Size**: ~8.8M passages, ~6.9K dev queries
- **Format**: Passage retrieval with binary relevance labels
- **Download**: HuggingFace `datasets` library

#### BEIR Benchmark
- **Source**: Benchmark for Information Retrieval
- **Datasets**: SciFact, NFCorpus, FiQA, etc.
- **Format**: Zero-shot retrieval across domains
- **Download**: BEIR library

### 8.3 Data Preprocessing

**Required Steps**:
1. **Text cleaning**: Remove special characters, normalize whitespace
2. **Chunking**: Split long documents (if applicable)
3. **Embedding generation**: Encode all passages
4. **Caching**: Store embeddings in HDF5 or memmap
5. **Indexing**: Build vector index (FAISS) for fast retrieval

---

## 9. Evaluation Framework

### 9.1 Relevance Metrics

```python
def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """Precision@K = |relevant ∩ retrieved[:k]| / k"""

def recall_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int
) -> float:
    """Recall@K = |relevant ∩ retrieved[:k]| / |relevant|"""

def ndcg_at_k(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    k: int
) -> float:
    """
    Normalized Discounted Cumulative Gain.

    DCG@K = Σ (2^rel_i - 1) / log2(i + 2)
    NDCG@K = DCG@K / IDCG@K
    """
```

### 9.2 Diversity Metrics

```python
def pairwise_diversity(
    chunks: List[Chunk],
    similarity_fn: Callable = cosine_similarity
) -> float:
    """
    Average pairwise dissimilarity.

    diversity = 1 - mean(similarity(chunk_i, chunk_j)) for all i ≠ j

    Range: [0, 1], higher = more diverse
    """

def cluster_coverage(
    retrieved_chunks: List[Chunk],
    cluster_labels: Dict[str, int]
) -> float:
    """
    Fraction of clusters covered.

    coverage = num_unique_clusters(retrieved) / total_clusters

    Range: [0, 1], higher = better coverage
    """

def alpha_ndcg(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    similarity_matrix: np.ndarray,
    alpha: float = 0.5,
    k: int = 10
) -> float:
    """
    α-NDCG: Balances relevance and novelty.

    From paper: "Novelty and Diversity in Information Retrieval Evaluation"

    Args:
        alpha: Novelty weight (0 = pure relevance, 1 = pure novelty)
    """
```

### 9.3 Composite Metrics

```python
def f1_diversity(
    precision: float,
    diversity: float
) -> float:
    """
    Harmonic mean of precision and diversity.

    F1 = 2 * (P * D) / (P + D)
    """
```

### 9.4 Baseline Retrievers

**Implementation Requirements**:

1. **Cosine Similarity**: Pure vector similarity ranking
2. **MMR**: With λ ∈ {0.3, 0.5, 0.7, 0.9} for comparison
3. **Random**: Random passage selection
4. **Oracle**: Perfect retrieval (upper bound)

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1)
**Goal**: Set up project structure and core utilities.

**Tasks**:
- [x] Create project directory structure
- [ ] Implement utility functions (log-space ops, similarity metrics)
- [ ] Set up embedding model wrapper
- [ ] Generate token vocabulary
- [ ] Write unit tests for utilities

**Deliverables**:
- Working utility module with 100% test coverage
- Token embeddings file cached on disk

### Phase 2: Core Algorithm (Week 1-2)
**Goal**: Implement Dartboard retrieval.

**Tasks**:
- [ ] Implement Gaussian kernel in log-space
- [ ] Implement greedy selection algorithm
- [ ] Implement triage module (KNN)
- [ ] Add three reranker variants
- [ ] Comprehensive unit testing

**Deliverables**:
- `DartboardRetriever` class fully functional
- ≥80% test coverage
- Passes numerical stability tests

### Phase 3: Synthetic Datasets (Week 2)
**Goal**: Generate synthetic evaluation data.

**Tasks**:
- [ ] Implement clustered dataset generator
- [ ] Implement query-focused dataset generator
- [ ] Implement adversarial dataset generator
- [ ] Add ground truth generation
- [ ] Write dataset validation tests

**Deliverables**:
- Three synthetic dataset types
- Dataset generation scripts
- Validation tests passing

### Phase 4: Evaluation Framework (Week 2-3)
**Goal**: Build comprehensive evaluation system.

**Tasks**:
- [ ] Implement all 6 metrics (precision, recall, NDCG, diversity, etc.)
- [ ] Implement baseline retrievers
- [ ] Create benchmark runner
- [ ] Add statistical significance tests
- [ ] Generate comparison reports

**Deliverables**:
- Metrics module with tests
- Baseline implementations
- Benchmark runner producing markdown reports

### Phase 5: Real Datasets (Week 3)
**Goal**: Integrate real-world datasets.

**Tasks**:
- [ ] Implement MS MARCO loader
- [ ] Implement BEIR loader
- [ ] Set up embedding cache
- [ ] Add preprocessing pipeline
- [ ] Integration tests

**Deliverables**:
- Dataset loaders functional
- Embeddings cached efficiently
- Can run evaluations on real data

### Phase 6: FastAPI Integration (Week 3-4)
**Goal**: Production API deployment.

**Tasks**:
- [ ] Create API routes
- [ ] Add request/response models (Pydantic)
- [ ] Implement error handling
- [ ] Add API tests
- [ ] Write API documentation

**Deliverables**:
- Working REST API
- API documentation (auto-generated)
- API integration tests

### Phase 7: Optimization & Documentation (Week 4)
**Goal**: Polish and productionize.

**Tasks**:
- [ ] Profile and optimize performance
- [ ] Add FAISS integration (optional)
- [ ] Write comprehensive documentation
- [ ] Create example notebooks
- [ ] Final validation and testing

**Deliverables**:
- Meets all performance requirements
- Complete documentation
- Example usage notebooks

---

## 11. Testing Strategy

### 11.1 Unit Tests

**Coverage Target**: ≥80%

**Test Files**:
- `tests/test_core.py`: Core algorithm functions
- `tests/test_utils.py`: Utility functions
- `tests/test_embeddings.py`: Embedding models
- `tests/test_datasets.py`: Dataset generation
- `tests/test_metrics.py`: Evaluation metrics

**Key Test Cases**:
```python
def test_log_gaussian_kernel_properties():
    """Verify log N(a,b,σ) is valid log-probability."""

def test_dartboard_score_first_iteration():
    """When G=∅, score should equal cosine similarity."""

def test_diversity_emerges():
    """Redundant passages should get lower scores."""

def test_sigma_effect():
    """Higher σ should increase diversity."""

def test_numerical_stability():
    """No NaN/Inf with extreme values."""
```

### 11.2 Integration Tests

**Test Full Pipeline**:
```python
def test_end_to_end_retrieval():
    """Query → Triage → Rerank → Results."""

def test_api_endpoints():
    """FastAPI routes return valid responses."""

def test_evaluation_pipeline():
    """Dataset → Retrieve → Evaluate → Report."""
```

### 11.3 Property-Based Tests

**Using Hypothesis**:
```python
@given(
    num_chunks=st.integers(5, 100),
    k=st.integers(1, 10),
    sigma=st.floats(0.1, 5.0)
)
def test_retrieval_invariants(num_chunks, k, sigma):
    """Properties that should always hold."""
    results = retriever.retrieve(query, chunks, k, sigma)

    assert len(results) <= k  # Never return more than k
    assert len(results) == len(set(results))  # All unique
    assert all(r in chunks for r in results)  # All from corpus
```

### 11.4 Regression Tests

**Performance Baselines**:
```python
def test_beats_cosine_on_diversity():
    """Dartboard diversity ≥ Cosine diversity + 10%."""

def test_maintains_relevance():
    """Dartboard precision ≥ Cosine precision - 5%."""

def test_latency_requirement():
    """Query latency < 100ms for 1000 candidates."""
```

---

## 12. Performance Requirements

### 12.1 Latency Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Query processing (1K candidates) | 50ms | 100ms |
| Embedding lookup (cached) | 5ms | 10ms |
| Triage (KNN, 10K corpus) | 20ms | 50ms |
| Reranking (100 candidates, k=5) | 30ms | 100ms |
| End-to-end (query → results) | 100ms | 200ms |

### 12.2 Memory Footprint

| Component | Typical | Maximum |
|-----------|---------|---------|
| Embeddings (100K docs, 384-dim) | 150MB | 500MB |
| Token vocabulary (5K, 384-dim) | 8MB | 20MB |
| Distance matrices (100×100) | 40KB | 1MB |
| Total application | 500MB | 2GB |

### 12.3 Scalability

- **Corpus Size**: Support up to 1M documents (with FAISS indexing)
- **Concurrent Queries**: 100 queries/second (with load balancing)
- **Batch Processing**: 1000 queries in <60 seconds

### 12.4 Optimization Strategies

1. **Embedding Cache**: Pre-compute and cache all embeddings
2. **FAISS Index**: Use approximate nearest neighbors for triage
3. **Vectorization**: NumPy/PyTorch for batch operations
4. **Token Similarity Cache**: Pre-compute chunk-token similarities
5. **Multiprocessing**: Parallel evaluation of multiple queries

---

## 13. Security Considerations

### 13.1 Input Validation

- **Query Length**: Limit to 512 tokens (prevent DoS)
- **Corpus Size**: Limit API requests to reasonable corpus sizes
- **Parameter Ranges**: Validate σ, k, triage_k within safe bounds

### 13.2 Data Privacy

- **No User Data Logging**: Don't log queries or results by default
- **Embedding Security**: Ensure embeddings can't reverse-engineer text
- **API Authentication**: Add authentication for production (future)

### 13.3 Resource Limits

- **Memory Limits**: Cap maximum memory usage per request
- **Timeout**: Set request timeout (e.g., 30 seconds)
- **Rate Limiting**: Prevent abuse with request rate limits

---

## 14. Open Questions

### 14.1 Token Vocabulary Construction

**Question**: What is the optimal way to construct token vocabulary T?

**Options**:
1. Use word-level tokens from embedding model
2. Use k-means clustering of token embeddings
3. Use subword tokens from tokenizer
4. Use a fixed vocabulary (e.g., GloVe)

**Current Approach**: Start with word-level, experiment with clustering.

**Decision Needed**: After initial implementation, benchmark performance.

### 14.2 Cross-Encoder Score Rescaling

**Question**: How to normalize cross-encoder scores to [0, 1] range?

**Options**:
1. Use fixed min/max bounds (as in reference implementation)
2. Normalize per-query using min-max scaling
3. Use sigmoid transformation
4. Learn normalization parameters

**Current Approach**: Use fixed bounds from paper (-11.6, 11.4).

**Decision Needed**: Validate with experiments on different cross-encoders.

### 14.3 FAISS Integration

**Question**: When should we use FAISS for triage?

**Threshold**: Corpus size > 10K documents?

**Index Type**: IVF (Inverted File Index) or HNSW (Hierarchical NSW)?

**Decision Needed**: After performance profiling.

### 14.4 Embedding Model Selection

**Question**: What is the default embedding model?

**Options**:
- `all-MiniLM-L6-v2`: Fast, 384-dim (default)
- `all-mpnet-base-v2`: Better quality, 768-dim
- OpenAI `text-embedding-3-small`: Proprietary, high quality

**Current Approach**: Use `all-MiniLM-L6-v2` for development.

**Decision Needed**: User-configurable with recommended defaults.

---

## Appendix A: Mathematical Derivations

### A.1 Log-Space Conversion

Starting from the Gaussian kernel:
```
N(a, b, σ) = (1 / (σ√(2π))) * exp(-||a - b||² / (2σ²))
```

Taking logarithm:
```
log N(a, b, σ) = log(1 / (σ√(2π))) - ||a - b||² / (2σ²)
                = -log(σ) - 0.5 * log(2π) - ||a - b||² / (2σ²)
```

### A.2 LogSumExp Trick

To compute `log(Σ exp(x_i))` stably:
```
log(Σ exp(x_i)) = log(exp(max(x)) * Σ exp(x_i - max(x)))
                 = max(x) + log(Σ exp(x_i - max(x)))
```

This prevents overflow when `max(x)` is large.

---

## Appendix B: References

1. **Dartboard Paper**: "Better RAG using Relevant Information Gain"
   - ArXiv: 2407.12101
   - GitHub: https://github.com/EmergenceAI/dartboard

2. **MMR**: "The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries"
   - Carbonell & Goldstein, 1998

3. **BEIR Benchmark**: "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"
   - Thakur et al., 2021

4. **Sentence Transformers**: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - Reimers & Gurevych, 2019

---

**End of Specification Document**

**Version History**:
- v1.0 (2025-11-20): Initial specification
