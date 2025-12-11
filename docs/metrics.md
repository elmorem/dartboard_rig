# Evaluation Metrics

## Overview

This repository computes **relevance metrics** (how accurate is retrieval?) and **diversity metrics** (how diverse are results?) to comprehensively evaluate retrieval systems.

## Relevance Metrics

### 1. Mean Reciprocal Rank (MRR@K)

**Definition**: Average of reciprocal ranks of first relevant document

**Formula**:
```
MRR@K = (1/|Q|) * Σ (1 / rank_i)
```

Where `rank_i` = position of first relevant doc for query i (within top K)

**Range**: [0, 1], higher is better

**Interpretation**:
- MRR@10 = 1.0: First relevant doc always at rank 1
- MRR@10 = 0.5: First relevant doc at avg rank 2
- MRR@10 = 0.1: First relevant doc at avg rank 10

**Example**:
```
Query 1: Relevant doc at rank 1 → 1/1 = 1.0
Query 2: Relevant doc at rank 3 → 1/3 = 0.333
Query 3: No relevant in top-10 → 0.0

MRR@10 = (1.0 + 0.333 + 0.0) / 3 = 0.444
```

**Use Case**: Single-answer questions ("What year did X happen?")

**Code**:
```python
from dartboard.evaluation.metrics import mean_reciprocal_rank

mrr = mean_reciprocal_rank(
    results=["doc1", "doc2", "doc3"],  # Retrieved doc IDs
    relevant_docs={"doc2"},  # Ground truth
    k=10
)
# mrr = 0.5 (relevant doc at rank 2)
```

### 2. Recall@K

**Definition**: Fraction of relevant documents retrieved in top-K

**Formula**:
```
Recall@K = |Relevant ∩ Retrieved@K| / |Relevant|
```

**Range**: [0, 1], higher is better

**Interpretation**:
- Recall@10 = 1.0: All relevant docs in top-10
- Recall@10 = 0.5: Half of relevant docs in top-10
- Recall@10 = 0.0: No relevant docs in top-10

**Example**:
```
Relevant docs: {doc2, doc4, doc7}  (3 docs)
Retrieved top-10: [doc1, doc2, doc3, doc4, doc5, ...]

Retrieved relevant: {doc2, doc4}  (2 docs)
Recall@10 = 2 / 3 = 0.667
```

**Use Case**: Comprehensive search (need all relevant docs)

**Code**:
```python
from dartboard.evaluation.metrics import recall_at_k

recall = recall_at_k(
    results=retrieved_ids,
    relevant_docs=relevant_ids,
    k=10
)
```

### 3. Precision@K

**Definition**: Fraction of retrieved documents that are relevant

**Formula**:
```
Precision@K = |Relevant ∩ Retrieved@K| / K
```

**Range**: [0, 1], higher is better

**Interpretation**:
- Precision@10 = 1.0: All 10 retrieved docs are relevant
- Precision@10 = 0.5: Half of retrieved docs are relevant
- Precision@10 = 0.0: No retrieved docs are relevant

**Example**:
```
Relevant docs: {doc2, doc4, doc7}
Retrieved top-10: [doc1, doc2, doc3, doc4, doc5, ...]

Retrieved relevant: {doc2, doc4}  (2 out of 10)
Precision@10 = 2 / 10 = 0.2
```

**Use Case**: Minimize irrelevant results (user sees only top results)

**Code**:
```python
from dartboard.evaluation.metrics import precision_at_k

precision = precision_at_k(
    results=retrieved_ids,
    relevant_docs=relevant_ids,
    k=10
)
```

### 4. Normalized Discounted Cumulative Gain (NDCG@K)

**Definition**: Ranking quality with position weighting (top-ranked docs matter more)

**Formula**:
```
DCG@K = Σ_{i=1}^K (gain_i / log2(i + 1))
NDCG@K = DCG@K / IDCG@K
```

Where:
- `gain_i` = relevance of doc at rank i (1 for relevant, 0 for not)
- `IDCG@K` = ideal DCG (if all relevant docs ranked first)

**Range**: [0, 1], higher is better

**Interpretation**:
- NDCG@10 = 1.0: Perfect ranking (all relevant docs at top)
- NDCG@10 = 0.7: Good ranking
- NDCG@10 = 0.3: Poor ranking

**Example**:
```
Retrieved: [doc1(0), doc2(1), doc3(0), doc4(1), doc5(0)]
Relevance: [0, 1, 0, 1, 0]

DCG = 0/log2(2) + 1/log2(3) + 0/log2(4) + 1/log2(5) + ...
    = 0 + 0.631 + 0 + 0.431
    = 1.062

Ideal (relevant docs first): [doc2(1), doc4(1), doc1(0), ...]
IDCG = 1/log2(2) + 1/log2(3) + 0/... = 1 + 0.631 = 1.631

NDCG = 1.062 / 1.631 = 0.651
```

**Use Case**: Ranking quality (most common metric in IR)

**Code**:
```python
from dartboard.evaluation.metrics import ndcg_at_k

ndcg = ndcg_at_k(
    results=retrieved_ids,
    relevant_docs=relevant_ids,
    k=10
)
```

### 5. Mean Average Precision (MAP@K)

**Definition**: Mean of average precision scores across queries

**Formula**:
```
AP@K = (1/|Relevant|) * Σ_{i=1}^K (Precision@i × is_relevant_i)
MAP@K = mean(AP@K across all queries)
```

**Range**: [0, 1], higher is better

**Interpretation**:
- MAP@100 = 0.8: Excellent overall ranking
- MAP@100 = 0.5: Good ranking
- MAP@100 = 0.2: Poor ranking

**Example**:
```
Retrieved: [doc2✓, doc3✗, doc4✓, doc5✗]
Relevant: {doc2, doc4, doc7}

Precision@1 = 1/1 = 1.0 (doc2 relevant)
Precision@2 = 1/2 = 0.5 (doc3 not relevant)
Precision@3 = 2/3 = 0.667 (doc4 relevant)
Precision@4 = 2/4 = 0.5 (doc5 not relevant)

AP = (1.0×1 + 0.667×1) / 3 = 0.556
```

**Use Case**: Overall ranking quality (accounts for all ranks)

**Code**:
```python
from dartboard.evaluation.metrics import average_precision, mean_average_precision

# Single query
ap = average_precision(results, relevant_docs, k=100)

# Multiple queries
map_score = mean_average_precision(
    all_results=[results1, results2, ...],
    all_relevant_docs=[relevant1, relevant2, ...],
    k=100
)
```

## Diversity Metrics

### 6. Intra-List Diversity (ILD)

**Definition**: Average dissimilarity between all pairs of retrieved documents

**Formula**:
```
ILD = (2 / (|R| × (|R|-1))) * Σ_{i<j} dissimilarity(doc_i, doc_j)
```

Where `dissimilarity` = 1 - cosine_similarity(embedding_i, embedding_j)

**Range**: [0, 1], higher is better (more diverse)

**Interpretation**:
- ILD = 0.9: Highly diverse results (documents cover different topics)
- ILD = 0.5: Medium diversity
- ILD = 0.2: Low diversity (redundant, similar results)

**Example**:
```
Results: [doc1, doc2, doc3]

Pairwise similarities:
  sim(doc1, doc2) = 0.9  → dissim = 0.1
  sim(doc1, doc3) = 0.3  → dissim = 0.7
  sim(doc2, doc3) = 0.4  → dissim = 0.6

ILD = (0.1 + 0.7 + 0.6) / 3 = 0.467
```

**Use Case**: Measuring result diversity, detecting redundancy

**Code**:
```python
from dartboard.evaluation.metrics import intra_list_diversity

# Requires embeddings
embeddings_dict = {doc.id: doc.embedding for doc in all_docs}

ild = intra_list_diversity(
    results=retrieved_ids,
    embeddings_dict=embeddings_dict
)
```

**Benchmark Results** (SciFact):
- BM25: ILD = 0.34
- Dense: ILD = 0.38
- Hybrid: ILD = 0.36
- **Dartboard: ILD = 0.89** ✅ (much more diverse)

### 7. Alpha-NDCG@K

**Definition**: NDCG variant that penalizes redundant documents

**Formula**:
```
gain_i = relevance_i × (α + (1-α) × novelty_i)
novelty_i = 1 - max_similarity(doc_i, {previously selected docs})

α-NDCG = DCG(gains) / IDCG
```

Where `α` ∈ [0, 1]:
- α = 1: Pure relevance (standard NDCG)
- α = 0.5: Balanced relevance + diversity
- α = 0: Pure diversity

**Range**: [0, 1], higher is better

**Interpretation**:
- α-NDCG@10 = 0.8: Good diversity-aware ranking
- Higher α-NDCG than NDCG → diverse results without hurting relevance

**Example**:
```
Retrieved: [doc1✓(0.9 sim to prev), doc2✓(0.2 sim), doc3✗(0.1 sim)]
α = 0.5

gain1 = 1 × (0.5 + 0.5 × 1.0) = 1.0  (no previous docs)
gain2 = 1 × (0.5 + 0.5 × 0.8) = 0.9  (novel, sim to doc1 = 0.2)
gain3 = 0 × ... = 0  (not relevant)

α-DCG = 1.0/log2(2) + 0.9/log2(3) + 0/log2(4)
      = 1.0 + 0.568 = 1.568

α-NDCG = 1.568 / IDCG
```

**Use Case**: Evaluating diversity-aware retrievers (like Dartboard)

**Code**:
```python
from dartboard.evaluation.metrics import alpha_ndcg

a_ndcg = alpha_ndcg(
    results=retrieved_ids,
    relevant_docs=relevant_ids,
    embeddings_dict=embeddings_dict,
    k=10,
    alpha=0.5  # Balance relevance/diversity
)
```

## Metric Comparison

| Metric | Focus | Range | Best For |
|--------|-------|-------|----------|
| **MRR@K** | First relevant doc | [0, 1] | Single-answer queries |
| **Recall@K** | Coverage | [0, 1] | Comprehensive search |
| **Precision@K** | Accuracy of top-K | [0, 1] | Minimizing irrelevant |
| **NDCG@K** | Ranking quality | [0, 1] | **General-purpose (most common)** |
| **MAP@K** | Overall ranking | [0, 1] | Ranking across all positions |
| **ILD** | Diversity | [0, 1] | Detecting redundancy |
| **α-NDCG@K** | Diversity-aware ranking | [0, 1] | Evaluating diverse retrievers |

## Computing Metrics

### Single Query Evaluation

```python
from dartboard.evaluation.metrics import evaluate_retrieval

# Retrieve documents
results = retriever.retrieve(query, corpus, k=10)
result_ids = [chunk.id for chunk in results.chunks]
relevant_ids = dataset.get_relevant_docs(query.id)

# Compute all metrics
metrics = evaluate_retrieval(
    results=result_ids,
    relevant_docs=relevant_ids,
    k_values=[1, 5, 10, 20, 100]
)

# Print results
for metric_name, score in metrics.items():
    print(f"{metric_name}: {score:.4f}")

# Output:
# MRR@100: 0.5230
# Recall@1: 0.0000
# Recall@5: 0.6667
# Recall@10: 1.0000
# Precision@1: 0.0000
# Precision@5: 0.4000
# Precision@10: 0.3000
# NDCG@1: 0.0000
# NDCG@5: 0.6245
# NDCG@10: 0.7123
# AP: 0.5112
```

### Batch Evaluation (Multiple Queries)

```python
from dartboard.evaluation.metrics import evaluate_batch

# Retrieve for all queries
all_results = []
all_relevant = []

for query in dataset.queries:
    results = retriever.retrieve(query.text, corpus, k=10)
    all_results.append([c.id for c in results.chunks])
    all_relevant.append(dataset.get_relevant_docs(query.id))

# Compute average metrics
avg_metrics = evaluate_batch(
    all_results=all_results,
    all_relevant_docs=all_relevant,
    k_values=[1, 5, 10, 20, 100]
)

print(f"Average NDCG@10: {avg_metrics['NDCG@10']:.4f}")
print(f"Average Recall@10: {avg_metrics['Recall@10']:.4f}")
print(f"Average MAP@100: {avg_metrics['MAP@100']:.4f}")
```

## Metric Interpretation Guidelines

### NDCG@10 Ranges

| NDCG@10 | Quality | Interpretation |
|---------|---------|----------------|
| 0.9 - 1.0 | Excellent | Near-perfect ranking |
| 0.7 - 0.9 | Very Good | Strong performance |
| 0.5 - 0.7 | Good | Acceptable for most tasks |
| 0.3 - 0.5 | Fair | Room for improvement |
| < 0.3 | Poor | Needs significant work |

### Recall@10 Ranges

| Recall@10 | Coverage | Interpretation |
|-----------|----------|----------------|
| > 0.8 | Excellent | Finding most relevant docs |
| 0.6 - 0.8 | Good | Missing some relevant docs |
| 0.4 - 0.6 | Fair | Missing many relevant docs |
| < 0.4 | Poor | Low coverage |

### ILD Ranges

| ILD | Diversity | Interpretation |
|-----|-----------|----------------|
| > 0.8 | High | Very diverse results |
| 0.6 - 0.8 | Medium | Moderate diversity |
| 0.4 - 0.6 | Low | Some redundancy |
| < 0.4 | Very Low | Highly redundant |

## Benchmark Results (Dec 2025)

### SciFact

| Method | MRR@100 | NDCG@10 | Recall@10 | MAP@100 | ILD |
|--------|---------|---------|-----------|---------|-----|
| BM25 | 0.6234 | 0.6156 | 0.812 | 0.5892 | 0.34 |
| Dense | 0.7445 | 0.7412 | 0.865 | 0.7123 | 0.38 |
| **Hybrid** | **0.7923** | **0.7826** | **0.872** | **0.7534** | 0.36 |
| Dartboard | 0.6892 | 0.7103 | 0.823 | 0.6723 | **0.89** |

**Key Insights**:
- Hybrid achieves best relevance metrics
- Dartboard has 2.5x higher diversity (ILD = 0.89 vs ~0.35)
- Dense outperforms BM25 by ~12% NDCG

### Trade-offs

**Relevance vs. Diversity**:
- Hybrid: High relevance, low diversity
- Dartboard: Good relevance, very high diversity
- Choice depends on use case (single answer vs. comprehensive exploration)

## Statistical Significance Testing

```python
from scipy.stats import ttest_rel

# Compare two methods across queries
method1_ndcgs = [compute_ndcg(method1, q) for q in queries]
method2_ndcgs = [compute_ndcg(method2, q) for q in queries]

# Paired t-test
t_stat, p_value = ttest_rel(method1_ndcgs, method2_ndcgs)

if p_value < 0.05:
    print(f"Method 1 significantly better (p={p_value:.4f})")
elif p_value < 0.1:
    print(f"Marginally significant (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

## References

### Papers

1. **NDCG**: Järvelin & Kekäläinen (2002) - "Cumulated gain-based evaluation of IR techniques"
2. **MAP**: Buckley & Voorhees (2004) - "Retrieval evaluation with incomplete information"
3. **ILD**: Clarke et al. (2008) - "Novelty and diversity in information retrieval evaluation"
4. **α-NDCG**: Clarke et al. (2008) - "Diversity-aware ranking"

### Implementation

- **Metrics Code**: [dartboard/evaluation/metrics.py](../dartboard/evaluation/metrics.py)
- **Benchmark Runner**: [benchmarks/scripts/run_benchmark.py](../benchmarks/scripts/run_benchmark.py)
- **Tests**: [tests/test_metrics.py](../tests/test_metrics.py)

### Further Reading

- [TREC Evaluation](https://trec.nist.gov/data/): Standard IR evaluation methodology
- [BEIR Paper](https://arxiv.org/abs/2104.08663): Heterogeneous IR benchmarks

## Summary

We compute **7 metrics** (5 relevance + 2 diversity) to comprehensively evaluate retrieval. **NDCG@10** is the primary relevance metric, while **ILD** measures diversity.

**Key Takeaways**:
- ✅ **NDCG@10**: Primary metric for ranking quality
- ✅ **Recall@10**: Measures coverage of relevant docs
- ✅ **MRR@100**: Best for single-answer queries
- ✅ **ILD**: Measures result diversity (Dartboard: 0.89 vs others: ~0.35)
- ✅ **α-NDCG**: Diversity-aware ranking metric
- 📊 **Benchmark results**: Hybrid best for relevance, Dartboard best for diversity
- 🔬 Use **statistical testing** to validate improvements (p < 0.05)
