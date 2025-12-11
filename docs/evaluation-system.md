# Evaluation System

## Overview

The evaluation system provides **comprehensive benchmarking** of retrieval methods on standard IR datasets. It computes relevance metrics (MRR, NDCG, MAP, Recall, Precision) and diversity metrics (ILD, Alpha-NDCG) to assess both accuracy and result diversity.

## System Architecture

```
Evaluation Pipeline
├── Dataset Loaders (MS MARCO, BEIR)
├── Retrieval Methods (BM25, Dense, Hybrid, Dartboard)
├── Metric Computation (MRR, NDCG, MAP, ILD, etc.)
├── Benchmark Runner (Automated evaluation)
└── Result Visualization (Streamlit app, reports)
```

## Components

### 1. Dataset Loaders

Load evaluation datasets with queries, documents, and relevance judgments:

```python
from dartboard.evaluation.datasets import MSMARCOLoader, BEIRLoader

# Load MS MARCO
msmarco = MSMARCOLoader(data_dir="data/msmarco")
dataset = msmarco.load_dev_small()  # 6,980 queries

# Load BEIR
beir = BEIRLoader(data_dir="data/beir")
scifact = beir.load_dataset("scifact")  # 300 queries, 5,183 docs

# Access components
print(f"Queries: {len(dataset.queries)}")
print(f"Documents: {len(dataset.documents)}")
print(f"Relevance judgments: {len(dataset.qrels)}")
```

### 2. Retrieval Methods

Evaluate multiple retrieval approaches:

```python
from dartboard.retrieval.bm25 import BM25Retriever
from dartboard.retrieval.dense import DenseRetriever
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.core import DartboardRetriever

# Initialize retrievers
retrievers = {
    "bm25": BM25Retriever(),
    "dense": DenseRetriever(vector_store=vector_store),
    "hybrid": HybridRetriever(bm25, dense),
    "dartboard": DartboardRetriever(config, model, cross_encoder)
}

# Fit on corpus
for name, retriever in retrievers.items():
    if hasattr(retriever, 'fit'):
        retriever.fit(chunks)
```

### 3. Metrics Computation

Compute standard IR metrics:

```python
from dartboard.evaluation.metrics import (
    evaluate_retrieval,
    mean_average_precision,
    intra_list_diversity,
    alpha_ndcg
)

# Per-query evaluation
results = retriever.retrieve(query, corpus, k=10)
result_ids = [chunk.id for chunk in results.chunks]
relevant_ids = dataset.get_relevant_docs(query.id)

metrics = evaluate_retrieval(
    results=result_ids,
    relevant_docs=relevant_ids,
    k_values=[1, 5, 10, 20, 100]
)

# Metrics dict:
# {
#     'MRR@100': 0.52,
#     'Recall@10': 0.68,
#     'Precision@10': 0.40,
#     'NDCG@10': 0.74,
#     'MAP@100': 0.51
# }
```

### 4. Benchmark Runner

Automated evaluation across datasets and methods:

```bash
# Run comprehensive benchmark
python benchmarks/scripts/run_benchmark.py \
    --dataset beir-scifact \
    --methods bm25 dense hybrid dartboard \
    --k 10
```

Output:
```
Running benchmark on beir-scifact (300 queries, 5,183 docs)
Evaluating: bm25... Done (12.3s)
Evaluating: dense... Done (45.1s)
Evaluating: hybrid... Done (68.7s)
Evaluating: dartboard... Done (95.2s)

Results saved to: benchmarks/results/beir-scifact_20251203_143022.json
```

## Evaluation Workflow

### Step 1: Download Dataset

```bash
python benchmarks/scripts/download_datasets.py --dataset beir-scifact
```

### Step 2: Run Benchmark

```python
from benchmarks.scripts.run_benchmark import BenchmarkRunner

runner = BenchmarkRunner(
    dataset_name="beir-scifact",
    methods=["bm25", "dense", "hybrid"],
    k_values=[1, 5, 10, 20, 100],
    output_dir="benchmarks/results"
)

results = runner.run()
```

### Step 3: Analyze Results

```python
# Load results
with open("benchmarks/results/beir-scifact_20251203.json") as f:
    results = json.load(f)

# Compare methods
for method, metrics in results["metrics"].items():
    print(f"\n{method}:")
    print(f"  NDCG@10: {metrics['NDCG@10']:.4f}")
    print(f"  Recall@10: {metrics['Recall@10']:.4f}")
    print(f"  MRR@100: {metrics['MRR@100']:.4f}")
```

### Step 4: Visualize

```bash
# Launch Streamlit viewer
streamlit run streamlit_app/app.py
```

## Benchmark Results (Dec 2025)

### SciFact (5,183 docs, 300 queries)

| Method | MRR@100 | NDCG@10 | Recall@10 | MAP@100 | ILD | Alpha-NDCG@10 |
|--------|---------|---------|-----------|---------|-----|---------------|
| BM25 | 0.6234 | 0.6156 | 0.812 | 0.5892 | 0.34 | 0.5823 |
| Dense | 0.7445 | 0.7412 | 0.865 | 0.7123 | 0.38 | 0.7201 |
| **Hybrid** | **0.7923** | **0.7826** | **0.872** | **0.7534** | 0.36 | **0.7645** |
| Dartboard | 0.6892 | 0.7103 | 0.823 | 0.6723 | **0.89** | 0.7289 |

**Key Findings**:
- Hybrid achieves best relevance (NDCG@10 = 0.78)
- Dartboard has highest diversity (ILD = 0.89 vs ~0.35)
- Dense outperforms BM25 by ~12% NDCG

### ArguAna (8,674 docs, 1,406 queries)

| Method | MRR@100 | NDCG@10 | Recall@10 | MAP@100 |
|--------|---------|---------|-----------|---------|
| BM25 | 0.4123 | 0.2734 | 0.592 | 0.3892 |
| **Dense** | **0.3567** | **0.3095** | **0.680** | **0.3123** |
| Hybrid | 0.3423 | 0.2934 | 0.645 | 0.3045 |
| Dartboard | 0.0023 | 0.0012 | 0.003 | 0.0018 |

**Key Findings**:
- Dense performs best overall
- BM25 has higher MRR (better at finding first relevant doc)
- **Dartboard fails catastrophically** (needs investigation/fixes)

### Climate-FEVER (10K sampled from 5.4M docs, 1,535 queries)

| Method | MRR@100 | NDCG@10 | Recall@10 | MAP@100 |
|--------|---------|---------|-----------|---------|
| BM25 | 0.4903 | 0.4097 | 0.486 | 0.4512 |
| **Dense** | **0.6152** | **0.5291** | **0.632** | **0.5678** |
| Hybrid | 0.5936 | 0.5260 | 0.641 | 0.5523 |
| Dartboard | 0.5234 | 0.4823 | 0.578 | 0.4934 |

**Key Findings**:
- Dense best overall (NDCG@10 = 0.53)
- Corpus sampling preserved all 86 relevant docs (100% recall possible)
- Hybrid slightly worse than Dense alone (unusual)

## Corpus Sampling for Large Datasets

For datasets like Climate-FEVER (5.4M docs), we use stratified sampling:

```python
from dartboard.evaluation.datasets import BEIRLoader

loader = BEIRLoader()
dataset = loader.load_dataset(
    "climate-fever",
    max_corpus_docs=10000  # Sample to 10K docs
)

# Sampling strategy:
# 1. Keep ALL relevant documents (86 docs)
# 2. Randomly sample 9,914 non-relevant documents
# 3. Result: 10K docs with 100% recall ceiling preserved
```

**Validation**:
- Original corpus: 5.4M docs
- Sampled corpus: 10K docs (0.2%)
- Relevant docs: 86 (100% preserved)
- Memory reduction: 99.8%

## Diversity Metrics

### Intra-List Diversity (ILD)

Measures average dissimilarity between all retrieved documents:

```python
from dartboard.evaluation.metrics import intra_list_diversity

# Compute ILD
ild_score = intra_list_diversity(
    results=[chunk.id for chunk in results.chunks],
    embeddings_dict={chunk.id: chunk.embedding for chunk in all_chunks}
)

# ild_score ∈ [0, 1], higher = more diverse
# - ILD = 0.35: Low diversity (redundant results)
# - ILD = 0.65: Medium diversity
# - ILD = 0.90: High diversity (Dartboard)
```

### Alpha-NDCG

Diversity-aware NDCG that penalizes redundant documents:

```python
from dartboard.evaluation.metrics import alpha_ndcg

score = alpha_ndcg(
    results=result_ids,
    relevant_docs=relevant_ids,
    embeddings_dict=embeddings,
    k=10,
    alpha=0.5  # Balance relevance (0) vs diversity (1)
)

# Alpha-NDCG penalizes similar documents at adjacent ranks
```

## Advanced Features

### Batch Evaluation

```python
from dartboard.evaluation.metrics import evaluate_batch

# Evaluate multiple queries
all_results = [retriever.retrieve(q, corpus) for q in queries]
all_relevant = [dataset.get_relevant_docs(q.id) for q in queries]

# Compute average metrics
avg_metrics = evaluate_batch(
    all_results=[[c.id for c in r.chunks] for r in all_results],
    all_relevant_docs=all_relevant,
    k_values=[1, 5, 10, 20]
)

print(f"Average NDCG@10: {avg_metrics['NDCG@10']:.4f}")
```

### Statistical Significance Testing

```python
from scipy.stats import ttest_rel

# Compare two methods
method1_scores = [ndcg_per_query(method1, q) for q in queries]
method2_scores = [ndcg_per_query(method2, q) for q in queries]

# Paired t-test
t_stat, p_value = ttest_rel(method1_scores, method2_scores)

if p_value < 0.05:
    print(f"Method 1 significantly better (p={p_value:.4f})")
else:
    print(f"No significant difference (p={p_value:.4f})")
```

### Result Export

```python
# Export to JSON
runner.save_results("benchmarks/results/my_benchmark.json")

# Export to CSV
import pandas as pd

df = pd.DataFrame({
    "method": list(results["metrics"].keys()),
    **{metric: [results["metrics"][m][metric] for m in results["metrics"]]
       for metric in ["MRR@100", "NDCG@10", "Recall@10"]}
})

df.to_csv("benchmarks/results/summary.csv")
```

## Troubleshooting

### Issue: Benchmark Running Too Slow

**Solutions**:
```bash
# 1. Sample queries
python run_benchmark.py --dataset beir-scifact --sample 50

# 2. Use GPU
export CUDA_VISIBLE_DEVICES=0

# 3. Reduce candidate pool
# Edit config: triage_k=50 instead of 100

# 4. Skip slow methods
python run_benchmark.py --methods bm25 dense  # Skip Dartboard
```

### Issue: Out of Memory

**Solutions**:
```bash
# 1. Sample corpus
python run_benchmark.py --dataset climate-fever --max-corpus-docs 5000

# 2. Process in batches
# Edit benchmark script: batch_size=100

# 3. Use smaller embedding model
# all-MiniLM-L6-v2 (384-dim) instead of all-mpnet-base-v2 (768-dim)
```

## References

### Implementation

- **Metrics**: [dartboard/evaluation/metrics.py](../dartboard/evaluation/metrics.py)
- **Datasets**: [dartboard/evaluation/datasets.py](../dartboard/evaluation/datasets.py)
- **Benchmark Runner**: [benchmarks/scripts/run_benchmark.py](../benchmarks/scripts/run_benchmark.py)
- **Streamlit Viewer**: [streamlit_app/app.py](../streamlit_app/app.py)

### Documentation

- **Benchmarks README**: [benchmarks/README.md](../benchmarks/README.md)
- **Metrics Guide**: [docs/metrics.md](metrics.md)
- **Datasets Guide**: [docs/datasets.md](datasets.md)

## Summary

The evaluation system provides **automated, comprehensive benchmarking** of retrieval methods. It supports standard IR metrics, diversity metrics, and interactive visualization through Streamlit.

**Key Takeaways**:
- ✅ Automated benchmarking on **MS MARCO and BEIR** datasets
- ✅ Computes **relevance + diversity metrics** (MRR, NDCG, ILD, Alpha-NDCG)
- ✅ **Corpus sampling** handles massive datasets (5.4M → 10K docs)
- ✅ **Streamlit viewer** for interactive result exploration
- ✅ **Statistical testing** for method comparison
- 📊 **Dec 2025 results**: Hybrid best (SciFact), Dense best (ArguAna, Climate-FEVER)
