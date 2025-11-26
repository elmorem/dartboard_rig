# Benchmark Evaluation Suite

Comprehensive benchmarking framework for comparing retrieval methods (BM25, Dense, Hybrid, Dartboard) on standard IR datasets.

## Overview

This suite provides:
- 📊 Automated benchmark runner for multiple retrieval methods
- 📥 Dataset downloaders for MS MARCO and BEIR
- 📈 Visualization tools for result analysis
- 📝 HTML report generation with interactive charts

## Quick Start

### 1. Download Datasets

```bash
# Download MS MARCO dev.small (6,980 queries)
python benchmarks/scripts/download_datasets.py --dataset msmarco

# Download BEIR datasets
python benchmarks/scripts/download_datasets.py --dataset beir --beir-datasets scifact nfcorpus fiqa

# Download all datasets
python benchmarks/scripts/download_datasets.py --dataset all
```

### 2. Run Benchmarks

```bash
# Quick test with sample of 100 queries
python benchmarks/scripts/run_benchmark.py \
    --dataset msmarco \
    --methods bm25 dense hybrid \
    --sample 100

# Full evaluation on MS MARCO
python benchmarks/scripts/run_benchmark.py \
    --dataset msmarco \
    --methods bm25 dense hybrid dartboard

# BEIR dataset evaluation
python benchmarks/scripts/run_benchmark.py \
    --dataset beir-scifact \
    --methods bm25 dense hybrid
```

### 3. Visualize Results

```bash
# Generate HTML report from benchmark results
python benchmarks/scripts/visualize_results.py \
    --results benchmarks/results/msmarco_*.json \
    --output benchmarks/results/report.html

# Open in browser
open benchmarks/results/report.html
```

## Supported Datasets

### MS MARCO
- **Size**: 6,980 queries, ~8.8M passages
- **Domain**: Web search queries
- **Metrics**: MRR@10, Recall@K, NDCG@K
- **Use case**: General-purpose passage ranking

### BEIR Datasets

| Dataset | Queries | Documents | Domain |
|---------|---------|-----------|--------|
| SciFact | 300 | 5K | Scientific fact verification |
| NFCorpus | 323 | 3.6K | Biomedical literature |
| FiQA | 648 | 57K | Financial QA |
| TREC-COVID | 50 | 171K | COVID-19 research |
| Natural Questions | 3.5K | 2.7M | Google search QA |

## Benchmark Scripts

### `run_benchmark.py`

Main benchmark runner that evaluates retrieval methods.

**Arguments**:
- `--dataset`: Dataset to evaluate on (`msmarco`, `beir-scifact`, etc.)
- `--methods`: Retrieval methods to compare (default: `bm25 dense hybrid`)
- `--sample`: Sample N queries for faster testing
- `--k`: Number of results to retrieve per query (default: 10)
- `--data-dir`: Directory for cached datasets
- `--results-dir`: Directory for benchmark results

**Example**:
```bash
python benchmarks/scripts/run_benchmark.py \
    --dataset msmarco \
    --methods bm25 dense hybrid dartboard \
    --sample 1000 \
    --k 10
```

**Output**:
- Console table with metrics comparison
- JSON file with complete results in `benchmarks/results/`

### `download_datasets.py`

Downloads and caches evaluation datasets.

**Arguments**:
- `--dataset`: Dataset to download (`msmarco`, `beir`, `all`)
- `--beir-datasets`: Specific BEIR datasets (default: `scifact nfcorpus fiqa`)
- `--data-dir`: Directory to save datasets

**Example**:
```bash
python benchmarks/scripts/download_datasets.py \
    --dataset beir \
    --beir-datasets scifact nfcorpus fiqa trec-covid
```

### `visualize_results.py`

Generates interactive HTML reports from benchmark results.

**Arguments**:
- `--results`: Path(s) to benchmark result JSON files
- `--output`: Output HTML file path

**Example**:
```bash
python benchmarks/scripts/visualize_results.py \
    --results benchmarks/results/*.json \
    --output report.html
```

**Features**:
- Summary metrics table
- Grouped bar charts for metric comparison
- NDCG@K line charts
- Recall-Precision curves
- Interactive Plotly charts

## Evaluation Metrics

All benchmarks compute standard IR metrics:

- **MRR@K**: Mean Reciprocal Rank - rank of first relevant document
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Fraction of retrieved documents that are relevant
- **NDCG@K**: Normalized Discounted Cumulative Gain - ranking quality
- **MAP@K**: Mean Average Precision - overall ranking performance

Metrics are computed at K ∈ {1, 5, 10, 20, 100} by default.

## Directory Structure

```
benchmarks/
├── README.md              # This file
├── scripts/
│   ├── run_benchmark.py   # Main benchmark runner
│   ├── download_datasets.py  # Dataset downloader
│   └── visualize_results.py  # Result visualizer
├── data/                  # Downloaded datasets (cached)
│   ├── msmarco/
│   └── beir/
└── results/               # Benchmark results (JSON + HTML)
    ├── msmarco_20241125_143022.json
    ├── beir-scifact_20241125_144533.json
    └── report.html
```

## Workflow Example

Complete workflow for running and visualizing benchmarks:

```bash
# 1. Download datasets
python benchmarks/scripts/download_datasets.py --dataset msmarco

# 2. Run quick test (100 queries)
python benchmarks/scripts/run_benchmark.py \
    --dataset msmarco \
    --methods bm25 dense hybrid \
    --sample 100

# 3. Run full evaluation (6,980 queries)
python benchmarks/scripts/run_benchmark.py \
    --dataset msmarco \
    --methods bm25 dense hybrid dartboard

# 4. Generate visualization
python benchmarks/scripts/visualize_results.py \
    --results benchmarks/results/msmarco_*.json \
    --output benchmarks/results/msmarco_report.html

# 5. Open report in browser
open benchmarks/results/msmarco_report.html
```

## Performance Tips

### Fast Testing
Use `--sample` to test on a subset of queries:
```bash
python benchmarks/scripts/run_benchmark.py --dataset msmarco --sample 100
```

### Parallel Evaluation
Run multiple datasets in parallel:
```bash
# Terminal 1
python benchmarks/scripts/run_benchmark.py --dataset msmarco --methods bm25 dense

# Terminal 2
python benchmarks/scripts/run_benchmark.py --dataset beir-scifact --methods bm25 dense
```

### Resource Management
- BM25: Requires fitting on full corpus (~2GB RAM for MS MARCO)
- Dense: GPU accelerated if available
- Hybrid: Runs both BM25 and Dense (highest resource usage)

## Result Files

### JSON Format

Benchmark results are saved as JSON with this structure:

```json
{
  "dataset": "msmarco",
  "sample_size": 1000,
  "num_queries": 1000,
  "num_documents": 8841823,
  "methods": ["bm25", "dense", "hybrid"],
  "k": 10,
  "k_values": [1, 5, 10, 20, 100],
  "metrics": {
    "bm25": {
      "MRR@100": 0.185,
      "Recall@10": 0.512,
      "NDCG@10": 0.314,
      ...
    },
    "dense": {...},
    "hybrid": {...}
  },
  "timestamp": "2024-11-25T14:30:22"
}
```

### HTML Report

Interactive report includes:
- Summary table with key metrics
- Grouped bar chart comparing all metrics
- NDCG@K line chart
- Recall-Precision curve
- Detailed breakdown by method

## Interpreting Results

### Key Metrics to Watch

1. **MRR@10**: How quickly you find the first relevant result
   - Higher is better
   - Best for single-answer questions

2. **Recall@10**: How many relevant results you retrieve
   - Higher is better
   - Best for comprehensive search

3. **NDCG@10**: Overall ranking quality
   - Higher is better
   - Balances relevance and ranking position

### Typical Performance Ranges

| Metric | BM25 | Dense | Hybrid | Target |
|--------|------|-------|--------|--------|
| MRR@10 | 0.15-0.20 | 0.20-0.30 | 0.25-0.35 | >0.30 |
| Recall@10 | 0.40-0.50 | 0.50-0.60 | 0.55-0.65 | >0.60 |
| NDCG@10 | 0.25-0.35 | 0.35-0.45 | 0.40-0.50 | >0.45 |

*Note: Values vary significantly by dataset and domain*

## Troubleshooting

### Out of Memory

**Problem**: BM25 fails with OOM on large datasets

**Solution**: Use smaller sample or increase system RAM
```bash
python benchmarks/scripts/run_benchmark.py --dataset msmarco --sample 1000
```

### Slow Evaluation

**Problem**: Dense retrieval is very slow

**Solution**: Enable GPU acceleration or reduce k
```bash
# Use GPU
CUDA_VISIBLE_DEVICES=0 python benchmarks/scripts/run_benchmark.py ...

# Or reduce k
python benchmarks/scripts/run_benchmark.py --k 5
```

### Dataset Download Fails

**Problem**: BEIR dataset download times out

**Solution**: Download individually or check network
```bash
# Download one at a time
python benchmarks/scripts/download_datasets.py --dataset beir --beir-datasets scifact
```

## Citations

If you use these benchmarks, please cite:

```bibtex
@inproceedings{msmarco,
  title={MS MARCO: A Human Generated MAchine Reading COmprehension Dataset},
  author={Bajaj, Payal and Campos, Daniel and Craswell, Nick and others},
  year={2016}
}

@inproceedings{beir,
  title={BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models},
  author={Thakur, Nandan and Reimers, Nils and Rücklé, Andreas and others},
  year={2021}
}
```

## Contributing

To add a new dataset:

1. Implement loader in `dartboard/evaluation/datasets.py`
2. Add dataset option to `run_benchmark.py`
3. Update this README with dataset details
4. Submit PR with example results

## License

Same as parent project.
