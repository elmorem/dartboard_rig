# Streamlit Benchmark Viewer

An interactive web interface for viewing and analyzing benchmark results from the Dartboard RAG system.

## Features

- **Benchmark Result Explorer**: View detailed results from MS MARCO and BEIR datasets
- **Interactive Visualizations**: Metric comparison bar charts, dataset performance tables, score distributions
- **Metric Explanations**: Educational tooltips for MRR, MAP, NDCG, Recall, Precision, ILD, Alpha-NDCG
- **Dataset Comparison**: Compare performance across SciFact, ArguAna, Climate-FEVER
- **Result Filtering**: Filter by dataset, method, and metrics

## Running Locally

### Prerequisites

- Python 3.13+
- Benchmark results in `benchmarks/results/` directory
- Required packages installed

### Installation

```bash
# From project root
pip install -r requirements.txt
```

### Start the App

```bash
# Run Streamlit
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Viewing Results

The app automatically loads benchmark results from:

- `benchmarks/results/*.json` - Individual benchmark runs
- Displays latest results from MS MARCO and BEIR datasets
- Shows comparative analysis across methods

## Usage

### Navigation

1. **Home**: Overview and quick stats
2. **Benchmark Results**: Detailed metric comparison with explanatory tooltips
3. **Dataset Comparison**: Cross-dataset analysis

### Interactive Features

- Click on metric names with ❓ icons to see explanations
- Use dropdowns to filter by dataset or method
- Interactive Plotly charts for visual comparison

## Available Pages

### 1. Home

Overview of available benchmarks and quick statistics.

### 2. Benchmark Results

Detailed view of individual benchmark runs:

- Metric tables with all standard IR metrics
- Method comparison bar charts
- Educational tooltips for each metric
- Performance breakdown by k-value

### 3. Dataset Comparison

Compare performance across multiple BEIR datasets:

- Side-by-side metric comparison
- Dataset characteristics table
- Best method per dataset
- Insights and recommendations

## Supported Metrics

The viewer displays the following metrics with explanations:

- **MRR@K**: Mean Reciprocal Rank - rank of first relevant result
- **MAP@K**: Mean Average Precision - overall ranking quality
- **NDCG@K**: Normalized Discounted Cumulative Gain - ranking with position weighting
- **Recall@K**: Fraction of relevant documents retrieved
- **Precision@K**: Fraction of retrieved documents that are relevant
- **ILD**: Intra-List Diversity - diversity of retrieved results
- **Alpha-NDCG@10**: Diversity-aware NDCG metric

## Supported Retrieval Methods

1. **BM25**: Sparse lexical retrieval (keyword matching)
2. **Dense**: Semantic vector similarity
3. **Hybrid**: RRF fusion of BM25 + Dense
4. **Dartboard**: Diversity-aware retrieval with information gain

## Troubleshooting

### No Results Displayed

- Ensure you have run benchmarks and results exist in `benchmarks/results/`
- Check that JSON files are properly formatted
- Verify file permissions

### App Won't Start

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version is 3.13+
- Verify Streamlit is installed: `streamlit --version`

## Development

To modify the interface:

1. Edit `streamlit_app/app.py` or page files in `streamlit_app/pages/`
2. Streamlit auto-reloads on file changes
3. Use `st.write()` for debugging
4. Check browser console for errors

## Architecture

```text
Streamlit App (Port 8501)
    ↓
Load JSON Results from benchmarks/results/
    ↓
Display Interactive Visualizations
    ↓
User Exploration & Analysis
```

## Latest Results (Dec 2025)

The viewer currently displays results from:

- **SciFact** (5,183 docs): Scientific claim verification
- **ArguAna** (8,674 docs): Counter-argument retrieval
- **Climate-FEVER** (10K sampled): Climate change fact-checking

View these results by running the app and navigating to the Benchmark Results or Dataset Comparison pages.
