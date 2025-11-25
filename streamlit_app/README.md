# Streamlit Comparison Interface

A web-based interface for comparing different retrieval methods in the Dartboard RAG system.

## Features

- **Side-by-Side Comparison**: View results from multiple retrieval methods simultaneously
- **Interactive Visualizations**:
  - Score distribution box plots
  - Latency comparison bar charts
  - Overlap heatmaps between methods
- **Configurable Parameters**:
  - Select retrieval methods (BM25, Dense, Hybrid, Dartboard)
  - Adjust top-k results
  - Toggle cross-encoder reranking
- **Real-Time Metrics**: Performance insights and overlap analysis

## Running Locally

### Prerequisites

- Python 3.13+
- FastAPI backend running (see main README)
- Required packages installed

### Installation

```bash
# From project root
pip install -r requirements.txt
```

### Start the App

```bash
# Set environment variables (optional)
export API_BASE_URL="http://localhost:8000"
export API_KEY="your-api-key"

# Run Streamlit
streamlit run streamlit_app/app.py
```

The app will open in your browser at `http://localhost:8501`

## Running with Docker

```bash
# From project root
docker-compose up streamlit
```

## Configuration

Environment variables:

- `API_BASE_URL`: Base URL of the Dartboard API (default: `http://localhost:8000`)
- `API_KEY`: API key for authentication (default: `dev-key-12345`)

## Usage

1. **Enter Query**: Type your search question in the text area
2. **Select Methods**: Check the retrieval methods you want to compare
3. **Configure Parameters**: Adjust top-k and reranking options
4. **Run Comparison**: Click "Compare Methods" to execute
5. **Explore Results**: Navigate through tabs to view:
   - **Results**: Retrieved documents side-by-side
   - **Metrics**: Score distributions and detailed statistics
   - **Overlap**: Heatmap showing result overlap between methods
   - **Performance**: Latency comparison and insights

## Interface Overview

### Sidebar
- API connection status
- Method selection checkboxes
- Parameter sliders
- Reranking toggle

### Main Area
- Query input
- Summary metrics
- Tabbed results view

### Tabs
- **📝 Results**: Side-by-side document display (up to 4 methods)
- **📈 Metrics**: Score distributions, averages, and detailed tables
- **🔗 Overlap**: Heatmap and table showing chunk overlap
- **⚡ Performance**: Latency comparison and performance insights

## Comparison Workflow

```
User Input → API /compare → Process Results → Visualize
    ↓             ↓              ↓               ↓
  Query      [BM25, Dense,   Metrics &      Charts &
  Methods     Hybrid...]     Overlap        Tables
```

## Supported Retrieval Methods

1. **BM25**: Sparse lexical retrieval (keyword matching)
2. **Dense**: Semantic vector similarity
3. **Hybrid**: RRF fusion of BM25 + Dense
4. **Dartboard**: Proprietary diversification algorithm

## Example Queries

- "What is machine learning?"
- "Explain neural networks"
- "How does gradient descent work?"
- "Compare supervised and unsupervised learning"

## Troubleshooting

### API Connection Failed
- Ensure FastAPI backend is running
- Check `API_BASE_URL` environment variable
- Verify API key is correct

### No Documents Indexed
- Use the `/ingest` endpoint to add documents first
- Check vector store has been populated

### Empty Results
- Try different retrieval methods
- Adjust top-k parameter
- Verify documents are relevant to query

## Development

To modify the interface:

1. Edit `streamlit_app/app.py`
2. Streamlit auto-reloads on file changes
3. Use `st.write()` for debugging
4. Check browser console for errors

## Architecture

```
Streamlit App (Port 8501)
    ↓ HTTP Request
FastAPI Backend (Port 8000)
    ↓ Retrieval
Vector Store + Retrievers
    ↓ Results
Comparison & Visualization
```
