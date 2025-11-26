#!/bin/bash
#
# Quick Start Script for Benchmark Evaluation
#
# This script runs a complete benchmark workflow:
# 1. Downloads datasets (if not already cached)
# 2. Runs benchmarks on sample data
# 3. Generates visualization report
#

set -e  # Exit on error

echo "========================================"
echo "Dartboard RAG - Benchmark Quick Start"
echo "========================================"
echo ""

# Configuration
SAMPLE_SIZE=100
METHODS="bm25 dense hybrid"
DATASET="msmarco"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --full)
            SAMPLE_SIZE=""
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--sample N] [--methods 'bm25 dense'] [--dataset msmarco] [--full]"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Methods: $METHODS"
if [ -n "$SAMPLE_SIZE" ]; then
    echo "  Sample: $SAMPLE_SIZE queries"
else
    echo "  Sample: FULL dataset"
fi
echo ""

# Step 1: Check if dataset exists, download if not
echo "[1/3] Checking datasets..."
if [ ! -d "benchmarks/data/$DATASET" ]; then
    echo "Dataset not found. Downloading..."
    python benchmarks/scripts/download_datasets.py --dataset $DATASET
else
    echo "Dataset found. Skipping download."
fi
echo ""

# Step 2: Run benchmark
echo "[2/3] Running benchmark..."
SAMPLE_ARG=""
if [ -n "$SAMPLE_SIZE" ]; then
    SAMPLE_ARG="--sample $SAMPLE_SIZE"
fi

python benchmarks/scripts/run_benchmark.py \
    --dataset $DATASET \
    --methods $METHODS \
    $SAMPLE_ARG

echo ""

# Step 3: Generate visualization
echo "[3/3] Generating visualization report..."
LATEST_RESULT=$(ls -t benchmarks/results/${DATASET}_*.json | head -1)

if [ -f "$LATEST_RESULT" ]; then
    python benchmarks/scripts/visualize_results.py \
        --results "$LATEST_RESULT" \
        --output benchmarks/results/latest_report.html

    echo ""
    echo "========================================"
    echo "✅ Benchmark Complete!"
    echo "========================================"
    echo ""
    echo "Results saved to:"
    echo "  JSON: $LATEST_RESULT"
    echo "  HTML: benchmarks/results/latest_report.html"
    echo ""
    echo "Open report in browser:"
    echo "  open benchmarks/results/latest_report.html"
    echo ""
else
    echo "❌ Error: Could not find result file"
    exit 1
fi
