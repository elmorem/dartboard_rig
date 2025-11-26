"""
Download BEIR datasets for benchmarking.

This script downloads BEIR datasets and organizes them in the expected format.
"""

import argparse
import logging
from pathlib import Path
from beir.datasets.data_loader import GenericDataLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_beir_dataset(dataset_name: str, data_dir: str = "benchmarks/data/beir"):
    """
    Download a BEIR dataset using GenericDataLoader.

    Args:
        dataset_name: Name of the dataset (e.g., 'arguana', 'scifact', 'climate-fever')
        data_dir: Directory to save the dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading BEIR dataset: {dataset_name}")
    logger.info(f"Saving to: {data_path}")

    # Download using BEIR's GenericDataLoader which handles download automatically
    logger.info("Loading dataset with GenericDataLoader...")
    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(
        custom_data_path=None, name=dataset_name, url=None
    )

    logger.info(f"Dataset downloaded successfully:")
    logger.info(f"  Corpus: {len(corpus)} documents")
    logger.info(f"  Queries: {len(queries)} queries")
    logger.info(f"  Qrels: {len(qrels)} query-document pairs")
    logger.info(f"Dataset saved to: {data_path / dataset_name}")

    return data_path / dataset_name


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Download BEIR datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., arguana, scifact, climate-fever)",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/data/beir",
        help="Directory to save datasets",
    )

    args = parser.parse_args()

    download_beir_dataset(args.dataset, args.data_dir)


if __name__ == "__main__":
    main()
