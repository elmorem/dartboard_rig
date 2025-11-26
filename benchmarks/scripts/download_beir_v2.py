"""
Download BEIR datasets using direct Python approach.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_beir_dataset(dataset_name: str, data_dir: str = "benchmarks/data/beir"):
    """
    Download a BEIR dataset.

    Args:
        dataset_name: Name of the dataset (e.g., 'arguana', 'scifact', 'climate-fever')
        data_dir: Directory to save the dataset
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading BEIR dataset: {dataset_name}")
    logger.info(f"Saving to: {data_path}")

    # Try to download and unzip
    try:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        logger.info(f"Attempting download from: {url}")
        dataset_path = util.download_and_unzip(url, str(data_path))
        logger.info(f"Dataset downloaded to: {dataset_path}")

        # Load to verify
        corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(
            split="test"
        )

        logger.info(f"Dataset verified:")
        logger.info(f"  Corpus: {len(corpus)} documents")
        logger.info(f"  Queries: {len(queries)} queries")
        logger.info(f"  Qrels: {len(qrels)} query-document pairs")

        return dataset_path

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("This may be due to network connectivity issues.")
        logger.info(
            "Please try downloading manually from: https://github.com/beir-cellar/beir"
        )
        raise


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
