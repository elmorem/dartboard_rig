"""
Dataset Downloader for Benchmark Evaluation.

Downloads and prepares MS MARCO and BEIR datasets for evaluation.

Usage:
    python benchmarks/scripts/download_datasets.py --dataset msmarco
    python benchmarks/scripts/download_datasets.py --dataset beir --beir-datasets scifact nfcorpus
"""

import argparse
import logging
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from dartboard.evaluation.datasets import download_msmarco, download_beir

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_ms_marco(data_dir: str = "benchmarks/data/msmarco"):
    """
    Download MS MARCO dev.small dataset.

    Args:
        data_dir: Directory to save dataset
    """
    logger.info("Downloading MS MARCO dev.small dataset...")
    logger.info(f"Destination: {data_dir}")

    try:
        download_msmarco(data_dir=data_dir)
        logger.info("✅ MS MARCO download complete!")

    except Exception as e:
        logger.error(f"❌ Error downloading MS MARCO: {e}", exc_info=True)
        raise


def download_beir_datasets(datasets: list, data_dir: str = "benchmarks/data/beir"):
    """
    Download BEIR datasets.

    Args:
        datasets: List of BEIR dataset names
        data_dir: Directory to save datasets
    """
    logger.info(f"Downloading {len(datasets)} BEIR dataset(s)...")
    logger.info(f"Destination: {data_dir}")

    for dataset_name in datasets:
        try:
            logger.info(f"Downloading {dataset_name}...")
            download_beir(dataset_name=dataset_name, data_dir=data_dir)
            logger.info(f"✅ {dataset_name} complete!")

        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_name}: {e}", exc_info=True)
            # Continue with other datasets
            continue

    logger.info("✅ All BEIR downloads complete!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Download benchmark datasets")

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["msmarco", "beir", "all"],
        help="Dataset to download",
    )

    parser.add_argument(
        "--beir-datasets",
        nargs="+",
        default=["scifact", "nfcorpus", "fiqa"],
        help="Specific BEIR datasets to download",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="benchmarks/data",
        help="Base directory for datasets",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "msmarco" or args.dataset == "all":
        download_ms_marco(data_dir=str(data_dir / "msmarco"))

    if args.dataset == "beir" or args.dataset == "all":
        download_beir_datasets(
            datasets=args.beir_datasets, data_dir=str(data_dir / "beir")
        )

    logger.info("=" * 80)
    logger.info("📦 All downloads complete!")
    logger.info(f"📁 Data saved to: {data_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
