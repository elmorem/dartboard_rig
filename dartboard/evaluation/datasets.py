"""
Dataset loaders for retrieval evaluation.

Provides loaders for:
- MS MARCO: Microsoft Machine Reading Comprehension dataset
- BEIR: Benchmarking IR datasets (SciFact, NFCorpus, FiQA, etc.)
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Represents a search query with metadata."""

    id: str
    text: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Document:
    """Represents a document in the corpus."""

    id: str
    text: str
    title: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class QRel:
    """Query-document relevance judgment."""

    query_id: str
    doc_id: str
    relevance: int  # 0 = not relevant, 1+ = relevant (higher = more relevant)


@dataclass
class EvaluationDataset:
    """
    Complete evaluation dataset with queries, documents, and relevance judgments.
    """

    name: str
    queries: List[Query]
    documents: List[Document]
    qrels: List[QRel]
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_relevant_docs(self, query_id: str, min_relevance: int = 1) -> Set[str]:
        """
        Get relevant document IDs for a query.

        Args:
            query_id: Query ID
            min_relevance: Minimum relevance score (default: 1)

        Returns:
            Set of relevant document IDs
        """
        return {
            qrel.doc_id
            for qrel in self.qrels
            if qrel.query_id == query_id and qrel.relevance >= min_relevance
        }

    def get_query_by_id(self, query_id: str) -> Optional[Query]:
        """Get query by ID."""
        for query in self.queries:
            if query.id == query_id:
                return query
        return None

    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def stats(self) -> Dict:
        """Get dataset statistics."""
        return {
            "name": self.name,
            "num_queries": len(self.queries),
            "num_documents": len(self.documents),
            "num_qrels": len(self.qrels),
            "avg_relevant_per_query": (
                len(self.qrels) / len(self.queries) if self.queries else 0
            ),
        }


class MSMARCOLoader:
    """
    Loader for MS MARCO dataset.

    MS MARCO (Microsoft Machine Reading Comprehension) is a large-scale
    dataset for passage ranking and question answering.

    Dataset structure:
    - queries.dev.small.tsv: 6,980 queries
    - qrels.dev.small.tsv: Query-passage relevance judgments
    - collection.tsv: Full passage collection (8.8M passages)
    """

    def __init__(self, data_dir: str = "data/msmarco"):
        """
        Initialize MS MARCO loader.

        Args:
            data_dir: Directory containing MS MARCO files
        """
        self.data_dir = Path(data_dir)

    def load_dev_small(
        self, max_docs: int = None, use_cache: bool = True
    ) -> EvaluationDataset:
        """
        Load MS MARCO dev.small dataset (6,980 queries).

        Args:
            max_docs: Limit number of documents loaded (default: all)
            use_cache: Use cached version if available

        Returns:
            EvaluationDataset
        """
        logger.info("Loading MS MARCO dev.small dataset...")

        # Check for cached version
        cache_file = self.data_dir / "dev_small_cache.json"
        if use_cache and cache_file.exists():
            logger.info("Loading from cache...")
            return self._load_from_cache(cache_file)

        # Load queries
        queries = self._load_queries("queries.dev.small.tsv")
        logger.info(f"Loaded {len(queries)} queries")

        # Load qrels
        qrels = self._load_qrels("qrels.dev.small.tsv")
        logger.info(f"Loaded {len(qrels)} relevance judgments")

        # Get unique doc IDs from qrels
        relevant_doc_ids = {qrel.doc_id for qrel in qrels}

        # Load only relevant documents (much faster than loading all 8.8M)
        documents = self._load_documents(
            "collection.tsv", doc_ids=relevant_doc_ids, max_docs=max_docs
        )
        logger.info(f"Loaded {len(documents)} documents")

        dataset = EvaluationDataset(
            name="MS MARCO dev.small",
            queries=queries,
            documents=documents,
            qrels=qrels,
            metadata={"source": "ms_marco", "split": "dev_small"},
        )

        # Save cache
        if use_cache:
            self._save_to_cache(dataset, cache_file)

        return dataset

    def _load_queries(self, filename: str) -> List[Query]:
        """Load queries from TSV file."""
        queries = []
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Queries file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    query_id, query_text = parts[0], parts[1]
                    queries.append(Query(id=query_id, text=query_text))

        return queries

    def _load_qrels(self, filename: str) -> List[QRel]:
        """Load query-document relevance judgments from TSV file."""
        qrels = []
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Qrels file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts
                    qrels.append(
                        QRel(query_id=query_id, doc_id=doc_id, relevance=int(relevance))
                    )

        return qrels

    def _load_documents(
        self,
        filename: str,
        doc_ids: Set[str] = None,
        max_docs: int = None,
    ) -> List[Document]:
        """
        Load documents from TSV file.

        Args:
            filename: Document file name
            doc_ids: Only load documents with these IDs (for efficiency)
            max_docs: Maximum number of documents to load

        Returns:
            List of Document objects
        """
        documents = []
        filepath = self.data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Documents file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    doc_id, doc_text = parts[0], parts[1]

                    # Skip if not in requested doc_ids
                    if doc_ids and doc_id not in doc_ids:
                        continue

                    documents.append(Document(id=doc_id, text=doc_text))

                    # Check max_docs limit
                    if max_docs and len(documents) >= max_docs:
                        break

        return documents

    def _save_to_cache(self, dataset: EvaluationDataset, cache_file: Path):
        """Save dataset to cache file."""
        logger.info(f"Saving cache to {cache_file}...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "name": dataset.name,
            "metadata": dataset.metadata,
            "queries": [
                {"id": q.id, "text": q.text, "metadata": q.metadata}
                for q in dataset.queries
            ],
            "documents": [
                {"id": d.id, "text": d.text, "title": d.title, "metadata": d.metadata}
                for d in dataset.documents
            ],
            "qrels": [
                {
                    "query_id": qr.query_id,
                    "doc_id": qr.doc_id,
                    "relevance": qr.relevance,
                }
                for qr in dataset.qrels
            ],
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    def _load_from_cache(self, cache_file: Path) -> EvaluationDataset:
        """Load dataset from cache file."""
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        queries = [Query(**q) for q in cache_data["queries"]]
        documents = [Document(**d) for d in cache_data["documents"]]
        qrels = [QRel(**qr) for qr in cache_data["qrels"]]

        return EvaluationDataset(
            name=cache_data["name"],
            queries=queries,
            documents=documents,
            qrels=qrels,
            metadata=cache_data["metadata"],
        )


class BEIRLoader:
    """
    Loader for BEIR (Benchmarking IR) datasets.

    BEIR is a heterogeneous benchmark containing 18 diverse datasets:
    - Bio-medical: TREC-COVID, BioASQ, NFCorpus
    - Financial: FiQA
    - Scientific: SciFact, SCIDOCS
    - News: TREC-NEWS, Robust04
    - Question Answering: Natural Questions, HotpotQA, etc.

    Each dataset has queries, documents, and qrels.
    """

    AVAILABLE_DATASETS = [
        "trec-covid",
        "nfcorpus",
        "bioasq",
        "scifact",
        "fiqa",
        "scidocs",
        "nq",
        "hotpotqa",
        "climate-fever",
        "dbpedia-entity",
        "fever",
        "arguana",
        "quora",
        "cqadupstack",
        "touche-2020",
        "webis-touche2020",
        "trec-news",
        "robust04",
    ]

    def __init__(self, data_dir: str = "data/beir"):
        """
        Initialize BEIR loader.

        Args:
            data_dir: Directory containing BEIR datasets
        """
        self.data_dir = Path(data_dir)

    def load_dataset(
        self, dataset_name: str, split: str = "test", use_cache: bool = True
    ) -> EvaluationDataset:
        """
        Load a BEIR dataset.

        Args:
            dataset_name: Name of BEIR dataset (e.g., 'scifact', 'nfcorpus')
            split: Dataset split ('train', 'dev', 'test')
            use_cache: Use cached version if available

        Returns:
            EvaluationDataset

        Raises:
            ValueError: If dataset name is not recognized
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {', '.join(self.AVAILABLE_DATASETS)}"
            )

        logger.info(f"Loading BEIR dataset: {dataset_name} ({split} split)...")

        dataset_dir = self.data_dir / dataset_name

        # Check cache
        cache_file = dataset_dir / f"{split}_cache.json"
        if use_cache and cache_file.exists():
            logger.info("Loading from cache...")
            return self._load_from_cache(cache_file)

        # Load from BEIR format
        queries = self._load_beir_queries(dataset_dir / f"{split}_queries.jsonl")
        documents = self._load_beir_documents(dataset_dir / "corpus.jsonl")
        qrels = self._load_beir_qrels(dataset_dir / f"qrels/{split}.tsv")

        dataset = EvaluationDataset(
            name=f"BEIR {dataset_name} ({split})",
            queries=queries,
            documents=documents,
            qrels=qrels,
            metadata={"source": "beir", "dataset": dataset_name, "split": split},
        )

        # Save cache
        if use_cache:
            self._save_to_cache(dataset, cache_file)

        return dataset

    def _load_beir_queries(self, filepath: Path) -> List[Query]:
        """Load queries from BEIR JSONL format."""
        queries = []

        if not filepath.exists():
            raise FileNotFoundError(f"Queries file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                queries.append(
                    Query(
                        id=data["_id"],
                        text=data["text"],
                        metadata=data.get("metadata", {}),
                    )
                )

        return queries

    def _load_beir_documents(self, filepath: Path) -> List[Document]:
        """Load documents from BEIR JSONL format."""
        documents = []

        if not filepath.exists():
            raise FileNotFoundError(f"Documents file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                documents.append(
                    Document(
                        id=data["_id"],
                        text=data["text"],
                        title=data.get("title", ""),
                        metadata=data.get("metadata", {}),
                    )
                )

        return documents

    def _load_beir_qrels(self, filepath: Path) -> List[QRel]:
        """Load qrels from BEIR TSV format."""
        qrels = []

        if not filepath.exists():
            raise FileNotFoundError(f"Qrels file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            # Skip header if present
            header = f.readline()
            if not header.startswith("query-id"):
                f.seek(0)

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id, doc_id, relevance = parts[0], parts[1], parts[2]
                    qrels.append(
                        QRel(query_id=query_id, doc_id=doc_id, relevance=int(relevance))
                    )

        return qrels

    def _save_to_cache(self, dataset: EvaluationDataset, cache_file: Path):
        """Save dataset to cache file."""
        logger.info(f"Saving cache to {cache_file}...")
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "name": dataset.name,
            "metadata": dataset.metadata,
            "queries": [
                {"id": q.id, "text": q.text, "metadata": q.metadata}
                for q in dataset.queries
            ],
            "documents": [
                {"id": d.id, "text": d.text, "title": d.title, "metadata": d.metadata}
                for d in dataset.documents
            ],
            "qrels": [
                {
                    "query_id": qr.query_id,
                    "doc_id": qr.doc_id,
                    "relevance": qr.relevance,
                }
                for qr in dataset.qrels
            ],
        }

        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

    def _load_from_cache(self, cache_file: Path) -> EvaluationDataset:
        """Load dataset from cache file."""
        with open(cache_file, "r") as f:
            cache_data = json.load(f)

        queries = [Query(**q) for q in cache_data["queries"]]
        documents = [Document(**d) for d in cache_data["documents"]]
        qrels = [QRel(**qr) for qr in cache_data["qrels"]]

        return EvaluationDataset(
            name=cache_data["name"],
            queries=queries,
            documents=documents,
            qrels=qrels,
            metadata=cache_data["metadata"],
        )


def download_msmarco(data_dir: str = "data/msmarco"):
    """
    Download MS MARCO dev.small dataset.

    Downloads:
    - queries.dev.small.tsv
    - qrels.dev.small.tsv
    - collection.tsv (warning: 8.8M passages, ~3GB)

    Args:
        data_dir: Directory to save files
    """
    from datasets import load_dataset

    logger.info("Downloading MS MARCO dataset...")
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Load from HuggingFace datasets
    dataset = load_dataset("ms_marco", "v1.1", split="validation")

    logger.info(f"MS MARCO dataset downloaded to {data_dir}")
    logger.info(
        "Note: For full usage, you may need to download collection.tsv separately"
    )


def download_beir(dataset_name: str, data_dir: str = "data/beir"):
    """
    Download a BEIR dataset.

    Args:
        dataset_name: Name of BEIR dataset
        data_dir: Directory to save files
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    logger.info(f"Downloading BEIR dataset: {dataset_name}...")
    data_path = Path(data_dir)

    # Download and unzip dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    out_dir = data_path / dataset_name
    util.download_and_unzip(url, str(data_path))

    logger.info(f"BEIR {dataset_name} downloaded to {out_dir}")
