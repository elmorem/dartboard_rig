"""
Vector store abstraction for Dartboard RAG.

Supports FAISS, Pinecone, and other vector databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
from dartboard.datasets.models import Chunk


class VectorStore(ABC):
    """Abstract base class for vector storage."""

    @abstractmethod
    def add(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to vector store.

        Args:
            chunks: List of Chunk objects with embeddings
        """
        pass

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, k: int, filters: Optional[Dict] = None
    ) -> List[Chunk]:
        """
        Search for similar chunks.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of top-k most similar chunks
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks in store."""
        pass


class FAISSStore(VectorStore):
    """FAISS-based vector store (in-memory or disk-persisted)."""

    def __init__(self, embedding_dim: int, persist_path: Optional[str] = None):
        """
        Initialize FAISS vector store.

        Args:
            embedding_dim: Dimension of embeddings
            persist_path: Optional path to save/load index
        """
        import faiss

        self.embedding_dim = embedding_dim
        self.persist_path = persist_path

        # Use inner product (cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []
        self.id_to_idx: Dict[str, int] = {}

        # Load from disk if exists
        if persist_path:
            self._load()

    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to FAISS index."""
        if not chunks:
            return

        # Extract embeddings
        embeddings = np.array([c.embedding for c in chunks], dtype=np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)

        # Add to index
        start_idx = len(self.chunks)
        self.index.add(embeddings)

        # Store chunks and ID mapping
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self.chunks.append(chunk)
            self.id_to_idx[chunk.id] = idx

        # Persist if configured
        if self.persist_path:
            self._save()

    def search(
        self, query_embedding: np.ndarray, k: int, filters: Optional[Dict] = None
    ) -> List[Chunk]:
        """Search FAISS index."""
        if len(self.chunks) == 0:
            return []

        # Normalize query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        import faiss

        faiss.normalize_L2(query)

        # Search
        k = min(k, len(self.chunks))
        distances, indices = self.index.search(query, k)

        # Get chunks
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                chunk = self.chunks[idx]

                # Apply filters if provided
                if filters:
                    match = all(
                        chunk.metadata.get(key) == value
                        for key, value in filters.items()
                    )
                    if not match:
                        continue

                results.append(chunk)

        return results

    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks (rebuild index without them)."""
        # Filter out deleted chunks
        remaining = [c for c in self.chunks if c.id not in chunk_ids]

        # Rebuild index
        self.chunks = []
        self.id_to_idx = {}
        self.index.reset()

        # Re-add remaining chunks
        if remaining:
            self.add(remaining)

    def count(self) -> int:
        """Return number of chunks."""
        return len(self.chunks)

    def _save(self) -> None:
        """Save index to disk."""
        import faiss
        import pickle

        if not self.persist_path:
            return

        # Save FAISS index
        faiss.write_index(self.index, f"{self.persist_path}.index")

        # Save chunks and metadata
        with open(f"{self.persist_path}.meta", "wb") as f:
            pickle.dump({"chunks": self.chunks, "id_to_idx": self.id_to_idx}, f)

    def _load(self) -> None:
        """Load index from disk."""
        import faiss
        import pickle
        import os

        if not self.persist_path:
            return

        index_path = f"{self.persist_path}.index"
        meta_path = f"{self.persist_path}.meta"

        if os.path.exists(index_path) and os.path.exists(meta_path):
            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(meta_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.id_to_idx = data["id_to_idx"]


class PineconeStore(VectorStore):
    """Pinecone cloud vector store."""

    def __init__(
        self, api_key: str, environment: str, index_name: str, namespace: str = ""
    ):
        """
        Initialize Pinecone store.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment
            index_name: Name of Pinecone index
            namespace: Optional namespace for multi-tenancy
        """
        from pinecone import Pinecone

        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace

        # Cache chunks locally for retrieval
        self.chunk_cache: Dict[str, Chunk] = {}

    def add(self, chunks: List[Chunk]) -> None:
        """Add chunks to Pinecone."""
        if not chunks:
            return

        # Prepare vectors for Pinecone
        vectors = []
        for chunk in chunks:
            vectors.append(
                {
                    "id": chunk.id,
                    "values": chunk.embedding.tolist(),
                    "metadata": {
                        "text": chunk.text[:1000],  # Truncate for Pinecone limits
                        **chunk.metadata,
                    },
                }
            )

            # Cache chunk locally
            self.chunk_cache[chunk.id] = chunk

        # Upsert to Pinecone
        self.index.upsert(vectors=vectors, namespace=self.namespace)

    def search(
        self, query_embedding: np.ndarray, k: int, filters: Optional[Dict] = None
    ) -> List[Chunk]:
        """Search Pinecone index."""
        # Build filter
        pinecone_filter = None
        if filters:
            pinecone_filter = {key: {"$eq": value} for key, value in filters.items()}

        # Query
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            namespace=self.namespace,
            filter=pinecone_filter,
            include_metadata=True,
        )

        # Convert results to Chunks
        chunks = []
        for match in results.matches:
            chunk_id = match.id

            # Try cache first
            if chunk_id in self.chunk_cache:
                chunks.append(self.chunk_cache[chunk_id])
            else:
                # Reconstruct from metadata
                metadata = match.metadata
                text = metadata.pop("text", "")
                chunk = Chunk(
                    id=chunk_id,
                    text=text,
                    embedding=np.array(match.values),
                    metadata=metadata,
                )
                chunks.append(chunk)
                self.chunk_cache[chunk_id] = chunk

        return chunks

    def delete(self, chunk_ids: List[str]) -> None:
        """Delete chunks from Pinecone."""
        self.index.delete(ids=chunk_ids, namespace=self.namespace)

        # Remove from cache
        for chunk_id in chunk_ids:
            self.chunk_cache.pop(chunk_id, None)

    def count(self) -> int:
        """Return number of vectors in namespace."""
        stats = self.index.describe_index_stats()
        if self.namespace:
            return stats.namespaces.get(self.namespace, {}).get("vector_count", 0)
        return stats.total_vector_count
