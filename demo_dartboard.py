"""
Simple demo of Dartboard RAG retrieval.
"""

import numpy as np
from dartboard.core import DartboardConfig, DartboardRetriever
from dartboard.config import get_embedding_config
from dartboard.embeddings import SentenceTransformerModel
from dartboard.datasets.models import Chunk


def main():
    print("🎯 Dartboard RAG Demo\n")

    # Initialize embedding model
    print("Loading embedding model...")
    embedding_model = SentenceTransformerModel(get_embedding_config().model_name)
    print(f"✓ Model loaded (dim={embedding_model.embedding_dim})\n")

    # Create sample documents
    documents = [
        "The Dartboard algorithm optimizes for relevant information gain in RAG systems.",
        "Machine learning models can be used for natural language processing tasks.",
        "Retrieval-augmented generation combines retrieval with language models.",
        "Python is a popular programming language for data science and ML.",
        "Information retrieval is crucial for question answering systems.",
        "The Gaussian kernel measures similarity between embeddings.",
        "Diversity in retrieved passages improves answer quality.",
        "FastAPI is a modern web framework for building APIs with Python.",
    ]

    print(f"Creating {len(documents)} document chunks...")
    chunks = []
    for i, text in enumerate(documents):
        embedding = embedding_model.encode(text)
        chunk = Chunk(
            id=f"doc_{i}", text=text, embedding=embedding, metadata={"source": "demo"}
        )
        chunks.append(chunk)
    print(f"✓ Created {len(chunks)} chunks\n")

    # Initialize Dartboard retriever
    config = DartboardConfig(sigma=1.0, top_k=3, triage_k=8, reranker_type="cosine")
    retriever = DartboardRetriever(config, embedding_model)
    print(f"✓ Dartboard retriever initialized (σ={config.sigma}, k={config.top_k})\n")

    # Retrieve documents for a query
    query = "How does Dartboard improve RAG systems?"
    print(f"Query: '{query}'\n")
    print("Retrieving relevant passages...\n")

    result = retriever.retrieve(query, chunks)

    print(f"Top {len(result.chunks)} results:\n")
    for i, (chunk, score) in enumerate(zip(result.chunks, result.scores), 1):
        print(f"{i}. [Score: {score:.4f}]")
        print(f"   {chunk.text}\n")

    print("✓ Demo complete!")


if __name__ == "__main__":
    main()
