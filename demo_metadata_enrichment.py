#!/usr/bin/env python3
"""
Demonstration of metadata enrichment for chunked documents.

Shows how to use MetadataEnricher to add:
- Section titles from Markdown headings
- Heading paths (breadcrumbs)
- Page numbers from PDFs
- Document hierarchy
"""

from dartboard.ingestion.chunking import SentenceChunker, Document
from dartboard.ingestion.metadata_extractors import (
    MetadataEnricher,
    MarkdownMetadataExtractor,
)


# Sample Markdown document with hierarchy
SAMPLE_MARKDOWN = """
# Machine Learning Fundamentals

Machine learning is transforming how we build software.

## Supervised Learning

Supervised learning uses labeled data to train models.

### Classification

Classification assigns discrete labels to inputs.

### Regression

Regression predicts continuous values.

## Unsupervised Learning

Unsupervised learning discovers patterns in unlabeled data.

### Clustering

Clustering groups similar data points together.

## Deep Learning

Deep learning uses neural networks with multiple layers.

### Convolutional Networks

CNNs excel at image processing tasks.

### Recurrent Networks

RNNs handle sequential data effectively.
""".strip()


def main():
    print("=" * 80)
    print("METADATA ENRICHMENT DEMONSTRATION")
    print("=" * 80)

    # Step 1: Chunk the document
    print("\n[1] Chunking document...")
    document = Document(
        content=SAMPLE_MARKDOWN,
        metadata={"source": "ml_guide.md", "source_type": "markdown"},
        source="ml_guide.md",
    )

    chunker = SentenceChunker(chunk_size=128, overlap=20)
    chunks = chunker.chunk(document)

    print(f"    Created {len(chunks)} chunks\n")

    # Step 2: Initialize metadata enricher
    print("[2] Initializing metadata enricher...")
    enricher = MetadataEnricher()

    # Step 3: Enrich chunks with metadata
    print("[3] Enriching chunks with section titles and hierarchy...\n")

    enriched_chunks = []
    current_pos = 0

    for chunk in chunks:
        # Find chunk position in original document
        chunk_start = SAMPLE_MARKDOWN.find(chunk.text[:50])  # Find by first 50 chars

        if chunk_start == -1:
            # Fallback: use current position estimate
            chunk_start = current_pos

        # Enrich metadata
        enriched_metadata = enricher.enrich_markdown_chunk(
            chunk.text, chunk_start, SAMPLE_MARKDOWN, chunk.metadata
        )

        enriched_chunks.append(
            {
                "text": chunk.text,
                "metadata": enriched_metadata,
                "chunk_index": chunk.chunk_index,
            }
        )

        current_pos += len(chunk.text)

    # Step 4: Display enriched chunks
    print("=" * 80)
    print("ENRICHED CHUNKS")
    print("=" * 80)

    for i, chunk in enumerate(enriched_chunks, 1):
        print(f"\n[Chunk {i}]")
        print(f"Text: {chunk['text'][:80]}...")

        if "section_title" in chunk["metadata"]:
            print(f"Section: {chunk['metadata']['section_title']}")

        if "heading_path" in chunk["metadata"]:
            print(f"Path: {chunk['metadata']['heading_path']}")

        if "heading_level" in chunk["metadata"]:
            print(f"Level: H{chunk['metadata']['heading_level']}")

        print("-" * 40)

    # Step 5: Demonstrate section extraction
    print("\n" + "=" * 80)
    print("DOCUMENT STRUCTURE")
    print("=" * 80)

    md_extractor = MarkdownMetadataExtractor()
    sections = md_extractor.extract_sections(SAMPLE_MARKDOWN)

    def print_section_tree(section, indent=0):
        """Print section hierarchy as a tree."""
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        print(f"{prefix}{section.title} (H{section.level})")

    # Group sections by hierarchy
    root_sections = [s for s in sections if s.parent is None]

    for root in root_sections:
        print_section_tree(root, 0)

        # Find children
        children = [s for s in sections if s.parent == root]
        for child in children:
            print_section_tree(child, 1)

            # Find grandchildren
            grandchildren = [s for s in sections if s.parent == child]
            for grandchild in grandchildren:
                print_section_tree(grandchild, 2)

    # Step 6: Show heading paths for key sections
    print("\n" + "=" * 80)
    print("HEADING PATHS (Breadcrumbs)")
    print("=" * 80)

    sample_positions = [
        ("Supervised Learning section", "Supervised learning uses"),
        ("Classification subsection", "Classification assigns"),
        ("CNNs section", "CNNs excel at"),
    ]

    for desc, search_text in sample_positions:
        pos = SAMPLE_MARKDOWN.find(search_text)
        if pos != -1:
            path = md_extractor.get_heading_path(SAMPLE_MARKDOWN, pos)
            print(f"\n{desc}:")
            print(f"  Position: {pos}")
            print(f"  Path: {' > '.join(path)}")

    print("\n" + "=" * 80)
    print("✓ METADATA ENRICHMENT COMPLETE")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Section title extraction from Markdown headings")
    print("  ✓ Hierarchical heading paths (breadcrumbs)")
    print("  ✓ Heading level tracking (H1, H2, H3)")
    print("  ✓ Document structure analysis")
    print("  ✓ Position-based section lookup")
    print()


if __name__ == "__main__":
    main()
