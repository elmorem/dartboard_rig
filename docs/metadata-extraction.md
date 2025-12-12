# Metadata Extraction

## Overview

Metadata extraction enriches document chunks with contextual information like section titles, page numbers, heading paths, and document hierarchy. This enables more precise retrieval and better user experience by providing source context.

**Key Benefits:**
- 📍 **Section Context**: Know which section a chunk came from
- 📄 **Page Numbers**: Reference specific pages in PDFs
- 🗂️ **Hierarchical Structure**: Understand document organization
- 🎯 **Better Retrieval**: Filter by metadata for targeted search
- 👤 **User Experience**: Show users where information came from

## Quick Start

### Basic Metadata Enrichment

```python
from dartboard.ingestion.metadata_extractors import MetadataEnricher
from dartboard.ingestion.loaders import MarkdownLoader

# Load document
loader = MarkdownLoader()
documents = loader.load("documentation.md")
doc = documents[0]

# Create enricher
enricher = MetadataEnricher()

# Enrich chunk metadata
enriched_metadata = enricher.enrich_markdown_chunk(
    chunk_text="This is a chunk...",
    chunk_start=500,  # Character position in document
    document_content=doc.content,
    base_metadata={"source": "documentation.md"}
)

print(enriched_metadata)
# {
#     'source': 'documentation.md',
#     'section_title': 'Installation',
#     'heading_path': 'Getting Started > Installation',
#     'heading_level': 2
# }
```

## Document Structure Extraction

### MarkdownMetadataExtractor

Extracts hierarchical structure from Markdown documents.

**Features:**
- Parses Markdown headings (`#`, `##`, `###`)
- Builds parent-child section relationships
- Extracts heading paths (breadcrumbs)
- Finds section containing any text position

#### Extracting Sections

```python
from dartboard.ingestion.metadata_extractors import MarkdownMetadataExtractor

extractor = MarkdownMetadataExtractor()

markdown_content = """
# Getting Started

## Installation

Install using pip...

## Configuration

Configure the system...

### Environment Variables

Set the following variables...
"""

sections = extractor.extract_sections(markdown_content)

for section in sections:
    print(f"Level {section.level}: {section.title}")
    print(f"  Position: {section.start_pos}-{section.end_pos}")
    print(f"  Parent: {section.parent.title if section.parent else 'None'}")
```

**Output:**
```
Level 1: Getting Started
  Position: 0-50
  Parent: None
Level 2: Installation
  Position: 50-120
  Parent: Getting Started
Level 2: Configuration
  Position: 120-200
  Parent: Getting Started
Level 3: Environment Variables
  Position: 200-300
  Parent: Configuration
```

#### Document Section Object

```python
@dataclass
class DocumentSection:
    title: str              # Heading text
    level: int             # Heading level (1-6)
    start_pos: int         # Start character position
    end_pos: Optional[int] # End character position
    content: str           # Section content (without heading)
    parent: Optional[DocumentSection]  # Parent section
    page_number: Optional[int]  # Page number (for PDFs)
```

#### Finding Section at Position

```python
# Find which section contains character position 150
heading = extractor.extract_heading_at_position(markdown_content, 150)
print(heading)  # "Installation"
```

#### Getting Heading Path (Breadcrumb)

```python
# Get full path to heading at position 250
path = extractor.get_heading_path(markdown_content, 250)
print(" > ".join(path))
# "Getting Started > Configuration > Environment Variables"
```

### PDFMetadataExtractor

Extracts metadata from PDF documents with page number mapping.

**Features:**
- Maps character positions to page numbers
- Extracts potential section headings
- Maintains page boundaries
- Supports multi-page documents

#### Page Number Mapping

```python
from dartboard.ingestion.metadata_extractors import PDFMetadataExtractor

extractor = PDFMetadataExtractor()

# From PDF loader
page_texts = [
    "Page 1 content...",
    "Page 2 content...",
    "Page 3 content..."
]

# Create position-to-page mapping
mappings = extractor.extract_page_mapping(page_texts)

# mappings = [
#     (0, 16, 1),      # chars 0-16 are on page 1
#     (18, 34, 2),     # chars 18-34 are on page 2
#     (36, 52, 3)      # chars 36-52 are on page 3
# ]

# Find page number for character position
page_num = extractor.get_page_number(25, mappings)
print(page_num)  # 2
```

#### Extracting PDF Sections

```python
# Full PDF text
pdf_content = "\n\n".join(page_texts)

# Extract sections with page numbers
sections = extractor.extract_pdf_sections(pdf_content, mappings)

for section in sections:
    print(f"{section.title} (Page {section.page_number})")
```

**Note:** PDF section extraction uses heuristics to detect headings:
- Short lines (< 100 chars)
- Start with capital letter
- No ending punctuation (except ?)
- Followed by newline

## Metadata Enrichment

### MetadataEnricher

Unified interface for enriching chunk metadata across document types.

```python
from dartboard.ingestion.metadata_extractors import MetadataEnricher

enricher = MetadataEnricher()
```

### Enriching Markdown Chunks

```python
# Enrich a chunk from a Markdown document
enriched = enricher.enrich_markdown_chunk(
    chunk_text="Installation is simple...",
    chunk_start=120,  # Position in document
    document_content=full_markdown_content,
    base_metadata={
        "source": "README.md",
        "file_type": "markdown"
    }
)

print(enriched)
# {
#     'source': 'README.md',
#     'file_type': 'markdown',
#     'section_title': 'Installation',
#     'heading_path': 'Getting Started > Installation',
#     'heading_level': 2
# }
```

**Metadata Fields Added:**
- `section_title`: Current section heading
- `heading_path`: Full breadcrumb (e.g., "Parent > Child > Section")
- `heading_level`: Depth in heading hierarchy (1-6)

### Enriching PDF Chunks

```python
# Enrich a chunk from a PDF
enriched = enricher.enrich_pdf_chunk(
    chunk_text="The attention mechanism...",
    chunk_start=850,  # Position in full PDF text
    page_mappings=page_mappings,
    base_metadata={
        "source": "paper.pdf",
        "file_type": "pdf"
    }
)

print(enriched)
# {
#     'source': 'paper.pdf',
#     'file_type': 'pdf',
#     'page_number': 3
# }
```

**Metadata Fields Added:**
- `page_number`: PDF page number (1-indexed)

### Auto-Detection

```python
# Automatically detect document type and enrich
enriched = enricher.enrich_chunk_metadata(
    chunk_text="...",
    chunk_start=position,
    document_content=content,
    document_type="markdown",  # or "pdf"
    base_metadata=base_meta,
    page_mappings=mappings  # Optional, for PDFs
)
```

## Integration with Chunking

### Enriching Chunks During Ingestion

```python
from dartboard.ingestion.chunking import RecursiveChunker
from dartboard.ingestion.metadata_extractors import MetadataEnricher
from dartboard.ingestion.loaders import MarkdownLoader

# Load document
loader = MarkdownLoader()
documents = loader.load("documentation.md")
doc = documents[0]

# Create chunker
chunker = RecursiveChunker(chunk_size=512, overlap=50)

# Chunk document
chunks = chunker.chunk(doc)

# Enrich metadata
enricher = MetadataEnricher()

for chunk in chunks:
    # Find chunk position in original document
    chunk_start = doc.content.find(chunk.text)

    # Enrich metadata
    enriched_metadata = enricher.enrich_markdown_chunk(
        chunk_text=chunk.text,
        chunk_start=chunk_start,
        document_content=doc.content,
        base_metadata=chunk.metadata
    )

    # Update chunk metadata
    chunk.metadata = enriched_metadata

# Now chunks have rich metadata
for chunk in chunks[:3]:
    print(f"Chunk: {chunk.text[:50]}...")
    print(f"  Section: {chunk.metadata.get('section_title', 'Unknown')}")
    print(f"  Path: {chunk.metadata.get('heading_path', 'Unknown')}")
```

### Custom Chunker with Metadata

```python
from dartboard.ingestion.chunking import RecursiveChunker
from dartboard.ingestion.metadata_extractors import MetadataEnricher
from typing import List

class MetadataAwareChunker(RecursiveChunker):
    """Chunker that automatically enriches metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enricher = MetadataEnricher()

    def chunk(self, document) -> List:
        # Get base chunks
        chunks = super().chunk(document)

        # Detect document type
        doc_type = document.metadata.get("source_type", "text")

        # Prepare page mappings for PDFs
        page_mappings = None
        if doc_type == "pdf":
            # Would need to extract from document
            page_mappings = []  # TODO: implement

        # Enrich each chunk
        for chunk in chunks:
            chunk_start = document.content.find(chunk.text)

            enriched_metadata = self.enricher.enrich_chunk_metadata(
                chunk_text=chunk.text,
                chunk_start=chunk_start,
                document_content=document.content,
                document_type=doc_type,
                base_metadata=chunk.metadata,
                page_mappings=page_mappings
            )

            chunk.metadata = enriched_metadata

        return chunks

# Use custom chunker
chunker = MetadataAwareChunker(chunk_size=512, overlap=50)
chunks = chunker.chunk(document)
```

## Use Cases

### 1. Section-Specific Search

```python
from dartboard.retrieval.dense import DenseRetriever
from dartboard.storage.vector_store import FAISSStore

# Retrieve only from "Installation" section
retriever = DenseRetriever(embedding_model, vector_store)

results = retriever.retrieve(
    query="How do I install?",
    k=20  # Get more results for filtering
)

# Filter by section
installation_chunks = [
    chunk for chunk in results
    if chunk.metadata.get("section_title") == "Installation"
][:5]
```

### 2. Page Reference in Responses

```python
from dartboard.generation import create_generator

# Generate answer
generator = create_generator(provider="openai")
result = generator.generate(query, chunks)

# Format with page numbers
for source in result.sources:
    page = source["metadata"].get("page_number", "?")
    print(f"[Source {source['number']}] Page {page}:")
    print(f"  {source['text'][:100]}...")
```

### 3. Structured Navigation

```python
# Build document table of contents from chunks
toc = {}
for chunk in all_chunks:
    path = chunk.metadata.get("heading_path", "")
    if path and path not in toc:
        toc[path] = {
            "level": chunk.metadata.get("heading_level", 0),
            "chunks": []
        }

    if path:
        toc[path]["chunks"].append(chunk.id)

# Display TOC
for path, info in sorted(toc.items()):
    indent = "  " * (info["level"] - 1)
    print(f"{indent}{path} ({len(info['chunks'])} chunks)")
```

### 4. Context-Aware Chunking

```python
# Prefer not to split sections
from dartboard.ingestion.metadata_extractors import MarkdownMetadataExtractor

def chunk_by_sections(markdown_content: str):
    """Chunk by Markdown sections instead of arbitrary positions."""
    extractor = MarkdownMetadataExtractor()
    sections = extractor.extract_sections(markdown_content)

    chunks = []
    for section in sections:
        # Each section becomes a chunk
        chunks.append({
            "text": section.content,
            "metadata": {
                "section_title": section.title,
                "level": section.level,
                "parent": section.parent.title if section.parent else None
            }
        })

    return chunks
```

## Advanced Features

### Building Section Hierarchy

```python
def build_section_tree(sections):
    """Build nested dictionary of section hierarchy."""
    root = {"title": "Document", "children": {}}
    stack = [root]

    for section in sections:
        # Pop until we find the parent level
        while len(stack) > section.level:
            stack.pop()

        # Create node
        node = {
            "title": section.title,
            "level": section.level,
            "content": section.content[:100] + "...",
            "children": {}
        }

        # Add to parent
        stack[-1]["children"][section.title] = node

        # Push to stack
        stack.append(node)

    return root

# Use it
sections = extractor.extract_sections(markdown_content)
tree = build_section_tree(sections)

# Tree structure:
# {
#     'title': 'Document',
#     'children': {
#         'Getting Started': {
#             'title': 'Getting Started',
#             'children': {
#                 'Installation': {...},
#                 'Configuration': {
#                     'children': {
#                         'Environment Variables': {...}
#                     }
#                 }
#             }
#         }
#     }
# }
```

### Extracting Code Block Metadata

```python
# Markdown loader can extract code blocks
from dartboard.ingestion.loaders import MarkdownLoader

loader = MarkdownLoader(extract_code_blocks=True)
documents = loader.load("README.md")

doc = documents[0]
code_blocks = doc.metadata.get("code_blocks", [])

for block in code_blocks:
    print(f"Language: {block['language']}")
    print(f"Code:\n{block['code']}\n")
```

### Custom Metadata Extractors

```python
from dartboard.ingestion.metadata_extractors import MetadataEnricher

class CustomEnricher(MetadataEnricher):
    """Custom metadata enricher with domain-specific logic."""

    def enrich_code_chunk(
        self,
        chunk_text: str,
        chunk_start: int,
        document_content: str,
        base_metadata: dict
    ) -> dict:
        """Enrich code chunks with function/class names."""
        enriched = base_metadata.copy()

        # Extract function name if this chunk contains a function
        if "def " in chunk_text:
            func_name = self._extract_function_name(chunk_text)
            enriched["function_name"] = func_name

        # Extract class name
        if "class " in chunk_text:
            class_name = self._extract_class_name(chunk_text)
            enriched["class_name"] = class_name

        # Determine module from file path
        file_path = base_metadata.get("file_path", "")
        if file_path:
            enriched["module"] = self._path_to_module(file_path)

        return enriched

    def _extract_function_name(self, text: str) -> str:
        import re
        match = re.search(r'def\s+(\w+)\s*\(', text)
        return match.group(1) if match else None

    def _extract_class_name(self, text: str) -> str:
        import re
        match = re.search(r'class\s+(\w+)\s*[(::]', text)
        return match.group(1) if match else None

    def _path_to_module(self, path: str) -> str:
        # Convert file path to Python module name
        # e.g., "src/utils/helpers.py" -> "utils.helpers"
        return path.replace("/", ".").replace(".py", "")
```

## Performance Considerations

### Caching Section Extractions

```python
from functools import lru_cache

class CachedMetadataEnricher(MetadataEnricher):
    """Enricher with cached section extraction."""

    @lru_cache(maxsize=128)
    def _get_sections(self, document_hash: str, content: str):
        """Cache section extraction by document hash."""
        return self.markdown_extractor.extract_sections(content)

    def enrich_markdown_chunk(
        self,
        chunk_text: str,
        chunk_start: int,
        document_content: str,
        base_metadata: dict
    ) -> dict:
        # Use cached sections
        doc_hash = hash(document_content)
        sections = self._get_sections(doc_hash, document_content)

        # Find section containing this chunk
        for section in sections:
            if section.start_pos <= chunk_start < (section.end_pos or len(document_content)):
                enriched = base_metadata.copy()
                enriched["section_title"] = section.title
                # Build heading path
                path = []
                current = section
                while current:
                    path.insert(0, current.title)
                    current = current.parent
                enriched["heading_path"] = " > ".join(path)
                enriched["heading_level"] = section.level
                return enriched

        return base_metadata
```

### Batch Enrichment

```python
def enrich_chunks_batch(chunks, document, enricher):
    """Enrich multiple chunks efficiently."""
    # Extract sections once
    sections = enricher.markdown_extractor.extract_sections(document.content)

    # Build position index
    position_to_section = {}
    for section in sections:
        for pos in range(section.start_pos, section.end_pos or len(document.content)):
            position_to_section[pos] = section

    # Enrich all chunks
    for chunk in chunks:
        chunk_start = document.content.find(chunk.text)
        section = position_to_section.get(chunk_start)

        if section:
            chunk.metadata["section_title"] = section.title
            # ... more metadata

    return chunks
```

## Best Practices

### 1. Always Enrich Metadata During Ingestion

```python
# ✅ Good: Enrich during ingestion
chunks = chunker.chunk(document)
enriched_chunks = [enrich(chunk, document) for chunk in chunks]
vector_store.add(enriched_chunks)

# ❌ Bad: No metadata enrichment
chunks = chunker.chunk(document)
vector_store.add(chunks)  # Missing context
```

### 2. Include Source Information

```python
# Always include at minimum:
base_metadata = {
    "source": document.source,
    "source_type": document.metadata["source_type"],
    "chunk_index": i
}
```

### 3. Use Consistent Metadata Fields

```python
# Define standard metadata schema
METADATA_SCHEMA = {
    "source": str,  # Required
    "source_type": str,  # Required: "pdf", "markdown", "code"
    "chunk_index": int,  # Required
    "section_title": str,  # Optional
    "heading_path": str,  # Optional
    "page_number": int,  # Optional (PDFs)
    "heading_level": int,  # Optional (Markdown)
}
```

### 4. Validate Metadata

```python
def validate_metadata(metadata: dict) -> bool:
    """Validate chunk metadata against schema."""
    required_fields = ["source", "source_type", "chunk_index"]

    for field in required_fields:
        if field not in metadata:
            return False

    return True
```

## Troubleshooting

### Issue: No section titles extracted

**Cause:** Document has no headings or irregular heading format

**Solution:**
```python
# Check if sections were found
sections = extractor.extract_sections(content)
if not sections:
    print("No sections found")
    # Fallback to basic metadata
```

### Issue: Incorrect page numbers

**Cause:** Page mapping doesn't account for PDF structure

**Solution:**
```python
# Verify page mappings
for start, end, page in page_mappings:
    print(f"Page {page}: chars {start}-{end}")

# Test with known position
page = extractor.get_page_number(500, page_mappings)
print(f"Char 500 is on page {page}")
```

### Issue: Missing heading paths

**Cause:** Hierarchy not built correctly

**Solution:**
```python
# Check if hierarchy was built
for section in sections:
    print(f"{section.title}: parent={section.parent}")

# Manually build hierarchy if needed
extractor._build_hierarchy(sections)
```

## See Also

- [Chunking Strategies](./chunking-strategies.md) - Chunk creation
- [Ingestion Pipeline](./ingestion-pipeline.md) - Full ingestion workflow
- [Retrieval Methods](./retrieval-methods.md) - Using metadata for retrieval
- [Vector Stores](./vector-stores.md) - Metadata filtering in search
