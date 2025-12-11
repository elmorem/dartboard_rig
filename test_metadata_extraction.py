"""
Tests for metadata extraction functionality.

Tests extraction of:
- Section titles from Markdown
- Heading hierarchy
- Page numbers from PDFs
- Metadata enrichment for chunks
"""

import pytest
from dartboard.ingestion.metadata_extractors import (
    MarkdownMetadataExtractor,
    PDFMetadataExtractor,
    MetadataEnricher,
    DocumentSection,
)


# Sample Markdown with hierarchy
SAMPLE_MARKDOWN = """
# Introduction

This is the introduction section.

## Background

Some background information here.

### Historical Context

Details about history.

## Motivation

Why this matters.

# Methods

Description of methods.

## Data Collection

How data was collected.

## Analysis

Analysis procedures.

# Results

The findings.

## Key Findings

Important results here.

# Conclusion

Final thoughts.
""".strip()


SAMPLE_FLAT_MARKDOWN = """
# Section One

Content for section one.

# Section Two

Content for section two.

# Section Three

Content for section three.
""".strip()


class TestMarkdownMetadataExtractor:
    """Tests for MarkdownMetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        return MarkdownMetadataExtractor()

    def test_extract_sections(self, extractor):
        """Test basic section extraction."""
        sections = extractor.extract_sections(SAMPLE_MARKDOWN)

        assert len(sections) > 0
        assert all(isinstance(s, DocumentSection) for s in sections)

        # Check we found the main headings
        titles = [s.title for s in sections]
        assert "Introduction" in titles
        assert "Methods" in titles
        assert "Results" in titles
        assert "Conclusion" in titles

    def test_section_hierarchy(self, extractor):
        """Test that hierarchy is correctly built."""
        sections = extractor.extract_sections(SAMPLE_MARKDOWN)

        # Find "Background" section (H2 under Introduction)
        background = next(s for s in sections if s.title == "Background")

        assert background.level == 2
        assert background.parent is not None
        assert background.parent.title == "Introduction"
        assert background.parent.level == 1

    def test_nested_hierarchy(self, extractor):
        """Test deeply nested headings."""
        sections = extractor.extract_sections(SAMPLE_MARKDOWN)

        # Find "Historical Context" (H3 under Background under Introduction)
        historical = next(s for s in sections if s.title == "Historical Context")

        assert historical.level == 3
        assert historical.parent is not None
        assert historical.parent.title == "Background"
        assert historical.parent.parent.title == "Introduction"

    def test_heading_at_position(self, extractor):
        """Test finding heading for a position."""
        # Find position of "Background" content
        background_pos = SAMPLE_MARKDOWN.find("Some background information")

        heading = extractor.extract_heading_at_position(SAMPLE_MARKDOWN, background_pos)

        assert heading == "Background"

    def test_heading_path(self, extractor):
        """Test getting full heading path (breadcrumb)."""
        # Find position in "Historical Context" section
        historical_pos = SAMPLE_MARKDOWN.find("Details about history")

        path = extractor.get_heading_path(SAMPLE_MARKDOWN, historical_pos)

        assert path == ["Introduction", "Background", "Historical Context"]

    def test_flat_hierarchy(self, extractor):
        """Test document with flat structure (all H1)."""
        sections = extractor.extract_sections(SAMPLE_FLAT_MARKDOWN)

        # All sections should be top-level (no parents)
        assert all(s.parent is None for s in sections)
        assert all(s.level == 1 for s in sections)

    def test_no_headings(self, extractor):
        """Test document without headings."""
        plain_text = "This is plain text with no headings."

        sections = extractor.extract_sections(plain_text)

        assert len(sections) == 1
        assert sections[0].title == "Document"
        assert sections[0].level == 0

    def test_section_content(self, extractor):
        """Test that section content is correctly extracted."""
        sections = extractor.extract_sections(SAMPLE_MARKDOWN)

        intro_section = next(s for s in sections if s.title == "Introduction")

        # Content should include text under Introduction but not under Methods
        assert "introduction section" in intro_section.content.lower()
        # Should NOT include content from Methods section
        assert "Description of methods" not in intro_section.content


class TestPDFMetadataExtractor:
    """Tests for PDFMetadataExtractor."""

    @pytest.fixture
    def extractor(self):
        return PDFMetadataExtractor()

    def test_page_mapping(self, extractor):
        """Test creation of page mappings."""
        page_texts = [
            "Page 1 content here.",
            "Page 2 content here.",
            "Page 3 content here.",
        ]

        mappings = extractor.extract_page_mapping(page_texts)

        assert len(mappings) == 3

        # Check first page
        assert mappings[0][2] == 1  # Page number
        assert mappings[0][0] == 0  # Start position

        # Check second page
        assert mappings[1][2] == 2

        # Check third page
        assert mappings[2][2] == 3

    def test_get_page_number(self, extractor):
        """Test getting page number for a position."""
        page_texts = [
            "First page text.",  # 0-16
            "Second page text.",  # 18-35
            "Third page text.",  # 37-53
        ]

        mappings = extractor.extract_page_mapping(page_texts)

        # Position in first page
        assert extractor.get_page_number(5, mappings) == 1

        # Position in second page
        assert extractor.get_page_number(25, mappings) == 2

        # Position in third page
        assert extractor.get_page_number(45, mappings) == 3

    def test_empty_pages(self, extractor):
        """Test handling of empty pages."""
        page_texts = ["Page 1", "", "Page 3"]  # Page 2 is empty

        mappings = extractor.extract_page_mapping(page_texts)

        # Should only map non-empty pages
        assert len(mappings) == 2
        assert mappings[0][2] == 1
        assert mappings[1][2] == 3  # Skips page 2


class TestMetadataEnricher:
    """Tests for MetadataEnricher."""

    @pytest.fixture
    def enricher(self):
        return MetadataEnricher()

    def test_enrich_markdown_chunk(self, enricher):
        """Test enriching Markdown chunk metadata."""
        # Find position in "Background" section
        chunk_start = SAMPLE_MARKDOWN.find("Some background information")
        chunk_text = "Some background information here."

        base_metadata = {"source": "test.md"}

        enriched = enricher.enrich_markdown_chunk(
            chunk_text, chunk_start, SAMPLE_MARKDOWN, base_metadata
        )

        assert "section_title" in enriched
        assert enriched["section_title"] == "Background"

        assert "heading_path" in enriched
        assert enriched["heading_path"] == "Introduction > Background"

        assert "heading_level" in enriched
        assert enriched["heading_level"] == 2

    def test_enrich_pdf_chunk(self, enricher):
        """Test enriching PDF chunk metadata."""
        page_texts = ["Page 1 text.", "Page 2 text.", "Page 3 text."]

        pdf_extractor = PDFMetadataExtractor()
        page_mappings = pdf_extractor.extract_page_mapping(page_texts)

        # Chunk in page 2
        chunk_start = 20  # In second page
        chunk_text = "Some text from page 2"

        base_metadata = {"source": "test.pdf"}

        enriched = enricher.enrich_pdf_chunk(
            chunk_text, chunk_start, page_mappings, base_metadata
        )

        assert "page_number" in enriched
        assert enriched["page_number"] == 2

    def test_auto_detect_markdown(self, enricher):
        """Test auto-detection for Markdown."""
        chunk_start = SAMPLE_MARKDOWN.find("Some background")
        chunk_text = "Some background information"

        base_metadata = {"source": "test.md", "source_type": "markdown"}

        enriched = enricher.enrich_chunk_metadata(
            chunk_text,
            chunk_start,
            SAMPLE_MARKDOWN,
            document_type="markdown",
            base_metadata=base_metadata,
        )

        assert "section_title" in enriched
        assert "heading_path" in enriched

    def test_auto_detect_pdf(self, enricher):
        """Test auto-detection for PDF."""
        page_texts = ["Page 1", "Page 2"]
        pdf_extractor = PDFMetadataExtractor()
        page_mappings = pdf_extractor.extract_page_mapping(page_texts)

        chunk_start = 10
        chunk_text = "Text from PDF"

        base_metadata = {"source": "test.pdf", "source_type": "pdf"}

        enriched = enricher.enrich_chunk_metadata(
            chunk_text,
            chunk_start,
            "Page 1\n\nPage 2",
            document_type="pdf",
            base_metadata=base_metadata,
            page_mappings=page_mappings,
        )

        assert "page_number" in enriched

    def test_unknown_document_type(self, enricher):
        """Test handling of unknown document type."""
        base_metadata = {"source": "test.txt"}

        enriched = enricher.enrich_chunk_metadata(
            "Some text",
            0,
            "Full document",
            document_type="unknown",
            base_metadata=base_metadata,
        )

        # Should return base metadata unchanged
        assert enriched == base_metadata


def test_end_to_end_metadata_enrichment():
    """Test complete metadata enrichment workflow."""
    enricher = MetadataEnricher()

    # Simulate chunking a Markdown document
    document = SAMPLE_MARKDOWN

    # Simulate chunks at different positions
    chunks = [
        {"text": "This is the introduction", "start": 0},
        {
            "text": "Some background information",
            "start": SAMPLE_MARKDOWN.find("Some background"),
        },
        {
            "text": "Details about history",
            "start": SAMPLE_MARKDOWN.find("Details about history"),
        },
    ]

    enriched_chunks = []

    for chunk_info in chunks:
        base_metadata = {
            "source": "document.md",
            "source_type": "markdown",
            "chunk_index": len(enriched_chunks),
        }

        enriched = enricher.enrich_markdown_chunk(
            chunk_info["text"],
            chunk_info["start"],
            document,
            base_metadata,
        )

        enriched_chunks.append(enriched)

    # Verify first chunk (in Introduction)
    assert "section_title" in enriched_chunks[0]
    assert enriched_chunks[0]["section_title"] == "Introduction"

    # Verify second chunk (in Background under Introduction)
    assert enriched_chunks[1]["section_title"] == "Background"
    assert enriched_chunks[1]["heading_path"] == "Introduction > Background"
    assert enriched_chunks[1]["heading_level"] == 2

    # Verify third chunk (in Historical Context)
    assert enriched_chunks[2]["section_title"] == "Historical Context"
    assert (
        "Introduction > Background > Historical Context"
        in enriched_chunks[2]["heading_path"]
    )
    assert enriched_chunks[2]["heading_level"] == 3


if __name__ == "__main__":
    print("Running metadata extraction tests...")
    pytest.main([__file__, "-v"])
