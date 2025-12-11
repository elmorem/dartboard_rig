"""
Metadata extraction utilities for document chunking.

Extracts rich metadata from documents including:
- Section titles and headings
- Page numbers (for PDFs)
- Document hierarchy (parent-child relationships)
- Heading levels (H1, H2, H3, etc.)
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DocumentSection:
    """Represents a section in a document with hierarchy."""

    title: str
    level: int  # Heading level (1 for H1, 2 for H2, etc.)
    start_pos: int  # Character position where section starts
    end_pos: Optional[int]  # Character position where section ends
    content: str
    parent: Optional["DocumentSection"] = None
    page_number: Optional[int] = None


class MarkdownMetadataExtractor:
    """Extract metadata from Markdown documents."""

    # Regex patterns for Markdown headings
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#{1,6})?\s*$")

    def extract_sections(self, content: str) -> List[DocumentSection]:
        """
        Extract hierarchical sections from Markdown content.

        Args:
            content: Markdown text

        Returns:
            List of DocumentSection objects with hierarchy
        """
        sections = []
        matches = list(self.HEADING_PATTERN.finditer(content))

        if not matches:
            # No headings found - treat entire document as one section
            return [
                DocumentSection(
                    title="Document",
                    level=0,
                    start_pos=0,
                    end_pos=len(content),
                    content=content,
                    parent=None,
                )
            ]

        # Create sections from headings
        for i, match in enumerate(matches):
            heading_marks = match.group(1)
            title = match.group(2).strip()
            level = len(heading_marks)
            start_pos = match.end()

            # Find end position (start of next heading or end of document)
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(content)

            section_content = content[start_pos:end_pos].strip()

            section = DocumentSection(
                title=title,
                level=level,
                start_pos=match.start(),
                end_pos=end_pos,
                content=section_content,
            )

            sections.append(section)

        # Build hierarchy (assign parents)
        self._build_hierarchy(sections)

        return sections

    def _build_hierarchy(self, sections: List[DocumentSection]) -> None:
        """
        Build parent-child relationships between sections.

        Modifies sections in place to set parent references.
        """
        stack = []  # Stack of (level, section) tuples

        for section in sections:
            # Pop sections from stack that are at same or deeper level
            while stack and stack[-1][0] >= section.level:
                stack.pop()

            # Current section's parent is the top of the stack
            if stack:
                section.parent = stack[-1][1]

            # Push current section onto stack
            stack.append((section.level, section))

    def extract_heading_at_position(
        self, content: str, char_position: int
    ) -> Optional[str]:
        """
        Find the heading that contains a given character position.

        Args:
            content: Markdown content
            char_position: Character position in document

        Returns:
            Heading text if found, None otherwise
        """
        sections = self.extract_sections(content)

        for section in sections:
            if section.start_pos <= char_position < (section.end_pos or len(content)):
                return section.title

        return None

    def get_heading_path(self, content: str, char_position: int) -> List[str]:
        """
        Get full heading path (breadcrumb) for a position.

        Args:
            content: Markdown content
            char_position: Character position

        Returns:
            List of heading titles from top level to current level
        """
        sections = self.extract_sections(content)

        # Find section containing this position
        target_section = None
        for section in sections:
            if section.start_pos <= char_position < (section.end_pos or len(content)):
                target_section = section
                break

        if not target_section:
            return []

        # Build path from root to target
        path = []
        current = target_section
        while current:
            path.insert(0, current.title)
            current = current.parent

        return path


class PDFMetadataExtractor:
    """Extract metadata from PDF documents."""

    def extract_page_mapping(self, page_texts: List[str]) -> List[Tuple[int, int, int]]:
        """
        Create mapping of character positions to page numbers.

        Args:
            page_texts: List of text strings, one per page

        Returns:
            List of (start_char, end_char, page_number) tuples
        """
        mappings = []
        current_pos = 0

        for page_num, page_text in enumerate(page_texts, start=1):
            start_pos = current_pos
            end_pos = current_pos + len(page_text)

            if page_text.strip():  # Only include non-empty pages
                mappings.append((start_pos, end_pos, page_num))

            # Account for newlines added between pages
            current_pos = end_pos + 2  # "\n\n" between pages

        return mappings

    def get_page_number(
        self, char_position: int, page_mappings: List[Tuple[int, int, int]]
    ) -> Optional[int]:
        """
        Get page number for a character position.

        Args:
            char_position: Character position in full document
            page_mappings: List of (start, end, page_num) tuples

        Returns:
            Page number (1-indexed) or None
        """
        for start, end, page_num in page_mappings:
            if start <= char_position < end:
                return page_num

        return None

    def extract_pdf_sections(
        self, content: str, page_mappings: List[Tuple[int, int, int]]
    ) -> List[DocumentSection]:
        """
        Extract sections from PDF content with page numbers.

        Uses simple heuristics to detect section headings in PDFs.

        Args:
            content: Full PDF text content
            page_mappings: Page number mappings

        Returns:
            List of DocumentSection objects
        """
        # Simple heuristic: lines that are:
        # 1. Short (< 100 chars)
        # 2. End with newline (not mid-paragraph)
        # 3. Start with capital letter
        # 4. No ending punctuation except ?
        # Might be section headings

        sections = []
        lines = content.split("\n")
        current_pos = 0

        potential_headings = []

        for line in lines:
            line_stripped = line.strip()

            if (
                line_stripped
                and len(line_stripped) < 100
                and line_stripped[0].isupper()
                and (not line_stripped[-1] in ".,:;" or line_stripped.endswith("?"))
            ):
                # Potential heading
                page_num = self.get_page_number(current_pos, page_mappings)
                potential_headings.append((current_pos, line_stripped, page_num))

            current_pos += len(line) + 1  # +1 for newline

        # Create sections from potential headings
        for i, (start_pos, title, page_num) in enumerate(potential_headings):
            end_pos = (
                potential_headings[i + 1][0]
                if i + 1 < len(potential_headings)
                else len(content)
            )

            section_content = content[start_pos:end_pos].strip()

            section = DocumentSection(
                title=title,
                level=1,  # PDFs don't have clear hierarchy
                start_pos=start_pos,
                end_pos=end_pos,
                content=section_content,
                page_number=page_num,
            )

            sections.append(section)

        return (
            sections
            if sections
            else [
                DocumentSection(
                    title="Document",
                    level=0,
                    start_pos=0,
                    end_pos=len(content),
                    content=content,
                    page_number=1,
                )
            ]
        )


class MetadataEnricher:
    """
    Enrich chunk metadata with contextual information.

    Adds section titles, page numbers, and hierarchy information
    to chunks based on their position in the document.
    """

    def __init__(self):
        self.markdown_extractor = MarkdownMetadataExtractor()
        self.pdf_extractor = PDFMetadataExtractor()

    def enrich_markdown_chunk(
        self,
        chunk_text: str,
        chunk_start: int,
        document_content: str,
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich metadata for a Markdown chunk.

        Args:
            chunk_text: The chunk's text content
            chunk_start: Starting character position in document
            document_content: Full document content
            base_metadata: Existing metadata dict

        Returns:
            Enriched metadata dict
        """
        enriched = base_metadata.copy()

        # Extract heading path
        heading_path = self.markdown_extractor.get_heading_path(
            document_content, chunk_start
        )

        if heading_path:
            enriched["section_title"] = heading_path[-1]  # Current section
            enriched["heading_path"] = " > ".join(heading_path)  # Full breadcrumb
            enriched["heading_level"] = len(heading_path)

        return enriched

    def enrich_pdf_chunk(
        self,
        chunk_text: str,
        chunk_start: int,
        page_mappings: List[Tuple[int, int, int]],
        base_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enrich metadata for a PDF chunk.

        Args:
            chunk_text: The chunk's text content
            chunk_start: Starting character position in document
            page_mappings: Page number mappings
            base_metadata: Existing metadata dict

        Returns:
            Enriched metadata dict
        """
        enriched = base_metadata.copy()

        # Add page number
        page_num = self.pdf_extractor.get_page_number(chunk_start, page_mappings)
        if page_num:
            enriched["page_number"] = page_num

        return enriched

    def enrich_chunk_metadata(
        self,
        chunk_text: str,
        chunk_start: int,
        document_content: str,
        document_type: str,
        base_metadata: Dict[str, Any],
        page_mappings: Optional[List[Tuple[int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Auto-detect document type and enrich metadata accordingly.

        Args:
            chunk_text: The chunk's text
            chunk_start: Starting position
            document_content: Full document
            document_type: "markdown", "pdf", or "text"
            base_metadata: Existing metadata
            page_mappings: Optional page mappings for PDFs

        Returns:
            Enriched metadata dict
        """
        if document_type == "markdown" or document_type == "md":
            return self.enrich_markdown_chunk(
                chunk_text, chunk_start, document_content, base_metadata
            )
        elif document_type == "pdf" and page_mappings:
            return self.enrich_pdf_chunk(
                chunk_text, chunk_start, page_mappings, base_metadata
            )
        else:
            # No enrichment for unknown types
            return base_metadata.copy()
