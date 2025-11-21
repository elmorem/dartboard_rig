"""
Text chunking utilities for RAG document processing.

Implements multiple chunking strategies:
- RecursiveChunker: Sentence-aware chunking with overlap
- SemanticChunker: Paragraph/section-based chunking
- FixedSizeChunker: Simple token-based chunking
"""

import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:
    tiktoken = None


@dataclass
class Document:
    """Document to be chunked."""

    content: str
    metadata: dict
    source: str


@dataclass
class Chunk:
    """Text chunk with metadata."""

    text: str
    metadata: dict
    chunk_index: int
    token_count: Optional[int] = None


class TokenCounter:
    """Utility for counting tokens using tiktoken."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initialize token counter.

        Args:
            model: Model name for tiktoken encoding (default: gpt-3.5-turbo)
        """
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install with: pip install tiktoken"
            )
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def estimate_tokens(self, text: str) -> int:
        """
        Fast token estimation (4 chars ≈ 1 token).

        More accurate than simple word count, faster than tiktoken.
        """
        return len(text) // 4


class RecursiveChunker:
    """
    Sentence-aware chunker with overlap.

    Splits text on sentence boundaries while respecting:
    - Maximum chunk size (in tokens)
    - Overlap between chunks
    - Code block integrity
    - Paragraph boundaries
    """

    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

    # Code block patterns
    CODE_BLOCK_START = re.compile(r"^```[\w]*\s*$", re.MULTILINE)
    CODE_BLOCK_END = re.compile(r"^```\s*$", re.MULTILINE)

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        respect_code_blocks: bool = True,
        model: str = "gpt-3.5-turbo",
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            respect_code_blocks: Don't split code blocks mid-block
            model: Model for token counting
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_code_blocks = respect_code_blocks

        # Initialize token counter if available
        try:
            self.token_counter = TokenCounter(model)
        except ImportError:
            self.token_counter = None

    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk document into overlapping chunks.

        Args:
            document: Document to chunk

        Returns:
            List of chunks with metadata
        """
        text = document.content

        # Extract and protect code blocks if needed
        if self.respect_code_blocks and "```" in text:
            text, code_blocks = self._extract_code_blocks(text)
        else:
            code_blocks = []

        # Split into sentences
        sentences = self._split_sentences(text)

        # Group sentences into chunks
        chunks = self._create_chunks(sentences, document.metadata)

        # Restore code blocks
        if code_blocks:
            chunks = self._restore_code_blocks(chunks, code_blocks)

        return chunks

    def _extract_code_blocks(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Extract code blocks and replace with placeholders.

        Returns:
            (text_with_placeholders, [(placeholder, code_block)])
        """
        code_blocks = []
        placeholder_pattern = "<<<CODE_BLOCK_{}>>>"

        # Find all code blocks
        parts = []
        last_end = 0
        in_code_block = False
        block_start = 0

        lines = text.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for code block start
            if self.CODE_BLOCK_START.match(line) and not in_code_block:
                # Add text before code block
                if i > last_end:
                    parts.append("\n".join(lines[last_end:i]))

                in_code_block = True
                block_start = i
                i += 1
                continue

            # Check for code block end
            if self.CODE_BLOCK_END.match(line) and in_code_block:
                # Extract code block
                code_block = "\n".join(lines[block_start : i + 1])
                placeholder = placeholder_pattern.format(len(code_blocks))
                code_blocks.append((placeholder, code_block))
                parts.append(placeholder)

                in_code_block = False
                last_end = i + 1
                i += 1
                continue

            i += 1

        # Add remaining text
        if last_end < len(lines):
            parts.append("\n".join(lines[last_end:]))

        return "\n".join(parts), code_blocks

    def _restore_code_blocks(
        self, chunks: List[Chunk], code_blocks: List[Tuple[str, str]]
    ) -> List[Chunk]:
        """Restore code blocks to chunks."""
        for chunk in chunks:
            for placeholder, code_block in code_blocks:
                chunk.text = chunk.text.replace(placeholder, code_block)
        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _create_chunks(self, sentences: List[str], base_metadata: dict) -> List[Chunk]:
        """Group sentences into overlapping chunks."""
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self._count_tokens(sentence)

            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        metadata={
                            **base_metadata,
                            "chunk_index": chunk_index,
                            "chunk_size": current_tokens,
                        },
                        chunk_index=chunk_index,
                        token_count=current_tokens,
                    )
                )

                # Calculate overlap
                overlap_tokens = 0
                overlap_sentences = []

                # Include sentences from end for overlap
                for j in range(len(current_chunk) - 1, -1, -1):
                    sent = current_chunk[j]
                    sent_tokens = self._count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break

                # Start new chunk with overlap
                current_chunk = overlap_sentences
                current_tokens = overlap_tokens
                chunk_index += 1
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
                i += 1

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_index,
                        "chunk_size": current_tokens,
                    },
                    chunk_index=chunk_index,
                    token_count=current_tokens,
                )
            )

        return chunks

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.token_counter:
            return self.token_counter.count_tokens(text)
        else:
            # Fallback: estimate 4 chars per token
            return len(text) // 4


class SemanticChunker:
    """
    Paragraph/section-based chunker.

    Splits on paragraph boundaries and section headings,
    preserving semantic structure.
    """

    # Heading patterns (Markdown)
    HEADING_PATTERN = re.compile(r"^#+\s+.+$", re.MULTILINE)

    def __init__(self, max_chunk_size: int = 512, model: str = "gpt-3.5-turbo"):
        """
        Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum tokens per chunk
            model: Model for token counting
        """
        self.max_chunk_size = max_chunk_size

        try:
            self.token_counter = TokenCounter(model)
        except ImportError:
            self.token_counter = None

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document on semantic boundaries."""
        text = document.content

        # Split on headings and paragraphs
        sections = self._split_sections(text)

        # Create chunks from sections
        chunks = []
        chunk_index = 0

        for section in sections:
            section_tokens = self._count_tokens(section)

            if section_tokens <= self.max_chunk_size:
                # Section fits in one chunk
                chunks.append(
                    Chunk(
                        text=section,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunk_size": section_tokens,
                        },
                        chunk_index=chunk_index,
                        token_count=section_tokens,
                    )
                )
                chunk_index += 1
            else:
                # Section too large, use recursive chunking
                recursive_chunker = RecursiveChunker(
                    chunk_size=self.max_chunk_size, overlap=50
                )
                section_doc = Document(
                    content=section, metadata=document.metadata, source=document.source
                )
                section_chunks = recursive_chunker.chunk(section_doc)

                for chunk in section_chunks:
                    chunk.chunk_index = chunk_index
                    chunk.metadata["chunk_index"] = chunk_index
                    chunks.append(chunk)
                    chunk_index += 1

        return chunks

    def _split_sections(self, text: str) -> List[str]:
        """Split text on headings and double newlines."""
        # Split on double newlines first (paragraphs)
        paragraphs = text.split("\n\n")
        sections = [p.strip() for p in paragraphs if p.strip()]

        return sections

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.token_counter:
            return self.token_counter.count_tokens(text)
        else:
            return len(text) // 4


class FixedSizeChunker:
    """
    Simple fixed-size chunker.

    Splits text into fixed-size chunks without respecting
    sentence or paragraph boundaries.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Tokens per chunk
            overlap: Token overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document into fixed-size chunks."""
        text = document.content

        # Simple character-based chunking (4 chars ≈ 1 token)
        chars_per_chunk = self.chunk_size * 4
        chars_overlap = self.overlap * 4

        chunks = []
        chunk_index = 0
        start = 0

        while start < len(text):
            end = start + chars_per_chunk
            chunk_text = text[start:end]

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "end_char": end,
                    },
                    chunk_index=chunk_index,
                    token_count=self.chunk_size,
                )
            )

            chunk_index += 1
            start = end - chars_overlap

        return chunks
