"""
Document loaders for Dartboard RAG system.

Focused on essential use cases:
- PDF documents
- Markdown files
- Code repositories (GitHub, GitLab)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Document:
    """Container for loaded document with metadata."""

    def __init__(
        self,
        content: str,
        metadata: Dict[str, Any],
        source: str,
    ):
        """
        Initialize document.

        Args:
            content: Full text content
            metadata: Document metadata (title, author, date, etc.)
            source: Source identifier (file path, URL, etc.)
        """
        self.content = content
        self.metadata = metadata
        self.source = source

    def __repr__(self) -> str:
        return f"Document(source={self.source}, length={len(self.content)})"


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load documents from source.

        Args:
            source: Path or identifier for documents

        Returns:
            List of Document objects
        """
        pass


class PDFLoader(DocumentLoader):
    """Load and parse PDF documents."""

    def __init__(self, extract_images: bool = False):
        """
        Initialize PDF loader.

        Args:
            extract_images: Whether to extract images (not implemented yet)
        """
        self.extract_images = extract_images

    def load(self, source: str) -> List[Document]:
        """
        Load PDF file.

        Args:
            source: Path to PDF file

        Returns:
            List containing single Document
        """
        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf required for PDF loading. Install: pip install pypdf"
            )

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {source}")

        # Read PDF
        with open(path, "rb") as f:
            pdf_reader = pypdf.PdfReader(f)

            # Extract text from all pages
            text_parts = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            # Extract metadata
            metadata = {
                "source_type": "pdf",
                "file_name": path.name,
                "file_path": str(path.absolute()),
                "num_pages": len(pdf_reader.pages),
            }

            # Add PDF metadata if available
            if pdf_reader.metadata:
                pdf_meta = pdf_reader.metadata
                if pdf_meta.title:
                    metadata["title"] = pdf_meta.title
                if pdf_meta.author:
                    metadata["author"] = pdf_meta.author
                if pdf_meta.creator:
                    metadata["creator"] = pdf_meta.creator
                if pdf_meta.subject:
                    metadata["subject"] = pdf_meta.subject

            return [Document(content=full_text, metadata=metadata, source=source)]


class MarkdownLoader(DocumentLoader):
    """Load Markdown documents."""

    def __init__(self, extract_code_blocks: bool = True):
        """
        Initialize Markdown loader.

        Args:
            extract_code_blocks: Whether to preserve code blocks separately
        """
        self.extract_code_blocks = extract_code_blocks

    def load(self, source: str) -> List[Document]:
        """
        Load Markdown file.

        Args:
            source: Path to Markdown file

        Returns:
            List containing single Document
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Markdown file not found: {source}")

        # Read file
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata from frontmatter (YAML or TOML)
        metadata = self._extract_frontmatter(content)

        # Add file metadata
        metadata.update(
            {
                "source_type": "markdown",
                "file_name": path.name,
                "file_path": str(path.absolute()),
            }
        )

        # Extract code blocks if requested
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                metadata["code_blocks"] = code_blocks

        return [Document(content=content, metadata=metadata, source=source)]

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from Markdown."""
        metadata = {}

        # Check for YAML frontmatter (---)
        if content.startswith("---\n"):
            try:
                import yaml

                end = content.find("\n---\n", 4)
                if end != -1:
                    frontmatter = content[4:end]
                    metadata = yaml.safe_load(frontmatter) or {}
            except ImportError:
                logger.warning("pyyaml not installed, skipping frontmatter extraction")
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter: {e}")

        return metadata

    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from Markdown."""
        code_blocks = []
        import re

        # Match ```language\ncode\n```
        pattern = r"```(\w+)?\n(.*?)```"
        matches = re.finditer(pattern, content, re.DOTALL)

        for match in matches:
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({"language": language, "code": code})

        return code_blocks


class CodeRepositoryLoader(DocumentLoader):
    """Load code files from a repository."""

    def __init__(
        self,
        file_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize code repository loader.

        Args:
            file_extensions: List of extensions to include (e.g., ['.py', '.js'])
            exclude_patterns: Patterns to exclude (e.g., ['node_modules', '__pycache__'])
        """
        self.file_extensions = file_extensions or [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".rs",
            ".go",
            ".rb",
            ".php",
            ".md",
        ]
        self.exclude_patterns = exclude_patterns or [
            "node_modules",
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
        ]

    def load(self, source: str) -> List[Document]:
        """
        Load code files from repository.

        Args:
            source: Path to repository root

        Returns:
            List of Document objects (one per file)
        """
        repo_path = Path(source)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository not found: {source}")

        documents = []

        # Walk directory tree
        for file_path in repo_path.rglob("*"):
            # Skip if not a file
            if not file_path.is_file():
                continue

            # Skip if excluded
            if self._should_exclude(file_path, repo_path):
                continue

            # Skip if wrong extension
            if file_path.suffix not in self.file_extensions:
                continue

            # Load file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Create metadata
                relative_path = file_path.relative_to(repo_path)
                metadata = {
                    "source_type": "code",
                    "file_name": file_path.name,
                    "file_path": str(file_path.absolute()),
                    "relative_path": str(relative_path),
                    "language": self._detect_language(file_path.suffix),
                    "extension": file_path.suffix,
                    "repository": str(repo_path.absolute()),
                }

                documents.append(
                    Document(
                        content=content,
                        metadata=metadata,
                        source=str(file_path.absolute()),
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue

        logger.info(f"Loaded {len(documents)} files from {source}")
        return documents

    def _should_exclude(self, file_path: Path, repo_root: Path) -> bool:
        """Check if file should be excluded."""
        relative_path = str(file_path.relative_to(repo_root))

        for pattern in self.exclude_patterns:
            if pattern in relative_path:
                return True

        return False

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from extension."""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".md": "markdown",
        }
        return language_map.get(extension, "unknown")


class DirectoryLoader(DocumentLoader):
    """
    Load multiple documents from a directory.

    Automatically detects file types and uses appropriate loader.
    """

    def __init__(self):
        """Initialize directory loader."""
        self.loaders = {
            ".pdf": PDFLoader(),
            ".md": MarkdownLoader(),
        }

    def load(self, source: str) -> List[Document]:
        """
        Load all supported documents from directory.

        Args:
            source: Path to directory

        Returns:
            List of Document objects
        """
        dir_path = Path(source)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {source}")

        documents = []

        # Walk directory
        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue

            # Get appropriate loader
            extension = file_path.suffix.lower()
            loader = self.loaders.get(extension)

            if loader:
                try:
                    docs = loader.load(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {source}")
        return documents
