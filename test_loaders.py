#!/usr/bin/env python3
"""Test document loaders - validates PDF, Markdown, and Code loaders."""

from dartboard.ingestion.loaders import (
    PDFLoader,
    MarkdownLoader,
    CodeRepositoryLoader,
    DirectoryLoader,
)


def test_markdown():
    """Test Markdown loader on existing README."""
    print("📄 Testing Markdown Loader...")
    loader = MarkdownLoader()

    try:
        docs = loader.load("README_DARTBOARD.md")
        assert len(docs) == 1
        doc = docs[0]

        print(f"  ✓ Loaded: {doc.metadata['file_name']}")
        print(f"  ✓ Length: {len(doc.content):,} chars")
        print(f"  ✓ Metadata keys: {list(doc.metadata.keys())}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_code_repository():
    """Test code repository loader on dartboard package."""
    print("🔧 Testing Code Repository Loader...")
    loader = CodeRepositoryLoader(file_extensions=[".py"])

    try:
        docs = loader.load("./dartboard")
        print(f"  ✓ Loaded {len(docs)} Python files")

        if docs:
            example = docs[0]
            print(f"  ✓ Example: {example.metadata['relative_path']}")
            print(f"  ✓ Language: {example.metadata['language']}")
            print(f"  ✓ Content preview: {example.content[:100]}...")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_directory_loader():
    """Test directory loader on current directory."""
    print("📁 Testing Directory Loader...")
    loader = DirectoryLoader()

    try:
        docs = loader.load(".")
        print(f"  ✓ Loaded {len(docs)} documents")

        # Count by type
        pdf_count = sum(1 for d in docs if d.metadata.get("source_type") == "pdf")
        md_count = sum(1 for d in docs if d.metadata.get("source_type") == "markdown")

        print(f"  ✓ PDFs: {pdf_count}")
        print(f"  ✓ Markdown: {md_count}")
        print()
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_pdf_error_handling():
    """Test PDF loader error handling."""
    print("📕 Testing PDF Error Handling...")
    loader = PDFLoader()

    try:
        # Try to load a Python file as PDF (should fail gracefully)
        docs = loader.load("test_loaders.py")
        print(f"  ✗ Should have raised error")
        return False
    except Exception as e:
        print(f"  ✓ Correctly raised error: {type(e).__name__}")
        print()
        return True


def main():
    """Run all loader tests."""
    print("=" * 60)
    print("🧪 Document Loader Tests")
    print("=" * 60)
    print()

    results = {
        "Markdown Loader": test_markdown(),
        "Code Repository Loader": test_code_repository(),
        "Directory Loader": test_directory_loader(),
        "PDF Error Handling": test_pdf_error_handling(),
    }

    print("=" * 60)
    print("📊 Test Results")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")

    print()
    total = len(results)
    passed = sum(results.values())
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
