"""
Tests for LLM generation and RAG answer generation.

Tests:
- Prompt template formatting
- Source citation extraction
- RAG generation with mock LLM
- Error handling
- Batch generation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from dartboard.generation.generator import (
    RAGGenerator,
    ClaudeGenerator,
    GenerationConfig,
    GenerationResult,
    create_generator,
)
from dartboard.generation.prompts import (
    RAGPromptTemplate,
    ConversationalRAGPrompt,
    SummarizationPrompt,
    ExtractiveQAPrompt,
    extract_citations,
    build_response_with_sources,
    format_sources,
)
from dartboard.datasets.models import Chunk


# Sample chunks for testing
SAMPLE_CHUNKS = [
    Chunk(
        id="chunk1",
        text="Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={"source": "ml_intro.pdf", "chunk_index": 0},
    ),
    Chunk(
        id="chunk2",
        text="Deep learning uses neural networks with multiple layers to learn complex patterns.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={"source": "ml_intro.pdf", "chunk_index": 1},
    ),
    Chunk(
        id="chunk3",
        text="Natural language processing (NLP) is a branch of AI that deals with human language.",
        embedding=np.random.rand(384).astype(np.float32),
        metadata={"source": "nlp_guide.pdf", "chunk_index": 0},
    ),
]


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_rag_prompt_template(self):
        """Test basic RAG prompt template."""
        template = RAGPromptTemplate()
        query = "What is machine learning?"
        chunks = SAMPLE_CHUNKS[:2]

        prompt = template.format(query, chunks)

        # Should contain query
        assert "What is machine learning?" in prompt

        # Should contain sources
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt

        # Should contain chunk text
        assert "Machine learning" in prompt
        assert "Deep learning" in prompt

    def test_conversational_prompt(self):
        """Test conversational RAG prompt."""
        template = ConversationalRAGPrompt()
        query = "Tell me more"
        chunks = SAMPLE_CHUNKS[:1]
        history = [
            {"role": "user", "content": "What is ML?"},
            {"role": "assistant", "content": "ML is machine learning..."},
        ]

        prompt = template.format(query, chunks, history=history)

        # Should contain history
        assert "What is ML?" in prompt
        assert "ML is machine learning" in prompt

        # Should contain current query
        assert "Tell me more" in prompt

    def test_summarization_prompt(self):
        """Test summarization prompt."""
        template = SummarizationPrompt()
        chunks = SAMPLE_CHUNKS

        prompt = template.format("", chunks)

        # Should contain all document texts
        assert "Machine learning" in prompt
        assert "Deep learning" in prompt
        assert "Natural language processing" in prompt

    def test_extractive_qa_prompt(self):
        """Test extractive QA prompt."""
        template = ExtractiveQAPrompt()
        query = "What is deep learning?"
        chunks = SAMPLE_CHUNKS

        prompt = template.format(query, chunks)

        # Should request extraction
        assert "extract" in prompt.lower()
        assert query in prompt


class TestCitationExtraction:
    """Tests for citation extraction."""

    def test_extract_single_citation(self):
        """Test extracting single citation."""
        text = "Machine learning is important [Source 1]."
        citations = extract_citations(text)

        assert citations == [1]

    def test_extract_multiple_citations(self):
        """Test extracting multiple citations."""
        text = "ML [Source 1] and DL [Source 2] are related [Source 1]."
        citations = extract_citations(text)

        # Should deduplicate
        assert citations == [1, 2]

    def test_extract_no_citations(self):
        """Test text with no citations."""
        text = "This is just plain text."
        citations = extract_citations(text)

        assert citations == []

    def test_citation_case_insensitive(self):
        """Test case-insensitive citation extraction."""
        text = "Content [source 1] and [SOURCE 2]."
        citations = extract_citations(text)

        assert citations == [1, 2]

    def test_build_response_with_sources(self):
        """Test building structured response."""
        answer = "ML is important [Source 1]. DL is powerful [Source 2]."
        chunks = SAMPLE_CHUNKS[:2]
        citations = [1, 2]

        result = build_response_with_sources(answer, chunks, citations)

        assert result["answer"] == answer
        assert len(result["sources"]) == 2
        assert result["num_sources_cited"] == 2
        assert result["total_sources_available"] == 2

    def test_format_sources(self):
        """Test source formatting."""
        chunks = SAMPLE_CHUNKS[:2]
        formatted = format_sources(chunks)

        # Should contain source numbers
        assert "[Source 1]" in formatted
        assert "[Source 2]" in formatted

        # Should contain file names
        assert "ml_intro.pdf" in formatted


class TestRAGGenerator:
    """Tests for RAGGenerator."""

    @patch("openai.OpenAI")
    def test_generator_initialization(self, mock_openai):
        """Test generator initialization."""
        generator = RAGGenerator(api_key="test-key")

        assert generator.api_key == "test-key"
        assert generator.config.model == "gpt-3.5-turbo"
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch("openai.OpenAI")
    def test_generate_with_mock(self, mock_openai):
        """Test generation with mocked OpenAI."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Machine learning is a subset of AI [Source 1]."
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Create generator
        generator = RAGGenerator(api_key="test-key")

        # Generate
        query = "What is machine learning?"
        chunks = SAMPLE_CHUNKS[:2]
        result = generator.generate(query, chunks)

        # Verify result
        assert isinstance(result, GenerationResult)
        assert "Machine learning" in result.answer
        assert result.num_sources_cited == 1
        assert result.total_sources_available == 2

    @patch("openai.OpenAI")
    def test_generate_with_custom_config(self, mock_openai):
        """Test generation with custom config."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Answer"))]
        mock_client.chat.completions.create.return_value = mock_response

        # Custom config
        config = GenerationConfig(model="gpt-4", temperature=0.5, max_tokens=1000)
        generator = RAGGenerator(api_key="test-key", config=config)

        result = generator.generate("Question", SAMPLE_CHUNKS)

        # Verify API was called with custom params
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 1000

    @patch("openai.OpenAI")
    def test_batch_generation(self, mock_openai):
        """Test batch generation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Answer [Source 1]"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        generator = RAGGenerator(api_key="test-key")

        queries = ["Q1", "Q2", "Q3"]
        chunks_list = [SAMPLE_CHUNKS[:1], SAMPLE_CHUNKS[:2], SAMPLE_CHUNKS]

        results = generator.generate_batch(queries, chunks_list)

        assert len(results) == 3
        assert all(isinstance(r, GenerationResult) for r in results)

    @patch("openai.OpenAI")
    def test_error_handling(self, mock_openai):
        """Test error handling in generation."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # Simulate API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        generator = RAGGenerator(api_key="test-key")

        # Should raise exception
        with pytest.raises(Exception, match="API Error"):
            generator.generate("Question", SAMPLE_CHUNKS)

    @patch("openai.OpenAI")
    def test_batch_error_recovery(self, mock_openai):
        """Test batch generation continues after error."""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [
            Exception("API Error"),
            MagicMock(
                choices=[MagicMock(message=MagicMock(content="Success [Source 1]"))]
            ),
        ]

        generator = RAGGenerator(api_key="test-key")

        queries = ["Q1", "Q2"]
        chunks_list = [SAMPLE_CHUNKS, SAMPLE_CHUNKS]

        results = generator.generate_batch(queries, chunks_list)

        # Should have 2 results, first with error
        assert len(results) == 2
        assert "Error" in results[0].answer
        assert "Success" in results[1].answer


try:
    import anthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class TestClaudeGenerator:
    """Tests for ClaudeGenerator."""

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    @patch("anthropic.Anthropic")
    def test_claude_initialization(self, mock_anthropic):
        """Test Claude generator initialization."""
        generator = ClaudeGenerator(api_key="test-key")

        assert generator.api_key == "test-key"
        assert "claude" in generator.config.model.lower()
        mock_anthropic.assert_called_once_with(api_key="test-key")

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    @patch("anthropic.Anthropic")
    def test_claude_generate(self, mock_anthropic):
        """Test generation with Claude."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude's answer [Source 1]")]
        mock_client.messages.create.return_value = mock_response

        generator = ClaudeGenerator(api_key="test-key")

        result = generator.generate("Question", SAMPLE_CHUNKS)

        assert "Claude's answer" in result.answer
        assert result.num_sources_cited == 1


class TestGeneratorFactory:
    """Tests for generator factory function."""

    @patch("openai.OpenAI")
    def test_create_openai_generator(self, mock_openai):
        """Test creating OpenAI generator."""
        generator = create_generator(
            provider="openai", api_key="test-key", model="gpt-4", temperature=0.5
        )

        assert isinstance(generator, RAGGenerator)
        assert generator.config.model == "gpt-4"
        assert generator.config.temperature == 0.5

    @pytest.mark.skipif(not ANTHROPIC_AVAILABLE, reason="anthropic not installed")
    @patch("anthropic.Anthropic")
    def test_create_claude_generator(self, mock_anthropic):
        """Test creating Claude generator."""
        generator = create_generator(
            provider="claude", api_key="test-key", temperature=0.8
        )

        assert isinstance(generator, ClaudeGenerator)
        assert generator.config.temperature == 0.8

    @patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"})
    @patch("openai.OpenAI")
    def test_create_generator_from_env(self, mock_openai):
        """Test creating generator from environment variable."""
        generator = create_generator(provider="openai")

        assert generator.api_key == "env-key"

    def test_create_generator_unknown_provider(self):
        """Test error for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            create_generator(provider="unknown", api_key="test-key")

    def test_create_generator_missing_key(self):
        """Test error when API key is missing."""
        with pytest.raises(ValueError, match="API_KEY"):
            create_generator(provider="openai")


class TestIntegration:
    """Integration tests."""

    @patch("openai.OpenAI")
    def test_end_to_end_rag(self, mock_openai):
        """Test end-to-end RAG generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="Machine learning is a subset of AI [Source 1] that uses neural networks [Source 2]."
                )
            )
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Create generator
        generator = create_generator(provider="openai", api_key="test-key")

        # Generate answer
        query = "What is machine learning and how does it work?"
        chunks = SAMPLE_CHUNKS

        result = generator.generate(query, chunks)

        # Verify complete result
        assert result.answer
        assert result.num_sources_cited == 2
        assert len(result.sources) == 2
        assert result.model == "gpt-3.5-turbo"
        assert "query" in result.metadata


def test_prompt_template_extensibility():
    """Test that custom prompt templates can be created."""

    class CustomPrompt(RAGPromptTemplate):
        TEMPLATE = "Custom template: {context}\n\nQuestion: {query}"

    template = CustomPrompt()
    prompt = template.format("Test query", SAMPLE_CHUNKS[:1])

    assert "Custom template" in prompt
    assert "Test query" in prompt


if __name__ == "__main__":
    print("Running generation tests...")
    pytest.main([__file__, "-v"])
