"""
RAG generation with LLM integration.

Supports:
- OpenAI (GPT-3.5, GPT-4)
- Anthropic Claude (optional)
- Custom prompt templates
- Source citation extraction
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from dartboard.datasets.models import Chunk
from dartboard.generation.prompts import (
    DEFAULT_PROMPT,
    PromptTemplate,
    extract_citations,
    build_response_with_sources,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""

    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 500
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class GenerationResult:
    """Result of RAG generation."""

    answer: str
    sources: List[Dict[str, Any]]
    num_sources_cited: int
    total_sources_available: int
    model: str
    metadata: Dict[str, Any]


class RAGGenerator:
    """
    RAG answer generator using LLMs.

    Supports OpenAI GPT models with source citation.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[GenerationConfig] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize RAG generator.

        Args:
            api_key: OpenAI API key
            config: Generation configuration
            prompt_template: Custom prompt template
        """
        self.api_key = api_key
        self.config = config or GenerationConfig()
        self.prompt_template = prompt_template or DEFAULT_PROMPT

        # Initialize OpenAI client
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate answer from query and context chunks.

        Args:
            query: User question
            chunks: Context chunks retrieved by Dartboard
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)

        Returns:
            GenerationResult with answer and cited sources
        """
        try:
            # Build prompt
            prompt = self.prompt_template.format(query, chunks)

            # Generate response
            response = self._call_llm(
                prompt,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
            )

            # Extract answer
            answer = response["content"]

            # Extract citations
            citations = extract_citations(answer)

            # Build structured response
            result_dict = build_response_with_sources(answer, chunks, citations)

            return GenerationResult(
                answer=result_dict["answer"],
                sources=result_dict["sources"],
                num_sources_cited=result_dict["num_sources_cited"],
                total_sources_available=result_dict["total_sources_available"],
                model=self.config.model,
                metadata={
                    "query": query,
                    "temperature": temperature or self.config.temperature,
                    "prompt_length": len(prompt),
                    "response_length": len(answer),
                },
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            raise

    def generate_batch(
        self, queries: List[str], chunks_list: List[List[Chunk]]
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of user questions
            chunks_list: List of chunk lists (one per query)

        Returns:
            List of GenerationResult
        """
        if len(queries) != len(chunks_list):
            raise ValueError("Number of queries must match number of chunk lists")

        results = []
        for query, chunks in zip(queries, chunks_list):
            try:
                result = self.generate(query, chunks)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate for query '{query}': {e}")
                # Return error result
                results.append(
                    GenerationResult(
                        answer=f"Error: {str(e)}",
                        sources=[],
                        num_sources_cited=0,
                        total_sources_available=len(chunks),
                        model=self.config.model,
                        metadata={"error": str(e), "query": query},
                    )
                )

        return results

    def _call_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        """
        Call OpenAI API.

        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response dictionary with content
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
            )

            # Extract content
            content = response.choices[0].message.content

            return {"content": content, "raw_response": response}

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise


class ClaudeGenerator(RAGGenerator):
    """
    RAG generator using Anthropic Claude.

    Alternative to OpenAI with similar interface.
    """

    def __init__(
        self,
        api_key: str,
        config: Optional[GenerationConfig] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        """
        Initialize Claude generator.

        Args:
            api_key: Anthropic API key
            config: Generation configuration
            prompt_template: Custom prompt template
        """
        self.api_key = api_key
        self.config = config or GenerationConfig(model="claude-3-sonnet-20240229")
        self.prompt_template = prompt_template or DEFAULT_PROMPT

        # Initialize Anthropic client
        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    def _call_llm(
        self, prompt: str, temperature: float, max_tokens: int
    ) -> Dict[str, Any]:
        """
        Call Anthropic Claude API.

        Args:
            prompt: Formatted prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Response dictionary with content
        """
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract content
            content = response.content[0].text

            return {"content": content, "raw_response": response}

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise


def create_generator(
    provider: str = "openai",
    api_key: str = None,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    prompt_template: Optional[PromptTemplate] = None,
) -> RAGGenerator:
    """
    Factory function to create a RAG generator.

    Args:
        provider: LLM provider ("openai" or "claude")
        api_key: API key for the provider
        model: Model name (optional, uses defaults)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        prompt_template: Custom prompt template

    Returns:
        RAGGenerator instance
    """
    if api_key is None:
        import os

        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
        elif provider == "claude":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    # Set default models
    if model is None:
        model = "gpt-3.5-turbo" if provider == "openai" else "claude-3-sonnet-20240229"

    # Create config
    config = GenerationConfig(
        model=model, temperature=temperature, max_tokens=max_tokens
    )

    # Create generator
    if provider == "openai":
        return RAGGenerator(
            api_key=api_key, config=config, prompt_template=prompt_template
        )
    elif provider == "claude":
        return ClaudeGenerator(
            api_key=api_key, config=config, prompt_template=prompt_template
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
