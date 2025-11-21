"""
Prompt templates for RAG generation.

Provides structured prompts for question answering with source citations.
"""

from typing import List
from dartboard.datasets.models import Chunk


class PromptTemplate:
    """Base class for prompt templates."""

    def format(self, query: str, chunks: List[Chunk]) -> str:
        """Format prompt with query and context chunks."""
        raise NotImplementedError


class RAGPromptTemplate(PromptTemplate):
    """
    Standard RAG prompt template with source citations.

    Instructs the model to:
    - Answer using only provided context
    - Cite sources with [Source N] notation
    - Admit when information is not available
    """

    TEMPLATE = """You are a helpful AI assistant. Answer the question using ONLY the information provided in the context below. Do not use any external knowledge.

If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

When answering, cite your sources using [Source N] notation where N is the source number.

Context:
{context}

Question: {query}

Answer:"""

    def format(self, query: str, chunks: List[Chunk]) -> str:
        """Format prompt with query and chunks."""
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_text = f"[Source {i+1}]: {chunk.text}"
            context_parts.append(source_text)

        context = "\n\n".join(context_parts)

        # Format template
        return self.TEMPLATE.format(context=context, query=query)


class ConversationalRAGPrompt(PromptTemplate):
    """
    Conversational RAG prompt with chat history.

    Supports multi-turn conversations while maintaining
    source citation requirements.
    """

    TEMPLATE = """You are a helpful AI assistant engaged in a conversation. Answer the current question using ONLY the information provided in the context below and the conversation history.

Do not use any external knowledge. If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Always cite your sources using [Source N] notation.

Context:
{context}

Conversation History:
{history}

Current Question: {query}

Answer:"""

    def format(
        self, query: str, chunks: List[Chunk], history: List[dict] = None
    ) -> str:
        """Format prompt with query, chunks, and conversation history."""
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_text = f"[Source {i+1}]: {chunk.text}"
            context_parts.append(source_text)

        context = "\n\n".join(context_parts)

        # Build history
        if history:
            history_parts = []
            for turn in history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                history_parts.append(f"{role.capitalize()}: {content}")
            history_text = "\n".join(history_parts)
        else:
            history_text = "(No previous conversation)"

        return self.TEMPLATE.format(context=context, history=history_text, query=query)


class SummarizationPrompt(PromptTemplate):
    """Prompt for summarizing retrieved documents."""

    TEMPLATE = """Provide a concise summary of the following documents. Focus on the main points and key information.

Documents:
{context}

Summary:"""

    def format(self, query: str, chunks: List[Chunk]) -> str:
        """Format prompt for summarization."""
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"Document {i+1}:\n{chunk.text}")

        context = "\n\n".join(context_parts)

        return self.TEMPLATE.format(context=context)


class ExtractiveQAPrompt(PromptTemplate):
    """
    Extractive QA prompt.

    Requests exact quotes from the context as answers.
    """

    TEMPLATE = """Answer the question by extracting the relevant text directly from the context below. Your answer should be a direct quote from the context.

If no relevant text can be found, respond with "No relevant information found."

Include the source number [Source N] where you found the answer.

Context:
{context}

Question: {query}

Extracted Answer:"""

    def format(self, query: str, chunks: List[Chunk]) -> str:
        """Format extractive QA prompt."""
        context_parts = []
        for i, chunk in enumerate(chunks):
            source_text = f"[Source {i+1}]: {chunk.text}"
            context_parts.append(source_text)

        context = "\n\n".join(context_parts)

        return self.TEMPLATE.format(context=context, query=query)


# Default prompt template
DEFAULT_PROMPT = RAGPromptTemplate()


def format_sources(chunks: List[Chunk]) -> str:
    """
    Format chunks as numbered sources for display.

    Args:
        chunks: List of chunks to format

    Returns:
        Formatted source list
    """
    sources = []
    for i, chunk in enumerate(chunks):
        # Extract metadata
        source_file = chunk.metadata.get("source", "Unknown")
        chunk_idx = chunk.metadata.get("chunk_index", i)

        # Format source
        source_text = (
            f"[Source {i+1}] (from {source_file}, chunk {chunk_idx}):\n{chunk.text}"
        )
        sources.append(source_text)

    return "\n\n".join(sources)


def extract_citations(response_text: str) -> List[int]:
    """
    Extract source citation numbers from response.

    Looks for [Source N] patterns in the text.

    Args:
        response_text: Generated response text

    Returns:
        List of cited source numbers (1-indexed)
    """
    import re

    # Pattern for [Source N]
    pattern = r"\[Source\s+(\d+)\]"
    matches = re.findall(pattern, response_text, re.IGNORECASE)

    # Convert to integers and deduplicate
    citations = sorted(set(int(m) for m in matches))

    return citations


def build_response_with_sources(
    answer: str, chunks: List[Chunk], citations: List[int]
) -> dict:
    """
    Build structured response with answer and cited sources.

    Args:
        answer: Generated answer text
        chunks: All context chunks
        citations: List of cited source numbers (1-indexed)

    Returns:
        Dictionary with answer, sources, and metadata
    """
    # Get cited chunks
    cited_chunks = []
    for citation_num in citations:
        # Convert to 0-indexed
        idx = citation_num - 1
        if 0 <= idx < len(chunks):
            cited_chunks.append(chunks[idx])

    # Format sources
    sources = []
    for i, chunk in enumerate(cited_chunks):
        source_info = {
            "number": citations[i] if i < len(citations) else i + 1,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        sources.append(source_info)

    return {
        "answer": answer,
        "sources": sources,
        "num_sources_cited": len(citations),
        "total_sources_available": len(chunks),
    }
