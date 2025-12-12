# Answer Generation with LLMs

## Overview

The generation layer converts retrieved context chunks into natural language answers using Large Language Models (LLMs). It handles prompt construction, LLM API calls, and source citation extraction.

**Key Components:**
- **RAG Generator**: Integrates with OpenAI GPT or Anthropic Claude
- **Prompt Templates**: Structured prompts for different use cases
- **Source Citation**: Automatic extraction of cited sources
- **Multi-LLM Support**: Pluggable architecture for different providers

**Workflow:**
```
Query + Retrieved Chunks → Format Prompt → Call LLM → Extract Citations → Return Answer
```

## Quick Start

### Basic RAG Generation

```python
from dartboard.generation import create_generator
from dartboard.retrieval.dense import DenseRetriever
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Setup retrieval
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(embedding_dim=384, persist_path="./data/store")
retriever = DenseRetriever(embedding_model, vector_store)

# Create generator
generator = create_generator(
    provider="openai",
    api_key="sk-...",
    model="gpt-3.5-turbo",
    temperature=0.7
)

# Full RAG pipeline
query = "What is the attention mechanism?"
chunks = retriever.retrieve(query, k=5)
result = generator.generate(query, chunks)

print(f"Answer: {result.answer}")
print(f"Cited {result.num_sources_cited} of {result.total_sources_available} sources")

for source in result.sources:
    print(f"\n[Source {source['number']}]")
    print(f"Text: {source['text'][:100]}...")
```

### With Environment Variables

```python
import os

# Set API key via environment
os.environ["OPENAI_API_KEY"] = "sk-..."

# Generator automatically uses environment variable
generator = create_generator(provider="openai")

result = generator.generate(query, chunks)
```

## RAG Generator

### Initialization

```python
from dartboard.generation import RAGGenerator, GenerationConfig

# Simple initialization
generator = RAGGenerator(api_key="sk-...")

# With custom configuration
config = GenerationConfig(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

generator = RAGGenerator(api_key="sk-...", config=config)

# With custom prompt template
from dartboard.generation.prompts import ExtractiveQAPrompt

generator = RAGGenerator(
    api_key="sk-...",
    config=config,
    prompt_template=ExtractiveQAPrompt()
)
```

**Parameters:**
- `api_key` (str): OpenAI API key
- `config` (GenerationConfig): LLM generation parameters
- `prompt_template` (PromptTemplate): Custom prompt template

### Generation Configuration

```python
from dartboard.generation import GenerationConfig

config = GenerationConfig(
    model="gpt-3.5-turbo",    # or "gpt-4", "gpt-4-turbo-preview"
    temperature=0.7,           # 0.0 = deterministic, 1.0 = creative
    max_tokens=500,            # Maximum answer length
    top_p=1.0,                 # Nucleus sampling
    frequency_penalty=0.0,     # Reduce repetition
    presence_penalty=0.0       # Encourage topic diversity
)
```

**Model Options:**
- `gpt-3.5-turbo`: Fast, cost-effective, good quality
- `gpt-4`: Highest quality, slower, more expensive
- `gpt-4-turbo-preview`: Balance of speed and quality

**Temperature Guidelines:**
- `0.0-0.3`: Factual Q&A, consistency important
- `0.4-0.7`: General RAG, balanced creativity
- `0.8-1.0`: Creative writing, brainstorming

### Generating Answers

```python
# Single query
result = generator.generate(
    query="What is machine learning?",
    chunks=retrieved_chunks,
    temperature=0.5,      # Optional override
    max_tokens=300        # Optional override
)

# Access result
print(result.answer)
print(result.sources)
print(result.num_sources_cited)
print(result.metadata)
```

**Returns `GenerationResult`:**
```python
@dataclass
class GenerationResult:
    answer: str                      # Generated answer text
    sources: List[Dict[str, Any]]    # Cited sources
    num_sources_cited: int           # Number of citations
    total_sources_available: int     # Total chunks provided
    model: str                       # Model used
    metadata: Dict[str, Any]         # Query, temperature, lengths
```

### Batch Generation

```python
# Generate answers for multiple queries
queries = [
    "What is deep learning?",
    "How does backpropagation work?",
    "What are transformers?"
]

chunks_list = [
    retriever.retrieve(q, k=5) for q in queries
]

results = generator.generate_batch(queries, chunks_list)

for query, result in zip(queries, results):
    print(f"\nQ: {query}")
    print(f"A: {result.answer}")
```

## LLM Providers

### OpenAI

```python
from dartboard.generation import RAGGenerator, GenerationConfig

# Initialize
generator = RAGGenerator(
    api_key="sk-...",
    config=GenerationConfig(model="gpt-3.5-turbo")
)

# Or use factory
from dartboard.generation import create_generator

generator = create_generator(
    provider="openai",
    api_key="sk-...",
    model="gpt-3.5-turbo",
    temperature=0.7
)
```

**Requirements:**
```bash
pip install openai
```

**Environment Variable:**
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic Claude

```python
from dartboard.generation import ClaudeGenerator, GenerationConfig

# Initialize
generator = ClaudeGenerator(
    api_key="sk-ant-...",
    config=GenerationConfig(model="claude-3-sonnet-20240229")
)

# Or use factory
generator = create_generator(
    provider="claude",
    api_key="sk-ant-...",
    model="claude-3-sonnet-20240229",
    temperature=0.7
)
```

**Requirements:**
```bash
pip install anthropic
```

**Environment Variable:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Claude Models:**
- `claude-3-sonnet-20240229`: Fast, cost-effective
- `claude-3-opus-20240229`: Highest quality
- `claude-3-haiku-20240307`: Fastest, cheapest

### Factory Function

```python
from dartboard.generation import create_generator

# Auto-detects API key from environment
generator = create_generator(
    provider="openai",  # or "claude"
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500
)
```

**Benefits:**
- Automatically reads API key from environment
- Consistent interface across providers
- Default model selection

## Prompt Templates

### Overview

Prompt templates control how the LLM is instructed to use the retrieved context.

**Available Templates:**
- `RAGPromptTemplate`: Standard RAG with source citations
- `ConversationalRAGPrompt`: Multi-turn conversations
- `ExtractiveQAPrompt`: Extract exact quotes
- `SummarizationPrompt`: Summarize documents

### RAGPromptTemplate (Default)

Standard RAG prompt with source citation requirements.

```python
from dartboard.generation.prompts import RAGPromptTemplate

template = RAGPromptTemplate()
prompt = template.format(query, chunks)
```

**Template Structure:**
```
You are a helpful AI assistant. Answer the question using ONLY
the information provided in the context below.

If the answer cannot be found in the context, respond with
"I don't have enough information to answer this question."

When answering, cite your sources using [Source N] notation.

Context:
[Source 1]: {chunk_1_text}
[Source 2]: {chunk_2_text}
...

Question: {query}

Answer:
```

**Example Output:**
```
The attention mechanism allows models to focus on relevant parts
of the input [Source 1]. It computes weights for each input element
based on the query [Source 2], enabling better context understanding.
```

### ConversationalRAGPrompt

Support multi-turn conversations with chat history.

```python
from dartboard.generation.prompts import ConversationalRAGPrompt

template = ConversationalRAGPrompt()

# With conversation history
history = [
    {"role": "user", "content": "What is attention?"},
    {"role": "assistant", "content": "Attention is a mechanism..."},
    {"role": "user", "content": "How is it used in transformers?"}
]

prompt = template.format(
    query="How is it used in transformers?",
    chunks=chunks,
    history=history
)
```

**Use Cases:**
- Chatbots with context
- Follow-up questions
- Clarification requests
- Multi-turn dialogues

### ExtractiveQAPrompt

Request exact quotes from context as answers.

```python
from dartboard.generation.prompts import ExtractiveQAPrompt

template = ExtractiveQAPrompt()
prompt = template.format(query, chunks)
```

**Instruction:**
```
Answer the question by extracting the relevant text directly
from the context below. Your answer should be a direct quote
from the context.

Include the source number [Source N] where you found the answer.
```

**Use Cases:**
- Factoid questions
- Quote extraction
- Verbatim answers
- Legal/compliance requirements

### SummarizationPrompt

Summarize retrieved documents.

```python
from dartboard.generation.prompts import SummarizationPrompt

template = SummarizationPrompt()
prompt = template.format(query="", chunks=chunks)
```

**Use Cases:**
- Document summaries
- Topic overviews
- Information aggregation
- Report generation

### Custom Prompt Templates

```python
from dartboard.generation.prompts import PromptTemplate
from typing import List
from dartboard.datasets.models import Chunk

class CustomPrompt(PromptTemplate):
    """Custom prompt for domain-specific RAG."""

    TEMPLATE = """You are an expert in {domain}.

Answer the question using the provided context.
Be technical and precise. Cite sources with [Source N].

Context:
{context}

Question: {query}

Technical Answer:"""

    def format(self, query: str, chunks: List[Chunk], domain: str = "AI") -> str:
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}]: {chunk.text}")

        context = "\n\n".join(context_parts)

        return self.TEMPLATE.format(
            domain=domain,
            context=context,
            query=query
        )

# Use custom template
custom_prompt = CustomPrompt()
generator = RAGGenerator(api_key=api_key, prompt_template=custom_prompt)
```

## Source Citation

### Citation Extraction

Automatically extract `[Source N]` citations from LLM responses.

```python
from dartboard.generation.prompts import extract_citations

answer = """
The attention mechanism [Source 1] allows models to focus on
relevant parts [Source 3] of the input sequence [Source 1].
"""

citations = extract_citations(answer)
print(citations)  # [1, 3]
```

**Pattern:**
- Looks for `[Source N]` where N is a number
- Returns sorted, deduplicated list of source numbers
- Case-insensitive matching

### Building Response with Sources

```python
from dartboard.generation.prompts import build_response_with_sources

answer = "The answer is in [Source 1] and [Source 3]."
chunks = [chunk1, chunk2, chunk3, chunk4, chunk5]
citations = [1, 3]

response = build_response_with_sources(answer, chunks, citations)

print(response["answer"])
print(response["sources"])  # Only chunks 1 and 3
print(response["num_sources_cited"])  # 2
print(response["total_sources_available"])  # 5
```

**Returns:**
```python
{
    "answer": "...",
    "sources": [
        {
            "number": 1,
            "text": "...",
            "metadata": {...}
        },
        {
            "number": 3,
            "text": "...",
            "metadata": {...}
        }
    ],
    "num_sources_cited": 2,
    "total_sources_available": 5
}
```

### Formatting Sources for Display

```python
from dartboard.generation.prompts import format_sources

formatted = format_sources(chunks)
print(formatted)
```

**Output:**
```
[Source 1] (from paper.pdf, chunk 0):
The attention mechanism allows neural networks to focus...

[Source 2] (from paper.pdf, chunk 1):
Transformers use multi-head attention to process...
```

## Complete Examples

### Example 1: Basic RAG System

```python
from dartboard.generation import create_generator
from dartboard.retrieval.dense import DenseRetriever
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore

# Setup
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(embedding_dim=384, persist_path="./data/kb")
retriever = DenseRetriever(embedding_model, vector_store)
generator = create_generator(provider="openai")

# RAG pipeline
def answer_question(query: str) -> str:
    # Retrieve
    chunks = retriever.retrieve(query, k=5)

    # Generate
    result = generator.generate(query, chunks)

    # Format response
    response = f"{result.answer}\n\n"
    response += "Sources:\n"
    for source in result.sources:
        response += f"  [Source {source['number']}]: {source['metadata'].get('source', 'Unknown')}\n"

    return response

# Use it
answer = answer_question("What is deep learning?")
print(answer)
```

### Example 2: Conversational RAG

```python
from dartboard.generation import create_generator
from dartboard.generation.prompts import ConversationalRAGPrompt

# Setup with conversational template
generator = create_generator(
    provider="openai",
    prompt_template=ConversationalRAGPrompt()
)

# Conversation state
conversation_history = []

def chat(user_message: str, retriever) -> str:
    # Retrieve relevant chunks
    chunks = retriever.retrieve(user_message, k=5)

    # Generate with history
    result = generator.generate(
        query=user_message,
        chunks=chunks,
        history=conversation_history
    )

    # Update history
    conversation_history.append({
        "role": "user",
        "content": user_message
    })
    conversation_history.append({
        "role": "assistant",
        "content": result.answer
    })

    return result.answer

# Multi-turn conversation
print(chat("What is attention?", retriever))
print(chat("How is it used in transformers?", retriever))
print(chat("Can you explain self-attention?", retriever))
```

### Example 3: Claude with Custom Temperature

```python
from dartboard.generation import create_generator, GenerationConfig

# Conservative temperature for factual answers
config = GenerationConfig(
    model="claude-3-sonnet-20240229",
    temperature=0.2,  # More deterministic
    max_tokens=300
)

generator = create_generator(
    provider="claude",
    api_key="sk-ant-...",
    model=config.model,
    temperature=config.temperature,
    max_tokens=config.max_tokens
)

# Generate
result = generator.generate(query, chunks)
```

### Example 4: Extractive QA

```python
from dartboard.generation import create_generator
from dartboard.generation.prompts import ExtractiveQAPrompt

# Use extractive prompt
generator = create_generator(
    provider="openai",
    prompt_template=ExtractiveQAPrompt()
)

# Get exact quotes
query = "Who invented the transformer architecture?"
chunks = retriever.retrieve(query, k=5)
result = generator.generate(query, chunks)

# Result will be a direct quote
print(result.answer)
# "The transformer architecture was invented by Vaswani et al. [Source 1]"
```

### Example 5: Document Summarization

```python
from dartboard.generation import create_generator
from dartboard.generation.prompts import SummarizationPrompt

# Setup for summarization
generator = create_generator(
    provider="openai",
    model="gpt-4",  # Better for summarization
    prompt_template=SummarizationPrompt()
)

# Retrieve all chunks from a document
chunks = vector_store.search(
    query_embedding=embedding_model.encode("machine learning overview"),
    k=20  # Get more chunks for comprehensive summary
)

# Generate summary (query is ignored for summarization)
result = generator.generate(query="", chunks=chunks, max_tokens=800)

print("Summary:")
print(result.answer)
```

## Configuration Best Practices

### Temperature Selection

```python
# Factual Q&A (prioritize accuracy)
config = GenerationConfig(temperature=0.1)

# General RAG (balanced)
config = GenerationConfig(temperature=0.5)

# Creative tasks (more variation)
config = GenerationConfig(temperature=0.8)
```

### Max Tokens

```python
# Short answers (tweets, captions)
config = GenerationConfig(max_tokens=100)

# Standard answers (Q&A)
config = GenerationConfig(max_tokens=300)

# Long-form (explanations, summaries)
config = GenerationConfig(max_tokens=800)

# Very long (reports, articles)
config = GenerationConfig(max_tokens=2000)
```

### Model Selection

**For Cost Efficiency:**
```python
generator = create_generator(
    provider="openai",
    model="gpt-3.5-turbo"  # Cheapest, fastest
)
```

**For Quality:**
```python
generator = create_generator(
    provider="openai",
    model="gpt-4"  # Best quality
)
```

**For Claude:**
```python
# Fast and cheap
generator = create_generator(provider="claude", model="claude-3-haiku-20240307")

# Balanced
generator = create_generator(provider="claude", model="claude-3-sonnet-20240229")

# Highest quality
generator = create_generator(provider="claude", model="claude-3-opus-20240229")
```

## Error Handling

### API Errors

```python
from openai import OpenAIError

try:
    result = generator.generate(query, chunks)
except OpenAIError as e:
    print(f"OpenAI API error: {e}")
    # Fallback or retry logic

# For Claude
from anthropic import APIError

try:
    result = generator.generate(query, chunks)
except APIError as e:
    print(f"Anthropic API error: {e}")
```

### Missing API Keys

```python
import os

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

generator = create_generator(provider="openai")
```

### Empty Context

```python
if not chunks:
    return "I don't have any relevant information to answer this question."

result = generator.generate(query, chunks)
```

## Integration with Full RAG Pipeline

```python
from dartboard.ingestion import create_pipeline
from dartboard.ingestion.loaders import PDFLoader
from dartboard.embeddings import SentenceTransformerModel
from dartboard.storage.vector_store import FAISSStore
from dartboard.retrieval.hybrid import HybridRetriever
from dartboard.generation import create_generator

# 1. Ingestion (one-time setup)
embedding_model = SentenceTransformerModel("all-MiniLM-L6-v2")
vector_store = FAISSStore(embedding_dim=384, persist_path="./data/kb")

ingestion_pipeline = create_pipeline(
    loader=PDFLoader(),
    embedding_model=embedding_model,
    vector_store=vector_store
)

ingestion_pipeline.ingest_batch(["doc1.pdf", "doc2.pdf"])

# 2. Retrieval
retriever = HybridRetriever(
    embedding_model=embedding_model,
    vector_store=vector_store
)

# 3. Generation
generator = create_generator(provider="openai")

# 4. End-to-end RAG function
def rag_query(question: str) -> dict:
    chunks = retriever.retrieve(question, k=5)
    result = generator.generate(question, chunks)

    return {
        "question": question,
        "answer": result.answer,
        "sources": result.sources,
        "model": result.model
    }

# Use it
response = rag_query("What is the attention mechanism?")
print(response["answer"])
```

## Performance Optimization

### Caching Generators

```python
# Don't create generator for each query
# ❌ Bad
def answer(query):
    generator = create_generator(provider="openai")  # Slow!
    return generator.generate(query, chunks)

# ✅ Good
generator = create_generator(provider="openai")  # Once

def answer(query):
    return generator.generate(query, chunks)
```

### Batch Processing

```python
# Process multiple queries efficiently
queries = ["Q1", "Q2", "Q3"]
chunks_list = [retriever.retrieve(q, k=5) for q in queries]

# Single batch call
results = generator.generate_batch(queries, chunks_list)
```

### Streaming (Future Enhancement)

```python
# Future: streaming responses
for chunk in generator.generate_stream(query, chunks):
    print(chunk, end="", flush=True)
```

## Monitoring and Logging

```python
import logging

logging.basicConfig(level=logging.INFO)

# Generation includes metadata
result = generator.generate(query, chunks)

print(f"Model: {result.model}")
print(f"Prompt length: {result.metadata['prompt_length']} chars")
print(f"Response length: {result.metadata['response_length']} chars")
print(f"Temperature: {result.metadata['temperature']}")
print(f"Citations: {result.num_sources_cited}/{result.total_sources_available}")
```

## Troubleshooting

### Issue: No citations in answer

**Cause:** LLM didn't use `[Source N]` notation

**Solution:**
```python
# Use more explicit prompt template
# Or check if chunks contain relevant information
if result.num_sources_cited == 0:
    print("Warning: No sources cited")
    # May need better retrieval or more specific prompt
```

### Issue: API rate limits

**Solution:**
```python
import time
from openai import RateLimitError

try:
    result = generator.generate(query, chunks)
except RateLimitError:
    time.sleep(60)  # Wait before retry
    result = generator.generate(query, chunks)
```

### Issue: Hallucinations (answers not from context)

**Solution:**
- Use lower temperature (0.1-0.3)
- Use `ExtractiveQAPrompt` for strict grounding
- Validate answer against chunks programmatically

```python
# Strict configuration
config = GenerationConfig(
    temperature=0.1,
    presence_penalty=0.5  # Discourage new topics
)

generator = RAGGenerator(
    api_key=api_key,
    config=config,
    prompt_template=ExtractiveQAPrompt()  # Force extraction
)
```

## See Also

- [Retrieval Methods](./retrieval-methods.md) - Getting relevant chunks
- [Ingestion Pipeline](./ingestion-pipeline.md) - Populating the knowledge base
- [API Guide](./api-guide.md) - Using generation via API
- [Evaluation System](./evaluation-system.md) - Measuring generation quality
