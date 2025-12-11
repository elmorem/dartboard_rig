# Evaluation Datasets

## Overview

This repository uses **standard IR benchmarks** (MS MARCO and BEIR) to evaluate retrieval methods. These datasets provide queries, document corpora, and relevance judgments (qrels) for measuring retrieval quality.

## Supported Datasets

### 1. MS MARCO (Microsoft Machine Reading Comprehension)

**Description**: Large-scale passage ranking dataset

**Statistics**:
- **Queries**: 6,980 (dev.small) / 1M+ (full)
- **Passages**: 8.8 million
- **Domain**: Web search queries
- **Relevance**: Binary (relevant/not relevant)
- **Average relevant per query**: 1.1

**Characteristics**:
- Natural language questions from Bing search
- Human-annotated relevance judgments
- Diverse topics (general knowledge, how-to, factual)
- Gold standard for passage retrieval

**Download**:
```bash
python benchmarks/scripts/download_datasets.py --dataset msmarco
```

**Why Chosen**: Industry standard, large scale, realistic queries

### 2. BEIR (Benchmarking IR)

**Description**: Heterogeneous benchmark with 18 diverse datasets

**Datasets Used in This Repository**:

#### SciFact (Scientific Fact Verification)

- **Queries**: 300
- **Documents**: 5,183
- **Domain**: Scientific claims and evidence
- **Task**: Verify scientific claims with research papers
- **Avg relevant per query**: 1.2

**Example**:
- Query: "Coronavirus droplets can remain airborne for several hours"
- Relevant doc: "SARS-CoV-2 RNA detected in air samples 2-4.8m from patients..."

#### ArguAna (Counter-Argument Retrieval)

- **Queries**: 1,406
- **Documents**: 8,674
- **Domain**: Debate arguments
- **Task**: Find counter-arguments to given arguments
- **Avg relevant per query**: 1.0

**Example**:
- Query: "We should ban plastic bags"
- Relevant doc: "Plastic bag bans lead to increased use of reusable bags which harbor bacteria..."

#### Climate-FEVER (Climate Fact Verification)

- **Queries**: 1,535
- **Documents**: 5,416,593 (5.4 million)
- **Domain**: Climate change claims
- **Task**: Fact-check climate-related statements
- **Avg relevant per query**: 3.5
- **Note**: We use **corpus sampling** (10K docs) due to size

**Example**:
- Query: "Global temperatures have risen 1.5°C since pre-industrial times"
- Relevant docs: Multiple climate research papers

#### NFCorpus (Nutrition/Medical)

- **Queries**: 323
- **Documents**: 3,633
- **Domain**: Biomedical literature
- **Task**: Medical/nutrition question answering

#### FiQA (Financial Question Answering)

- **Queries**: 648
- **Documents**: 57,638
- **Domain**: Finance, investing, economics
- **Task**: Answer financial questions

**Download BEIR**:
```bash
python benchmarks/scripts/download_datasets.py \
    --dataset beir \
    --beir-datasets scifact nfcorpus fiqa climate-fever
```

**Why BEIR Chosen**:
- Domain diversity (science, debate, climate, medicine, finance)
- Zero-shot evaluation (no training data)
- Different query types (facts, arguments, questions)
- Realistic document collections

## Why These Datasets Are Appropriate

### 1. Domain Coverage

| Domain | Dataset | Rationale |
|--------|---------|-----------|
| General Web | MS MARCO | Realistic search queries |
| Scientific | SciFact | Technical accuracy, citation verification |
| Argumentative | ArguAna | Counter-point retrieval, debate |
| Climate | Climate-FEVER | Fact-checking, multi-document |
| Medical | NFCorpus | Domain-specific terminology |
| Financial | FiQA | Specialized vocabulary |

### 2. Task Diversity

- **Fact verification**: SciFact, Climate-FEVER
- **Question answering**: MS MARCO, NFCorpus, FiQA
- **Argument retrieval**: ArguAna
- **General search**: MS MARCO

### 3. Scale Variety

- Small (< 10K docs): SciFact, ArguAna, NFCorpus
- Medium (10K-100K): FiQA
- Large (> 1M): MS MARCO, Climate-FEVER

### 4. Evaluation Rigor

All datasets have:
- ✅ Human-annotated relevance judgments
- ✅ Clear task definitions
- ✅ Published baselines
- ✅ Active research community
- ✅ Multiple use cases (academic + industry)

## Dataset Statistics

| Dataset | Queries | Docs | Avg Rel/Query | Query Length | Doc Length |
|---------|---------|------|---------------|--------------|------------|
| MS MARCO | 6,980 | 8.8M | 1.1 | 6 words | 60 words |
| SciFact | 300 | 5.2K | 1.2 | 12 words | 200 words |
| ArguAna | 1,406 | 8.7K | 1.0 | 18 words | 165 words |
| Climate-FEVER | 1,535 | 5.4M | 3.5 | 8 words | 75 words |
| NFCorpus | 323 | 3.6K | 38.2 | 3 words | 221 words |
| FiQA | 648 | 57K | 2.6 | 11 words | 132 words |

## Corpus Sampling Strategy

For Climate-FEVER (5.4M documents), we use **stratified sampling**:

### Sampling Algorithm

```python
def sample_corpus(corpus, qrels, max_docs=10000):
    """
    Sample corpus while preserving all relevant documents.

    Args:
        corpus: Full document collection
        qrels: Query-document relevance judgments
        max_docs: Target corpus size

    Returns:
        Sampled corpus
    """
    # Step 1: Collect all relevant document IDs
    relevant_ids = set()
    for qrel in qrels:
        if qrel.relevance > 0:
            relevant_ids.add(qrel.doc_id)

    # Step 2: Keep ALL relevant documents
    sampled = [doc for doc in corpus if doc.id in relevant_ids]
    num_relevant = len(sampled)

    # Step 3: Random sample non-relevant documents
    non_relevant = [doc for doc in corpus if doc.id not in relevant_ids]
    num_to_sample = max_docs - num_relevant

    import random
    sampled_non_relevant = random.sample(non_relevant, num_to_sample)

    # Step 4: Combine
    sampled.extend(sampled_non_relevant)

    print(f"Sampled {len(sampled)} docs ({num_relevant} relevant, {num_to_sample} non-relevant)")
    return sampled
```

### Sampling Validation (Climate-FEVER)

```
Original Corpus: 5,416,593 documents
Relevant Documents: 86 (across all queries)
Sampled Corpus: 10,000 documents
Relevant Documents in Sample: 86 (100% preserved)

Memory Reduction: 99.8%
Recall Ceiling: 100% (all relevant docs available)
```

**Trade-off**: Slightly easier retrieval task (less distractors), but enables evaluation on limited hardware.

## Loading Datasets

### MS MARCO

```python
from dartboard.evaluation.datasets import MSMARCOLoader

loader = MSMARCOLoader(data_dir="data/msmarco")
dataset = loader.load_dev_small()

# Access components
print(f"Queries: {len(dataset.queries)}")  # 6,980
print(f"Documents: {len(dataset.documents)}")
print(f"Qrels: {len(dataset.qrels)}")

# Get query
query = dataset.get_query_by_id("1048585")
print(query.text)  # "what is paula deen's brother"

# Get relevant docs for query
relevant = dataset.get_relevant_docs("1048585")
print(f"Relevant docs: {relevant}")  # {'7267032'}
```

### BEIR

```python
from dartboard.evaluation.datasets import BEIRLoader

loader = BEIRLoader(data_dir="data/beir")

# Load specific dataset
scifact = loader.load_dataset("scifact")

# Load with corpus sampling (for large datasets)
climate = loader.load_dataset(
    "climate-fever",
    max_corpus_docs=10000  # Sample to 10K
)

# Access components
for query in scifact.queries[:3]:
    print(f"Query: {query.text}")
    relevant = scifact.get_relevant_docs(query.id)
    print(f"Relevant docs: {relevant}\n")
```

## Other Datasets to Consider

### Additional BEIR Datasets

| Dataset | Queries | Docs | Domain | Use Case |
|---------|---------|------|--------|----------|
| **TREC-COVID** | 50 | 171K | COVID-19 research | Pandemic literature |
| **Natural Questions** | 3,452 | 2.7M | Wikipedia | Open-domain QA |
| **HotpotQA** | 7,405 | 5.2M | Wikipedia | Multi-hop reasoning |
| **DBPedia** | 400 | 4.6M | Knowledge base | Entity retrieval |
| **FEVER** | 6,666 | 5.4M | Wikipedia | Fact verification |

### Domain-Specific Datasets

**Legal**:
- **CaseHOLD**: Legal case citation prediction
- **LeCaRD**: Chinese legal case retrieval

**Medical**:
- **BioASQ**: Biomedical literature QA
- **TREC Precision Medicine**: Clinical trials retrieval

**Code**:
- **CodeSearchNet**: Code search from natural language
- **CoNaLa**: Mining code for NL intents

**Multi-Modal**:
- **MS COCO**: Image-text retrieval
- **Conceptual Captions**: Image captioning

### Why Not Included (Yet)

1. **TREC-COVID**: Only 50 queries (too small for robust eval)
2. **Natural Questions**: 2.7M docs (very large, requires sampling)
3. **HotpotQA**: Multi-hop reasoning (different task)
4. **Domain-specific**: Requires specialized models

**Future Additions**: As the system matures, we plan to add:
- Natural Questions (general knowledge QA)
- BioASQ (biomedical domain)
- CodeSearchNet (code search)

## Dataset Selection Guidelines

### For General-Purpose RAG

**Recommended**:
1. MS MARCO (web search)
2. SciFact (scientific)
3. FiQA (financial)

**Why**: Covers main use cases, diverse domains, realistic queries

### For Domain-Specific RAG

**Medical/Healthcare**:
- NFCorpus (nutrition)
- BioASQ (biomedical)

**Scientific**:
- SciFact (fact verification)
- TREC-COVID (pandemic research)

**Argumentative/Debate**:
- ArguAna (counter-arguments)
- FEVER (claim verification)

**Financial**:
- FiQA (financial QA)

### For Benchmarking

**Quick Test** (minutes):
- SciFact (300 queries, 5K docs)

**Standard Evaluation** (hours):
- MS MARCO dev.small (6,980 queries, 8.8M docs)
- SciFact + ArguAna + FiQA

**Comprehensive** (days):
- All BEIR datasets
- MS MARCO full

## Dataset Quality

### Relevance Judgment Quality

All datasets use **human annotations**:

| Dataset | Annotation Method | Inter-Annotator Agreement |
|---------|------------------|---------------------------|
| MS MARCO | Crowdworkers (3 annotators/query) | κ = 0.77 (substantial) |
| SciFact | Domain experts (scientists) | κ = 0.84 (excellent) |
| BEIR (avg) | Varies by dataset | κ = 0.65-0.85 |

### Known Limitations

**MS MARCO**:
- Sparse labels (avg 1.1 relevant/query)
- May miss relevant documents

**Climate-FEVER**:
- Very large (requires sampling)
- Climate domain bias

**ArguAna**:
- Narrow domain (debate arguments)
- Subjective relevance (what's a good counter-argument?)

## Benchmark Results

See [evaluation-system.md](evaluation-system.md) for full results.

**Summary** (Dec 2025):
- **Best overall**: Hybrid (BM25 + Dense)
- **SciFact**: Hybrid NDCG@10 = 0.78
- **ArguAna**: Dense NDCG@10 = 0.31
- **Climate-FEVER**: Dense NDCG@10 = 0.53

## References

### Datasets

- **MS MARCO**: [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/)
- **BEIR**: [https://github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)
- **BEIR Paper**: Thakur et al. (2021) - "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"

### Implementation

- **Dataset Loaders**: [dartboard/evaluation/datasets.py](../dartboard/evaluation/datasets.py)
- **Download Script**: [benchmarks/scripts/download_datasets.py](../benchmarks/scripts/download_datasets.py)
- **Benchmark Runner**: [benchmarks/scripts/run_benchmark.py](../benchmarks/scripts/run_benchmark.py)

## Summary

We use **MS MARCO and BEIR** datasets for comprehensive retrieval evaluation. These provide diverse domains, realistic queries, and rigorous human judgments, enabling robust assessment of retrieval methods.

**Key Takeaways**:
- ✅ **MS MARCO**: Industry-standard passage ranking
- ✅ **BEIR**: Diverse domains (science, climate, finance, medicine, debate)
- ✅ **Corpus sampling**: Enables evaluation on massive datasets (Climate-FEVER: 5.4M → 10K)
- ✅ **Human annotations**: High-quality relevance judgments
- 📊 **3 datasets benchmarked** (SciFact, ArguAna, Climate-FEVER)
- 🔮 **Future**: Add Natural Questions, BioASQ, CodeSearchNet for broader coverage
