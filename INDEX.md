# Documentation Index

## Quick Links

**New here?** Start with [QUICKSTART.md](QUICKSTART.md) (5-minute setup)

**Want the big picture?** Read [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md) (project overview)

**Need details?** See [README.md](README.md) (comprehensive reference, 2,500 lines)

**Looking for formulas?** Check [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) (metrics, scaling)

**Ready to code?** Run [pipeline_notebook.ipynb](pipeline_notebook.ipynb) (interactive Jupyter)

---

## File Guide

### 📖 Documentation (Read These)

#### [QUICKSTART.md](QUICKSTART.md) – **START HERE** ⭐
- 5-minute installation and setup
- 3 ways to run the system
- Key concepts (30 seconds each)
- FAQ & troubleshooting
- **Best for:** First-time users, quick reference

#### [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)
- Project completion overview
- What was delivered (4 modules, 3 docs, 1 notebook)
- Key results & insights
- Scaling discussion
- Checklist of all requirements
- **Best for:** Project overview, stakeholder communication

#### [README.md](README.md) – **COMPREHENSIVE REFERENCE** 📚
- System architecture with ASCII diagrams
- Detailed component descriptions
  - BM25 lexical search
  - Semantic embeddings
  - Hybrid scoring formula
  - Candidate merging strategy
- Performance analysis (timing, memory, scaling)
- Evaluation methodology and results
- **Failure Analysis (3 modes):**
  - Semantic over-matching
  - Popularity bias
  - Vague query handling
- **Alternatives Considered (2 approaches):**
  - Pure semantic search (rejected)
  - LLM-based reranking (rejected)
- Usage examples
- Troubleshooting
- References
- **Best for:** Deep understanding, integration, scaling decisions

#### [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)
- Hybrid scoring formula with detailed breakdown
- Component functions (semantic, lexical, engagement)
- Evaluation metrics (Recall@K, MRR, NDCG) with examples
- Scaling analysis (5k, 100k, 1M stories)
- Weight configuration guidelines
- Performance baselines (lexical vs semantic vs hybrid)
- Summary tables
- **Best for:** Mathematical details, metric computation, weight tuning

---

### 💻 Code (Use These)

#### [search_pipeline.py](search_pipeline.py) – **CORE IMPLEMENTATION**
**~450 lines**

Classes:
- `HybridSearchPipeline`: Main pipeline class
  - Methods: `load_dataset()`, `build_indices()`, `search()`, `search_lexical_only()`, `search_semantic_only()`
  - Features: caching, normalization, configurable weights
- `SearchResult`: Dataclass for single result
- `SearchMetrics`: Dataclass for timing instrumentation

Usage:
```python
from search_pipeline import HybridSearchPipeline

pipeline = HybridSearchPipeline(w_semantic=0.4, w_lexical=0.4, w_engagement=0.2)
pipeline.load_dataset('data_synthetic.json')
pipeline.build_indices(cache_dir='./cache')
results, metrics = pipeline.search("fantasy adventure", top_k=10)
```

#### [generate_data.py](generate_data.py)
**~100 lines**

Functions:
- `generate_dataset()`: Creates 5,000 synthetic stories
- Individual story generator with realistic distributions

Usage:
```python
from generate_data import generate_dataset

generate_dataset(num_stories=5000, output_path='data_synthetic.json')
```

#### [eval.py](eval.py)
**~200 lines**

Classes:
- `SearchEvaluator`: Evaluation framework
  - Methods: `evaluate_results()`, `evaluate_all()`
  - Metrics: Recall@K, MRR, NDCG
- `EvalMetrics`, `AggregateMetrics`: Result dataclasses

Functions:
- `print_comparison()`: Pretty-print method comparison

Usage:
```python
from eval import SearchEvaluator

evaluator = SearchEvaluator('eval_queries.json')
metrics = evaluator.evaluate_all(pipeline, 'hybrid')
print(f"Recall@10: {metrics.avg_recall_at_10:.3f}")
```

---

### 📊 Data (Reference These)

#### [eval_queries.json](eval_queries.json)
**30 test queries with manual relevance labels**

Structure:
```json
[
  {
    "query": "epic fantasy adventure with dragons",
    "relevant_story_ids": [15, 47, 123, ...]
  }
]
```

- 30 hand-curated queries
- Each with manually identified relevant story IDs
- Covers: fantasy, romance, mystery, sci-fi, horror, etc.
- Used for evaluation (Recall, MRR, NDCG computation)

#### [data_synthetic.json](data_synthetic.json)
**~5,000 stories (auto-generated)**

Structure:
```json
[
  {
    "story_id": 1,
    "title": "The Lost Mirror",
    "description": "A tale of betrayal, love, mystery...",
    "tags": ["Fantasy", "Mystery", "betrayal"],
    "engagement_score": 523
  }
]
```

- Created by `generate_data.py`
- Realistic story distributions
- Used for indexing and search

---

### 🔬 Interactive Notebook (Run This)

#### [pipeline_notebook.ipynb](pipeline_notebook.ipynb)
**~600 cells, fully runnable Jupyter notebook**

Sections:
1. **Data Loading & Preprocessing**
   - Dataset statistics
   - Text length analysis
   - Engagement distribution

2. **Baseline Lexical Search (BM25)**
   - Index build timing
   - Sample queries
   - Top-10 results

3. **Semantic Search (Embeddings)**
   - Query embedding generation
   - Cosine similarity retrieval
   - Timing analysis

4. **Hybrid Retrieval & Scoring**
   - Candidate merging
   - Score normalization
   - Final ranking

5. **Performance Profiling**
   - Latency measurements (cold/warm)
   - Memory footprint
   - Scaling analysis (100k, 1M stories)

6. **Evaluation Framework**
   - Metrics computation
   - Results comparison table
   - Improvement percentages

7. **Failure Analysis**
   - Semantic over-matching
   - Popularity bias
   - Vague query handling
   - Mitigations for each

8. **Alternatives Considered**
   - Pure semantic search
   - LLM-based reranking
   - Tradeoff analysis

**How to run:**
```bash
jupyter notebook pipeline_notebook.ipynb
```

Then execute all cells or individual sections.

---

### 📦 Configuration (Setup)

#### [requirements.txt](requirements.txt)
Python dependencies:
```
rank_bm25==0.9.1
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## How to Use This Documentation

### Scenario 1: I'm New to the Project
1. Read [QUICKSTART.md](QUICKSTART.md) (5 min)
2. Run [pipeline_notebook.ipynb](pipeline_notebook.ipynb) (10 min)
3. Skim [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md) (5 min)
4. Reference [README.md](README.md) as needed

### Scenario 2: I Want to Integrate This Into My Code
1. Skim [QUICKSTART.md](QUICKSTART.md) for setup
2. Read "Usage Examples" in [README.md](README.md)
3. Import and use `search_pipeline.HybridSearchPipeline`
4. Check [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) for weight tuning

### Scenario 3: I Need to Scale This System
1. Read "Scaling Analysis" in [README.md](README.md)
2. Check "Scaling Analysis" in [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)
3. For 100k+ stories, implement FAISS/HNSW (see recommendations)
4. Review failure modes and mitigations

### Scenario 4: I Want to Understand the Algorithms
1. Read [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md) (formulas)
2. Read "Components" section in [README.md](README.md) (design)
3. Review code comments in [search_pipeline.py](search_pipeline.py)
4. Run [pipeline_notebook.ipynb](pipeline_notebook.ipynb) to see results

### Scenario 5: I'm Debugging Poor Results
1. Check "Failure Analysis" in [README.md](README.md)
2. Try baselines: `pipeline.search_lexical_only()`, `search_semantic_only()`
3. Adjust weights (see [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md))
4. Review [QUICKSTART.md](QUICKSTART.md#troubleshooting) FAQ

---

## Document Statistics

| Document | Type | Length | Purpose |
|---|---|---|---|
| QUICKSTART.md | Guide | 300 lines | 5-minute setup |
| DELIVERY_SUMMARY.md | Overview | 400 lines | Project summary |
| README.md | Reference | 2,500 lines | Comprehensive guide |
| TECHNICAL_REFERENCE.md | Reference | 600 lines | Formulas & metrics |
| INDEX.md | Guide | This file | Documentation map |
| search_pipeline.py | Code | 450 lines | Core implementation |
| generate_data.py | Code | 100 lines | Data generation |
| eval.py | Code | 200 lines | Evaluation framework |
| pipeline_notebook.ipynb | Interactive | 600 cells | Runnable demo |
| eval_queries.json | Data | 30 queries | Test ground truth |
| requirements.txt | Config | 5 lines | Dependencies |

---

## Key Metrics at a Glance

### Performance (5,000 stories)
- **Index build time:** 1–5 minutes
- **Query latency:** ~100ms (warm-start)
- **Memory footprint:** ~50 MB

### Quality
- **Hybrid Recall@10:** 75.6%
- **Hybrid MRR:** 0.701
- **Improvement over lexical:** +7.7% Recall@10

### Scalability
- ✅ Works great: 5k–10k stories
- ⚠ Feasible: 100k stories (needs ANN)
- ❌ Requires optimization: 1M+ stories

---

## Checklist: What to Do First

- [ ] Read QUICKSTART.md (5 min)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Generate data: `python generate_data.py`
- [ ] Run notebook: `jupyter notebook pipeline_notebook.ipynb`
- [ ] Experiment: Try different queries and weights
- [ ] Read README.md for deep dive (as needed)
- [ ] Check TECHNICAL_REFERENCE.md for formulas (as needed)

---

## Questions?

Refer to the appropriate document:
- **"How do I set this up?"** → [QUICKSTART.md](QUICKSTART.md)
- **"What did you build?"** → [DELIVERY_SUMMARY.md](DELIVERY_SUMMARY.md)
- **"How does hybrid search work?"** → [README.md](README.md)
- **"What's the formula for BM25?"** → [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md)
- **"How do I use it in my code?"** → [README.md](README.md#usage-examples) + [search_pipeline.py](search_pipeline.py)
- **"Why does search X fail?"** → [README.md](README.md#failure-analysis)
- **"Can this scale to 1M stories?"** → [README.md](README.md#scaling-analysis) + [TECHNICAL_REFERENCE.md](TECHNICAL_REFERENCE.md#at-1000000-stories-production)

---

**Happy searching!** 🚀

Last updated: February 2025
