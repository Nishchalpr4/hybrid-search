# Hybrid Search: BM25 + Semantic Embeddings

A working prototype demonstrating **hybrid search** combining lexical (BM25) and semantic (embeddings) approaches on 5,000 real books.

> **What this is:** Educational prototype showing how two search methods work together.  
> **What this isn't:** Production system. [See Real-World Notes](#real-world-notes).

## Quick Start (2 minutes)

```bash
git clone https://github.com/yourusername/hybrid-search.git
cd hybrid-search
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python quick_demo.py
```

## How It Works

```
User Query
    ↓
[BM25 Search] + [Semantic Search]
    ↓
[Hybrid Fusion: 0.4×semantic + 0.4×lexical + 0.2×engagement]
    ↓
Ranked Results
```

## Results

| Method | Precision@10 | Query Time |
|--------|-------------|-----------|
| BM25 only | 65% | 2ms |
| Embeddings only | 72% | 100ms |
| **Hybrid** | **88%** | 130ms |

## Project Structure

```
hybrid-search/
├── quick_demo.py          # 👈 START HERE
├── eval_demo.py           # Evaluation
├── search_pipeline.py     # Core engine (409 lines)
├── generate_data.py       # Data loading (200 lines)
├── eval.py               # Metrics (200 lines)
├── data_books.json       # 5,000 books
├── eval_queries.json     # Test queries
└── cache/
    └── embeddings_cache.npy
```

## Code Example

```python
from search_pipeline import HybridSearchPipeline

pipeline = HybridSearchPipeline()
pipeline.load_dataset('data_books.json')
pipeline.build_indices(cache_dir='./cache')

results, metrics = pipeline.search("fantasy adventure", top_k=10)
for r in results:
    print(f"{r.rank}. {r.title} (score: {r.final_score:.3f})")
```

## Why Hybrid Works

**Problem:** Single-method approaches fail different queries

- BM25: "The Dark Secret" ✓ (exact match)
- Embeddings: "The Demon-Haunted World" ✓ (semantic match)
- **Hybrid:** "A Caribbean Mystery" ✓ (both methods agree)

## Real-World Considerations

⚠️ **This is a prototype.** Production needs:

| Feature | Here | Production |
|---------|------|-----------|
| Scale | 5K books | 1M+ books |
| Storage | NumPy file | Vector DB (Pinecone) |
| Lexical | BM25 in-memory | Elasticsearch |
| Latency | 130ms | <50ms |
| Availability | Single machine | Distributed |
| Updates | Full rebuild | Zero-downtime |
| Filtering | None | Genre, price, rating |

### Why Not Production?

1. **Data:** Only title + author (1 sentence per book)
2. **Scale:** Breaks at 100K documents
3. **Updates:** Full rebuild needed for new books
4. **Cost:** Embedding inference expensive at scale
5. **Reliability:** No fault tolerance

### What You'd Actually Use

```
Lexical:  Elasticsearch (distributed BM25)
Semantic: Pinecone/Weaviate (vector DB)
Ranking:  Custom service
Cache:    Redis
Monitor:  Datadog
```

## Try Different Approaches

### Different Embedding Models

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
```

### Adjust Weights

```python
# Current: 0.4/0.4/0.2 (balanced)
# Try: 0.6/0.2/0.2 (prefer semantic)
pipeline = HybridSearchPipeline(
    w_semantic=0.6,
    w_lexical=0.2,
    w_engagement=0.2
)
```

## Dependencies

```
rank-bm25==0.4.1
sentence-transformers==2.2.2
scikit-learn==1.3.2
numpy==1.24.3
pandas==2.0.3
```

## Performance

```
Load books:        20ms
Build BM25:        100ms
Generate embeddings: 72s (first run)
Query latency:     130ms
Memory:            ~225 MB peak
```

## Educational Value

✅ How BM25 works  
✅ How embeddings work  
✅ Why hybrid helps (88% vs 65-72%)  
✅ Real-world trade-offs  
✅ When to use what  

## License

MIT - Use freely for learning

---

**Status:** Educational prototype ✓ | Working demo ✓ | Production-ready ✗

Last updated: February 2, 2026
