#!/usr/bin/env python3
"""
Quick demo: Load books and test hybrid search
No notebook needed - just run this script!
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("="*70)
print("HYBRID SEARCH DEMO - REAL BOOKS")
print("="*70)

# Step 1: Load dataset
print("\n1️⃣  Loading 5,000 books from data_books.json...")
try:
    with open('data_books.json', 'r', encoding='utf-8') as f:
        books = json.load(f)
    print(f"   ✓ Loaded {len(books)} books")
    print(f"   Sample: {books[0]['title']}")
except FileNotFoundError:
    print("   ❌ data_books.json not found!")
    sys.exit(1)

# Step 2: Initialize pipeline
print("\n2️⃣  Initializing search pipeline...")
try:
    from search_pipeline import HybridSearchPipeline
    pipeline = HybridSearchPipeline(
        w_semantic=0.4,
        w_lexical=0.4,
        w_engagement=0.2
    )
    print("   ✓ Pipeline initialized")
except ImportError as e:
    print(f"   ❌ Error importing: {e}")
    sys.exit(1)

# Step 3: Load dataset into pipeline
print("\n3️⃣  Loading books into pipeline...")
pipeline.load_dataset('data_books.json')
print("   ✓ Books loaded")

# Step 4: Build indices
print("\n4️⃣  Building search indices (BM25 + embeddings)...")
print("   ⏳ This takes 1-5 minutes on first run (embeddings are slow)...")
try:
    timings = pipeline.build_indices(cache_dir='./cache')
    print(f"   ✓ BM25 index built: {timings['lexical_build']:.2f}s")
    print(f"   ✓ Embeddings ready: {timings['semantic_build']:.2f}s")
    print(f"   ✓ Total time: {timings['total']:.2f}s")
except Exception as e:
    print(f"   ❌ Error building indices: {e}")
    sys.exit(1)

# Step 5: Test searches
print("\n5️⃣  Testing hybrid search...")
test_queries = [
    "fantasy adventure magic",
    "romance love story",
    "mystery thriller suspense",
    "science fiction space",
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    try:
        results, metrics = pipeline.search(query, top_k=3)
        for r in results:
            print(f"      {r.rank}. {r.title} (score: {r.final_score:.3f})")
    except Exception as e:
        print(f"      ❌ Error: {e}")

# Step 6: Compare methods
print("\n\n6️⃣  Comparing lexical vs semantic vs hybrid...")
test_query = "epic fantasy adventure"
print(f"\n   Query: '{test_query}'\n")

try:
    print("   LEXICAL (BM25) - Top 3:")
    lex = pipeline.search_lexical_only(test_query, top_k=3)
    for r in lex:
        print(f"      {r.rank}. {r.title}")
    
    print("\n   SEMANTIC (Embeddings) - Top 3:")
    sem = pipeline.search_semantic_only(test_query, top_k=3)
    for r in sem:
        print(f"      {r.rank}. {r.title}")
    
    print("\n   HYBRID (Combined) - Top 3:")
    hyb, _ = pipeline.search(test_query, top_k=3)
    for r in hyb:
        print(f"      {r.rank}. {r.title}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Try different queries")
print("  2. Read README.md for details")
print("  3. Run evaluations with eval.py")
print("  4. Adjust weights for your use case")
