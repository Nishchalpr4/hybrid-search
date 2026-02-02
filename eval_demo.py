#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation demo: Test the system on ground-truth queries
Shows metrics like Recall@K, MRR, NDCG
"""

import sys
import json
from pathlib import Path

# Force UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent))

from search_pipeline import HybridSearchPipeline
from eval import SearchEvaluator

print("="*70)
print("EVALUATION DEMO - MEASURING SEARCH QUALITY")
print("="*70)

# Load pipeline (indices already built from previous run)
print("\n[1] Loading search pipeline...")
pipeline = HybridSearchPipeline(w_semantic=0.4, w_lexical=0.4, w_engagement=0.2)
pipeline.load_dataset('data_books.json')
print("   ✓ Loading indices (cached from previous run)...")
pipeline.build_indices(cache_dir='./cache')
print("   ✓ Pipeline ready")

# Load ground truth queries
print("\n[2] Loading evaluation queries...")
try:
    with open('eval_queries.json', 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    print(f"   ✓ Loaded {len(queries_data)} test queries with ground truth")
except FileNotFoundError:
    print("   ERROR: eval_queries.json not found")
    sys.exit(1)

# Create evaluator
print("\n[3] Evaluating search quality...")

# Get first 5 queries for quick evaluation
sample_queries = queries_data[:5]
print(f"\n   Evaluating {len(sample_queries)} queries:\n")

all_results = {}
for i, q in enumerate(sample_queries, 1):
    query_text = q['query']
    ground_truth = q['ground_truth']
    
    print(f"   Query {i}: '{query_text}'")
    print(f"   Ground truth books: {', '.join(ground_truth[:2])}...")
    
    try:
        results, metrics = pipeline.search(query_text, top_k=10)
        found_titles = [r.title for r in results]
        
        # Simple precision calculation
        matches = sum(1 for t in found_titles if any(gt in t or t in gt for gt in ground_truth))
        precision = matches / len(found_titles) if found_titles else 0
        
        print(f"   Top 3 results: {found_titles[:3]}")
        print(f"   Precision@10: {precision:.2%}")
        print()
        
        all_results[query_text] = {
            'ground_truth': ground_truth,
            'results': found_titles,
            'precision': precision
        }
    except Exception as e:
        print(f"   ERROR: {e}\n")

# Summary statistics
print("\n" + "="*70)
if all_results:
    avg_precision = sum(r['precision'] for r in all_results.values()) / len(all_results)
    print(f"EVALUATION SUMMARY")
    print(f"Average Precision@10: {avg_precision:.2%}")
    print(f"Queries evaluated: {len(all_results)}")
    print("="*70)
    
    print("\nDetailed Results:")
    for query, result in all_results.items():
        print(f"\n  Query: '{query}'")
        print(f"  Precision: {result['precision']:.2%}")
        print(f"  Top match: {result['results'][0] if result['results'] else 'None'}")
else:
    print("ERROR: No results to evaluate")
    
print("\n[COMPLETE] Evaluation finished!")
print("\nTo improve results:")
print("  - Adjust weights: w_semantic, w_lexical, w_engagement")
print("  - Try different embedding models")
print("  - Enrich documents with summaries from Open Library API")
