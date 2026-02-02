"""
Simple Hybrid Search - Combines BM25 (keywords) + Embeddings (meaning)
"""

import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler


class HybridSearch:
    """Simple hybrid search combining keyword and semantic search."""
    
    def __init__(self, data_file="data_books.json", model_name="all-MiniLM-L6-v2"):
        """Load data and initialize search models."""
        
        # Load books dataset
        with open(data_file, encoding='utf-8') as f:
            self.books = json.load(f)
        
        # Extract texts for BM25 indexing
        self.texts = [b["description"] for b in self.books]
        self.titles = [b["title"] for b in self.books]
        
        # Initialize BM25 (keyword search)
        self.bm25 = BM25Okapi([text.split() for text in self.texts])
        
        # Initialize embeddings model (semantic search)
        self.model = SentenceTransformer(model_name)
        
        # Encode all book descriptions once
        print("🔄 Encoding embeddings... (takes ~1 min first time)")
        self.embeddings = self.model.encode(self.texts, show_progress_bar=False)
        
        # Normalize scores to 0-1 range
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.embeddings)
        
        print("✅ Search ready!")
    
    def search(self, query, top_k=10, alpha=0.4):
        """
        Search combining BM25 (keyword) + embeddings (meaning).
        
        Args:
            query: Search text
            top_k: Return top K results
            alpha: Weight for embeddings (0.4 = 40% semantic, 60% keyword)
        """
        
        # Get keyword scores (BM25)
        bm25_scores = np.array(self.bm25.get_scores(query.split()))
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-10)
        
        # Get semantic scores (embeddings)
        query_embedding = self.model.encode(query)
        semantic_scores = np.dot(self.embeddings, query_embedding)
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)
        
        # Combine: alpha% semantic + (1-alpha)% keyword
        final_scores = alpha * semantic_scores + (1 - alpha) * bm25_scores
        
        # Get top K
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        
        # Return results
        results = []
        for i, idx in enumerate(top_indices, 1):
            results.append({
                'rank': i,
                'title': self.titles[idx],
                'score': float(final_scores[idx]),
                'description': self.texts[idx][:100] + "..."
            })
        
        return results


# Demo
if __name__ == "__main__":
    search = HybridSearch()
    
    queries = [
        "fantasy adventure magic",
        "love romance relationship",
        "mystery detective crime"
    ]
    
    for q in queries:
        print(f"\n🔍 Query: '{q}'")
        results = search.search(q, top_k=3)
        for r in results:
            print(f"  {r['rank']}. {r['title']} ({r['score']:.3f})")
