"""
Hybrid Search Pipeline for Digital Storytelling Platform
Combines BM25 lexical search with semantic embeddings for intent-heavy queries.
Target: 5k-10k story documents.
"""

import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path

# External dependencies
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with ranking scores."""
    story_id: int
    title: str
    description: str
    tags: List[str]
    engagement_score: float
    lexical_score: float
    semantic_score: float
    engagement_score_norm: float
    final_score: float
    rank: int


@dataclass
class SearchMetrics:
    """Instrumentation for search operations."""
    query: str
    index_build_time: float = 0.0
    lexical_query_time: float = 0.0
    semantic_query_time: float = 0.0
    merge_time: float = 0.0
    total_query_time: float = 0.0


class HybridSearchPipeline:
    """
    Hybrid search system combining lexical (BM25) and semantic (embeddings) retrieval.
    
    Architecture:
    - Lexical: BM25 index built on tokenized searchable_text
    - Semantic: Dense embeddings (sentence-transformers) with cosine similarity
    - Hybrid: Configurable weighted combination of both signals + engagement
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        w_semantic: float = 0.4,
        w_lexical: float = 0.4,
        w_engagement: float = 0.2,
    ):
        """
        Initialize pipeline with hyperparameters.
        
        Args:
            model_name: Sentence-transformers model for embeddings
            w_semantic: Weight for semantic similarity score
            w_lexical: Weight for lexical (BM25) score
            w_engagement: Weight for engagement score
        """
        self.model_name = model_name
        self.w_semantic = w_semantic
        self.w_lexical = w_lexical
        self.w_engagement = w_engagement
        
        # Validation
        total_weight = w_semantic + w_lexical + w_engagement
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        # Lazy-loaded components
        self.embedding_model = None
        self.bm25_index = None
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.engagement_scores: Optional[np.ndarray] = None
        self.searchable_texts: List[str] = []

        logger.info(f"Pipeline initialized: weights (semantic={w_semantic}, lexical={w_lexical}, engagement={w_engagement})")

    def load_dataset(self, data_path: str) -> None:
        """
        Load stories from JSON file and cache processed text.
        
        Dataset format:
        [
          {
            "story_id": 1,
            "title": "...",
            "description": "...",
            "tags": ["genre1", "genre2"],
            "engagement_score": 123
          },
          ...
        ]
        """
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
        
        logger.info(f"Loaded {len(self.documents)} stories")
        
        # Preprocess: concatenate searchable text
        self._preprocess_documents()

    def _preprocess_documents(self) -> None:
        """
        Concatenate title, description, tags into searchable_text.
        Cache for repeated access.
        """
        logger.info("Preprocessing documents: concatenating searchable text")
        self.searchable_texts = []
        self.engagement_scores = np.zeros(len(self.documents), dtype=np.float32)

        for i, doc in enumerate(self.documents):
            # Concatenate fields
            title = doc.get('title', '')
            description = doc.get('description', '')
            tags = ' '.join(doc.get('tags', []))
            searchable_text = f"{title} {description} {tags}".strip()
            self.searchable_texts.append(searchable_text)
            
            # Cache engagement score
            self.engagement_scores[i] = float(doc.get('engagement_score', 0))

        logger.info(f"Preprocessed {len(self.searchable_texts)} documents")

    def build_indices(self, cache_dir: str = "./cache") -> Dict[str, float]:
        """
        Build BM25 index and generate embeddings.
        Returns timing information.
        
        Args:
            cache_dir: Directory to cache embeddings for future runs
        
        Returns:
            dict with timings for lexical_build, semantic_build, total
        """
        timings = {}
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        start = time.time()
        tokenized_texts = [text.lower().split() for text in self.searchable_texts]
        self.bm25_index = BM25Okapi(tokenized_texts)
        timings['lexical_build'] = time.time() - start
        logger.info(f"BM25 index built in {timings['lexical_build']:.2f}s")

        # Build embeddings (with caching)
        logger.info(f"Building embeddings with model {self.model_name}...")
        start = time.time()
        
        # Always load the model for encoding queries
        if self.embedding_model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
        
        cache_path = Path(cache_dir) / "embeddings_cache.npy"
        
        if cache_path.exists():
            logger.info("Loading cached embeddings...")
            self.embeddings = np.load(cache_path)
        else:
            logger.info(f"Generating {len(self.searchable_texts)} embeddings...")
            self.embeddings = self.embedding_model.encode(
                self.searchable_texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Cache for future runs
            Path(cache_dir).mkdir(exist_ok=True)
            np.save(cache_path, self.embeddings)
            logger.info(f"Embeddings cached to {cache_path}")
        
        timings['semantic_build'] = time.time() - start
        logger.info(f"Embeddings ready in {timings['semantic_build']:.2f}s")
        
        timings['total'] = timings['lexical_build'] + timings['semantic_build']
        logger.info(f"Total index build time: {timings['total']:.2f}s")
        
        return timings

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 50,
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """
        Hybrid search: combines lexical and semantic retrieval.
        
        Args:
            query: Search query text
            top_k: Number of final results to return
            candidate_k: Number of candidates to retrieve from each method before merging
        
        Returns:
            (list of SearchResult, SearchMetrics)
        """
        if self.bm25_index is None or self.embeddings is None:
            raise RuntimeError("Indices not built. Call build_indices() first.")
        
        metrics = SearchMetrics(query=query)
        
        # 1. Lexical search
        start = time.time()
        query_tokens = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-K indices for lexical
        lexical_top_k = min(candidate_k, len(self.documents))
        lexical_indices = np.argsort(-bm25_scores)[:lexical_top_k]
        lexical_scores_dict = {idx: float(bm25_scores[idx]) for idx in lexical_indices}
        metrics.lexical_query_time = time.time() - start
        
        # 2. Semantic search
        start = time.time()
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Cosine similarity: normalized dot product (embeddings are already normalized by sentence-transformers)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-K indices for semantic
        semantic_top_k = min(candidate_k, len(self.documents))
        semantic_indices = np.argsort(-similarities)[:semantic_top_k]
        semantic_scores_dict = {idx: float(similarities[idx]) for idx in semantic_indices}
        metrics.semantic_query_time = time.time() - start
        
        # 3. Merge candidate sets
        start = time.time()
        all_indices = set(lexical_indices) | set(semantic_indices)
        metrics.merge_time = time.time() - start
        
        # 4. Compute hybrid scores
        results = []
        for idx in all_indices:
            # Get individual scores (normalize if missing)
            lexical_score = lexical_scores_dict.get(idx, 0.0)
            semantic_score = semantic_scores_dict.get(idx, 0.0)
            engagement_score = self.engagement_scores[idx]
            
            # Normalize scores to [0, 1]
            # BM25 scores vary; normalize via MinMaxScaler on retrieved set
            lexical_scores_vals = np.array(list(lexical_scores_dict.values()))
            semantic_scores_vals = np.array(list(semantic_scores_dict.values()))
            
            lexical_norm = self._normalize_score(lexical_score, lexical_scores_vals)
            semantic_norm = self._normalize_score(semantic_score, semantic_scores_vals)
            
            # Engagement: log scale and normalize
            engagement_norm = np.log1p(engagement_score)
            max_engagement = np.log1p(self.engagement_scores.max())
            engagement_norm = engagement_norm / max_engagement if max_engagement > 0 else 0
            
            # Hybrid score
            final_score = (
                self.w_semantic * semantic_norm +
                self.w_lexical * lexical_norm +
                self.w_engagement * engagement_norm
            )
            
            results.append({
                'idx': idx,
                'lexical_score': lexical_norm,
                'semantic_score': semantic_norm,
                'engagement_score': engagement_norm,
                'final_score': final_score,
            })
        
        # Sort by final score
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Format output
        search_results = []
        for rank, result in enumerate(results[:top_k], 1):
            idx = result['idx']
            doc = self.documents[idx]
            desc = doc.get('description', '') or ''
            search_results.append(SearchResult(
                story_id=int(doc.get('story_id', idx)),
                title=doc.get('title', 'Unknown'),
                description=desc[:100] + "..." if len(desc) > 100 else desc,
                tags=doc.get('tags', []),
                engagement_score=self.engagement_scores[idx],
                lexical_score=result['lexical_score'],
                semantic_score=result['semantic_score'],
                engagement_score_norm=result['engagement_score'],
                final_score=result['final_score'],
                rank=rank,
            ))
        
        metrics.total_query_time = metrics.lexical_query_time + metrics.semantic_query_time + metrics.merge_time
        
        return search_results, metrics

    @staticmethod
    def _normalize_score(score: float, all_scores: np.ndarray) -> float:
        """Normalize score to [0, 1] range."""
        min_score = all_scores.min()
        max_score = all_scores.max()
        
        if max_score == min_score:
            return 0.5 if score >= min_score else 0.0
        
        return float((score - min_score) / (max_score - min_score))

    def search_lexical_only(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Baseline: lexical search only."""
        if self.bm25_index is None:
            raise RuntimeError("BM25 index not built.")
        
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(-scores)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc = self.documents[idx]
            desc = doc.get('description', '') or ''
            results.append(SearchResult(
                story_id=int(doc.get('story_id', idx)),
                title=doc.get('title', 'Unknown'),
                description=desc[:100] + "..." if len(desc) > 100 else desc,
                tags=doc.get('tags', []),
                engagement_score=self.engagement_scores[idx],
                lexical_score=float(scores[idx]),
                semantic_score=0.0,
                engagement_score_norm=0.0,
                final_score=float(scores[idx]),
                rank=rank,
            ))
        return results

    def search_semantic_only(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Baseline: semantic search only."""
        if self.embeddings is None:
            raise RuntimeError("Embeddings not built.")
        
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            doc = self.documents[idx]
            desc = doc.get('description', '') or ''
            results.append(SearchResult(
                story_id=int(doc.get('story_id', idx)),
                title=doc.get('title', 'Unknown'),
                description=desc[:100] + "..." if len(desc) > 100 else desc,
                tags=doc.get('tags', []),
                engagement_score=self.engagement_scores[idx],
                lexical_score=0.0,
                semantic_score=float(similarities[idx]),
                engagement_score_norm=0.0,
                final_score=float(similarities[idx]),
                rank=rank,
            ))
        return results


def print_results(
    results: List[SearchResult],
    metrics: Optional[SearchMetrics] = None,
    method_name: str = "Hybrid"
) -> None:
    """Pretty-print search results."""
    print(f"\n{'='*80}")
    print(f"{method_name.upper()} SEARCH RESULTS")
    print(f"{'='*80}")
    
    if metrics:
        print(f"\nQuery: {metrics.query}")
        print(f"Lexical search time: {metrics.lexical_query_time*1000:.2f}ms")
        print(f"Semantic search time: {metrics.semantic_query_time*1000:.2f}ms")
        print(f"Total query time: {metrics.total_query_time*1000:.2f}ms")
    
    print(f"\n{'Rank':<5} {'Score':<8} {'Story ID':<10} {'Title':<40}")
    print(f"{'-'*70}")
    
    for result in results:
        score_str = f"{result.final_score:.3f}"
        title_str = result.title[:37] + "..." if len(result.title) > 37 else result.title
        print(f"{result.rank:<5} {score_str:<8} {result.story_id:<10} {title_str:<40}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    # Quick test (requires data_synthetic.json)
    print("Hybrid Search Pipeline initialized. Use import to integrate.")
