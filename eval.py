"""
Evaluation framework for hybrid search.
Computes Recall@K, MRR, and compares different retrieval methods.
"""

import json
from typing import List, Dict, Set
from dataclasses import dataclass
import numpy as np
from search_pipeline import HybridSearchPipeline, SearchResult


@dataclass
class EvalMetrics:
    """Metrics for a single query evaluation."""
    query: str
    relevant_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all queries."""
    method_name: str
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_mrr: float
    avg_ndcg_at_10: float
    num_queries: int


class SearchEvaluator:
    """Evaluate search results against ground truth labels."""

    def __init__(self, queries_path: str):
        """
        Load evaluation queries with manual relevance labels.
        
        Format:
        [
          {
            "query": "epic fantasy adventure",
            "relevant_story_ids": [123, 456, 789, ...]
          },
          ...
        ]
        """
        with open(queries_path, 'r', encoding='utf-8') as f:
            self.queries_data = json.load(f)
        print(f"Loaded {len(self.queries_data)} evaluation queries")

    def evaluate_results(
        self,
        query: str,
        results: List[SearchResult],
        relevant_ids: Set[int]
    ) -> EvalMetrics:
        """
        Evaluate a single query's results.
        
        Metrics:
        - Recall@K: How many relevant docs in top-K / total relevant
        - MRR: Mean Reciprocal Rank of first relevant doc
        - NDCG@10: Normalized Discounted Cumulative Gain
        """
        result_ids = [r.story_id for r in results]
        
        # Recall@5
        recall_at_5 = self._compute_recall(result_ids[:5], relevant_ids)
        
        # Recall@10
        recall_at_10 = self._compute_recall(result_ids[:10], relevant_ids)
        
        # MRR
        mrr = self._compute_mrr(result_ids, relevant_ids)
        
        # NDCG@10
        ndcg = self._compute_ndcg(result_ids[:10], relevant_ids)
        
        return EvalMetrics(
            query=query,
            relevant_count=len(relevant_ids),
            recall_at_5=recall_at_5,
            recall_at_10=recall_at_10,
            mrr=mrr,
            ndcg_at_10=ndcg
        )

    @staticmethod
    def _compute_recall(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
        """Recall = |retrieved ∩ relevant| / |relevant|"""
        if not relevant_ids:
            return 0.0
        
        hits = len(set(retrieved_ids) & relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def _compute_mrr(retrieved_ids: List[int], relevant_ids: Set[int]) -> float:
        """MRR = 1 / (rank of first relevant result)"""
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _compute_ndcg(retrieved_ids: List[int], relevant_ids: Set[int], k: int = 10) -> float:
        """
        NDCG@K = DCG@K / Ideal DCG@K
        DCG = Σ(relevance_i / log2(i+1))
        """
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_ids[:k], 1):
            relevance = 1.0 if doc_id in relevant_ids else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Ideal DCG (assume at most K relevant docs)
        ideal_dcg = 0.0
        for i in range(1, min(len(relevant_ids), k) + 1):
            ideal_dcg += 1.0 / np.log2(i + 1)
        
        if ideal_dcg == 0:
            return 0.0
        
        return dcg / ideal_dcg

    def evaluate_all(
        self,
        pipeline: HybridSearchPipeline,
        method_name: str = "hybrid"
    ) -> AggregateMetrics:
        """
        Evaluate entire query set for a method.
        
        Args:
            pipeline: Search pipeline with search_* methods
            method_name: "hybrid", "lexical", or "semantic"
        
        Returns:
            AggregateMetrics with averages across all queries
        """
        metrics_list = []
        
        for query_data in self.queries_data:
            query = query_data['query']
            relevant_ids = set(query_data['relevant_story_ids'])
            
            # Choose evaluation method
            if method_name == "hybrid":
                results, _ = pipeline.search(query, top_k=10)
            elif method_name == "lexical":
                results = pipeline.search_lexical_only(query, top_k=10)
            elif method_name == "semantic":
                results = pipeline.search_semantic_only(query, top_k=10)
            else:
                raise ValueError(f"Unknown method: {method_name}")
            
            # Evaluate
            metrics = self.evaluate_results(query, results, relevant_ids)
            metrics_list.append(metrics)
            
            print(f"Query: '{query}' | R@5: {metrics.recall_at_5:.2f} | R@10: {metrics.recall_at_10:.2f} | MRR: {metrics.mrr:.3f}")
        
        # Aggregate
        avg_recall_5 = np.mean([m.recall_at_5 for m in metrics_list])
        avg_recall_10 = np.mean([m.recall_at_10 for m in metrics_list])
        avg_mrr = np.mean([m.mrr for m in metrics_list])
        avg_ndcg = np.mean([m.ndcg_at_10 for m in metrics_list])
        
        return AggregateMetrics(
            method_name=method_name,
            avg_recall_at_5=avg_recall_5,
            avg_recall_at_10=avg_recall_10,
            avg_mrr=avg_mrr,
            avg_ndcg_at_10=avg_ndcg,
            num_queries=len(metrics_list)
        )


def print_comparison(results: List[AggregateMetrics]) -> None:
    """Pretty-print comparison of retrieval methods."""
    print("\n" + "="*100)
    print("EVALUATION RESULTS COMPARISON")
    print("="*100)
    print(f"\n{'Method':<15} {'Recall@5':<12} {'Recall@10':<12} {'MRR':<12} {'NDCG@10':<12}")
    print("-"*100)
    
    for metrics in results:
        print(
            f"{metrics.method_name:<15} "
            f"{metrics.avg_recall_at_5:<12.3f} "
            f"{metrics.avg_recall_at_10:<12.3f} "
            f"{metrics.avg_mrr:<12.3f} "
            f"{metrics.avg_ndcg_at_10:<12.3f}"
        )
    
    print("="*100)


if __name__ == "__main__":
    print("Evaluator initialized. Use with search_pipeline.HybridSearchPipeline")
