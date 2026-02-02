"""Quick demo of hybrid search."""

from search_simple import HybridSearch

# Initialize search (loads data + embeddings)
search = HybridSearch()

# Run 4 test queries
queries = [
    "fantasy adventure magic",
    "love romance heartbreak",
    "mystery detective crime",
    "sci-fi future technology"
]

print("\n" + "="*60)
print("HYBRID SEARCH DEMO")
print("="*60)

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    results = search.search(query, top_k=3)
    
    for result in results:
        print(f"   {result['rank']}. {result['title']:<40} Score: {result['score']:.3f}")
        print(f"      {result['description']}")

print("\n" + "="*60)
