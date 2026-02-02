"""
Load real open-source book/story dataset for hybrid search evaluation.

Supports multiple datasets:
1. Good Reads Books (CSV) - https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
2. Project Gutenberg (via gutendex API)
3. Custom CSV format
"""

import json
import csv
import requests
from pathlib import Path
from typing import List, Dict, Optional


def load_goodreads_csv(csv_path: str, num_stories: int = 5000, output_path: str = "data_books.json") -> None:
    """
    Load Good Reads books dataset from CSV.
    
    CSV should have columns: title, authors, average_rating, isbn, language_code, ...
    Download from: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks
    
    Args:
        csv_path: Path to books.csv file
        num_stories: Number of books to load (default 5000)
        output_path: Path to save converted JSON
    """
    print(f"Loading Good Reads dataset from {csv_path}...")
    
    stories = []
    story_id = 1
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if story_id > num_stories:
                    break
                
                # Extract fields with error handling
                title = row.get('title', 'Unknown').strip()
                if not title:
                    continue
                
                authors = row.get('authors', 'Unknown').strip()
                
                # Safe rating extraction
                try:
                    rating_str = str(row.get('average_rating', '0')).strip()
                    rating = float(rating_str) if rating_str and rating_str != '' else 0.0
                except (ValueError, TypeError):
                    rating = 0.0
                
                language = row.get('language_code', 'en').strip()
                
                # Skip non-English
                if language and language != 'en' and language != 'eng':
                    continue
                
                # Create description
                author_safe = authors.split(',')[0].strip()[:50] if authors else "Unknown"
                description = f"By {author_safe}. Average rating: {rating:.1f}/5.0"
                
                # Extract tags from authors (first author name)
                tags = [author_safe] if author_safe != "Unknown" else []
                
                # Engagement score: based on rating (scale up for visibility)
                engagement_score = int(rating * 1000) if rating > 0 else 100
                
                story = {
                    "story_id": story_id,
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "engagement_score": engagement_score
                }
                
                stories.append(story)
                story_id += 1
                
                if story_id % 500 == 0:
                    print(f"  Loaded {story_id-1} stories...")
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        print("📥 Download Good Reads dataset from:")
        print("   https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks")
        return
    
    # Save to JSON
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print(f"✓ Total books: {len(stories)}")
    if stories:
        print(f"✓ Sample book:\n  {json.dumps(stories[0], indent=2)}")


def load_custom_csv(csv_path: str, 
                    title_col: str = "title",
                    description_col: str = "description",
                    tags_col: Optional[str] = None,
                    engagement_col: Optional[str] = None,
                    num_stories: int = 5000,
                    output_path: str = "data_books.json") -> None:
    """
    Load any CSV dataset with custom column mapping.
    
    Args:
        csv_path: Path to CSV file
        title_col: Column name for story titles
        description_col: Column name for descriptions
        tags_col: Column name for tags (optional, comma-separated)
        engagement_col: Column name for engagement scores (optional)
        num_stories: Number of stories to load
        output_path: Path to save converted JSON
    """
    print(f"Loading custom CSV from {csv_path}...")
    
    stories = []
    story_id = 1
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if story_id > num_stories:
                    break
                
                # Extract fields
                title = row.get(title_col, 'Unknown').strip()
                description = row.get(description_col, '').strip()
                
                # Skip if missing title or description
                if not title or not description:
                    continue
                
                # Parse tags
                tags = []
                if tags_col and tags_col in row:
                    tags = [t.strip() for t in str(row[tags_col]).split(',') if t.strip()]
                
                # Parse engagement score
                engagement_score = 0
                if engagement_col and engagement_col in row:
                    try:
                        engagement_score = int(float(row[engagement_col]))
                    except (ValueError, TypeError):
                        engagement_score = 0
                
                story = {
                    "story_id": story_id,
                    "title": title,
                    "description": description,
                    "tags": tags,
                    "engagement_score": engagement_score
                }
                
                stories.append(story)
                story_id += 1
                
                if story_id % 500 == 0:
                    print(f"  Loaded {story_id-1} stories...")
    
    except FileNotFoundError:
        print(f"❌ Error: File not found at {csv_path}")
        return
    
    # Save to JSON
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stories, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print(f"✓ Total stories: {len(stories)}")
    if stories:
        print(f"✓ Sample story:\n  {json.dumps(stories[0], indent=2)}")


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print("OPEN-SOURCE STORY DATASET LOADER")
    print("="*70)
    print()
    print("Option 1: Good Reads Books Dataset")
    print("  Download: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks")
    print("  Usage: python generate_data.py")
    print()
    print("Option 2: Custom CSV")
    print("  Usage: python generate_data.py --csv <path> --title <col> --desc <col>")
    print()
    print("="*70)
    
    # Try to load Good Reads if it exists
    goodreads_path = "books.csv"
    if Path(goodreads_path).exists():
        print(f"\n✓ Found {goodreads_path}")
        load_goodreads_csv(goodreads_path, num_stories=5000, output_path="data_books.json")
    else:
        print(f"\n⚠ {goodreads_path} not found.")
        print("\nQuick Setup:")
        print("1. Download from: https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks")
        print("2. Extract books.csv to this directory")
        print("3. Run: python generate_data.py")
        print("\nOR provide your own CSV file:")
        print("  python generate_data.py --csv your_file.csv --title title_col --desc description_col")
