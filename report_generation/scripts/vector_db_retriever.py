"""
Simple vector database retriever for dummy RAG dataset.

This module provides a simple keyword-based retrieval system for the vector DB.
In a production system, you would use proper vector embeddings and similarity search.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import re


class VectorDBRetriever:
    """Simple retriever for vector database using keyword matching."""
    
    def __init__(self, db_path: str = "data/dummy_rag_vector_db.jsonl"):
        """Initialize the vector DB retriever."""
        self.db_path = Path(db_path)
        self.entries = []
        self._load_db()
    
    def _load_db(self):
        """Load vector database from JSONL file."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found: {self.db_path}")
        
        with open(self.db_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))
        
        print(f"Loaded {len(self.entries)} entries from vector DB")
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for matching."""
        # Remove common words and extract meaningful terms
        stop_words = {'will', 'the', 'a', 'an', 'be', 'in', 'on', 'at', 'by', 'for', 
                     'to', 'of', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
        
        # Extract words (alphanumeric, at least 3 chars)
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
        keywords = [w for w in words if w not in stop_words]
        
        return keywords
    
    def _score_relevance(self, entry: Dict, keywords: List[str]) -> float:
        """Score entry relevance based on keyword matches."""
        searchable_text = entry.get('searchable_text', '').lower()
        title = entry.get('title', '').lower()
        content = entry.get('content', '').lower()
        
        score = 0.0
        
        # Title matches are worth more
        for keyword in keywords:
            if keyword in title:
                score += 3.0
            elif keyword in content:
                score += 1.0
        
        # Boost for high relevance entries
        if entry.get('relevance') == 'high':
            score += 2.0
        
        return score
    
    def retrieve(self, query: str, market_id: Optional[str] = None, 
                 num_results: int = 5) -> List[Dict]:
        """
        Retrieve relevant entries from vector DB.
        
        Args:
            query: Search query
            market_id: Optional market ID to filter by
            num_results: Number of results to return
        
        Returns:
            List of relevant entries
        """
        keywords = self._extract_keywords(query)
        
        # Filter by market_id if provided
        candidates = self.entries
        if market_id:
            candidates = [e for e in self.entries if e.get('market_id') == market_id]
        
        # Score and sort
        scored = []
        for entry in candidates:
            score = self._score_relevance(entry, keywords)
            if score > 0:
                scored.append((score, entry))
        
        # Sort by score (descending) and return top results
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [entry for _, entry in scored[:num_results]]
        
        # Format results to match search_retriever format
        formatted_results = []
        for result in results:
            formatted_results.append({
                'title': result.get('title', ''),
                'snippet': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                'link': result.get('url', ''),
                'source': result.get('source', 'Vector DB'),
                'published': result.get('date', ''),
                'relevance': result.get('relevance', 'medium'),
                'domain': result.get('domain', 'general')
            })
        
        return formatted_results
    
    def get_by_market_id(self, market_id: str, num_results: int = 5) -> List[Dict]:
        """Get all entries for a specific market ID."""
        entries = [e for e in self.entries if e.get('market_id') == market_id]
        
        # Sort by relevance (high first)
        entries.sort(key=lambda x: 0 if x.get('relevance') == 'high' else 1)
        
        formatted_results = []
        for result in entries[:num_results]:
            formatted_results.append({
                'title': result.get('title', ''),
                'snippet': result.get('content', '')[:200] + '...' if len(result.get('content', '')) > 200 else result.get('content', ''),
                'link': result.get('url', ''),
                'source': result.get('source', 'Vector DB'),
                'published': result.get('date', ''),
                'relevance': result.get('relevance', 'medium'),
                'domain': result.get('domain', 'general')
            })
        
        return formatted_results


if __name__ == "__main__":
    # Test the retriever
    retriever = VectorDBRetriever()
    
    # Test query
    query = "Will Donald Trump win the 2024 US Presidential Election?"
    results = retriever.retrieve(query, market_id="DUMMY001", num_results=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Source: {result['source']}")
        print(f"   Relevance: {result['relevance']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print()

