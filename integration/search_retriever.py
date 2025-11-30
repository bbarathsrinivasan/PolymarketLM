"""
Search-based retrieval module for RAG-style inference.

Fetches relevant information from web search APIs (DuckDuckGo, SerpAPI, etc.)
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter
import re
import time
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchRetriever:
    """Retrieve and score relevant search results from web search APIs."""
    
    def __init__(self, provider: str = "duckduckgo", api_key: Optional[str] = None, cache_dir: Optional[str] = None, cache_ttl: int = 3600):
        """
        Initialize search retriever.
        
        Args:
            provider: Search provider ("duckduckgo" or "serpapi")
            api_key: API key for providers that require it (e.g., SerpAPI)
            cache_dir: Directory to cache search results (None to disable caching)
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate provider
        if self.provider not in ["duckduckgo", "serpapi"]:
            logger.warning(f"Unknown provider {provider}, defaulting to duckduckgo")
            self.provider = "duckduckgo"
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """
        Perform web search and return results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of result dictionaries with keys: title, snippet, link, source
        """
        # Check cache first
        cached_results = self._get_cached_search(query)
        if cached_results:
            logger.info(f"Using cached search results for: {query}")
            return cached_results[:num_results]
        
        try:
            if self.provider == "duckduckgo":
                results = self._search_duckduckgo(query, num_results)
            elif self.provider == "serpapi":
                results = self._search_serpapi(query, num_results)
            else:
                results = []
            
            # Cache results
            if results:
                self._cache_search(query, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using DuckDuckGo (free, no API key required)."""
        try:
            import duckduckgo_search
            
            logger.info(f"Searching DuckDuckGo for: {query}")
            with duckduckgo_search.DDGS() as ddgs:
                results = []
                for r in ddgs.text(query, max_results=num_results):
                    result = {
                        'title': r.get('title', ''),
                        'snippet': r.get('body', ''),
                        'link': r.get('href', ''),
                        'source': 'DuckDuckGo',
                        'published': datetime.now()  # DuckDuckGo doesn't provide dates
                    }
                    results.append(result)
                
                logger.info(f"Found {len(results)} results from DuckDuckGo")
                return results
                
        except ImportError:
            logger.error("duckduckgo_search not installed. Install with: pip install duckduckgo-search")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_serpapi(self, query: str, num_results: int = 10) -> List[Dict]:
        """Search using SerpAPI (requires API key, free tier: 250 searches/month)."""
        if not self.api_key:
            logger.error("SerpAPI requires an API key. Get one from https://serpapi.com/")
            return []
        
        try:
            from serpapi import GoogleSearch
            
            logger.info(f"Searching SerpAPI for: {query}")
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            }
            
            search = GoogleSearch(params)
            results_dict = search.get_dict()
            
            results = []
            organic_results = results_dict.get("organic_results", [])
            
            for r in organic_results:
                result = {
                    'title': r.get('title', ''),
                    'snippet': r.get('snippet', ''),
                    'link': r.get('link', ''),
                    'source': r.get('source', 'SerpAPI'),
                    'published': self._parse_date(r.get('date', ''))
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} results from SerpAPI")
            return results
            
        except ImportError:
            logger.error("serpapi not installed. Install with: pip install google-search-results")
            return []
        except Exception as e:
            logger.error(f"SerpAPI search error: {e}")
            return []
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        stop_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go',
            'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them',
            'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over',
            'think', 'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us', 'is', 'are', 'was', 'were', 'been', 'being', 'has', 'had', 'have',
            'do', 'does', 'did', 'will', 'would', 'should', 'may', 'might', 'must', 'can', 'could'
        }
        
        filtered_words = [w for w in words if w not in stop_words and len(w) >= min_length]
        word_counts = Counter(filtered_words)
        keywords = [word for word, count in word_counts.most_common(20)]
        return keywords
    
    def score_relevance(self, results: List[Dict], keywords: List[str]) -> List[Dict]:
        """Score search results by relevance to keywords."""
        if not keywords:
            for result in results:
                result['relevance_score'] = 1.0  # Default score if no keywords
            return results
        
        keyword_set = set(kw.lower() for kw in keywords)
        
        for result in results:
            score = 0.0
            
            # Check title (weight: 3)
            title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', result.get('title', '').lower()))
            title_matches = len(title_words & keyword_set)
            score += title_matches * 3
            
            # Check snippet (weight: 2)
            snippet_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', result.get('snippet', '').lower()))
            snippet_matches = len(snippet_words & keyword_set)
            score += snippet_matches * 2
            
            # Normalize score (0-1 range)
            max_possible = len(keywords) * 5  # Rough estimate
            normalized_score = min(score / max(max_possible, 1), 1.0)
            
            result['relevance_score'] = normalized_score
        
        # Sort by relevance score (descending)
        sorted_results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)
        return sorted_results
    
    def get_relevant_search_results(
        self,
        market_question: str,
        num_results: int = 5,
        enhance_query: bool = True
    ) -> List[Dict]:
        """
        Get relevant search results for a market question.
        
        Args:
            market_question: The market question/query
            num_results: Number of results to return
            enhance_query: Whether to enhance the query with additional context
            
        Returns:
            List of relevant search result dictionaries
        """
        # Enhance query if requested
        if enhance_query:
            # Add context to make search more relevant
            query = f"{market_question} news recent"
        else:
            query = market_question
        
        logger.info(f"Searching for: {query}")
        
        # Perform search (get more results than needed for scoring)
        search_results = self.search(query, num_results=num_results * 2)
        
        if not search_results:
            logger.warning("No search results found")
            return []
        
        # Extract keywords from market question
        keywords = self.extract_keywords(market_question)
        logger.info(f"Extracted keywords: {keywords[:10]}")
        
        # Score relevance
        scored_results = self.score_relevance(search_results, keywords)
        
        # Return top N
        top_results = scored_results[:num_results]
        
        logger.info(f"Returning {len(top_results)} relevant search results")
        return top_results
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from search results."""
        if not date_str:
            return datetime.now()
        
        try:
            # Try common formats
            for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%B %d, %Y']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return datetime.now()
        except Exception:
            return datetime.now()
    
    def _get_cached_search(self, query: str) -> Optional[List[Dict]]:
        """Get cached search results if available and not expired."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"search_{hash(query)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cache_time_str = cache_data.get('timestamp', '')
            if cache_time_str:
                cache_time = datetime.fromisoformat(cache_time_str)
                if cache_time.tzinfo is not None:
                    cache_time = cache_time.replace(tzinfo=None)
                
                if datetime.now() - cache_time > timedelta(seconds=self.cache_ttl):
                    return None
            
            # Restore results
            results = cache_data.get('results', [])
            # Convert date strings back to datetime
            for result in results:
                if isinstance(result.get('published'), str):
                    try:
                        dt = datetime.fromisoformat(result['published'])
                        if dt.tzinfo is not None:
                            dt = dt.replace(tzinfo=None)
                        result['published'] = dt
                    except (ValueError, AttributeError):
                        result['published'] = datetime.now()
            
            return results
        except Exception as e:
            logger.debug(f"Error reading cache: {e}")
            return None
    
    def _cache_search(self, query: str, results: List[Dict]):
        """Cache search results."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"search_{hash(query)}.json"
        
        try:
            # Convert datetime objects to ISO strings
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                published = serializable_result.get('published')
                if isinstance(published, datetime):
                    if published.tzinfo is not None:
                        published = published.replace(tzinfo=None)
                    serializable_result['published'] = published.isoformat()
                serializable_results.append(serializable_result)
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'results': serializable_results
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.debug(f"Error caching search: {e}")


def get_relevant_search_results(
    market_question: str,
    provider: str = "duckduckgo",
    api_key: Optional[str] = None,
    num_results: int = 5,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to get relevant search results.
    
    Args:
        market_question: The market question/query
        provider: Search provider ("duckduckgo" or "serpapi")
        api_key: API key for providers that require it
        num_results: Number of results to return
        cache_dir: Directory to cache search results
        
    Returns:
        List of relevant search result dictionaries
    """
    retriever = SearchRetriever(provider=provider, api_key=api_key, cache_dir=cache_dir)
    return retriever.get_relevant_search_results(market_question, num_results=num_results)

