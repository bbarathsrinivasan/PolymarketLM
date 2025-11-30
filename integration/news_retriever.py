"""
News retrieval module for RAG-style inference.

Fetches news articles from RSS feeds and matches them to market queries.
"""

import feedparser
import requests
import re
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import Counter
import logging
from pathlib import Path
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsRetriever:
    """Retrieve and score relevant news articles from RSS feeds."""
    
    def __init__(self, feed_urls: List[str], cache_dir: Optional[str] = None, cache_ttl: int = 3600):
        """
        Initialize news retriever.
        
        Args:
            feed_urls: List of RSS feed URLs to fetch from
            cache_dir: Directory to cache RSS feed results (None to disable caching)
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.feed_urls = feed_urls
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_ttl = cache_ttl
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_rss_feeds(self, feed_urls: Optional[List[str]] = None) -> List[Dict]:
        """
        Fetch and parse RSS feeds.
        
        Args:
            feed_urls: List of RSS feed URLs (uses self.feed_urls if None)
            
        Returns:
            List of article dictionaries with keys: title, description, link, published, source
        """
        if feed_urls is None:
            feed_urls = self.feed_urls
        
        all_articles = []
        
        for feed_url in feed_urls:
            try:
                # Check cache first
                cached_articles = self._get_cached_feed(feed_url)
                if cached_articles:
                    logger.info(f"Using cached feed: {feed_url}")
                    all_articles.extend(cached_articles)
                    continue
                
                # Fetch feed
                logger.info(f"Fetching RSS feed: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                if feed.bozo and feed.bozo_exception:
                    logger.warning(f"Error parsing feed {feed_url}: {feed.bozo_exception}")
                    continue
                
                # Parse articles
                articles = []
                for entry in feed.entries[:50]:  # Limit to 50 most recent articles per feed
                    article = {
                        'title': entry.get('title', ''),
                        'description': entry.get('description', '') or entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': self._parse_date(entry.get('published', '')),
                        'source': feed.feed.get('title', feed_url) or feed_url
                    }
                    articles.append(article)
                
                # Cache the feed
                self._cache_feed(feed_url, articles)
                all_articles.extend(articles)
                
                # Be polite to RSS servers
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error fetching feed {feed_url}: {e}")
                continue
        
        logger.info(f"Fetched {len(all_articles)} articles from {len(feed_urls)} feeds")
        return all_articles
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from text using simple word frequency.
        
        Args:
            text: Input text to extract keywords from
            min_length: Minimum keyword length
            
        Returns:
            List of keywords (sorted by importance)
        """
        if not text:
            return []
        
        # Convert to lowercase and split
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Remove common stop words
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
        
        # Filter stop words and count
        filtered_words = [w for w in words if w not in stop_words and len(w) >= min_length]
        word_counts = Counter(filtered_words)
        
        # Return top keywords
        keywords = [word for word, count in word_counts.most_common(20)]
        return keywords
    
    def score_relevance(self, articles: List[Dict], keywords: List[str]) -> List[Dict]:
        """
        Score articles by relevance to keywords.
        
        Args:
            articles: List of article dictionaries
            keywords: List of keywords to match against
            
        Returns:
            List of articles with 'relevance_score' added, sorted by score (descending)
        """
        if not keywords:
            # If no keywords, return articles with score 0
            for article in articles:
                article['relevance_score'] = 0
            return sorted(articles, key=lambda x: x.get('published', datetime.min), reverse=True)
        
        keyword_set = set(kw.lower() for kw in keywords)
        
        for article in articles:
            score = 0
            
            # Check title (weight: 3)
            title_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', article.get('title', '').lower()))
            title_matches = len(title_words & keyword_set)
            score += title_matches * 3
            
            # Check description (weight: 1)
            desc_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', article.get('description', '').lower()))
            desc_matches = len(desc_words & keyword_set)
            score += desc_matches * 1
            
            # Bonus for multiple keyword matches
            if title_matches + desc_matches > 1:
                score += 2
            
            article['relevance_score'] = score
        
        # Sort by relevance score (descending), then by date (descending)
        sorted_articles = sorted(
            articles,
            key=lambda x: (x['relevance_score'], x.get('published', datetime.min)),
            reverse=True
        )
        
        return sorted_articles
    
    def get_relevant_news(
        self,
        market_question: str,
        num_articles: int = 3,
        max_age_days: int = 7
    ) -> List[Dict]:
        """
        Get relevant news articles for a market question.
        
        Args:
            market_question: The market question/query
            num_articles: Number of articles to return
            max_age_days: Maximum age of articles in days
            
        Returns:
            List of relevant article dictionaries
        """
        # Extract keywords
        keywords = self.extract_keywords(market_question)
        logger.info(f"Extracted keywords: {keywords[:10]}")
        
        # Fetch articles
        articles = self.fetch_rss_feeds()
        
        # Filter by age
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        recent_articles = [
            article for article in articles
            if article.get('published', datetime.min) >= cutoff_date
        ]
        logger.info(f"Found {len(recent_articles)} articles from last {max_age_days} days")
        
        # Score relevance
        scored_articles = self.score_relevance(recent_articles, keywords)
        
        # Return top N
        top_articles = scored_articles[:num_articles]
        
        # Filter out articles with zero relevance if we have enough
        if len(top_articles) < num_articles:
            # Include some zero-score articles if needed
            pass
        else:
            # Only return articles with some relevance
            top_articles = [a for a in top_articles if a['relevance_score'] > 0]
            if len(top_articles) < num_articles:
                # Fill with highest scored articles regardless of score
                top_articles = scored_articles[:num_articles]
        
        logger.info(f"Returning {len(top_articles)} relevant articles")
        return top_articles
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string from RSS feed."""
        if not date_str:
            return datetime.now()
        
        try:
            # Try parsing with feedparser's parsed date
            if hasattr(date_str, 'timetuple'):
                return datetime(*date_str.timetuple()[:6])
            
            # Try common formats
            for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%a, %d %b %Y %H:%M:%S %Z', '%Y-%m-%dT%H:%M:%S%z']:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # Fallback to current time
            return datetime.now()
        except Exception:
            return datetime.now()
    
    def _get_cached_feed(self, feed_url: str) -> Optional[List[Dict]]:
        """Get cached feed if available and not expired."""
        if not self.cache_dir:
            return None
        
        cache_file = self.cache_dir / f"{hash(feed_url)}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > timedelta(seconds=self.cache_ttl):
                return None
            
            return cache_data['articles']
        except Exception as e:
            logger.debug(f"Error reading cache: {e}")
            return None
    
    def _cache_feed(self, feed_url: str, articles: List[Dict]):
        """Cache feed results."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{hash(feed_url)}.json"
        
        try:
            # Convert datetime objects to ISO strings for JSON
            serializable_articles = []
            for article in articles:
                serializable_article = article.copy()
                if isinstance(serializable_article.get('published'), datetime):
                    serializable_article['published'] = serializable_article['published'].isoformat()
                serializable_articles.append(serializable_article)
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'articles': serializable_articles
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.debug(f"Error caching feed: {e}")


def get_relevant_news(
    market_question: str,
    feed_urls: List[str],
    num_articles: int = 3,
    cache_dir: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function to get relevant news.
    
    Args:
        market_question: The market question/query
        feed_urls: List of RSS feed URLs
        num_articles: Number of articles to return
        cache_dir: Directory to cache RSS feed results
        
    Returns:
        List of relevant article dictionaries
    """
    retriever = NewsRetriever(feed_urls, cache_dir=cache_dir)
    return retriever.get_relevant_news(market_question, num_articles=num_articles)

