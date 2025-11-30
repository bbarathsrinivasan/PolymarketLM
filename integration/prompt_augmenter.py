"""
Prompt augmentation module for RAG-style inference.

Adds relevant news articles to prompts before model inference.
"""

import logging
from typing import Optional, List, Dict, Tuple
from .news_retriever import NewsRetriever, get_relevant_news

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_article(article: Dict, max_length: int = 200) -> str:
    """
    Format a single article for inclusion in prompt.
    
    Args:
        article: Article dictionary with title, description, source, published
        max_length: Maximum length of description to include
        
    Returns:
        Formatted article string
    """
    title = article.get('title', 'Untitled')
    description = article.get('description', '')
    source = article.get('source', 'Unknown Source')
    published = article.get('published', None)
    
    # Truncate description if too long
    if len(description) > max_length:
        description = description[:max_length] + "..."
    
    # Format date
    date_str = ""
    if published:
        if isinstance(published, str):
            date_str = published[:10]  # Just the date part
        else:
            date_str = published.strftime('%Y-%m-%d') if hasattr(published, 'strftime') else str(published)[:10]
    
    # Format article
    if date_str:
        header = f"{title} - {source} ({date_str})"
    else:
        header = f"{title} - {source}"
    
    if description:
        return f"{header}\n{description}"
    else:
        return header


def augment_prompt_with_news(
    instruction: str,
    input_text: str,
    feed_urls: List[str],
    num_articles: int = 3,
    news_retriever: Optional[NewsRetriever] = None,
    cache_dir: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Augment prompt with relevant news articles.
    
    Args:
        instruction: Original instruction text
        input_text: Original input text (contains market question/data)
        feed_urls: List of RSS feed URLs to search
        num_articles: Number of news articles to include
        news_retriever: Optional NewsRetriever instance (creates new one if None)
        cache_dir: Directory to cache RSS feed results
        
    Returns:
        Tuple[str, List[Dict]]: (augmented_input_text, retrieved_articles)
    """
    # Extract market question from input text
    market_question = extract_market_question(input_text)
    
    if not market_question:
        logger.warning("Could not extract market question, skipping news retrieval")
        return input_text, []
    
    # Retrieve relevant news
    try:
        if news_retriever is None:
            news_retriever = NewsRetriever(feed_urls, cache_dir=cache_dir)
        
        articles = news_retriever.get_relevant_news(
            market_question,
            num_articles=num_articles
        )
        
        if not articles:
            logger.info("No relevant news articles found")
            return input_text, []
        
        # Format news section
        news_section = format_news_section(articles)
        
        # Combine with original input
        # Add instruction to consider news in the reasoning
        augmented_input = f"{input_text}\n\nRelevant News:\n{news_section}\n\nPlease consider the above news articles when making your prediction and explain how they influence your reasoning."
        
        logger.info(f"Augmented prompt with {len(articles)} news articles")
        return augmented_input, articles
        
    except Exception as e:
        logger.error(f"Error retrieving news: {e}")
        logger.info("Falling back to prompt without news")
        return input_text, []


def extract_market_question(input_text: str) -> str:
    """
    Extract market question from input text.
    
    Looks for patterns like "Question: ..." or extracts from market data.
    
    Args:
        input_text: Input text containing market information
        
    Returns:
        Extracted market question string
    """
    if not input_text:
        return ""
    
    # Try to find "Question:" pattern
    question_patterns = [
        r'Question:\s*(.+?)(?:\n|$)',
        r'Market Question:\s*(.+?)(?:\n|$)',
        r'question:\s*(.+?)(?:\n|$)',
    ]
    
    for pattern in question_patterns:
        import re
        match = re.search(pattern, input_text, re.IGNORECASE | re.MULTILINE)
        if match:
            question = match.group(1).strip()
            if question and len(question) > 10:  # Reasonable question length
                return question
    
    # Fallback: use first few lines or whole text if short
    lines = input_text.split('\n')
    if len(lines) > 0:
        # Try to find a line that looks like a question
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and ('?' in line or len(line) > 20):
                # Remove common prefixes
                for prefix in ['Question:', 'Market ID:', 'Market Question:']:
                    if line.lower().startswith(prefix.lower()):
                        line = line[len(prefix):].strip()
                        break
                if line:
                    return line
    
    # Last resort: return first meaningful line
    for line in lines:
        line = line.strip()
        if line and len(line) > 10:
            return line
    
    return input_text[:200]  # Fallback to first 200 chars


def format_news_section(articles: List[Dict], max_length: int = 200) -> str:
    """
    Format a list of articles into a news section for the prompt.
    
    Args:
        articles: List of article dictionaries
        max_length: Maximum length per article description
        
    Returns:
        Formatted news section string
    """
    if not articles:
        return "No relevant news found."
    
    formatted_articles = []
    for i, article in enumerate(articles, 1):
        formatted = format_article(article, max_length=max_length)
        formatted_articles.append(formatted)
    
    return "\n\n".join(formatted_articles)

