"""
Prompt augmentation module for RAG-style inference.

Adds relevant news articles to prompts before model inference.
"""

import logging
from typing import Optional, List, Dict, Tuple
from .search_retriever import SearchRetriever, get_relevant_search_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def format_search_result(result: Dict, max_length: int = 300) -> str:
    """
    Format a single search result for inclusion in prompt.
    
    Args:
        result: Search result dictionary with title, snippet, link, source
        max_length: Maximum length of snippet to include
        
    Returns:
        Formatted search result string
    """
    title = result.get('title', 'Untitled')
    snippet = result.get('snippet', '') or result.get('description', '')
    link = result.get('link', '')
    source = result.get('source', 'Unknown Source')
    published = result.get('published', None)
    
    # Truncate snippet if too long
    if len(snippet) > max_length:
        snippet = snippet[:max_length] + "..."
    
    # Format date
    date_str = ""
    if published:
        if isinstance(published, str):
            date_str = published[:10]  # Just the date part
        else:
            date_str = published.strftime('%Y-%m-%d') if hasattr(published, 'strftime') else str(published)[:10]
    
    # Format result
    if date_str:
        header = f"{title} - {source} ({date_str})"
    else:
        header = f"{title} - {source}"
    
    if link:
        header += f" [{link}]"
    
    if snippet:
        return f"{header}\n{snippet}"
    else:
        return header


def augment_prompt_with_search(
    instruction: str,
    input_text: str,
    provider: str = "duckduckgo",
    api_key: Optional[str] = None,
    num_results: int = 5,
    search_retriever: Optional[SearchRetriever] = None,
    cache_dir: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Augment prompt with relevant search results.
    
    Args:
        instruction: Original instruction text
        input_text: Original input text (contains market question/data)
        provider: Search provider ("duckduckgo" or "serpapi")
        api_key: API key for providers that require it
        num_results: Number of search results to include
        search_retriever: Optional SearchRetriever instance (creates new one if None)
        cache_dir: Directory to cache search results
        
    Returns:
        Tuple[str, List[Dict]]: (augmented_input_text, retrieved_results)
    """
    # Extract market question from input text
    market_question = extract_market_question(input_text)
    
    if not market_question:
        logger.warning("Could not extract market question, skipping search")
        return input_text, []
    
    # Retrieve relevant search results
    try:
        if search_retriever is None:
            search_retriever = SearchRetriever(provider=provider, api_key=api_key, cache_dir=cache_dir)
        
        results = search_retriever.get_relevant_search_results(
            market_question,
            num_results=num_results
        )
        
        if not results:
            logger.info("No relevant search results found")
            return input_text, []
        
        # Format search results section
        search_section = format_search_section(results)
        
        # Combine with original input
        # Add instruction to consider search results in the reasoning
        augmented_input = f"{input_text}\n\nRelevant Information from Web Search:\n{search_section}\n\nPlease analyze the above search results and explain how they inform your prediction. Consider the credibility of sources, recency of information, and how the information relates to the market question."
        
        logger.info(f"Augmented prompt with {len(results)} search results")
        return augmented_input, results
        
    except Exception as e:
        logger.error(f"Error retrieving search results: {e}")
        logger.info("Falling back to prompt without search results")
        return input_text, []


# Keep old function name for backward compatibility
def augment_prompt_with_news(
    instruction: str,
    input_text: str,
    feed_urls: List[str] = None,
    num_articles: int = 3,
    news_retriever: Optional = None,
    cache_dir: Optional[str] = None
) -> Tuple[str, List[Dict]]:
    """
    Deprecated: Use augment_prompt_with_search instead.
    Kept for backward compatibility.
    """
    logger.warning("augment_prompt_with_news is deprecated. Use augment_prompt_with_search instead.")
    return augment_prompt_with_search(
        instruction, input_text,
        provider="duckduckgo",
        num_results=num_articles,
        cache_dir=cache_dir
    )


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


def format_search_section(results: List[Dict], max_length: int = 300) -> str:
    """
    Format a list of search results into a section for the prompt.
    
    Args:
        results: List of search result dictionaries
        max_length: Maximum length per result snippet
        
    Returns:
        Formatted search results section string
    """
    if not results:
        return "No relevant information found."
    
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted = format_search_result(result, max_length=max_length)
        formatted_results.append(f"[Result {i}]\n{formatted}")
    
    return "\n\n".join(formatted_results)

