"""
Integration module for PolymarketLM.

Provides search-based retrieval and prompt augmentation for RAG-style inference.
"""

from .search_retriever import SearchRetriever, get_relevant_search_results
from .prompt_augmenter import augment_prompt_with_search, augment_prompt_with_news

__all__ = [
    'SearchRetriever',
    'get_relevant_search_results',
    'augment_prompt_with_search',
    'augment_prompt_with_news',  # For backward compatibility
]

