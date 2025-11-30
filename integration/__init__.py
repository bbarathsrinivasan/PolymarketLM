"""
Integration module for PolymarketLM.

Provides news retrieval and prompt augmentation for RAG-style inference.
"""

from .news_retriever import NewsRetriever, get_relevant_news
from .prompt_augmenter import augment_prompt_with_news

__all__ = [
    'NewsRetriever',
    'get_relevant_news',
    'augment_prompt_with_news',
]

