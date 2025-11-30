"""
Configuration for news retrieval and RAG integration.
"""

from pathlib import Path
from typing import List
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default RSS feed URLs (financial, political, crypto news)
DEFAULT_RSS_FEEDS = [
    # Financial News
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    
    # Political News
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.cnn.com/rss/edition.rss",
    
    # Crypto News
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    
    # General News
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]

# Default configuration
DEFAULT_CONFIG = {
    'rss_feeds': DEFAULT_RSS_FEEDS,
    'num_articles': 3,
    'article_max_length': 200,
    'cache_dir': 'integration/.news_cache',
    'cache_ttl': 3600,  # 1 hour
    'max_age_days': 7,
}


def load_config(config_path: str = "integration/news_config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.info(f"Config file not found at {config_path}, using defaults")
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(config_file, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        
        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(user_config)
        
        logger.info(f"Loaded config from {config_path}")
        return config
        
    except Exception as e:
        logger.warning(f"Error loading config: {e}, using defaults")
        return DEFAULT_CONFIG.copy()


def get_rss_feeds(config: dict = None) -> List[str]:
    """Get RSS feed URLs from config."""
    if config is None:
        config = load_config()
    return config.get('rss_feeds', DEFAULT_RSS_FEEDS)


def get_num_articles(config: dict = None) -> int:
    """Get number of articles to retrieve from config."""
    if config is None:
        config = load_config()
    return config.get('num_articles', 3)


def get_cache_dir(config: dict = None) -> str:
    """Get cache directory from config."""
    if config is None:
        config = load_config()
    return config.get('cache_dir', 'integration/.news_cache')

