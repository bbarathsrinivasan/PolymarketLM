"""
Configuration for news retrieval and RAG integration.
"""

from pathlib import Path
from typing import List
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Default search configuration
DEFAULT_CONFIG = {
    'search_provider': 'duckduckgo',  # Options: 'duckduckgo' or 'serpapi'
    'serpapi_key': None,  # Set your SerpAPI key here or in environment variable SERPAPI_KEY
    'num_results': 5,
    'result_max_length': 300,
    'cache_dir': 'integration/.search_cache',
    'cache_ttl': 3600,  # 1 hour
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


def get_search_provider(config: dict = None) -> str:
    """Get search provider from config."""
    if config is None:
        config = load_config()
    return config.get('search_provider', 'duckduckgo')


def get_serpapi_key(config: dict = None) -> Optional[str]:
    """Get SerpAPI key from config or environment variable."""
    import os
    if config is None:
        config = load_config()
    # Check environment variable first
    env_key = os.getenv('SERPAPI_KEY')
    if env_key:
        return env_key
    return config.get('serpapi_key')


def get_num_results(config: dict = None) -> int:
    """Get number of search results to retrieve from config."""
    if config is None:
        config = load_config()
    return config.get('num_results', 5)


def get_cache_dir(config: dict = None) -> str:
    """Get cache directory from config."""
    if config is None:
        config = load_config()
    return config.get('cache_dir', 'integration/.search_cache')

