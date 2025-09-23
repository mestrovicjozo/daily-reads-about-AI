"""
Utility functions for the Daily Tech Digest Agent.
"""
import os
import sys
import json
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
from urllib.parse import urlparse

import requests
from loguru import logger
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set default values if not specified
    config.setdefault('crawler', {})
    config['crawler'].setdefault('user_agent', 'Mozilla/5.0 (compatible; DigestAgent/1.0)')
    config['crawler'].setdefault('request_timeout', 10)
    config['crawler'].setdefault('max_retries', 3)
    config['crawler'].setdefault('rate_limit_delay', 1.0)
    config['crawler'].setdefault('max_articles_per_topic', 100)
    
    config.setdefault('processing', {})
    config['processing'].setdefault('min_article_length', 500)
    config['processing'].setdefault('max_article_length', 20000)
    config['processing'].setdefault('summary_length', 5)
    config['processing'].setdefault('language', 'en')
    config['processing'].setdefault('dedupe_threshold', 0.9)
    
    config.setdefault('output', {})
    config['output'].setdefault('digest_dir', 'digests')
    config['output'].setdefault('provenance_dir', 'provenance')
    config['output'].setdefault('date_format', '%Y-%m-%d')
    config['output'].setdefault('max_articles_per_topic', 5)
    
    config.setdefault('ranking', {})
    config['ranking'].setdefault('recency', 0.4)
    config['ranking'].setdefault('relevance', 0.3)
    config['ranking'].setdefault('authoritativeness', 0.2)
    config['ranking'].setdefault('engagement', 0.1)
    
    return config


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with loguru."""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def get_content_hash(content: str) -> str:
    """Generate a consistent hash for content deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def normalize_url(url: str) -> str:
    """Normalize URL to improve matching."""
    parsed = urlparse(url)
    # Remove common tracking parameters
    query = '&'.join(
        f"{k}={v}" for k, v in 
        [(k, v) for k, v in [p.split('=') for p in parsed.query.split('&')] 
         if k and k not in ['utm_', 'ref_', 'source']]
    ) if parsed.query else ''
    
    # Rebuild URL without tracking parameters
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{'?' + query if query else ''}"


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse a date string into a datetime object."""
    if not date_str:
        return None
    try:
        return date_parser.parse(date_str, ignoretz=True)
    except (ValueError, TypeError):
        return None


def setup_directories(config: Dict[str, Any]) -> None:
    """Ensure all required directories exist."""
    os.makedirs(config['output']['digest_dir'], exist_ok=True)
    os.makedirs(config['output']['provenance_dir'], exist_ok=True)


def save_provenance(data: Dict[str, Any], config: Dict[str, Any], target_date: date) -> Path:
    """Save provenance data to a JSON file."""
    provenance_dir = Path(config['output']['provenance_dir'])
    provenance_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{target_date.strftime(config['output']['date_format'])}.json"
    filepath = provenance_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filepath


def load_provenance(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load provenance data from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_clean_text(html: str) -> str:
    """Extract clean text from HTML content."""
    if not html:
        return ""
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text and clean up whitespace
    text = soup.get_text(separator=' ', strip=True)
    
    # Normalize whitespace
    return ' '.join(text.split())


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes."""
    if not text:
        return 0
    
    word_count = len(text.split())
    return max(1, round(word_count / wpm))
