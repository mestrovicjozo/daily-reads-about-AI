#!/usr/bin/env python3
"""
Test script to validate all configured feeds are working.
Run this to debug feed issues before running the full digest generation.
"""
import feedparser
import requests
import time
import sys
import logging
import yaml
from urllib.parse import urlparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("feed-tester")

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_feed(feed_url: str) -> dict:
    """Test a single feed and return results"""
    result = {
        'url': feed_url,
        'status': 'unknown',
        'http_status': None,
        'response_length': 0,
        'entries_count': 0,
        'first_title': None,
        'error': None
    }
    
    try:
        logger.info(f"Testing feed: {feed_url}")
        headers = {
            "User-Agent": "DigestAgent/1.0 (+https://github.com/yourusername/digest-agent)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        r = requests.get(feed_url, timeout=15, headers=headers)
        result['http_status'] = r.status_code
        result['response_length'] = len(r.text)
        
        logger.info(f"HTTP {r.status_code} length={len(r.text)}")
        
        if r.status_code == 200:
            parsed = feedparser.parse(r.text)
            result['entries_count'] = len(parsed.entries)
            logger.info(f"Parsed entries: {len(parsed.entries)}")
            
            if parsed.entries:
                first_entry = parsed.entries[0]
                result['first_title'] = first_entry.get("title", "No title")
                logger.info(f"Top title: {result['first_title']}")
                result['status'] = 'success'
            else:
                logger.warning(f"No entries found in feed: {feed_url}")
                result['status'] = 'no_entries'
        else:
            logger.warning(f"Non-200 status: {feed_url} - {r.status_code}")
            result['status'] = 'http_error'
            
    except Exception as e:
        logger.exception(f"Failed feed {feed_url}: {e}")
        result['error'] = str(e)
        result['status'] = 'exception'
    
    return result

def test_html_source(html_url: str) -> dict:
    """Test an HTML-only source"""
    result = {
        'url': html_url,
        'status': 'unknown',
        'http_status': None,
        'response_length': 0,
        'links_found': 0,
        'error': None
    }
    
    try:
        logger.info(f"Testing HTML source: {html_url}")
        headers = {
            "User-Agent": "DigestAgent/1.0 (+https://github.com/yourusername/digest-agent)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        r = requests.get(html_url, timeout=15, headers=headers)
        result['http_status'] = r.status_code
        result['response_length'] = len(r.text)
        
        if r.status_code == 200:
            # Simple link counting (would need BeautifulSoup for proper parsing)
            link_count = r.text.count('<a href=')
            result['links_found'] = link_count
            logger.info(f"HTML source {html_url}: {link_count} potential links found")
            result['status'] = 'success'
        else:
            logger.warning(f"HTML source failed: {html_url} - {r.status_code}")
            result['status'] = 'http_error'
            
    except Exception as e:
        logger.exception(f"Failed HTML source {html_url}: {e}")
        result['error'] = str(e)
        result['status'] = 'exception'
    
    return result

def main():
    """Test all configured feeds"""
    config = load_config()
    
    # Collect all feed URLs
    feed_urls = []
    html_urls = []
    
    for topic in config.get('topics', []):
        topic_name = topic['name']
        logger.info(f"\n=== Testing topic: {topic_name} ===")
        
        for url in topic.get('seed_urls', []):
            if any(url.endswith(ext) for ext in ['.rss', '.rdf', '.xml', '/feed', '/rss']):
                feed_urls.append(url)
            else:
                html_urls.append(url)
    
    # Test RSS feeds
    logger.info(f"\n=== Testing {len(feed_urls)} RSS feeds ===")
    feed_results = []
    for feed_url in feed_urls:
        result = test_feed(feed_url)
        feed_results.append(result)
        time.sleep(0.5)  # Rate limiting
    
    # Test HTML sources
    logger.info(f"\n=== Testing {len(html_urls)} HTML sources ===")
    html_results = []
    for html_url in html_urls:
        result = test_html_source(html_url)
        html_results.append(result)
        time.sleep(0.5)  # Rate limiting
    
    # Summary
    logger.info(f"\n=== SUMMARY ===")
    
    successful_feeds = [r for r in feed_results if r['status'] == 'success' and r['entries_count'] > 0]
    failed_feeds = [r for r in feed_results if r['status'] != 'success' or r['entries_count'] == 0]
    
    successful_html = [r for r in html_results if r['status'] == 'success']
    failed_html = [r for r in html_results if r['status'] != 'success']
    
    logger.info(f"RSS Feeds: {len(successful_feeds)} successful, {len(failed_feeds)} failed")
    logger.info(f"HTML Sources: {len(successful_html)} successful, {len(failed_html)} failed")
    
    if failed_feeds:
        logger.warning("Failed RSS feeds:")
        for result in failed_feeds:
            logger.warning(f"  {result['url']}: {result['status']} - {result.get('error', 'No error details')}")
    
    if failed_html:
        logger.warning("Failed HTML sources:")
        for result in failed_html:
            logger.warning(f"  {result['url']}: {result['status']} - {result.get('error', 'No error details')}")
    
    total_entries = sum(r['entries_count'] for r in successful_feeds)
    logger.info(f"Total entries available: {total_entries}")
    
    return len(successful_feeds) > 0 or len(successful_html) > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
