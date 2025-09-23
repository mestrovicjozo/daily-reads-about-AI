"""
Web crawler for fetching articles from various sources.
"""
import time
import random
import feedparser
from typing import Dict, List, Optional, Set, Any, Tuple
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin

import requests
from loguru import logger
from newspaper import Article, ArticleException
from bs4 import BeautifulSoup

from utils import get_content_hash, normalize_url, parse_date, get_clean_text
from sources_reddit import RedditSource


class WebCrawler:
    """Crawler for fetching and processing articles from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the crawler with configuration."""
        self.config = config
        self.user_agent = config['crawler']['user_agent']
        self.request_timeout = config['crawler']['request_timeout']
        self.max_retries = config['crawler']['max_retries']
        self.rate_limit_delay = config['crawler']['rate_limit_delay']
        self.max_articles_per_topic = config['crawler']['max_articles_per_topic']
        self.seen_urls: Set[str] = set()
        
        # Initialize session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })

        # Allowlist rules for HTML blog indexes (non-RSS)
        self.allowlist_rules: Dict[str, Dict[str, Any]] = {
            'cohere.com': {
                'index_paths': ['/blog'],
                'link_contains': ['/blog/'],
            },
            'weaviate.io': {
                'index_paths': ['/blog'],
                'link_contains': ['/blog/'],
            },
            'pinecone.io': {
                'index_paths': ['/learn'],
                'link_contains': ['/learn/'],
            },
            'milvus.io': {
                'index_paths': ['/blog'],
                'link_contains': ['/blog/'],
            },
            'langchain.dev': {
                'index_paths': ['/blog'],
                'link_contains': ['/blog/'],
            },
            'quantumai.google': {
                'index_paths': ['/'],
                'link_contains': ['/'],
            },
            'quantamagazine.org': {
                'index_paths': ['/physics/quantum-computing/'],
                'link_contains': ['/physics/'],
            },
            'research.ibm.com': {
                'index_paths': ['/blog/category/quantum'],
                'link_contains': ['/blog/'],
            },
            'cloudblogs.microsoft.com': {
                'index_paths': ['/quantum'],
                'link_contains': ['/quantum/'],
            },
            'rigetti.com': {
                'index_paths': ['/news'],
                'link_contains': ['/news/'],
            },
            'ionq.com': {
                'index_paths': ['/newsroom'],
                'link_contains': ['/newsroom/'],
            },
            'xanadu.ai': {
                'index_paths': ['/blog'],
                'link_contains': ['/blog/'],
            },
        }
    
    def crawl(self) -> List[Dict[str, Any]]:
        """Crawl all configured sources and return articles."""
        articles = []

        # Fetch Reddit once per run, then distribute to topics
        reddit_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        try:
            reddit_source = RedditSource(self.config)
            reddit_items = reddit_source.fetch()
            for r in reddit_items:
                t = r.get('topic')
                if not t:
                    continue
                reddit_by_topic.setdefault(t, []).append(r)
            if reddit_items:
                logger.info(f"Reddit contributed {len(reddit_items)} items across topics")
        except Exception as e:
            logger.warning(f"Reddit integration error: {e}")
        
        for topic in self.config['topics']:
            logger.info(f"Crawling topic: {topic['name']}")
            
            # Collect articles from all seed URLs for this topic
            topic_articles = []
            for url in topic.get('seed_urls', []):
                try:
                    if any(url.endswith(ext) for ext in ['.rss', '.rdf', '.xml', '/feed', '/rss']):
                        # Handle RSS/Atom feeds
                        feed_articles = self._parse_feed(url, topic)
                        topic_articles.extend(feed_articles)
                    else:
                        # Handle regular web pages (allowlist index crawling)
                        page_articles = self._crawl_webpage(url, topic)
                        topic_articles.extend(page_articles)
                    
                    # Respect rate limiting
                    time.sleep(self.rate_limit_delay)
                    
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    continue
            
            # Include Reddit items for this topic (if any), every run
            if topic['name'] in reddit_by_topic:
                topic_articles.extend(reddit_by_topic[topic['name']])
                logger.info(f"Added {len(reddit_by_topic[topic['name']])} Reddit items to topic {topic['name']}")
            
            # Deduplicate and limit articles per topic
            unique_articles = self._deduplicate_articles(topic_articles)
            topic_articles = sorted(
                unique_articles,
                key=lambda x: x.get('published', datetime.min),
                reverse=True
            )[:self.max_articles_per_topic]
            
            articles.extend(topic_articles)
            logger.info(f"Found {len(topic_articles)} articles for {topic['name']}")
        
        return articles
    
    def _parse_feed(self, feed_url: str, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse an RSS/Atom feed and return articles."""
        try:
            feed = feedparser.parse(feed_url)
            if feed.bozo:
                logger.warning(f"Feed parsing error for {feed_url}: {feed.bozo_exception}")
                return []
            
            articles = []
            for entry in feed.entries:
                try:
                    # Skip if we've already seen this URL
                    url = normalize_url(entry.get('link', ''))
                    if not url or url in self.seen_urls:
                        continue
                    
                    # Extract article metadata
                    published = parse_date(entry.get('published') or entry.get('updated') or entry.get('dc:date'))
                    if not published:
                        published = datetime.utcnow()
                    
                    # Create article object
                    article = {
                        'url': url,
                        'title': entry.get('title', '').strip(),
                        'summary': entry.get('summary', '').strip(),
                        'published': published,
                        'authors': self._extract_authors(entry),
                        'source': urlparse(url).netloc,
                        'topic': topic['name'],
                        'topic_keywords': topic['keywords'],
                        'content': '',  # Will be fetched later if needed
                        'content_hash': '',
                        'fetch_time': datetime.utcnow(),
                        'metadata': {
                            'feed_url': feed_url,
                            'feed_title': feed.feed.get('title', '')
                        }
                    }
                    
                    # Mark URL as seen
                    self.seen_urls.add(url)
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Error processing feed entry {entry.get('link')}: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching feed {feed_url}: {e}")
            return []
    
    def _crawl_webpage(self, url: str, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Crawl a webpage and extract articles.
        For allowlisted domains, crawl the index page and collect article links; otherwise, no-op.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            rules = self.allowlist_rules.get(domain)
            if not rules:
                return []

            base = f"{parsed.scheme}://{parsed.netloc}"
            articles: List[Dict[str, Any]] = []

            for path in rules.get('index_paths', []):
                index_url = urljoin(base, path)
                resp = self.session.get(index_url, timeout=self.request_timeout)
                if resp.status_code != 200:
                    continue
                soup = BeautifulSoup(resp.text, 'html.parser')
                anchors = soup.find_all('a', href=True)
                for a in anchors:
                    href = a['href']
                    if any(token in href for token in rules.get('link_contains', [])):
                        art_url = urljoin(base, href)
                        art_url = normalize_url(art_url)
                        if art_url in self.seen_urls:
                            continue
                        # Build article stub; content fetched later by processor
                        articles.append({
                            'url': art_url,
                            'title': a.get_text(strip=True) or '',
                            'summary': '',
                            'published': datetime.utcnow(),  # unknown; processor may refine
                            'authors': [],
                            'source': domain,
                            'topic': topic['name'],
                            'topic_keywords': topic['keywords'],
                            'content': '',
                            'content_hash': '',
                            'fetch_time': datetime.utcnow(),
                            'metadata': {'discovered_on': index_url},
                        })
                        self.seen_urls.add(art_url)
                time.sleep(self.rate_limit_delay)
            return articles
        except Exception as e:
            logger.warning(f"Allowlist crawl failed for {url}: {e}")
            return []
    
    def _extract_authors(self, entry) -> List[str]:
        """Extract authors from a feed entry."""
        authors = []
        
        # Handle different feed formats
        if hasattr(entry, 'author'):
            authors.append(entry.author)
        elif hasattr(entry, 'authors') and entry.authors:
            authors.extend([a.get('name', '') for a in entry.authors if a.get('name')])
        elif hasattr(entry, 'dc_creator'):
            if isinstance(entry.dc_creator, list):
                authors.extend(entry.dc_creator)
            else:
                authors.append(entry.dc_creator)
        
        # Clean up and deduplicate
        authors = [a.strip() for a in authors if a and a.strip()]
        return list(dict.fromkeys(authors))  # Preserve order while deduplicating
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on URL and content hash."""
        seen_urls = set()
        seen_hashes = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            content = article.get('content', '')
            content_hash = get_content_hash(content) if content else ''
            
            if not url or url in seen_urls or (content_hash and content_hash in seen_hashes):
                continue
                
            seen_urls.add(url)
            if content_hash:
                seen_hashes.add(content_hash)
            
            unique_articles.append(article)
        
        return unique_articles
    
    def _fetch_article_content(self, url: str) -> Dict[str, Any]:
        """Fetch and parse article content using newspaper3k."""
        try:
            article = Article(url, fetch_images=False, request_timeout=self.request_timeout)
            article.download()
            article.parse()
            
            return {
                'title': article.title,
                'text': article.text,
                'authors': article.authors,
                'publish_date': article.publish_date or datetime.utcnow(),
                'top_image': article.top_image,
                'videos': article.movies,
                'keywords': article.keywords,
                'summary': article.summary,
                'html': article.html,
                'meta_description': article.meta_description,
                'meta_lang': article.meta_lang,
                'meta_favicon': article.meta_favicon,
                'meta_data': article.meta_data,
            }
            
        except ArticleException as e:
            logger.warning(f"Failed to parse article {url}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching article {url}: {e}")
            return {}
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'session'):
            self.session.close()
