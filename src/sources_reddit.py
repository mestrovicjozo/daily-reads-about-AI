"""
Reddit source integration for fetching posts relevant to configured topics.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from loguru import logger

try:
    import praw
except Exception:  # pragma: no cover
    praw = None

import requests
from urllib.parse import urljoin


class RedditSource:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client_id = os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "digest-agent/0.1")
        self.enabled = bool(self.client_id and self.client_secret and praw is not None)
        if self.enabled:
            try:
                self.reddit = praw.Reddit(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    user_agent=self.user_agent,
                )
            except Exception as e:
                logger.warning(f"Failed to init Reddit client: {e}")
                self.enabled = False
        # requests session for anonymous JSON fallback
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})

    def fetch(self) -> List[Dict[str, Any]]:
        topics = self.config.get("topics", [])
        reddit_cfg = self.config.get("reddit", {})
        days_back = int(reddit_cfg.get("days_back", 3))
        after_ts = datetime.now(timezone.utc) - timedelta(days=days_back)
        per_topic_cap = int(self.config.get("output", {}).get("max_articles_per_topic", 5))

        results: List[Dict[str, Any]] = []
        per_topic_counts: Dict[str, int] = {}
        for topic in topics:
            name = topic["name"]
            subs = reddit_cfg.get("subreddits_per_topic", {}).get(name, [])
            keywords = [k.lower() for k in topic.get("keywords", [])]
            if not subs:
                continue
            # Stop early if we already hit cap for this topic
            per_topic_counts.setdefault(name, 0)

            for sub in subs:
                if per_topic_counts[name] >= per_topic_cap:
                    break
                try:
                    if self.enabled:
                        # Authenticated via PRAW
                        subreddit = self.reddit.subreddit(sub)
                        for post in subreddit.new(limit=200):
                            if per_topic_counts[name] >= per_topic_cap:
                                break
                            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                            if created < after_ts:
                                continue
                            title = (post.title or "").strip()
                            text = (post.selftext or "").strip()
                            url = post.url or ""
                            combined = f"{title} {text}".lower()
                            if keywords and not any(k in combined for k in keywords):
                                continue
                            source = f"reddit.com/r/{sub}"
                            results.append({
                                "url": url or f"https://reddit.com{post.permalink}",
                                "title": title,
                                "summary": text[:280],
                                "published": created.replace(tzinfo=None),
                                "authors": [str(post.author)] if post.author else [],
                                "source": source,
                                "topic": name,
                                "topic_keywords": topic.get("keywords", []),
                                "content": text,
                                "content_hash": "",
                                "fetch_time": datetime.utcnow(),
                                "metadata": {
                                    "reddit_permalink": f"https://reddit.com{post.permalink}",
                                    "score": getattr(post, "score", None),
                                },
                            })
                            per_topic_counts[name] += 1
                    else:
                        # Anonymous JSON fallback
                        api_url = f"https://www.reddit.com/r/{sub}/new.json?limit=100"
                        resp = self.session.get(api_url, timeout=10)
                        if resp.status_code != 200:
                            logger.warning(f"Reddit JSON API failed for r/{sub}: status {resp.status_code}")
                            continue
                        data = resp.json()
                        children = (data.get("data", {}) or {}).get("children", [])
                        for child in children:
                            if per_topic_counts[name] >= per_topic_cap:
                                break
                            post = (child or {}).get("data", {})
                            created_utc = post.get("created_utc")
                            if not created_utc:
                                continue
                            created = datetime.fromtimestamp(float(created_utc), tz=timezone.utc)
                            if created < after_ts:
                                continue
                            title = (post.get("title") or "").strip()
                            text = (post.get("selftext") or "").strip()
                            url = post.get("url") or ""
                            combined = f"{title} {text}".lower()
                            if keywords and not any(k in combined for k in keywords):
                                continue
                            permalink = post.get("permalink") or ""
                            source = f"reddit.com/r/{sub}"
                            results.append({
                                "url": url or urljoin("https://reddit.com", permalink),
                                "title": title,
                                "summary": text[:280],
                                "published": created.replace(tzinfo=None),
                                "authors": [post.get("author")] if post.get("author") else [],
                                "source": source,
                                "topic": name,
                                "topic_keywords": topic.get("keywords", []),
                                "content": text,
                                "content_hash": "",
                                "fetch_time": datetime.utcnow(),
                                "metadata": {
                                    "reddit_permalink": urljoin("https://reddit.com", permalink),
                                    "score": post.get("score"),
                                },
                            })
                            per_topic_counts[name] += 1
                except Exception as e:
                    logger.warning(f"Reddit fetch failed for r/{sub}: {e}")
                    continue
        return results
