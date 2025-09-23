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

    def fetch(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            logger.info("Reddit integration disabled or missing credentials.")
            return []
        
        topics = self.config.get("topics", [])
        reddit_cfg = self.config.get("reddit", {})
        days_back = int(reddit_cfg.get("days_back", 3))
        after_ts = datetime.now(timezone.utc) - timedelta(days=days_back)

        results: List[Dict[str, Any]] = []
        for topic in topics:
            name = topic["name"]
            subs = reddit_cfg.get("subreddits_per_topic", {}).get(name, [])
            keywords = [k.lower() for k in topic.get("keywords", [])]
            if not subs:
                continue

            for sub in subs:
                try:
                    subreddit = self.reddit.subreddit(sub)
                    for post in subreddit.new(limit=200):
                        created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                        if created < after_ts:
                            continue
                        title = (post.title or "").strip()
                        text = (post.selftext or "").strip()
                        url = post.url or ""
                        combined = f"{title} {text}".lower()
                        if keywords and not any(k in combined for k in keywords):
                            continue
                        # Map to article dict; treat Reddit link as source if external URL looks non-article
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
                except Exception as e:
                    logger.warning(f"Reddit fetch failed for r/{sub}: {e}")
                    continue
        return results
