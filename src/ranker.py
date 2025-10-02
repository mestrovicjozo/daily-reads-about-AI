"""
Article ranking: compute scores and select top-N per topic with diversity.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

from loguru import logger


class ArticleRanker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        rk = config.get("ranking", {})
        self.w_recency = float(rk.get("recency", 0.4))
        self.w_relevance = float(rk.get("relevance", 0.3))
        self.w_author = float(rk.get("authoritativeness", 0.2))
        self.w_engage = float(rk.get("engagement", 0.1))
        self.max_per_topic = int(config.get("output", {}).get("max_articles_per_topic", 5))

        # Known authoritative domains (simple heuristic)
        self.authoritative_domains = set([
            "arxiv.org", "openai.com", "ai.googleblog.com", "huggingface.co",
            "weaviate.io", "milvus.io", "blog.langchain.dev", "cohere.com",
            "research.google", "quantumai.google", "developers.googleblog.com",
        ])

    def rank(self, processed_by_topic: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        now = datetime.utcnow()
        ranked: Dict[str, List[Dict[str, Any]]] = {}

        for topic, items in processed_by_topic.items():
            if not items:
                ranked[topic] = []
                continue

            scored: List[Tuple[float, Dict[str, Any]]] = []
            topic_keywords = set()
            # Aggregate keywords we saw during processing for this topic
            for it in items:
                for kw in it.get("topic_keywords", []) or []:
                    if isinstance(kw, str):
                        topic_keywords.add(kw.lower())

            for it in items:
                s_recency = self._score_recency(it.get("published"), now)
                s_relevance = self._score_relevance(it.get("title"), it.get("raw_text"), topic_keywords)
                s_author = self._score_authoritativeness(it.get("source"), it.get("authors"))
                s_engage = self._score_engagement(it)
                total = (
                    self.w_recency * s_recency
                    + self.w_relevance * s_relevance
                    + self.w_author * s_author
                    + self.w_engage * s_engage
                )
                scored.append((total, it))

            # Deterministic base ordering: score desc, published desc, url asc
            scored.sort(key=lambda x: (
                -x[0],
                -self._published_ts_safe(x[1].get("published")),
                str(x[1].get("url", ""))
            ))

            # Diversity selection: penalize repeated domains when picking top N
            selection: List[Dict[str, Any]] = []
            domain_counts: Dict[str, int] = {}
            for score, it in scored:
                if len(selection) >= self.max_per_topic:
                    break
                dom = self._domain_of(it.get("url", ""))
                penalty = 0.0
                if dom:
                    count = domain_counts.get(dom, 0)
                    if count > 0:
                        # each repeat penalizes by 10% per prior appearance
                        penalty = min(0.5, 0.1 * count)
                adj_score = score * (1.0 - penalty)
                # Threshold: avoid adding items with near-zero adjusted scores
                if adj_score <= 0.0 and len(selection) > 0:
                    continue
                selection.append(it)
                if dom:
                    domain_counts[dom] = domain_counts.get(dom, 0) + 1

            ranked[topic] = selection

        return ranked

    # --- Scoring helpers ---

    def _score_recency(self, published: Any, now: datetime) -> float:
        if not published:
            return 0.2
        p = published
        if hasattr(p, "tzinfo") and p.tzinfo is not None:
            p = p.astimezone(timezone.utc).replace(tzinfo=None)
        age_hours = max(0.0, (now - p).total_seconds() / 3600.0)
        # Exponential decay with 24h half-life
        half_life = 24.0
        return max(0.0, min(1.0, 0.5 ** (age_hours / half_life)))

    def _score_relevance(self, title: Any, text: Any, topic_keywords: set) -> float:
        if not topic_keywords:
            return 0.5
        hay = ((title or "") + "\n" + (text or "")).lower()
        if not hay:
            return 0.0
        hits = sum(1 for kw in topic_keywords if kw in hay)
        # saturating function
        return max(0.0, min(1.0, hits / (len(topic_keywords) or 1)))

    def _score_authoritativeness(self, source: Any, authors: Any) -> float:
        dom = str(source or "").lower()
        score = 0.3
        if dom in self.authoritative_domains:
            score += 0.5
        # Bonus for named authors
        if isinstance(authors, list) and any(a and len(str(a)) > 0 for a in authors):
            score += 0.2
        return max(0.0, min(1.0, score))

    def _score_engagement(self, item: Dict[str, Any]) -> float:
        md = item.get("metadata") or {}
        # Reddit score if present
        score = md.get("score")
        if isinstance(score, (int, float)):
            # log-like scaling
            return max(0.0, min(1.0, math.log1p(max(0.0, float(score))) / 10.0))
        return 0.0

    def _published_ts_safe(self, published: Any) -> float:
        try:
            p = published
            if hasattr(p, "tzinfo") and p.tzinfo is not None:
                p = p.astimezone(timezone.utc).replace(tzinfo=None)
            return p.timestamp()
        except Exception:
            return 0.0

    def _domain_of(self, url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""
