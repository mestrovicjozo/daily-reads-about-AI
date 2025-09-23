"""
Deterministic ranking of articles per topic with diversity and cross-topic deduplication.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from loguru import logger


class ArticleRanker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get("ranking", {})
        self.max_per_topic = config["output"]["max_articles_per_topic"]
        # Simple static domain reputation map; can be expanded or learned.
        self.domain_reputation = {
            "arxiv.org": 0.95,
            "openai.com": 0.9,
            "ai.googleblog.com": 0.9,
            "meta.com": 0.85,
            "deepmind.google": 0.9,
            "huggingface.co": 0.85,
            "microsoft.com": 0.85,
            "anthropic.com": 0.85,
            # RAG/vector DB ecosystem
            "weaviate.io": 0.85,
            "milvus.io": 0.8,
            "pinecone.io": 0.85,
            "langchain.dev": 0.85,
            "cohere.com": 0.85,
            # Quantum
            "quantumcomputingreport.com": 0.85,
            "quantumai.google": 0.9,
            "quantamagazine.org": 0.9,
            "cloudblogs.microsoft.com": 0.85,
            "xanadu.ai": 0.8,
            "research.ibm.com": 0.85,
            "ionq.com": 0.8,
            "rigetti.com": 0.8,
            "nature.com": 0.95,
        }

    def rank(self, grouped: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
        # Score within each topic
        scored_per_topic: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        now = datetime.utcnow()

        for topic, items in grouped.items():
            scored: List[Tuple[float, Dict[str, Any]]] = []
            for it in items:
                score = self._score(it, now, topic)
                scored.append((score, it))
            # Deterministic ordering: score desc, published desc, url asc
            scored.sort(key=lambda x: (-x[0], -(x[1]["published"].timestamp() if it.get("published") else 0), x[1]["url"]))
            # Diversity: penalize repeated domains when selecting top N
            selected: List[Tuple[float, Dict[str, Any]]] = []
            seen_domains = defaultdict(int)
            for s, it in scored:
                dom = it.get("source", "")
                penalty = 0.0 if seen_domains[dom] == 0 else 0.1 * seen_domains[dom]
                effective = s - penalty
                selected.append((effective, it))
                seen_domains[dom] += 1
            selected.sort(key=lambda x: -x[0])
            scored_per_topic[topic] = selected

        # Cross-topic deduplication by content_hash
        chosen_hash_to_topic: Dict[str, str] = {}
        final: Dict[str, List[Dict[str, Any]]] = {t: [] for t in grouped.keys()}

        # Select in round-robin per topic to improve balance
        topics = list(scored_per_topic.keys())
        pointer = {t: 0 for t in topics}
        remaining = True
        while remaining:
            remaining = False
            for t in topics:
                lst = scored_per_topic[t]
                while pointer[t] < len(lst) and len(final[t]) < self.max_per_topic:
                    remaining = True
                    _, cand = lst[pointer[t]]
                    pointer[t] += 1
                    h = cand.get("content_hash")
                    if not h:
                        final[t].append(cand)
                        break
                    if h in chosen_hash_to_topic:
                        # keep the one with higher topic relevance; here proxied by keyword match count vs target topic
                        prev_topic = chosen_hash_to_topic[h]
                        # Prefer to keep in the topic where it first appeared (determinism) – or compute relevance per topic if available
                        continue
                    chosen_hash_to_topic[h] = t
                    final[t].append(cand)
                    break

        # Truncate to max per topic
        for t in final:
            final[t] = final[t][: self.max_per_topic]
        return final

    def _score(self, it: Dict[str, Any], now: datetime, topic: str) -> float:
        w_recency = float(self.weights.get("recency", 0.4))
        w_relevance = float(self.weights.get("relevance", 0.3))
        w_auth = float(self.weights.get("authoritativeness", 0.2))
        w_eng = float(self.weights.get("engagement", 0.1))

        recency = self._recency_score(it.get("published"), now)
        relevance = self._relevance_score(it.get("raw_text", ""), it.get("title", ""), it.get("topic_keywords", []))
        auth = self._authority_score(it.get("source", ""))
        engagement = 0.0  # placeholder

        score = w_recency * recency + w_relevance * relevance + w_auth * auth + w_eng * engagement
        return round(score, 6)

    def _recency_score(self, published: datetime, now: datetime) -> float:
        if not published:
            return 0.3
        # Normalize timezone awareness to avoid TypeError on subtraction
        if getattr(published, "tzinfo", None) is not None:
            published = published.astimezone(timezone.utc).replace(tzinfo=None)
        if getattr(now, "tzinfo", None) is not None:
            now = now.astimezone(timezone.utc).replace(tzinfo=None)
        delta = now - published
        # Full score within 72h, then exponential decay
        hours = max(0.0, delta.total_seconds() / 3600.0)
        if hours <= 72:
            return 1.0 - (hours / 72.0) * 0.2  # slight decay within window
        # decay after 72h
        import math
        return max(0.1, math.exp(-(hours - 72) / 168))  # week-scale decay

    def _relevance_score(self, text: str, title: str, keywords: List[str]) -> float:
        # Simple keyword coverage measure; can be replaced by embeddings
        all_text = f"{title} {text}".lower()
        hits = 0
        for kw in keywords:
            if kw.lower() in all_text:
                hits += 1
        if not keywords:
            return 0.3
        cov = hits / len(keywords)
        # Boost if keywords appear in title
        title_hits = sum(1 for kw in keywords if kw.lower() in title.lower())
        return min(1.0, cov + 0.1 * title_hits)

    def _authority_score(self, domain: str) -> float:
        return float(self.domain_reputation.get(domain, 0.5))
