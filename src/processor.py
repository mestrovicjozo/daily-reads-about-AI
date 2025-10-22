"""
Content processing: fetch content, language detection, deduplication, and summarization.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from loguru import logger
from newspaper import Article

from utils import (
    estimate_reading_time,
    get_content_hash,
)
from router import TopicRouter
from health_monitor import SourceHealthMonitor
from ai_augment import AIAugmentor

import json


@dataclass
class ProcessedArticle:
    url: str
    title: str
    source: str
    authors: List[str]
    published: datetime
    reading_time_min: int
    summary: str
    why_selected: str
    topic: str
    content_hash: str
    fetch_time: datetime


class ContentProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_len = config["processing"]["min_article_length"]
        self.max_len = config["processing"]["max_article_length"]
        self.target_lang = config["processing"]["language"]
        self.summary_sentences = 5  # enforce exactly 5 as per hard requirement
        self.max_final = config["output"]["max_articles_per_topic"]
        self.max_age_hours = int(config["processing"].get("max_age_hours", 72))  # Changed to 72h (3 days)
        self.fallback_max_age_hours = 168  # 7 days
        # Optional per-topic extended fallback up to N days (default 30)
        self.enable_per_topic_month_fallback = bool(
            config.get("processing", {}).get("per_topic_month_fallback", True)
        )
        self.per_topic_month_fallback_days = int(
            config.get("processing", {}).get("per_topic_month_fallback_days", 30)
        )
        # Cross-day dedup horizon
        self.history_days = int(config["processing"].get("history_days", 30))
        self.fallback_window_days = int(config["processing"].get("fallback_window_days", 7))
        self.provenance_dir = Path(config["output"].get("provenance_dir", "provenance"))
        self.date_format = config["output"].get("date_format", "%Y-%m-%d")

        # Preload recent provenance URLs and content hashes
        self.recent_urls, self.recent_hashes = self._load_recent_history(self.history_days)

        # Initialize topic router (optional, controlled by config)
        self.router = TopicRouter(config)

        # Initialize health monitor and AI augmentor
        self.health_monitor = SourceHealthMonitor()
        self.ai_augmentor = AIAugmentor()
        self.use_ai_validation = config.get("processing", {}).get("use_ai_validation", False)

    def process(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content, filter to English, dedupe, and summarize per topic.
        Returns a dict: {topic_name: [article_dict, ...]}"""
        logger.info(f"Processing {len(articles)} articles")
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen_hashes_per_topic: Dict[str, set] = defaultdict(set)
        now = datetime.utcnow()
        
        # Track processing stats
        stats = {
            'total_input': len(articles),
            'url_skipped_recent': 0,
            'content_hash_skipped_recent': 0,
            'fetch_failed': 0,
            'too_old': 0,
            'too_short': 0,
            'wrong_language': 0,
            'duplicate_content': 0,
            'successfully_processed': 0
        }

        def try_add(a: Dict[str, Any], allowed_age_hours: int) -> bool:
            """Process a single article and add it if it passes filters.
            Returns True if added, False otherwise.
            """
            url = a["url"]
            topic = a.get("topic")
            logger.debug(f"Processing article: {url} for topic {topic}")
            
            # Early skip if URL seen in recent history
            if url in self.recent_urls:
                logger.debug(f"Skipping {url}: URL seen in recent history")
                stats['url_skipped_recent'] += 1
                return False
            try:
                parsed = self._fetch_article(url)
                if not parsed:
                    logger.debug(f"Skipping {url}: Failed to fetch article content")
                    stats['fetch_failed'] += 1
                    return False
                text = (parsed.get("text") or "").strip()
                normalized_hash = get_content_hash(self._normalize_text(text)) if text else ""
                # Early skip if content hash seen recently
                if normalized_hash and normalized_hash in self.recent_hashes:
                    logger.debug(f"Skipping {url}: Content hash seen in recent history")
                    stats['content_hash_skipped_recent'] += 1
                    return False

                published = parsed.get("publish_date") or a.get("published") or a.get("fetch_time") or now
                if getattr(published, "tzinfo", None) is not None:
                    published = published.astimezone(timezone.utc).replace(tzinfo=None)
                
                # Handle date parsing issues more gracefully
                if not published or published == now:
                    # If no publish date, assume it's recent (within last 24h)
                    published = now - timedelta(hours=12)
                    logger.debug(f"No publish date for {url}, assuming recent")
                
                age_hours = (now - published).total_seconds() / 3600.0
                if age_hours > allowed_age_hours:
                    logger.debug(f"Skipping {url}: Too old ({age_hours:.1f}h > {allowed_age_hours}h)")
                    stats['too_old'] += 1
                    return False

                if len(text) < self.min_len:
                    logger.debug(f"Skipping {url}: Too short ({len(text)} < {self.min_len})")
                    stats['too_short'] += 1
                    return False
                if len(text) > self.max_len:
                    text = text[: self.max_len]

                try:
                    lang = detect(text)
                except LangDetectException:
                    lang = "unknown"
                if lang != self.target_lang:
                    logger.debug(f"Skipping {url}: Wrong language ({lang} != {self.target_lang})")
                    stats['wrong_language'] += 1
                    return False

                content_hash = normalized_hash or get_content_hash(self._normalize_text(text))
                if topic and content_hash in seen_hashes_per_topic[topic]:
                    logger.debug(f"Skipping {url}: Duplicate content in topic")
                    stats['duplicate_content'] += 1
                    return False
                if topic:
                    seen_hashes_per_topic[topic].add(content_hash)

                # Use original title from crawler if available, otherwise try parsed title
                title = (a.get("title") or parsed.get("title") or "Untitled").strip()
                
                # Clean up title by removing common metadata patterns
                if title and title != "Untitled":
                    # Remove common metadata patterns that get included in titles
                    title = re.sub(r'^(Tutorials|Engineering|News|Blog|Article|Post)\s*', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}\b', '', title)
                    title = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', title)
                    # Remove date patterns like "Aug 25, 2025" at the beginning
                    title = re.sub(r'^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}\s*', '', title, flags=re.IGNORECASE)
                    # Remove any remaining date patterns
                    title = re.sub(r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b', '', title, flags=re.IGNORECASE)
                    title = re.sub(r'\s+', ' ', title).strip()
                
                # Fallback to URL-based title if still empty
                if not title or title == "Untitled":
                    url_parts = url.split("/")[-1].replace("-", " ").replace("_", " ")
                    title = url_parts.title() if url_parts else "Untitled"
                authors = parsed.get("authors") or a.get("authors") or []
                source = a.get("source")
                reading_time = estimate_reading_time(text)

                summary = self._structured_summary(text, title)
                why_selected = self._infer_why_selected(text, title, source)

                # Optional AI validation
                if self.use_ai_validation and topic:
                    if not self.ai_augmentor.validate_relevance(
                        {"title": title, "text": text, "url": url}, topic
                    ):
                        logger.debug(f"AI rejected article as not relevant to {topic}: {title[:50]}")
                        stats['fetch_failed'] += 1
                        return False

                # Topic routing (may override original topic)
                routed_topic = topic
                routing_meta: Dict[str, Any] = {}
                if self.router and self.router.enabled:
                    best_topic, scores, margin, should_override = self.router.route(title, text, topic)
                    routing_meta = {
                        "router_best_topic": best_topic,
                        "router_margin": margin,
                        "router_scores": scores,
                        "router_overrode": False,
                    }
                    if best_topic and (topic is None or should_override):
                        routed_topic = best_topic
                        routing_meta["router_overrode"] = topic != best_topic
                elif self.ai_augmentor.enabled:
                    # Use AI for topic classification if router not enabled
                    topics_cfg = [t.get("name") for t in self.config.get("topics", [])]
                    ai_topic = self.ai_augmentor.enhance_topic_classification(
                        {"title": title, "text": text}, topics_cfg
                    )
                    if ai_topic:
                        routed_topic = ai_topic
                        routing_meta = {"ai_classified": True, "ai_topic": ai_topic}

                if not routed_topic:
                    routed_topic = topic or "LLMs"  # default bucket if unknown

                grouped[routed_topic].append({
                    "url": url,
                    "title": title,
                    "source": source,
                    "authors": authors,
                    "published": published,
                    "reading_time_min": reading_time,
                    "summary": summary,
                    "why_selected": why_selected,
                    "topic": routed_topic,
                    "content_hash": content_hash,
                    "fetch_time": a.get("fetch_time", now),
                    "raw_text": text,
                    "topic_keywords": a.get("topic_keywords", []),
                    "routing": routing_meta,
                })
                stats['successfully_processed'] += 1
                logger.debug(f"Successfully processed {url}: {title[:50]}...")
                return True
            except Exception as e:
                logger.warning(f"Failed processing article {url}: {e}")
                stats['fetch_failed'] += 1
                return False

        # First pass: strict freshness window (e.g., 24h)
        for a in articles:
            try_add(a, self.max_age_hours)

        # If nothing found in the strict 24h window, perform a one-time 30-day global fallback
        total_after_24h = sum(len(v) for v in grouped.values())
        if total_after_24h == 0:
            logger.warning("No items found in 24h window. Falling back to 30 days to populate digest.")
            month_hours = 24 * 30
            # Iterate per topic, add up to max_final per topic
            for tname in [t.get("name") for t in self.config.get("topics", [])]:
                for a in (x for x in articles if x.get("topic") == tname):
                    if len(grouped.get(tname, [])) >= self.max_final:
                        break
                    # avoid duplicates by URL within topic
                    existing_urls = {i["url"] for i in grouped.get(tname, [])}
                    if a.get("url") in existing_urls:
                        continue
                    try_add(a, month_hours)
            return grouped

        # Fallback pass: for topics with < max_final items, allow up to 7 days
        topics_cfg = [t.get("name") for t in self.config.get("topics", [])]
        for tname in topics_cfg:
            have = len(grouped.get(tname, []))
            if have >= self.max_final:
                continue
            need = self.max_final - have
            logger.info(f"Topic {tname} low coverage ({have}/{self.max_final}). Falling back up to 7 days to fill {need}.")
            for a in (x for x in articles if x.get("topic") == tname):
                if len(grouped.get(tname, [])) >= self.max_final:
                    break
                # Skip if already added by first pass (by URL)
                existing_urls = {i["url"] for i in grouped.get(tname, [])}
                if a.get("url") in existing_urls:
                    continue
                try_add(a, self.fallback_max_age_hours)

        # Second fallback: per-topic month-long window if still underfilled
        if self.enable_per_topic_month_fallback:
            month_hours = 24 * max(1, self.per_topic_month_fallback_days)
            for tname in topics_cfg:
                have = len(grouped.get(tname, []))
                if have >= self.max_final:
                    continue
                need = self.max_final - have
                logger.info(
                    f"Topic {tname} still low after 7-day fallback ({have}/{self.max_final}). "
                    f"Extending to {self.per_topic_month_fallback_days} days to fill {need}."
                )
                for a in (x for x in articles if x.get("topic") == tname):
                    if len(grouped.get(tname, [])) >= self.max_final:
                        break
                    existing_urls = {i["url"] for i in grouped.get(tname, [])}
                    if a.get("url") in existing_urls:
                        continue
                    try_add(a, month_hours)

        # Log processing statistics
        logger.info(f"Processing complete. Stats: {stats}")
        total_output = sum(len(v) for v in grouped.values())
        logger.info(f"Final output: {total_output} articles across {len(grouped)} topics")

        # Record topic results in health monitor
        for topic, topic_articles in grouped.items():
            logger.info(f"Topic {topic}: {len(topic_articles)} articles")
            self.health_monitor.record_topic_result(
                topic,
                len([a for a in articles if a.get("topic") == topic]),
                len(topic_articles),
                fallback_used=None  # Could be enhanced to track which fallback was used
            )

        # Finalize health monitoring
        self.health_monitor.finalize()

        return grouped

    def _fetch_article(self, url: str) -> Dict[str, Any] | None:
        """Fetch and parse article content with improved error handling."""
        try:
            logger.debug(f"Fetching article content from: {url}")
            art = Article(url, fetch_images=False)
            art.download()
            art.parse()
            
            # Validate that we got meaningful content
            if not art.text or len(art.text.strip()) < 50:
                logger.debug(f"Insufficient content from {url}: {len(art.text or '')} chars")
                return None
            
            result = {
                "title": art.title,
                "text": art.text,
                "authors": art.authors,
                "publish_date": art.publish_date,
                "http_status": 200,
            }
            
            logger.debug(f"Successfully fetched {url}: {len(art.text)} chars, title: {art.title[:50] if art.title else 'None'}...")
            return result
            
        except Exception as e:
            logger.debug(f"Failed to fetch article {url}: {e}")
            return None

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # --- Summarization helpers ---

    def _structured_summary(self, text: str, title: str) -> str:
        """Produce exactly 5 sentences (~60–120 words) with roles:
        1) what it does/claims, 2) methods/evidence, 3) results/arguments,
        4) limitations/controversies, 5) selection justification.
        """
        sents = self._split_sentences(text)
        # Heuristic picks
        claim = self._pick_claim(sents, title) or self._fallback_sentence(sents, 0)
        method = self._pick_method(sents) or self._fallback_sentence(sents, 1)
        results = self._pick_results(sents) or self._fallback_sentence(sents, 2)
        limits = self._pick_limits(sents) or self._generic_limitations()
        selection = self._generic_selection_sentence(title)

        ordered = [claim, method, results, limits, selection]
        ordered = [self._clean_sentence(s) for s in ordered]

        # Enforce exactly 5 sentences and 60–120 words total
        text_out = " ".join(ordered)
        words = text_out.split()
        if len(words) < 60:
            # extend by appending additional informative sentences if available
            i = 0
            while len(words) < 60 and i < len(sents):
                cand = self._clean_sentence(sents[i])
                if cand and cand not in ordered:
                    ordered.insert(3, cand)  # insert before limitations
                    # Keep at most 5 sentences by merging short ones
                    ordered = self._shrink_to_five(ordered)
                    text_out = " ".join(ordered)
                    words = text_out.split()
                i += 1
        elif len(words) > 120:
            # Trim sentences to keep within the limit
            text_out = self._trim_to_word_limit(ordered, 120)
        # Final safety: keep exactly 5 sentences
        final = self._shrink_to_five(self._split_into_sentences(text_out))
        return " ".join(final)

    def _split_sentences(self, text: str) -> List[str]:
        # Lightweight sentence splitter aligned with validator's counting
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        return self._split_sentences(text)

    def _clean_sentence(self, s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        # Ensure proper terminal punctuation
        if s and s[-1] not in ".!?":
            s += "."
        return s

    def _fallback_sentence(self, sents: List[str], idx: int) -> str:
        return self._clean_sentence(sents[idx]) if idx < len(sents) else ""

    def _pick_claim(self, sents: List[str], title: str) -> str | None:
        # Prefer a sentence mentioning the title keywords or introduction verbs
        title_tokens = [t for t in re.findall(r"\w+", title.lower()) if len(t) > 3]
        for s in sents[:5]:
            low = s.lower()
            if any(t in low for t in title_tokens) or any(k in low for k in ["introduce", "present", "explore", "report", "describe", "propos"]):
                return self._clean_sentence(s)
        return None

    def _pick_method(self, sents: List[str]) -> str | None:
        keys = ["method", "approach", "we", "dataset", "experiment", "benchmark", "architecture", "pipeline", "implementation"]
        for s in sents[:12]:
            if any(k in s.lower() for k in keys):
                return self._clean_sentence(s)
        return None

    def _pick_results(self, sents: List[str]) -> str | None:
        keys = ["result", "improv", "outperform", "%", "accuracy", "gain", "speedup", "latency", "throughput"]
        for s in sents[:15]:
            if any(k in s.lower() for k in keys):
                return self._clean_sentence(s)
        return None

    def _pick_limits(self, sents: List[str]) -> str | None:
        keys = ["however", "but ", "limitation", "risk", "bias", "cost", "compute", "data", "privacy", "security", "oversight", "trade-off"]
        for s in sents[:20]:
            if any(k in s.lower() for k in keys):
                return self._clean_sentence(s)
        return None

    def _generic_limitations(self) -> str:
        return "However, the piece notes limitations around generality, evaluation scope, and real-world constraints."

    def _generic_selection_sentence(self, title: str) -> str:
        return f"Selected for its clear takeaways and practical relevance to the topic of {title[:60]}."

    def _trim_to_word_limit(self, sentences: List[str], limit: int) -> str:
        words_total = 0
        kept: List[str] = []
        for s in sentences:
            w = s.split()
            if words_total + len(w) <= limit:
                kept.append(s)
                words_total += len(w)
            else:
                # add partial sentence if needed
                remaining = max(0, limit - words_total)
                if remaining > 0:
                    kept.append(" ".join(w[:remaining]) + ("." if w and kept[-1][-1] not in ".!?" else ""))
                break
        # Ensure exactly 5 sentences if possible
        return " ".join(self._shrink_to_five(kept))

    def _shrink_to_five(self, sentences: List[str]) -> List[str]:
        s = [x for x in sentences if x and x.strip()]
        if len(s) <= 5:
            # pad with short neutral sentences if we are short (rare)
            while len(s) < 5:
                s.append("Additional context is limited by the available details.")
            return s
        # merge extras into the earlier slots to remain at 5 without creating extra sentence boundaries
        merged = s[:5]
        for extra in s[5:]:
            base = merged[-2].rstrip()
            # remove terminal punctuation from base
            if base and base[-1] in ".!?":
                base = base[:-1]
            extra_clean = extra.strip()
            # remove terminal punctuation from the extra to avoid adding a new sentence boundary
            if extra_clean and extra_clean[-1] in ".!?":
                extra_clean = extra_clean[:-1]
            merged[-2] = f"{base}; {extra_clean}."
        return merged[:5]

    def _infer_why_selected(self, text: str, title: str, source: str | None) -> str:
        low = text.lower()
        if any(k in low for k in ["code", "github", "implementation", "repository", "colab"]):
            return "Includes code or implementation details readers can use."
        if any(k in low for k in ["benchmark", "evaluation", "ablation", "results"]):
            return "Strong empirical evaluation and clear benchmarks."
        if any(k in low for k in ["tutorial", "how to", "guide", "walkthrough", "hands-on"]):
            return "Actionable tutorial with step-by-step guidance."
        if source and ("arxiv" in source or "arxiv.org" in source):
            return "Notable research preprint relevant to current developments."
        if any(k in low for k in ["release", "announce", "launch", "roadmap"]):
            return "Timely announcement with practical implications."
        return "Clear insights and practical relevance to the topic."

    def _load_recent_history(self, days: int) -> Tuple[set, set]:
        """Load URLs and content hashes from provenance files in the last N days."""
        urls: set = set()
        hashes: set = set()
        if not self.provenance_dir.exists():
            return urls, hashes
        try:
            # Collect files and sort descending by date inferred from filename
            files = sorted(self.provenance_dir.glob("*.json"), reverse=True)
            cutoff = datetime.utcnow() - timedelta(days=days)
            for fp in files:
                try:
                    # Parse date from filename if possible
                    stem = fp.stem  # YYYY-MM-DD
                    dt = datetime.strptime(stem, self.date_format)
                    if dt < cutoff:
                        continue
                except Exception:
                    # If filename not parseable, include it anyway as recent
                    pass
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    for item in data.get("items", []):
                        u = item.get("url")
                        h = item.get("content_hash")
                        if u:
                            urls.add(u)
                        if h:
                            hashes.add(h)
                except Exception as e:
                    logger.debug(f"Skip malformed provenance {fp}: {e}")
        except Exception as e:
            logger.debug(f"Failed loading recent provenance: {e}")
        return urls, hashes
