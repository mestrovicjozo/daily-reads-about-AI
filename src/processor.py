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
        self.max_age_hours = int(config["processing"].get("max_age_hours", 24))
        self.fallback_max_age_hours = 168  # 7 days

    def process(self, articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch content, filter to English, dedupe, and summarize per topic.
        Returns a dict: {topic_name: [article_dict, ...]}"""
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen_hashes_per_topic: Dict[str, set] = defaultdict(set)
        now = datetime.utcnow()

        def try_add(a: Dict[str, Any], allowed_age_hours: int) -> bool:
            """Process a single article and add it if it passes filters.
            Returns True if added, False otherwise.
            """
            url = a["url"]
            topic = a["topic"]
            try:
                parsed = self._fetch_article(url)
                if not parsed:
                    return False

                published = parsed.get("publish_date") or a.get("published") or a.get("fetch_time") or now
                if getattr(published, "tzinfo", None) is not None:
                    published = published.astimezone(timezone.utc).replace(tzinfo=None)
                age_hours = (now - published).total_seconds() / 3600.0
                if age_hours > allowed_age_hours:
                    return False

                text = (parsed.get("text") or "").strip()
                if len(text) < self.min_len:
                    return False
                if len(text) > self.max_len:
                    text = text[: self.max_len]

                try:
                    lang = detect(text)
                except LangDetectException:
                    lang = "unknown"
                if lang != self.target_lang:
                    return False

                content_hash = get_content_hash(self._normalize_text(text))
                if content_hash in seen_hashes_per_topic[topic]:
                    return False
                seen_hashes_per_topic[topic].add(content_hash)

                title = (parsed.get("title") or a.get("title") or "Untitled").strip()
                authors = parsed.get("authors") or a.get("authors") or []
                source = a.get("source")
                reading_time = estimate_reading_time(text)

                summary = self._structured_summary(text, title)
                why_selected = self._infer_why_selected(text, title, source)

                grouped[topic].append({
                    "url": url,
                    "title": title,
                    "source": source,
                    "authors": authors,
                    "published": published,
                    "reading_time_min": reading_time,
                    "summary": summary,
                    "why_selected": why_selected,
                    "topic": topic,
                    "content_hash": content_hash,
                    "fetch_time": a.get("fetch_time", now),
                    "raw_text": text,
                    "topic_keywords": a.get("topic_keywords", []),
                })
                return True
            except Exception as e:
                logger.warning(f"Failed processing article {url}: {e}")
                return False

        # First pass: strict freshness window (e.g., 24h)
        for a in articles:
            try_add(a, self.max_age_hours)

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

        return grouped

    def _fetch_article(self, url: str) -> Dict[str, Any] | None:
        art = Article(url, fetch_images=False)
        art.download()
        art.parse()
        return {
            "title": art.title,
            "text": art.text,
            "authors": art.authors,
            "publish_date": art.publish_date,
            "http_status": 200,
        }

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
        # Lightweight sentence splitter without external downloads
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
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
        # merge extras into the earlier slots to remain at 5
        merged = s[:5]
        for extra in s[5:]:
            merged[-2] = (merged[-2].rstrip(".")) + "; " + extra
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
