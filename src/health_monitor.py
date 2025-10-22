"""
Source health monitoring system for tracking feed reliability.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from loguru import logger
import json
from pathlib import Path


class SourceHealthMonitor:
    """Monitors and tracks the health of content sources."""

    def __init__(self, history_file: str = "health_history.json"):
        self.history_file = Path(history_file)
        self.current_session = {
            "timestamp": datetime.now().isoformat(),
            "feeds": {},
            "topics": {},
        }
        self.history = self._load_history()

    def _load_history(self) -> List[Dict]:
        """Load historical health data."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load health history: {e}")
        return []

    def _save_history(self):
        """Save health history to disk."""
        try:
            # Keep last 30 days of history
            cutoff = datetime.now() - timedelta(days=30)
            filtered_history = [
                entry for entry in self.history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]
            filtered_history.append(self.current_session)

            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(filtered_history, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save health history: {e}")

    def record_feed_result(self, feed_url: str, topic: str,
                          article_count: int, error: str = None):
        """Record the result of a feed fetch attempt."""
        feed_key = feed_url
        if feed_key not in self.current_session["feeds"]:
            self.current_session["feeds"][feed_key] = {
                "url": feed_url,
                "topic": topic,
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "total_articles": 0,
                "errors": []
            }

        feed_data = self.current_session["feeds"][feed_key]
        feed_data["attempts"] += 1

        if error:
            feed_data["failures"] += 1
            feed_data["errors"].append(error)
        else:
            feed_data["successes"] += 1
            feed_data["total_articles"] += article_count

    def record_topic_result(self, topic: str, articles_found: int,
                           articles_after_filter: int, fallback_used: str = None):
        """Record the processing result for a topic."""
        if topic not in self.current_session["topics"]:
            self.current_session["topics"][topic] = {
                "articles_found": 0,
                "articles_filtered": 0,
                "fallbacks_used": []
            }

        topic_data = self.current_session["topics"][topic]
        topic_data["articles_found"] += articles_found
        topic_data["articles_filtered"] += articles_after_filter

        if fallback_used:
            topic_data["fallbacks_used"].append(fallback_used)

    def get_failing_feeds(self, min_failure_rate: float = 0.5) -> List[Tuple[str, Dict]]:
        """Get feeds with high failure rates over recent history."""
        failing = []

        # Analyze last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        recent_sessions = [
            s for s in self.history
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]

        # Aggregate feed statistics
        feed_stats = {}
        for session in recent_sessions:
            for feed_url, data in session.get("feeds", {}).items():
                if feed_url not in feed_stats:
                    feed_stats[feed_url] = {
                        "url": feed_url,
                        "topic": data.get("topic", "unknown"),
                        "attempts": 0,
                        "failures": 0,
                        "total_articles": 0
                    }
                feed_stats[feed_url]["attempts"] += data.get("attempts", 0)
                feed_stats[feed_url]["failures"] += data.get("failures", 0)
                feed_stats[feed_url]["total_articles"] += data.get("total_articles", 0)

        # Find feeds with high failure rates
        for feed_url, stats in feed_stats.items():
            if stats["attempts"] > 0:
                failure_rate = stats["failures"] / stats["attempts"]
                if failure_rate >= min_failure_rate:
                    failing.append((feed_url, stats))

        return failing

    def get_sparse_topics(self, min_articles: int = 5) -> List[Tuple[str, int]]:
        """Get topics that consistently produce few articles."""
        sparse = []

        # Analyze last 7 days
        cutoff = datetime.now() - timedelta(days=7)
        recent_sessions = [
            s for s in self.history
            if datetime.fromisoformat(s["timestamp"]) > cutoff
        ]

        if not recent_sessions:
            return sparse

        # Aggregate topic statistics
        topic_stats = {}
        for session in recent_sessions:
            for topic, data in session.get("topics", {}).items():
                if topic not in topic_stats:
                    topic_stats[topic] = {
                        "total_articles": 0,
                        "sessions": 0
                    }
                topic_stats[topic]["total_articles"] += data.get("articles_filtered", 0)
                topic_stats[topic]["sessions"] += 1

        # Find topics with low average article counts
        for topic, stats in topic_stats.items():
            avg_articles = stats["total_articles"] / stats["sessions"] if stats["sessions"] > 0 else 0
            if avg_articles < min_articles:
                sparse.append((topic, int(avg_articles)))

        return sparse

    def generate_report(self) -> str:
        """Generate a human-readable health report."""
        lines = [
            "=" * 60,
            "SOURCE HEALTH REPORT",
            "=" * 60,
            ""
        ]

        # Current session summary
        lines.append("Current Session:")
        lines.append(f"  Timestamp: {self.current_session['timestamp']}")
        lines.append(f"  Feeds checked: {len(self.current_session['feeds'])}")

        total_articles = sum(
            f.get("total_articles", 0)
            for f in self.current_session["feeds"].values()
        )
        lines.append(f"  Total articles found: {total_articles}")
        lines.append("")

        # Failing feeds
        failing = self.get_failing_feeds()
        if failing:
            lines.append("⚠️  Consistently Failing Feeds (last 7 days):")
            for feed_url, stats in failing[:10]:  # Top 10
                failure_rate = stats["failures"] / stats["attempts"] * 100
                lines.append(f"  - {feed_url[:60]}")
                lines.append(f"    Topic: {stats['topic']}, "
                           f"Failure Rate: {failure_rate:.1f}%, "
                           f"Attempts: {stats['attempts']}")
            lines.append("")

        # Sparse topics
        sparse = self.get_sparse_topics()
        if sparse:
            lines.append("📉 Topics with Low Article Counts (last 7 days avg):")
            for topic, avg_count in sparse:
                lines.append(f"  - {topic}: {avg_count} articles/day")
            lines.append("")

        # Topic summary from current session
        if self.current_session["topics"]:
            lines.append("Current Session Topic Summary:")
            for topic, data in self.current_session["topics"].items():
                lines.append(f"  {topic}:")
                lines.append(f"    Found: {data['articles_found']}")
                lines.append(f"    After filtering: {data['articles_filtered']}")
                if data.get("fallbacks_used"):
                    lines.append(f"    Fallbacks used: {', '.join(data['fallbacks_used'])}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def finalize(self):
        """Finalize the current session and save history."""
        logger.info("Health monitoring report:")
        logger.info("\n" + self.generate_report())
        self._save_history()
