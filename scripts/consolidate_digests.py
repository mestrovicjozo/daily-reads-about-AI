#!/usr/bin/env python3
"""
Consolidate daily digests into a single 30-day digest file.

This script reads all digest markdown files from the last 30 days,
extracts articles, deduplicates them, and creates a single consolidated file.
"""

import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Article:
    """Represents an article from a digest."""
    title: str
    url: str
    source: str
    author: str
    published: str
    reading_time: str
    summary: str
    why_selected: str
    topic: str
    digest_date: str

    def __hash__(self):
        return hash(self.url)

    def __eq__(self, other):
        return isinstance(other, Article) and self.url == other.url


def parse_digest_file(file_path: Path) -> List[Article]:
    """Parse a digest markdown file and extract all articles."""
    articles = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the date from the file name
        digest_date = file_path.stem  # YYYY-MM-DD

        # Current topic being processed
        current_topic = None

        # Split into sections by topic headers
        topic_pattern = r'^## (.+) — Top \d+$'
        lines = content.split('\n')

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for topic header
            topic_match = re.match(topic_pattern, line)
            if topic_match:
                current_topic = topic_match.group(1)
                # Map display names back to standard names
                if "LLMs" in current_topic:
                    current_topic = "LLMs"
                elif "RAG" in current_topic:
                    current_topic = "RAG"
                elif "MCP" in current_topic or "Model Context Protocol" in current_topic:
                    current_topic = "MCP"
                elif "Quantum" in current_topic:
                    current_topic = "Quantum Computing"
                i += 1
                continue

            # Check for article entry (starts with number and **[title](url)**)
            article_match = re.match(r'^\d+\.\s+\*\*\[(.+?)\]\((.+?)\)\*\*', line)
            if article_match and current_topic:
                title = article_match.group(1)
                url = article_match.group(2)

                # Parse the next few lines for metadata
                source = author = published = reading_time = ""
                summary = why_selected = ""

                # Next line should have metadata
                if i + 1 < len(lines):
                    metadata_line = lines[i + 1].strip()
                    # _Source_: ... · _Author_: ... · _Published_: ... · _Reading time_: ...
                    source_match = re.search(r'_Source_:\s*([^·]+)', metadata_line)
                    author_match = re.search(r'_Author_:\s*([^·]+)', metadata_line)
                    published_match = re.search(r'_Published_:\s*([^·]+)', metadata_line)
                    reading_match = re.search(r'_Reading time_:\s*(.+)', metadata_line)

                    if source_match:
                        source = source_match.group(1).strip()
                    if author_match:
                        author = author_match.group(1).strip()
                    if published_match:
                        published = published_match.group(1).strip()
                    if reading_match:
                        reading_time = reading_match.group(1).strip()

                # Look for summary and why selected in the next lines
                j = i + 2
                while j < len(lines) and not lines[j].strip().startswith('##') and not re.match(r'^\d+\.', lines[j].strip()):
                    line_content = lines[j].strip()
                    if line_content.startswith('**Summary:**'):
                        summary = re.sub(r'\*\*Summary:\*\*\s*', '', line_content).strip()
                    elif line_content.startswith('**Why selected:**'):
                        why_selected = re.sub(r'\*\*Why selected:\*\*\s*', '', line_content).strip()
                    elif not line_content or line_content == '---':
                        break
                    j += 1

                # Create article object
                article = Article(
                    title=title,
                    url=url,
                    source=source,
                    author=author,
                    published=published,
                    reading_time=reading_time,
                    summary=summary,
                    why_selected=why_selected,
                    topic=current_topic,
                    digest_date=digest_date
                )
                articles.append(article)

            i += 1

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return articles


def consolidate_digests(digests_dir: Path, days: int = 30) -> Dict[str, List[Article]]:
    """
    Consolidate all digests from the last N days.

    Returns a dictionary mapping topic names to lists of unique articles.
    """
    cutoff_date = datetime.now() - timedelta(days=days)

    # Find all digest files within the date range
    digest_files = []
    for file_path in digests_dir.glob("*.md"):
        try:
            # Parse date from filename (YYYY-MM-DD.md)
            date_str = file_path.stem
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            if file_date >= cutoff_date:
                digest_files.append(file_path)
        except ValueError:
            # Skip files that don't match the date format
            continue

    print(f"Found {len(digest_files)} digest files in the last {days} days")

    # Parse all articles
    all_articles = []
    for file_path in sorted(digest_files, reverse=True):
        print(f"Processing {file_path.name}...")
        articles = parse_digest_file(file_path)
        all_articles.extend(articles)
        print(f"  Found {len(articles)} articles")

    # Deduplicate by URL
    unique_articles: Dict[str, Article] = {}
    for article in all_articles:
        if article.url not in unique_articles:
            unique_articles[article.url] = article

    print(f"\nTotal articles after deduplication: {len(unique_articles)}")

    # Group by topic
    by_topic: Dict[str, List[Article]] = defaultdict(list)
    for article in unique_articles.values():
        by_topic[article.topic].append(article)

    # Sort articles within each topic by published date (most recent first)
    for topic in by_topic:
        by_topic[topic].sort(key=lambda a: a.published or "", reverse=True)
        print(f"{topic}: {len(by_topic[topic])} unique articles")

    return dict(by_topic)


def generate_consolidated_markdown(articles_by_topic: Dict[str, List[Article]],
                                   output_path: Path):
    """Generate a consolidated markdown file."""
    lines = [
        "# Daily Tech Digest — Last 30 Days",
        "",
        "> Consolidated digest of the most interesting English-language articles about LLMs, RAG, MCP (Model Context Protocol), and Quantum computing from the last 30 days.",
        f"> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
    ]

    # Topic order
    topic_order = [
        ("LLMs", "LLMs"),
        ("RAG", "RAG"),
        ("MCP", "MCP (Model Context Protocol)"),
        ("Quantum Computing", "Quantum Computing"),
    ]

    for topic_key, topic_display in topic_order:
        articles = articles_by_topic.get(topic_key, [])

        lines.append(f"## {topic_display}")
        lines.append("")

        if not articles:
            lines.append("(No articles found in the last 30 days.)")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        lines.append(f"*{len(articles)} articles*")
        lines.append("")

        for idx, article in enumerate(articles, start=1):
            lines.append(f"{idx}. **[{article.title}]({article.url})**  ")
            lines.append(f"   _Source_: {article.source} · _Author_: {article.author} · _Published_: {article.published} · _Reading time_: {article.reading_time}  ")
            if article.summary:
                lines.append(f"   **Summary:** {article.summary}  ")
            if article.why_selected:
                lines.append(f"   **Why selected:** {article.why_selected}  ")
            lines.append(f"   _First appeared in digest_: {article.digest_date}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nConsolidated digest written to: {output_path}")


def main():
    """Main function."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    digests_dir = project_dir / "digests"
    output_file = digests_dir / "last-30-days.md"

    if not digests_dir.exists():
        print(f"Error: Digests directory not found: {digests_dir}")
        sys.exit(1)

    print("=" * 60)
    print("DIGEST CONSOLIDATION SCRIPT")
    print("=" * 60)
    print()

    # Consolidate digests
    articles_by_topic = consolidate_digests(digests_dir, days=30)

    # Generate consolidated markdown
    generate_consolidated_markdown(articles_by_topic, output_file)

    print()
    print("=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
