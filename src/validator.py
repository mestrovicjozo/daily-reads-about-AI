"""
Validator for generated digests.
Checks:
- Exactly 5 articles per topic section.
- Each item has required metadata lines.
- Summary has exactly 5 sentences.
- No duplicate links within a topic and across topics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

TOPIC_HEADINGS = [
    ("LLMs", "## LLMs — Top 5"),
    ("RAG", "## RAG — Top 5"),
    ("MCP", "## MCP (Model Context Protocol) — Top 5"),
    ("Quantum Computing", "## Quantum computing — Top 5"),
]

ITEM_RE = re.compile(r"^\d+\. \*\*\[([^\]]+)\]\(([^\)]+)\)\*\*\s*$")
META_RE = re.compile(r"^\s*_Source_:\s*(.+?)\s*·\s*_Author_:\s*(.+?)\s*·\s*_Published_:\s*(.+?)\s*·\s*_Reading time_:\s*(\d+)\s*min\s*$")
SUMMARY_RE = re.compile(r"^\s*\*\*Summary:\*\*\s*(.+?)\s*$")
WHY_RE = re.compile(r"^\s*\*\*Why selected:\*\*\s*(.+?)\s*$")


def count_sentences(text: str) -> int:
    # Simple sentence boundary count by punctuation followed by space or end
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    sents = [s for s in sents if s]
    return len(sents)


def parse_sections(lines: List[str]) -> Dict[str, List[List[str]]]:
    """Return mapping topic_name -> list of item blocks (each block is list of lines)."""
    sections: Dict[str, List[List[str]]] = {k: [] for k, _ in TOPIC_HEADINGS}

    # Build heading line to topic key map
    heading_to_topic = {h: k for k, h in TOPIC_HEADINGS}

    current_topic: str | None = None
    current_block: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line in heading_to_topic:
            current_topic = heading_to_topic[line]
            i += 1
            continue

        if current_topic:
            # Detect start of an item
            if ITEM_RE.match(line):
                # If there was a previous block not flushed, flush it
                if current_block:
                    sections[current_topic].append(current_block)
                    current_block = []
                current_block = [line]
                # Expect following lines for meta, summary, why
                j = i + 1
                while j < len(lines):
                    nxt = lines[j].rstrip("\n")
                    if nxt.strip().startswith("---") or ITEM_RE.match(nxt) or nxt.startswith("## "):
                        break
                    current_block.append(nxt)
                    j += 1
                sections[current_topic].append(current_block)
                current_block = []
                i = j
                continue
        i += 1

    # Edge-case: append any trailing block
    if current_topic and current_block:
        sections[current_topic].append(current_block)

    return sections


def validate_digest(path: str | Path) -> Tuple[bool, List[str]]:
    p = Path(path)
    if not p.exists():
        return False, [f"File not found: {p}"]
    lines = p.read_text(encoding="utf-8").splitlines()

    errors: List[str] = []

    # Ensure all headings are present
    for _, heading in TOPIC_HEADINGS:
        if heading not in lines:
            errors.append(f"Missing heading: {heading}")

    sections = parse_sections(lines)

    # Duplicate tracking
    global_links = set()

    for topic, items in sections.items():
        # Filter out empty placeholders like (No eligible articles found.)
        items = [blk for blk in items if blk and ITEM_RE.match(blk[0] or "")]
        if len(items) != 5:
            errors.append(f"Topic '{topic}' has {len(items)} items, expected 5")

        seen_links = set()
        for blk in items:
            # 1) Title + link
            m = ITEM_RE.match(blk[0])
            if not m:
                errors.append(f"{topic}: Invalid item format: {blk[0] if blk else 'EMPTY'}")
                continue
            title, link = m.group(1), m.group(2)
            if link in seen_links:
                errors.append(f"{topic}: Duplicate link within topic: {link}")
            seen_links.add(link)
            if link in global_links:
                errors.append(f"Duplicate link across topics: {link}")
            global_links.add(link)

            # 2) Metadata line
            meta_line = next((l for l in blk[1:] if l.strip().startswith("_Source_")), None)
            if not meta_line or not META_RE.match(meta_line):
                errors.append(f"{topic}: Missing or invalid metadata line for {link}")

            # 3) Summary line
            sum_line = next((l for l in blk[1:] if l.strip().startswith("**Summary:**")), None)
            if not sum_line:
                errors.append(f"{topic}: Missing summary line for {link}")
            else:
                sm = SUMMARY_RE.match(sum_line)
                if not sm:
                    errors.append(f"{topic}: Malformed summary for {link}")
                else:
                    summary_text = sm.group(1)
                    n = count_sentences(summary_text)
                    if n != 5:
                        errors.append(f"{topic}: Summary must have 5 sentences, found {n} for {link}")

            # 4) Why selected line
            why_line = next((l for l in blk[1:] if l.strip().startswith("**Why selected:**")), None)
            if not why_line or not WHY_RE.match(why_line):
                errors.append(f"{topic}: Missing or malformed 'Why selected' for {link}")

    return len(errors) == 0, errors
