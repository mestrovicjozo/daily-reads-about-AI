# Daily Tech Digest Agent

This repository hosts an AI agent that runs daily and collects the most interesting English-language articles on:

- LLMs (Large Language Models)  
- RAG (Retrieval-Augmented Generation)  
- MCP (Model Context Protocol)  
- Quantum Computing  

The agent fetches fresh content from research hubs, technical blogs, and industry news sites. It then ranks, deduplicates, and summarizes the top 3 articles for each topic.

---

## Features
- **Daily updates** — runs automatically with GitHub Actions.
- **English only** — non-English content is filtered out.
- **Deduplication** — avoids repeating the same article across topics.
- **Summaries** — each article is distilled into ~5 sentences.
- **Consolidated digest** — the last 30 days of articles are available in a single file.
- **Markdown output** — daily results are saved in `digests/archive/YYYY-MM-DD.md` for easy reading on GitHub.

---

## Digest Structure

- **`digests/`** — Main digest directory
  - **Current consolidated digest** — Single file containing all unique articles from the last 30 days
  - **`archive/`** — Historical daily digests organized by date (YYYY-MM-DD.md)

The consolidated digest provides a convenient overview of all content without duplicates, while the archive preserves the original daily digests for historical reference.

---
