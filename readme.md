# Daily Tech Digest Agent

This repository hosts an AI agent that runs daily and collects the most interesting English-language articles on:

- LLMs (Large Language Models)  
- RAG (Retrieval-Augmented Generation)  
- MCP (Model Context Protocol)  
- Quantum Computing  

The agent fetches fresh content from research hubs, technical blogs, and industry news sites. It then ranks, deduplicates, and summarizes the top 5 articles for each topic.

---

## Features
- **Daily updates** — runs automatically with GitHub Actions.  
- **English only** — non-English content is filtered out.  
- **Deduplication** — avoids repeating the same article across topics.  
- **Summaries** — each article is distilled into ~5 sentences.  
- **Markdown output** — results are saved in `digests/YYYY-MM-DD.md` for easy reading on GitHub.  

---
