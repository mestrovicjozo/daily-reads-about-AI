"""
AI-powered content discovery and validation using Gemini API.
"""

import os
from typing import List, Dict, Optional
from loguru import logger
import google.generativeai as genai
from datetime import datetime


class AIAugmentor:
    """Uses Gemini API to augment content discovery for sparse topics."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.enabled = True
            logger.info("AI augmentation enabled with Gemini API")
        else:
            self.enabled = False
            logger.warning("GEMINI_API_KEY not found - AI augmentation disabled")

    def discover_sources(self, topic: str, existing_sources: List[str]) -> List[str]:
        """
        Use AI to discover new relevant RSS feeds or sources for a topic.

        Args:
            topic: The topic to find sources for (e.g., "RAG", "MCP", "Quantum Computing")
            existing_sources: List of existing source URLs to avoid duplicates

        Returns:
            List of suggested source URLs
        """
        if not self.enabled:
            return []

        try:
            prompt = f"""You are helping to find reliable RSS feeds and blog sources about {topic}.

Current existing sources:
{chr(10).join(f"- {s}" for s in existing_sources[:10])}

Please suggest 5-10 high-quality RSS feed URLs or blog index pages that cover {topic}.
Focus on:
- Official project blogs
- Academic/research institutions
- Tech company blogs
- Community aggregators
- GitHub repository feeds

Return ONLY a JSON array of URLs, no other text:
["url1", "url2", "url3"]
"""

            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Extract URLs from response
            import json
            try:
                urls = json.loads(text)
                logger.info(f"AI discovered {len(urls)} potential sources for {topic}")
                return urls
            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI response for source discovery: {text[:100]}")
                return []

        except Exception as e:
            logger.error(f"Error in AI source discovery: {e}")
            return []

    def validate_relevance(self, article: Dict, topic: str, threshold: float = 0.7) -> bool:
        """
        Use AI to validate if an article is truly relevant to a topic.

        Args:
            article: Dictionary with 'title', 'text', 'url' keys
            topic: The topic to validate against
            threshold: Confidence threshold (0-1)

        Returns:
            True if article is relevant, False otherwise
        """
        if not self.enabled:
            return True  # Fallback to accepting article

        try:
            title = article.get("title", "")
            text = article.get("text", "")[:1000]  # First 1000 chars

            prompt = f"""Evaluate if this article is relevant to the topic: {topic}

Article Title: {title}

Article Excerpt:
{text}

Is this article substantially about {topic}? Consider:
- Direct discussion of {topic} concepts
- Practical applications of {topic}
- Research or developments in {topic}

Respond with ONLY a JSON object:
{{"relevant": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

            response = self.model.generate_content(prompt)
            text_response = response.text.strip()

            import json
            try:
                result = json.loads(text_response)
                is_relevant = result.get("relevant", False)
                confidence = result.get("confidence", 0.0)

                if is_relevant and confidence >= threshold:
                    logger.debug(f"AI validated article as relevant: {title[:50]} "
                               f"(confidence: {confidence:.2f})")
                    return True
                else:
                    logger.debug(f"AI rejected article: {title[:50]} "
                               f"(relevant: {is_relevant}, confidence: {confidence:.2f})")
                    return False

            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI validation response: {text_response[:100]}")
                return True  # Fallback to accepting

        except Exception as e:
            logger.error(f"Error in AI relevance validation: {e}")
            return True  # Fallback to accepting

    def enhance_topic_classification(self, article: Dict,
                                     possible_topics: List[str]) -> Optional[str]:
        """
        Use AI to classify an article into the most relevant topic.

        Args:
            article: Dictionary with 'title', 'text', 'url' keys
            possible_topics: List of possible topic labels

        Returns:
            Best matching topic or None
        """
        if not self.enabled:
            return None

        try:
            title = article.get("title", "")
            text = article.get("text", "")[:1000]

            prompt = f"""Classify this article into the most relevant topic.

Article Title: {title}

Article Excerpt:
{text}

Possible topics:
{chr(10).join(f"- {t}" for t in possible_topics)}

Respond with ONLY a JSON object:
{{"topic": "exact topic name from list", "confidence": 0.0-1.0}}

If no topic fits well, use {{"topic": null, "confidence": 0.0}}
"""

            response = self.model.generate_content(prompt)
            text_response = response.text.strip()

            import json
            try:
                result = json.loads(text_response)
                topic = result.get("topic")
                confidence = result.get("confidence", 0.0)

                if topic and confidence >= 0.7:
                    logger.debug(f"AI classified article to {topic} "
                               f"(confidence: {confidence:.2f})")
                    return topic

            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI classification response: {text_response[:100]}")

        except Exception as e:
            logger.error(f"Error in AI topic classification: {e}")

        return None

    def generate_search_queries(self, topic: str, num_queries: int = 5) -> List[str]:
        """
        Generate diverse search queries for finding content about a topic.

        Args:
            topic: The topic to generate queries for
            num_queries: Number of queries to generate

        Returns:
            List of search query strings
        """
        if not self.enabled:
            return []

        try:
            prompt = f"""Generate {num_queries} diverse search queries to find recent articles about {topic}.

Queries should cover:
- Latest developments and news
- Research papers and academic content
- Practical tutorials and guides
- Industry applications and case studies
- Open source projects and tools

Return ONLY a JSON array of query strings:
["query1", "query2", "query3"]
"""

            response = self.model.generate_content(prompt)
            text = response.text.strip()

            import json
            try:
                queries = json.loads(text)
                logger.info(f"AI generated {len(queries)} search queries for {topic}")
                return queries
            except json.JSONDecodeError:
                logger.warning(f"Could not parse AI search query response: {text[:100]}")
                return []

        except Exception as e:
            logger.error(f"Error generating search queries: {e}")
            return []

    def improve_summary(self, article: Dict, max_sentences: int = 5) -> Optional[str]:
        """
        Generate an improved summary of an article.

        Args:
            article: Dictionary with 'title', 'text' keys
            max_sentences: Maximum number of sentences in summary

        Returns:
            Improved summary or None
        """
        if not self.enabled:
            return None

        try:
            title = article.get("title", "")
            text = article.get("text", "")[:3000]  # First 3000 chars

            prompt = f"""Create a concise {max_sentences}-sentence summary of this article.

Title: {title}

Content:
{text}

Summary should:
1. State the main claim or contribution
2. Explain the approach or method
3. Highlight key results or findings
4. Note limitations or caveats (if applicable)
5. Explain why this is significant

Respond with ONLY the summary text, no JSON or formatting.
"""

            response = self.model.generate_content(prompt)
            summary = response.text.strip()

            if summary and len(summary) > 50:
                return summary

        except Exception as e:
            logger.error(f"Error improving summary: {e}")

        return None
