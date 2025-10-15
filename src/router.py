"""
Topic routing model: assigns an article to the most suitable topic using embeddings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer, util as st_util
except Exception:  # pragma: no cover - optional dependency environment
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore


class TopicRouter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        routing_cfg = (config.get("processing", {}).get("routing", {}) or {})
        self.enabled: bool = bool(routing_cfg.get("enabled", False))
        self.model_name: str = str(routing_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        # Confidence margin between best and second-best to consider decision confident
        self.confidence_margin: float = float(routing_cfg.get("confidence_margin", 0.08))
        # If the best topic differs from input topic and margin satisfied, allow override
        self.force_override_if_best_differs: bool = bool(routing_cfg.get("force_override_if_best_differs", True))

        # Build topic profiles
        self.topic_names: List[str] = [t.get("name") for t in config.get("topics", [])]
        self.topic_texts: List[str] = []
        for t in config.get("topics", []):
            name = t.get("name")
            keywords = t.get("keywords", []) or []
            profile = f"{name}. Keywords: " + ", ".join(keywords)
            self.topic_texts.append(profile)

        self._model = None
        self._topic_embeddings = None

        if self.enabled:
            self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if SentenceTransformer is None:
            logger.warning("TopicRouter enabled but sentence-transformers is not available; disabling router.")
            self.enabled = False
            return
        try:
            self._model = SentenceTransformer(self.model_name)
            self._topic_embeddings = self._model.encode(self.topic_texts, normalize_embeddings=True)
            logger.info(f"TopicRouter model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load TopicRouter model '{self.model_name}': {e}; disabling router.")
            self.enabled = False

    def route(self, title: str, text: str, input_topic: str | None) -> Tuple[str | None, Dict[str, float], float, bool]:
        """Return (best_topic, scores_by_topic, margin, should_override).

        - best_topic: name of the topic with highest score
        - scores_by_topic: mapping of topic name to cosine similarity score
        - margin: best - second_best similarity
        - should_override: whether router recommends overriding input_topic
        """
        if not self.enabled or not title and not text:
            return None, {}, 0.0, False
        self._ensure_model_loaded()
        if not self.enabled:
            return None, {}, 0.0, False

        try:
            doc = (title or "").strip()
            if text:
                # Prioritize title but include a bit of body context
                doc = f"{title.strip()}\n\n{text[:2000]}"
            doc_emb = self._model.encode([doc], normalize_embeddings=True)
            sims = st_util.cos_sim(doc_emb, self._topic_embeddings).tolist()[0]
            pairs = list(zip(self.topic_names, sims))
            pairs.sort(key=lambda x: x[1], reverse=True)
            best_topic, best_score = pairs[0]
            second_score = pairs[1][1] if len(pairs) > 1 else 0.0
            margin = float(best_score - second_score)
            scores = {name: float(score) for name, score in pairs}

            should_override = False
            if input_topic is None:
                should_override = True if margin >= self.confidence_margin else False
            else:
                if best_topic != input_topic and margin >= self.confidence_margin and self.force_override_if_best_differs:
                    should_override = True

            return best_topic, scores, margin, should_override
        except Exception as e:
            logger.debug(f"TopicRouter routing error: {e}")
            return None, {}, 0.0, False


