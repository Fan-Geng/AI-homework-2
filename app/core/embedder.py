import logging
import math

logger = logging.getLogger("model_router.embedder")

INTENT_CATEGORIES = {
    "coding": {
        "description": "Code generation, debugging, programming, software development, scripts, algorithms",
        "target_routing_key": "claude-sonnet-5",
    },
    "translation": {
        "description": "Language translation, translate text between languages, multilingual",
        "target_routing_key": "deepseek-v4-flash",
    },
    "reasoning": {
        "description": "Logic, math, analysis, chain-of-thought, problem solving, deduction, proofs",
        "target_routing_key": "deepseek-v4-pro",
    },
    "creative_writing": {
        "description": "Stories, essays, creative content, poetry, fiction, narrative, novel",
        "target_routing_key": "claude-sonnet-5",
    },
    "analysis": {
        "description": "Data analysis, summarization, comparison, research, evaluation, review",
        "target_routing_key": "gpt-5.5",
    },
    "general": {
        "description": "General chat, questions, casual conversation, information, help",
        "target_routing_key": "gpt-5.4-mini",
    },
}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.4,
                 cache_dir: str = ".embed_cache", categories: dict | None = None):
        self.model_name = model_name
        self.threshold = threshold
        self.cache_dir = cache_dir
        self.model = None
        self.category_names: list[str] = []
        self.category_embeddings: list[list[float]] = []
        self.category_keys: list[str] = []
        self._loaded = False

        self._category_config = categories or INTENT_CATEGORIES

    def load(self):
        if self._loaded:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Semantic routing disabled. "
                "Install with: pip install sentence-transformers"
            )
            return

        try:
            logger.info("Loading embedding model '%s' (cached in %s)...", self.model_name, self.cache_dir)
            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)

            self.category_names = list(self._category_config.keys())
            self.category_keys = [
                self._category_config[name]["target_routing_key"]
                for name in self.category_names
            ]
            descriptions = [
                self._category_config[name]["description"]
                for name in self.category_names
            ]

            embeddings = self.model.encode(descriptions, normalize_embeddings=True)
            self.category_embeddings = [e.tolist() for e in embeddings]
            self._loaded = True
            logger.info("Embedding model loaded. %d categories ready.", len(self.category_names))
        except Exception as e:
            logger.warning("Failed to load embedding model: %s. Semantic routing disabled.", e)
            self.model = None

    @property
    def enabled(self) -> bool:
        return self._loaded and self.model is not None

    def classify(self, text: str) -> tuple[str, float, str] | None:
        if not self.enabled or not text.strip():
            return None

        try:
            embedding = self.model.encode([text], normalize_embeddings=True)[0].tolist()

            best_idx = 0
            best_score = -1.0
            for i, cat_emb in enumerate(self.category_embeddings):
                score = _cosine_similarity(cat_emb, embedding)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_score < self.threshold:
                return None

            category = self.category_names[best_idx]
            routing_key = self.category_keys[best_idx]
            return (category, best_score, routing_key)
        except Exception as e:
            logger.warning("Semantic classification failed: %s", e)
            return None

    def get_categories(self) -> list[dict]:
        result = []
        for name in self.category_names:
            config = self._category_config[name]
            result.append({
                "category": name,
                "description": config["description"],
                "target_routing_key": config["target_routing_key"],
            })
        return result
