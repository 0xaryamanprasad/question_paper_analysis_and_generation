"""
Module 5: Semantic Similarity and Duplicate Detection Service.

Generates dense semantic embeddings using Sentence Transformers (all-MiniLM-L6-v2),
computes cosine similarity matrices, identifies duplicate and recurring questions,
and performs threshold-based semantic clustering.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from backend.api.schemas.similarity import (
    DuplicateCluster,
    SimilarityMatch,
    SimilarityReport,
)
from backend.config import get_settings
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class SimilarityEngineError(Exception):
    """Raised when semantic similarity computation fails."""
    pass


class SimilarityEngine:
    """
    Engine for generating sentence embeddings and identifying duplicate questions.

    Features:
    - 384-dimensional dense sentence embeddings using all-MiniLM-L6-v2.
    - Offline loading from local workspace model cache.
    - Automatic abbreviation expansion & NLP cleaning before embedding.
    - Pairwise cosine similarity matrix computation.
    - Identification of exact duplicates (>= 0.90), near duplicates (>= 0.75),
      and semantically related questions (>= 0.60).
    - Connected-component clustering of recurring questions across years.
    - Vector serialization/deserialization for PostgreSQL bytea storage.
    """

    def __init__(
        self,
        model_name: str | None = None,
        similarity_threshold: float | None = None,
        duplicate_threshold: float | None = None,
        related_threshold: float = 0.60,
        nlp_preprocessor: NLPPreprocessor | None = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else settings.similarity_threshold
        )
        self.duplicate_threshold = (
            duplicate_threshold if duplicate_threshold is not None else settings.duplicate_threshold
        )
        self.related_threshold = related_threshold
        self.nlp_preprocessor = nlp_preprocessor

        # Ensure offline cache path points to local workspace
        workspace_root = Path(__file__).resolve().parent.parent.parent
        self.model_cache = workspace_root / ".model_cache"
        if self.model_cache.exists():
            os.environ["HF_HOME"] = str(self.model_cache)
            os.environ["HF_HUB_OFFLINE"] = "1"

        self._model = None
        logger.info(
            "similarity_engine_initialized",
            model_name=self.model_name,
            sim_threshold=self.similarity_threshold,
            dup_threshold=self.duplicate_threshold,
        )

    def _prepare_text(self, text: str) -> str:
        """Apply NLP abbreviation expansion and cleaning if preprocessor available."""
        if not text.strip():
            return ""
        if self.nlp_preprocessor is not None:
            expanded, _ = self.nlp_preprocessor.expand_abbreviations(text)
            return self.nlp_preprocessor.clean_for_embedding(expanded)
        return text.strip()

    @property
    def model(self):
        """Lazy-load the SentenceTransformer model on first invocation."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                local_only = self.model_cache.exists()
                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=local_only,
                )
                logger.info("sentence_transformer_loaded", model=self.model_name)
            except Exception as e:
                logger.error("sentence_transformer_load_failed", error=str(e))
                raise SimilarityEngineError(
                    f"Failed to load embedding model '{self.model_name}': {e}"
                ) from e
        return self._model

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate a normalized 384-dimensional embedding for a single text.

        Returns:
            1D float32 NumPy array of length 384.
        """
        prepared = self._prepare_text(text)
        if not prepared:
            return np.zeros(384, dtype=np.float32)

        try:
            emb = self.model.encode(prepared, convert_to_numpy=True, normalize_embeddings=True)
            return emb.astype(np.float32)
        except Exception as e:
            logger.error("encode_text_failed", error=str(e))
            raise SimilarityEngineError(f"Embedding generation failed: {e}") from e

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a list of texts in a single batch.

        Returns:
            2D float32 NumPy array of shape (N, 384).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        prepared_texts = [self._prepare_text(t) or "empty question" for t in texts]
        try:
            embs = self.model.encode(
                prepared_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embs.astype(np.float32)
        except Exception as e:
            logger.error("encode_batch_failed", error=str(e))
            raise SimilarityEngineError(f"Batch embedding generation failed: {e}") from e

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Convenience method accepting either a single string or a list of strings."""
        if isinstance(texts, str):
            return self.encode_text(texts)
        return self.encode_batch(texts)

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Serialize a float32 NumPy array into raw bytes for PostgreSQL bytea storage."""
        return embedding.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_embedding(raw_bytes: bytes) -> np.ndarray:
        """Deserialize raw bytes from PostgreSQL bytea storage into a float32 NumPy array."""
        return np.frombuffer(raw_bytes, dtype=np.float32)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute the cosine similarity score between two question texts."""
        emb_a = self.encode_text(text_a).reshape(1, -1)
        emb_b = self.encode_text(text_b).reshape(1, -1)
        sim = float(cosine_similarity(emb_a, emb_b)[0][0])
        return round(max(0.0, min(1.0, sim)), 4)

    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute the full pairwise N x N cosine similarity matrix."""
        if len(embeddings) == 0:
            return np.empty((0, 0), dtype=np.float32)
        sim_matrix = cosine_similarity(embeddings, embeddings)
        # Clip to [0, 1] due to potential precision rounding
        return np.clip(sim_matrix, 0.0, 1.0)

    def classify_relation(self, score: float) -> str:
        """Classify similarity score into relation type."""
        if score >= self.duplicate_threshold:
            return "duplicate"
        elif score >= self.similarity_threshold:
            return "near_duplicate"
        elif score >= self.related_threshold:
            return "related"
        return "distinct"

    def find_duplicates(
        self,
        questions: list[dict[str, Any]],
    ) -> SimilarityReport:
        """
        Identify duplicates, near duplicates, and clusters across questions.

        Args:
            questions: List of dicts, each containing:
                - 'text': question text (str)
                - 'id': optional identifier (str/UUID)
                - 'year': optional year (str)

        Returns:
            Comprehensive SimilarityReport.
        """
        n = len(questions)
        if n < 2:
            return SimilarityReport(
                total_questions=n,
                total_pairs_evaluated=0,
                duplicates_count=0,
                near_duplicates_count=0,
                matches=[],
                clusters=[],
            )

        texts = [q["text"] for q in questions]
        embeddings = self.encode_batch(texts)
        sim_matrix = self.compute_similarity_matrix(embeddings)

        matches: list[SimilarityMatch] = []
        dup_count = 0
        near_dup_count = 0
        pairs_evaluated = 0

        # Adjacency for connected component clustering (for duplicates & near duplicates)
        adj: dict[int, list[int]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                pairs_evaluated += 1
                score = round(float(sim_matrix[i, j]), 4)
                rel_type = self.classify_relation(score)

                if rel_type in ("duplicate", "near_duplicate", "related"):
                    match = SimilarityMatch(
                        question_a=texts[i],
                        question_b=texts[j],
                        similarity_score=score,
                        relation_type=rel_type,
                        question_a_id=str(questions[i].get("id", "")) or None,
                        question_b_id=str(questions[j].get("id", "")) or None,
                    )
                    matches.append(match)

                if rel_type == "duplicate":
                    dup_count += 1
                    adj[i].append(j)
                    adj[j].append(i)
                elif rel_type == "near_duplicate":
                    near_dup_count += 1
                    adj[i].append(j)
                    adj[j].append(i)

        # Build clusters using Breadth First Search on graph
        visited = set()
        clusters: list[DuplicateCluster] = []
        cluster_idx = 1

        for i in range(n):
            if i not in visited and len(adj[i]) > 0:
                component = []
                queue = [i]
                visited.add(i)

                while queue:
                    curr = queue.pop(0)
                    component.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                if len(component) > 1:
                    member_texts = [texts[idx] for idx in component]
                    member_ids = [str(questions[idx].get("id", idx)) for idx in component]
                    years = list(
                        {str(questions[idx].get("year", "")) for idx in component if questions[idx].get("year")}
                    )
                    clusters.append(
                        DuplicateCluster(
                            cluster_id=f"cluster_{cluster_idx:03d}",
                            canonical_question=member_texts[0],
                            member_questions=member_texts,
                            member_ids=member_ids,
                            size=len(member_texts),
                            years_spanned=sorted(years),
                        )
                    )
                    cluster_idx += 1

        logger.info(
            "similarity_analysis_completed",
            total_questions=n,
            pairs_evaluated=pairs_evaluated,
            duplicates=dup_count,
            near_duplicates=near_dup_count,
            clusters_found=len(clusters),
        )

        return SimilarityReport(
            total_questions=n,
            total_pairs_evaluated=pairs_evaluated,
            duplicates_count=dup_count,
            near_duplicates_count=near_dup_count,
            matches=matches,
            clusters=clusters,
        )

    # Convenience alias
    find_duplicates_and_clusters = find_duplicates

    def search_similar(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
        min_threshold: float = 0.5,
    ) -> list[SimilarityMatch]:
        """
        Find top-k most semantically similar candidate questions to a query.

        Args:
            query: Query text string.
            candidates: List of candidate question dicts (must have 'text').
            top_k: Maximum number of matches to return.
            min_threshold: Minimum cosine similarity score.

        Returns:
            Ranked list of SimilarityMatch objects.
        """
        if not candidates or not query.strip():
            return []

        query_emb = self.encode_text(query).reshape(1, -1)
        candidate_texts = [c["text"] for c in candidates]
        cand_embs = self.encode_batch(candidate_texts)

        scores = cosine_similarity(query_emb, cand_embs)[0]
        ranked_indices = np.argsort(scores)[::-1]

        results: list[SimilarityMatch] = []
        for idx in ranked_indices[:top_k]:
            score = round(float(scores[idx]), 4)
            if score < min_threshold:
                continue

            results.append(
                SimilarityMatch(
                    question_a=query,
                    question_b=candidate_texts[idx],
                    similarity_score=score,
                    relation_type=self.classify_relation(score),
                    question_a_id=None,
                    question_b_id=str(candidates[idx].get("id", "")) or None,
                )
            )

        return results
