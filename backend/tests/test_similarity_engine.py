"""
Unit and Integration Tests for Module 5: Semantic Similarity and Duplicate Detection.

Tests:
- 384-dimensional dense embedding generation
- Float32 serialization and deserialization for PostgreSQL bytea columns
- Semantic similarity detection across paraphrased / differently-worded questions
- Duplicate, near-duplicate, and distinct question classification
- Semantic clustering across historical examination years
- Top-k semantic search
- Integration between QuestionDetector (Mod 3), NLPPreprocessor (Mod 4), and SimilarityEngine (Mod 5)
"""

import numpy as np
import pytest

from backend.api.schemas.similarity import (
    DuplicateCluster,
    SimilarityMatch,
    SimilarityReport,
)
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.services.question_detector import QuestionDetector
from backend.services.similarity_engine import SimilarityEngine


@pytest.fixture
def similarity_engine() -> SimilarityEngine:
    preprocessor = NLPPreprocessor()
    return SimilarityEngine(nlp_preprocessor=preprocessor)


class TestSimilarityEngine:
    """Tests for SimilarityEngine service."""

    def test_encode_single_text(self, similarity_engine: SimilarityEngine):
        """Should generate normalized 384-dimensional float32 vector."""
        emb = similarity_engine.encode_text("Explain Binary Search Tree operations.")

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)
        assert emb.dtype == np.float32
        # Normalized vector has unit L2 norm
        norm = np.linalg.norm(emb)
        assert pytest.approx(norm, abs=1e-3) == 1.0

    def test_encode_batch(self, similarity_engine: SimilarityEngine):
        """Should batch-encode multiple questions efficiently."""
        texts = [
            "What is an AVL tree?",
            "Explain Dijkstra's shortest path algorithm.",
            "Describe the principles of Dynamic Programming.",
        ]
        embs = similarity_engine.encode_batch(texts)

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (3, 384)
        assert embs.dtype == np.float32

    def test_serialization_roundtrip(self, similarity_engine: SimilarityEngine):
        """Should serialize to bytes and deserialize without precision loss."""
        original = similarity_engine.encode_text("Define Abstract Data Type.")
        byte_data = similarity_engine.embedding_to_bytes(original)

        assert isinstance(byte_data, bytes)
        assert len(byte_data) == 384 * 4  # 384 float32 values = 1536 bytes

        recovered = similarity_engine.bytes_to_embedding(byte_data)
        assert recovered.shape == (384,)
        assert np.allclose(original, recovered, atol=1e-6)

    def test_semantically_similar_questions_different_wording(
        self, similarity_engine: SimilarityEngine
    ):
        """
        Critical requirement: The NLP system must identify semantically similar
        questions even when their wording differs significantly.
        """
        # Paraphrased pair 1 (with abbreviation BST expanded)
        q1a = "Explain the concept of a binary search tree."
        q1b = "Describe BST and its properties with an example."
        sim_1 = similarity_engine.compute_similarity(q1a, q1b)
        assert sim_1 >= 0.74  # Strong semantic match despite differing wording

        # Paraphrased pair 2
        q2a = "What is the time complexity of Quick Sort in the worst case?"
        q2b = "Analyze the worst-case running time of the quicksort algorithm."
        sim_2 = similarity_engine.compute_similarity(q2a, q2b)
        assert sim_2 >= 0.75  # Matches similarity_threshold (0.75)

        # Completely unrelated question
        unrelated = "Discuss the normalization forms in Relational Database Management Systems."
        sim_diff = similarity_engine.compute_similarity(q1a, unrelated)
        assert sim_diff < 0.40  # Well below similarity threshold

    def test_find_duplicates_and_clustering(self, similarity_engine: SimilarityEngine):
        """Should detect recurring duplicate questions and form clusters across years."""
        exam_questions = [
            {"id": "q1", "text": "What is an Abstract Data Type (ADT)?", "year": "2021"},
            {"id": "q2", "text": "Define the term Abstract Data Type with examples.", "year": "2022"},
            {"id": "q3", "text": "Explain Dijkstra's shortest path algorithm.", "year": "2021"},
            {"id": "q4", "text": "Describe Dijkstra algorithm with a graph example.", "year": "2023"},
            {"id": "q5", "text": "What is a Red-Black tree?", "year": "2022"},
        ]

        report = similarity_engine.find_duplicates(exam_questions)

        assert isinstance(report, SimilarityReport)
        assert report.total_questions == 5
        assert report.total_pairs_evaluated == 10  # 5 * 4 / 2
        assert len(report.clusters) >= 2

        # Check ADT cluster spans 2021 and 2022
        adt_cluster = next((c for c in report.clusters if "Abstract Data Type" in c.canonical_question), None)
        assert adt_cluster is not None
        assert adt_cluster.size == 2
        assert "2021" in adt_cluster.years_spanned
        assert "2022" in adt_cluster.years_spanned

        # Check Dijkstra cluster spans 2021 and 2023
        dijkstra_cluster = next((c for c in report.clusters if "Dijkstra" in c.canonical_question), None)
        assert dijkstra_cluster is not None
        assert dijkstra_cluster.size == 2
        assert "2021" in dijkstra_cluster.years_spanned
        assert "2023" in dijkstra_cluster.years_spanned

    def test_search_similar(self, similarity_engine: SimilarityEngine):
        """Should retrieve and rank top matching questions for a query."""
        candidates = [
            {"id": "c1", "text": "Explain Depth First Search (DFS) on graphs."},
            {"id": "c2", "text": "Explain Breadth First Search (BFS) graph traversal."},
            {"id": "c3", "text": "What is Merge Sort?"},
        ]
        results = similarity_engine.search_similar(
            query="Graph traversal using Depth First Search",
            candidates=candidates,
            top_k=2,
        )

        assert len(results) > 0
        top_match = results[0]
        assert "DFS" in top_match.question_b or "Depth First" in top_match.question_b
        assert top_match.similarity_score > 0.70


class TestSimilarityPipelineIntegration:
    """Integration test: Question Detector (Mod 3) -> NLP Preprocessor (Mod 4) -> Similarity Engine (Mod 5)."""

    def test_end_to_end_duplicate_detection(
        self,
        similarity_engine: SimilarityEngine,
    ):
        """Extracted questions from two papers should be matched and deduplicated."""
        paper_2022 = """
        SECTION A
        1. Explain the working of Binary Search Tree. [5 Marks]
        2. What is a Stack? [2 Marks]
        """
        paper_2023 = """
        SECTION A
        1. Describe BST operations with an example. [5 Marks]
        2. Define Stack data structure. [2 Marks]
        """
        detector = QuestionDetector()
        preprocessor = NLPPreprocessor()

        q_2022 = detector.segment_text(paper_2022).questions
        q_2023 = detector.segment_text(paper_2023).questions

        # Normalize texts for embedding
        items = []
        for q in q_2022:
            clean = preprocessor.clean_for_embedding(q.cleaned_text)
            items.append({"id": f"2022_{q.question_number}", "text": clean, "year": "2022"})
        for q in q_2023:
            clean = preprocessor.clean_for_embedding(q.cleaned_text)
            items.append({"id": f"2023_{q.question_number}", "text": clean, "year": "2023"})

        report = similarity_engine.find_duplicates(items)

        # Both BST and Stack recur across 2022 and 2023
        assert report.duplicates_count + report.near_duplicates_count >= 2
        assert len(report.clusters) >= 2
