"""
Unit and Integration Tests for Module 7: Bloom's Taxonomy Classification.
"""

import pytest

from backend.api.schemas.classification import BloomsLevelEnum
from backend.services.blooms_classifier import BloomsClassifier
from backend.services.nlp_preprocessor import NLPPreprocessor


@pytest.fixture(scope="module")
def nlp_preprocessor():
    return NLPPreprocessor()


@pytest.fixture(scope="module")
def blooms_classifier(nlp_preprocessor):
    return BloomsClassifier(nlp_preprocessor=nlp_preprocessor)


class TestBloomsClassifier:
    """Test suite for BloomsClassifier service."""

    def test_init_loads_all_levels(self, blooms_classifier: BloomsClassifier):
        """Should configure and load all 6 Bloom's revised taxonomy levels."""
        assert len(blooms_classifier.level_verbs) == 6
        for level in BloomsLevelEnum:
            assert level in blooms_classifier.level_verbs
            assert len(blooms_classifier.level_verbs[level]) > 0

    def test_classify_remember_level_1(self, blooms_classifier: BloomsClassifier):
        """'Define' / 'State' / 'What is' should map to Remember (L1)."""
        q = "Define the term Abstract Data Type. State any two properties."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.REMEMBER
        assert res.level_code == "L1"
        assert res.confidence >= 0.65
        assert any(k in ["define", "state"] for k in res.action_verbs + res.matched_keywords)

    def test_classify_understand_level_2(self, blooms_classifier: BloomsClassifier):
        """'Explain' / 'Describe' / 'Illustrate' should map to Understand (L2)."""
        q = "Explain the working of collision resolution in hash tables using chaining."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.UNDERSTAND
        assert res.level_code == "L2"
        assert res.confidence >= 0.65
        assert any(k in ["explain", "explain the"] for k in res.action_verbs + res.matched_keywords)

    def test_classify_apply_level_3(self, blooms_classifier: BloomsClassifier):
        """'Calculate' / 'Solve' / 'Construct' / 'Find' should map to Apply (L3)."""
        q = "Calculate the shortest path from vertex S to all other vertices using Dijkstra's algorithm for the given weighted graph."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.APPLY
        assert res.level_code == "L3"
        assert res.confidence >= 0.65
        assert any(k in ["calculate", "calculate the"] for k in res.action_verbs + res.matched_keywords)

    def test_classify_analyze_level_4(self, blooms_classifier: BloomsClassifier):
        """'Compare and contrast' / 'Derive' / 'Analyze' should map to Analyze (L4)."""
        q = "Compare and contrast Quick Sort and Merge Sort. Derive the worst-case time complexity of Quick Sort."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.ANALYZE
        assert res.level_code == "L4"
        assert res.confidence >= 0.65
        assert any(k in ["compare and contrast", "derive", "analyze"] for k in res.action_verbs + res.matched_keywords)

    def test_classify_evaluate_level_5(self, blooms_classifier: BloomsClassifier):
        """'Critique' / 'Evaluate' / 'Justify' should map to Evaluate (L5)."""
        q = "Critique the use of an adjacency matrix versus an adjacency list for sparse graphs. Justify your recommendation."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.EVALUATE
        assert res.level_code == "L5"
        assert res.confidence >= 0.65
        assert any(k in ["critique", "justify"] for k in res.action_verbs + res.matched_keywords)

    def test_classify_create_level_6(self, blooms_classifier: BloomsClassifier):
        """'Design' / 'Formulate' / 'Write a program' should map to Create (L6)."""
        q = "Design an optimal cache replacement algorithm for a memory hierarchy and write a program implementing it."
        res = blooms_classifier.classify_question(q)
        assert res.blooms_level == BloomsLevelEnum.CREATE
        assert res.level_code == "L6"
        assert res.confidence >= 0.65
        assert any(k in ["design", "write a program"] for k in res.action_verbs + res.matched_keywords)

    def test_empty_question_handling(self, blooms_classifier: BloomsClassifier):
        """Empty input should produce a safe fallback without crashing."""
        res = blooms_classifier.classify_question("")
        assert res.blooms_level == BloomsLevelEnum.UNDERSTAND
        assert res.level_code == "L2"
        assert res.confidence == 0.0
        assert res.action_verbs == []

    def test_batch_classification(self, blooms_classifier: BloomsClassifier):
        """Batch should process all questions maintaining ordering and IDs."""
        questions = [
            {"id": "q1", "text": "List the advantages of circular linked lists."},
            {"id": "q2", "text": "Describe how AVL tree rotations maintain balance."},
            {"id": "q3", "text": "Compute the minimum spanning tree using Prim's method."},
            {"id": "q4", "text": "Prove that the lower bound for comparison sorting is Omega(n log n)."},
            {"id": "q5", "text": "Evaluate which sorting algorithm is best suited for almost sorted data."},
            {"id": "q6", "text": "Devise a data structure that supports insert and find_median in O(log n) time."},
        ]

        results = blooms_classifier.classify_batch(questions)
        assert len(results) == 6

        expected_levels = [
            BloomsLevelEnum.REMEMBER,
            BloomsLevelEnum.UNDERSTAND,
            BloomsLevelEnum.APPLY,
            BloomsLevelEnum.ANALYZE,
            BloomsLevelEnum.EVALUATE,
            BloomsLevelEnum.CREATE,
        ]

        for i, res in enumerate(results):
            assert res.question_id == questions[i]["id"]
            assert res.blooms_level == expected_levels[i], (
                f"Question {i+1} ('{res.question_text}') classified as {res.blooms_level}, "
                f"expected {expected_levels[i]}"
            )
            assert res.confidence >= 0.50
