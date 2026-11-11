"""
Unit and Integration Tests for Module 6: Topic and Unit Classification.
"""

import pytest

from backend.api.schemas.classification import SyllabusDefinition, UnitDefinition
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.services.similarity_engine import SimilarityEngine
from backend.services.topic_classifier import TopicClassifier


@pytest.fixture(scope="module")
def nlp_preprocessor():
    return NLPPreprocessor()


@pytest.fixture(scope="module")
def similarity_engine(nlp_preprocessor):
    return SimilarityEngine(nlp_preprocessor=nlp_preprocessor)


@pytest.fixture(scope="module")
def topic_classifier(nlp_preprocessor, similarity_engine):
    """Module-scoped TopicClassifier sharing loaded models for fast execution."""
    return TopicClassifier(
        nlp_preprocessor=nlp_preprocessor,
        similarity_engine=similarity_engine,
        enable_semantic=True,
    )


class TestTopicClassifier:
    """Test suite for TopicClassifier service."""

    def test_default_syllabus_loaded(self, topic_classifier: TopicClassifier):
        """Should load sample syllabus with 5 units correctly."""
        syllabus = topic_classifier.syllabus
        assert syllabus.subject == "Data Structures and Algorithms"
        assert syllabus.course_code == "CS201"
        assert len(syllabus.units) == 5
        assert len(topic_classifier.topic_profiles) > 15

    def test_custom_syllabus_loading(self, nlp_preprocessor):
        """Should initialize with a custom custom syllabus definition."""
        custom_syllabus = SyllabusDefinition(
            subject="Operating Systems",
            course_code="CS301",
            units=[
                UnitDefinition(
                    unit_number=1,
                    unit_name="Process Management",
                    topics=["CPU Scheduling", "Process Synchronization", "Deadlocks"],
                    keywords=["process", "thread", "scheduling", "semaphore", "deadlock"],
                )
            ],
        )
        classifier = TopicClassifier(
            syllabus=custom_syllabus,
            nlp_preprocessor=nlp_preprocessor,
            enable_semantic=False,
        )
        assert classifier.syllabus.subject == "Operating Systems"
        assert len(classifier.syllabus.units) == 1

        res = classifier.classify_question("Explain the banker's algorithm for deadlock avoidance.")
        assert res.unit_number == 1
        assert "deadlock" in [k.lower() for k in res.matched_keywords]
        assert res.confidence > 0.3

    def test_classify_unit_1_data_structures(self, topic_classifier: TopicClassifier):
        """Should correctly classify stacks/queues/linked lists into Unit 1."""
        q = "Explain the concept of an Abstract Data Type (ADT) and its implementation using Singly Linked Lists."
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 1
        assert res.confidence >= 0.50
        assert any(k in ["ADT", "linked list", "singly linked", "Abstract Data Types"] for k in res.matched_keywords)

    def test_classify_unit_2_trees(self, topic_classifier: TopicClassifier):
        """Should correctly classify AVL/BST/traversals into Unit 2."""
        q = "Construct an AVL tree by inserting elements [15, 20, 24, 10, 13] and explain RR and LL rotations."
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 2
        assert "Trees" in res.unit_name
        assert res.confidence >= 0.50

    def test_classify_unit_3_graphs(self, topic_classifier: TopicClassifier):
        """Should correctly classify Dijkstra / BFS / DFS into Unit 3."""
        q = "Explain Dijkstra's shortest path algorithm with a weighted directed graph example."
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 3
        assert "Graphs" in res.unit_name
        assert res.confidence >= 0.50

    def test_classify_unit_4_sorting(self, topic_classifier: TopicClassifier):
        """Should correctly classify Quick Sort / Merge Sort / Hashing into Unit 4."""
        q = "Describe the Quick Sort algorithm. How does the partition subroutine select the pivot?"
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 4
        assert "Sorting" in res.unit_name
        assert res.confidence >= 0.50

    def test_classify_unit_5_algorithm_analysis(self, topic_classifier: TopicClassifier):
        """Should correctly classify Big-O / Dynamic Programming into Unit 5."""
        q = "Define asymptotic notations Big-O, Omega, and Theta. Analyze the time complexity of recursive algorithms."
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 5
        assert "Algorithm" in res.unit_name
        assert res.confidence >= 0.50

    def test_acronym_expansion_integration(self, topic_classifier: TopicClassifier):
        """Domain acronyms like MST, BFS, DFS should match syllabus keywords."""
        q = "Find the MST of the given graph using Prim's algorithm."
        res = topic_classifier.classify_question(q)
        assert res.unit_number == 3
        assert res.confidence >= 0.50

    def test_empty_and_unknown_question(self, topic_classifier: TopicClassifier):
        """Empty string should return clean empty result without throwing exceptions."""
        res = topic_classifier.classify_question("")
        assert res.unit_number is None
        assert res.confidence == 0.0

    def test_batch_classification_all_units(self, topic_classifier: TopicClassifier):
        """Batch classification should classify each question in order across units."""
        questions = [
            {"id": "q1", "text": "What is a circular queue? Explain enqueue and dequeue operations."},
            {"id": "q2", "text": "What is a Binary Search Tree (BST)? Show inorder traversal."},
            {"id": "q3", "text": "Explain Kruskal's algorithm to compute minimum spanning tree."},
            {"id": "q4", "text": "Compare Merge Sort and Quick Sort in terms of worst case complexity."},
            {"id": "q5", "text": "Explain the principle of Dynamic Programming with optimal substructure."},
        ]

        results = topic_classifier.classify_batch(questions)
        assert len(results) == 5

        expected_units = [1, 2, 3, 4, 5]
        for i, res in enumerate(results):
            assert res.question_id == questions[i]["id"]
            assert res.unit_number == expected_units[i], (
                f"Question {i+1} ('{res.question_text}') classified as Unit {res.unit_number}, "
                f"expected Unit {expected_units[i]}"
            )
            assert res.confidence >= 0.45
            assert len(res.top_candidates) > 0
