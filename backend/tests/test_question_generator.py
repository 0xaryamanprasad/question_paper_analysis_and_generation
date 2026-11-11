"""
Unit Tests for Module 10: Question Generation Agent.
"""

import pytest

from backend.services.question_generator import QuestionGenerator


@pytest.fixture
def question_generator():
    return QuestionGenerator()


class TestQuestionGenerator:
    """Test suite for QuestionGenerator service."""

    def test_init_and_fallback_ready(self, question_generator: QuestionGenerator):
        """Should initialize cleanly with model configuration."""
        assert question_generator.model_name is not None

    def test_generate_single_question(self, question_generator: QuestionGenerator):
        """Should generate a valid, syllabus-compliant question item."""
        q = question_generator.generate_question(
            subject="Data Structures and Algorithms",
            unit_number=1,
            unit_name="Introduction to Data Structures",
            topic="Stacks and applications",
            keywords=["stack", "LIFO", "push", "pop"],
            blooms_level="Understand",
            marks=5.0,
            difficulty="medium",
            question_type="medium",
        )

        assert q.unit_number == 1
        assert q.topic == "Stacks and applications"
        assert q.blooms_level == "Understand"
        assert q.marks == 5.0
        assert len(q.question_text) > 15
        assert len(q.marking_scheme) > 0

    def test_generate_composite_question_with_subparts(
        self, question_generator: QuestionGenerator
    ):
        """High-marks questions (>=8m) should generate subparts whose marks sum to total."""
        q = question_generator.generate_question(
            subject="Data Structures and Algorithms",
            unit_number=3,
            unit_name="Graphs",
            topic="Shortest path algorithms (Dijkstra, Bellman-Ford)",
            keywords=["Dijkstra", "Bellman-Ford", "shortest path"],
            blooms_level="Apply",
            marks=10.0,
            difficulty="hard",
            question_type="long",
        )

        assert q.marks == 10.0
        assert len(q.sub_parts) >= 2

        # Subparts sum must equal total marks
        subparts_sum = sum(sp.marks for sp in q.sub_parts)
        assert abs(subparts_sum - 10.0) < 0.1

    def test_candidate_pool_generation(self, question_generator: QuestionGenerator):
        """Should generate candidate pool covering all provided units."""
        units = [
            {"unit_number": 1, "unit_name": "Unit 1", "topics": ["Arrays", "Stacks"], "keywords": ["stack"]},
            {"unit_number": 2, "unit_name": "Unit 2", "topics": ["BST", "AVL"], "keywords": ["tree"]},
        ]
        pool = question_generator.generate_candidate_pool(
            subject="DSA",
            units_data=units,
            target_marks_list=[2.0, 5.0, 10.0],
            blooms_levels=["Remember", "Understand", "Apply"],
            count_per_unit=2,
        )

        assert len(pool) == 4
        unit_numbers = {q.unit_number for q in pool}
        assert unit_numbers == {1, 2}
