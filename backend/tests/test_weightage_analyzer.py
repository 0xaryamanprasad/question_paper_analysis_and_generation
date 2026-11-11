"""
Unit and Integration Tests for Module 9: Marks and Unit Weightage Analysis.
"""

import pytest

from backend.api.schemas.classification import SyllabusDefinition, UnitDefinition
from backend.services.weightage_analyzer import WeightageAnalyzer


@pytest.fixture
def weightage_analyzer():
    return WeightageAnalyzer()


@pytest.fixture
def sample_syllabus():
    return SyllabusDefinition(
        subject="Data Structures and Algorithms",
        course_code="CS201",
        units=[
            UnitDefinition(unit_number=1, unit_name="Introduction to Data Structures", topics=[], keywords=[]),
            UnitDefinition(unit_number=2, unit_name="Trees", topics=[], keywords=[]),
            UnitDefinition(unit_number=3, unit_name="Graphs", topics=[], keywords=[]),
            UnitDefinition(unit_number=4, unit_name="Sorting and Searching", topics=[], keywords=[]),
            UnitDefinition(unit_number=5, unit_name="Algorithm Design and Analysis", topics=[], keywords=[]),
        ],
    )


@pytest.fixture
def sample_exam_questions():
    """Questions across 2 papers with explicit marks and units."""
    return [
        # Paper 1 (Total: 50 marks)
        {"paper_id": "p1", "year": "2022", "unit_number": 1, "unit_name": "Unit 1", "topic": "Stacks", "marks": 2.0},
        {"paper_id": "p1", "year": "2022", "unit_number": 1, "unit_name": "Unit 1", "topic": "Queues", "marks": 5.0},
        {"paper_id": "p1", "year": "2022", "unit_number": 2, "unit_name": "Unit 2", "topic": "AVL Trees", "marks": 10.0},
        {"paper_id": "p1", "year": "2022", "unit_number": 3, "unit_name": "Unit 3", "topic": "Dijkstra", "marks": 15.0},
        {"paper_id": "p1", "year": "2022", "unit_number": 4, "unit_name": "Unit 4", "topic": "Quick Sort", "marks": 10.0},
        {"paper_id": "p1", "year": "2022", "unit_number": 4, "unit_name": "Unit 4", "topic": "Binary Search", "marks": 8.0},

        # Paper 2 (Total: 50 marks)
        {"paper_id": "p2", "year": "2023", "unit_number": 1, "unit_name": "Unit 1", "topic": "Linked Lists", "marks": 5.0},
        {"paper_id": "p2", "year": "2023", "unit_number": 2, "unit_name": "Unit 2", "topic": "B-Trees", "marks": 10.0},
        {"paper_id": "p2", "year": "2023", "unit_number": 3, "unit_name": "Unit 3", "topic": "Prim's Algorithm", "marks": 15.0},
        {"paper_id": "p2", "year": "2023", "unit_number": 4, "unit_name": "Unit 4", "topic": "Merge Sort", "marks": 10.0},
        {"paper_id": "p2", "year": "2023", "unit_number": 4, "unit_name": "Unit 4", "topic": "Hashing", "marks": 10.0},
    ]


class TestWeightageAnalyzer:
    """Test suite for WeightageAnalyzer service."""

    def test_empty_analysis(self, weightage_analyzer: WeightageAnalyzer):
        """Should safely handle empty input."""
        res = weightage_analyzer.analyze([], subject="DSA")
        assert res.total_papers == 0
        assert res.total_marks_all_papers == 0.0
        assert res.unit_weightages == []
        assert res.topic_weightages == []

    def test_total_marks_and_averages(
        self, weightage_analyzer: WeightageAnalyzer, sample_exam_questions
    ):
        """Should compute total marks across papers and average paper marks accurately."""
        res = weightage_analyzer.analyze(sample_exam_questions, subject="DSA")
        assert res.total_papers == 2
        assert res.total_marks_all_papers == 100.0  # 50 + 50
        assert res.average_paper_total_marks == 50.0

    def test_unit_weightage_percentages(
        self, weightage_analyzer: WeightageAnalyzer, sample_exam_questions
    ):
        """Should compute unit-wise marks and percentage allocations."""
        res = weightage_analyzer.analyze(sample_exam_questions, subject="DSA")
        weights = {u.unit_number: u.weightage_percentage for u in res.unit_weightages}

        # Unit 1 marks: 2 + 5 + 5 = 12 (12%)
        assert weights[1] == 12.0
        # Unit 2 marks: 10 + 10 = 20 (20%)
        assert weights[2] == 20.0
        # Unit 3 marks: 15 + 15 = 30 (30%)
        assert weights[3] == 30.0
        # Unit 4 marks: 10 + 8 + 10 + 10 = 38 (38%)
        assert weights[4] == 38.0

        # Sum of actual percentages should equal 100%
        assert round(sum(weights.values()), 1) == 100.0

    def test_blueprint_deviations_and_unrepresented_units(
        self,
        weightage_analyzer: WeightageAnalyzer,
        sample_exam_questions,
        sample_syllabus: SyllabusDefinition,
    ):
        """Should detect over-represented and under-represented units against syllabus blueprint."""
        # Blueprint: equal 20% across all 5 units
        expected_bp = {1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0, 5: 20.0}

        res = weightage_analyzer.analyze(
            sample_exam_questions,
            syllabus=sample_syllabus,
            expected_blueprint=expected_bp,
            subject="DSA",
        )

        assert len(res.unit_weightages) == 5

        # Unit 4 is 38% (expected 20%) -> deviation +18% (most weighty)
        assert 4 in res.most_weighty_units

        # Unit 5 had 0 questions -> underrepresented
        assert 5 in res.underrepresented_units

        # Unit 1 is 12% (expected 20%) -> deviation -8% (underrepresented)
        assert 1 in res.underrepresented_units

    def test_marks_by_question_type(
        self, weightage_analyzer: WeightageAnalyzer, sample_exam_questions
    ):
        """Should categorize marks by short, medium, and long answer question types."""
        res = weightage_analyzer.analyze(sample_exam_questions, subject="DSA")
        by_type = res.marks_by_question_type

        # Short (<=2 marks): one 2-mark question = 2 marks
        assert by_type["short"] == 2.0
        # Medium (3-7 marks): two 5-mark questions = 10 marks
        assert by_type["medium"] == 10.0
        # Long (>=8 marks): 10 + 15 + 10 + 8 + 10 + 15 + 10 + 10 = 88 marks
        assert by_type["long"] == 88.0

        assert sum(by_type.values()) == 100.0

    def test_topic_weightage_sorting(
        self, weightage_analyzer: WeightageAnalyzer, sample_exam_questions
    ):
        """Topic weightage should be sorted descending by total marks."""
        res = weightage_analyzer.analyze(sample_exam_questions, subject="DSA")
        topics = res.topic_weightages

        assert len(topics) > 0
        for i in range(len(topics) - 1):
            assert topics[i].total_marks >= topics[i + 1].total_marks
