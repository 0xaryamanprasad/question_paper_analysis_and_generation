"""
Unit Tests for Module 11: Question Paper Generation Agent.
"""

import pytest

from backend.api.schemas.analysis import HistoricalAnalysisResult
from backend.api.schemas.classification import SyllabusDefinition, UnitDefinition
from backend.api.schemas.generation import PaperBlueprintConfig, SectionConstraint
from backend.services.paper_generator import PaperGenerator
from backend.services.question_generator import QuestionGenerator


@pytest.fixture
def sample_syllabus():
    return SyllabusDefinition(
        subject="Data Structures and Algorithms",
        course_code="CS201",
        units=[
            UnitDefinition(unit_number=1, unit_name="Unit 1: DS Basics", topics=["Stacks", "Queues"], keywords=["stack"]),
            UnitDefinition(unit_number=2, unit_name="Unit 2: Trees", topics=["BST", "AVL"], keywords=["tree"]),
            UnitDefinition(unit_number=3, unit_name="Unit 3: Graphs", topics=["Dijkstra", "BFS"], keywords=["graph"]),
            UnitDefinition(unit_number=4, unit_name="Unit 4: Sorting", topics=["Quick Sort", "Merge Sort"], keywords=["sort"]),
            UnitDefinition(unit_number=5, unit_name="Unit 5: Complexity", topics=["Big-O", "DP"], keywords=["complexity"]),
        ],
    )


@pytest.fixture
def sample_blueprint():
    return PaperBlueprintConfig(
        subject="Data Structures and Algorithms",
        course_code="CS201",
        total_marks=50.0,
        duration_hours=2.0,
        institution_name="Antigravity University",
        examination_name="Mid-Term Examination",
        sections=[
            SectionConstraint(
                section_name="Part A",
                question_count=5,
                marks_per_question=2.0,
                allowed_question_types=["short"],
                has_internal_choice=False,
            ),
            SectionConstraint(
                section_name="Part B",
                question_count=4,
                marks_per_question=10.0,
                allowed_question_types=["long"],
                has_internal_choice=True,  # 4 choice pairs
            ),
        ],
    )


class TestPaperGenerator:
    """Test suite for PaperGenerator service."""

    def test_generate_complete_paper(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Should assemble a complete paper matching sections, marks, and units."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        assert paper.subject == sample_blueprint.subject
        assert paper.total_marks == 50.0
        assert len(paper.sections) == 2

        # Part A has 5 questions without choice
        part_a = paper.sections[0]
        assert len(part_a.questions) == 5
        assert part_a.total_section_marks == 10.0

        # Part B has 4 positions with internal choice (4 primary + 4 alternative = 8 questions)
        part_b = paper.sections[1]
        assert len(part_b.questions) == 8
        assert part_b.total_section_marks == 40.0

    def test_internal_choice_or_linking(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Internal choice questions must be linked via or_choice_alternative."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        part_b = paper.sections[1]
        primary_q = part_b.questions[0]
        alt_q = part_b.questions[1]

        assert primary_q.or_choice_alternative == alt_q.id
        assert alt_q.or_choice_alternative == primary_q.id
        assert "(OR)" in alt_q.question_number

    def test_unit_coverage_balance(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Should distribute questions across syllabus units."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        # All units should receive questions
        covered = set(paper.unit_coverage_summary.keys())
        assert len(covered) >= 4
