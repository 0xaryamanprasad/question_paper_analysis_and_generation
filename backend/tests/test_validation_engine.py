"""
Unit Tests for Module 12: Question Validation Engine.
"""

import pytest

from backend.api.schemas.classification import SyllabusDefinition, UnitDefinition
from backend.api.schemas.generation import (
    GeneratedPaperResult,
    GeneratedQuestionItem,
    PaperBlueprintConfig,
    PaperSectionResult,
    SectionConstraint,
    SubPartItem,
)
from backend.services.paper_generator import PaperGenerator
from backend.services.similarity_engine import SimilarityEngine
from backend.services.validation_engine import ValidationEngine


@pytest.fixture
def sample_syllabus():
    return SyllabusDefinition(
        subject="Data Structures and Algorithms",
        course_code="CS201",
        units=[
            UnitDefinition(unit_number=1, unit_name="Unit 1", topics=["Stacks"], keywords=["stack"]),
            UnitDefinition(unit_number=2, unit_name="Unit 2", topics=["Trees"], keywords=["tree"]),
            UnitDefinition(unit_number=3, unit_name="Unit 3", topics=["Graphs"], keywords=["graph"]),
        ],
    )


@pytest.fixture
def sample_blueprint():
    return PaperBlueprintConfig(
        subject="Data Structures and Algorithms",
        total_marks=30.0,
        sections=[
            SectionConstraint(
                section_name="Section A",
                question_count=3,
                marks_per_question=10.0,
                has_internal_choice=False,
            ),
        ],
    )


class TestValidationEngine:
    """Test suite for ValidationEngine service."""

    def test_valid_paper_validation(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """A properly generated paper conforming to blueprint should validate cleanly."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        engine = ValidationEngine()
        report = engine.validate_paper(
            paper, sample_blueprint, sample_syllabus, check_semantic_duplicates=False
        )

        assert report.is_valid is True
        assert report.overall_quality_score >= 85.0
        assert len(report.failed_checks) == 0
        assert report.marks_check.is_match is True

    def test_marks_mismatch_detected(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Should detect and report total marks discrepancies."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        # Alter blueprint target to create mismatch
        tampered_blueprint = sample_blueprint.model_copy(update={"total_marks": 100.0})

        engine = ValidationEngine()
        report = engine.validate_paper(
            paper, tampered_blueprint, sample_syllabus, check_semantic_duplicates=False
        )

        assert report.is_valid is False
        assert any("mismatch" in f.lower() for f in report.failed_checks)
        assert report.marks_check.is_match is False

    def test_subparts_sum_discrepancy(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Should detect when sub-parts do not add up to question marks."""
        generator = PaperGenerator()
        paper = generator.generate_paper(sample_blueprint, sample_syllabus)

        # Tamper with first question's subparts
        q = paper.sections[0].questions[0]
        q.sub_parts = [
            SubPartItem(label="a", text="Part a", marks=2.0),
            SubPartItem(label="b", text="Part b", marks=2.0),
        ]  # Sum = 4, but q.marks = 10!

        engine = ValidationEngine()
        report = engine.validate_paper(
            paper, sample_blueprint, sample_syllabus, check_semantic_duplicates=False
        )

        assert report.is_valid is False
        assert len(report.marks_check.subpart_discrepancies) > 0

    def test_missing_unit_detection(
        self, sample_blueprint: PaperBlueprintConfig
    ):
        """Should flag when syllabus units are omitted from generated paper."""
        # Syllabus has 5 units, but paper only covers units 1 and 2
        syllabus_with_5_units = SyllabusDefinition(
            subject="DSA",
            units=[
                UnitDefinition(unit_number=i, unit_name=f"Unit {i}", topics=[], keywords=[])
                for i in range(1, 6)
            ],
        )

        paper = GeneratedPaperResult(
            id="p_test",
            subject="DSA",
            institution_name="Test",
            examination_name="Test",
            total_marks=30.0,
            duration_hours=2.0,
            sections=[
                PaperSectionResult(
                    section_name="Part A",
                    total_section_marks=30.0,
                    questions=[
                        GeneratedQuestionItem(
                            question_text="Q1 text",
                            marks=15.0,
                            unit_number=1,
                            topic="T1",
                            blooms_level="Understand",
                        ),
                        GeneratedQuestionItem(
                            question_text="Q2 text",
                            marks=15.0,
                            unit_number=2,
                            topic="T2",
                            blooms_level="Apply",
                        ),
                    ],
                )
            ],
            total_questions=2,
            created_at="2026-01-01T00:00:00",
        )

        engine = ValidationEngine()
        report = engine.validate_paper(
            paper, sample_blueprint, syllabus_with_5_units, check_semantic_duplicates=False
        )

        assert report.is_valid is False
        assert report.unit_coverage_check.missing_units == [3, 4, 5]
