"""
Unit Tests for Module 13: PDF Generation Module.
"""

from pathlib import Path
import pytest

from backend.api.schemas.classification import SyllabusDefinition, UnitDefinition
from backend.api.schemas.generation import PaperBlueprintConfig, SectionConstraint
from backend.services.paper_generator import PaperGenerator
from backend.services.pdf_renderer import PDFRenderer


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
        institution_name="Antigravity Institute of Technology",
        examination_name="B.Tech Degree Examination",
        sections=[
            SectionConstraint(
                section_name="Part A",
                description="Answer ALL questions. (5 x 2 = 10 Marks)",
                question_count=5,
                marks_per_question=2.0,
                has_internal_choice=False,
            ),
            SectionConstraint(
                section_name="Part B",
                description="Answer ALL questions. Each question has internal choice. (4 x 10 = 40 Marks)",
                question_count=4,
                marks_per_question=10.0,
                has_internal_choice=True,
            ),
        ],
    )


class TestPDFRenderer:
    """Test suite for PDFRenderer service."""

    def test_render_pdf_bytes(
        self, sample_syllabus: SyllabusDefinition, sample_blueprint: PaperBlueprintConfig
    ):
        """Should render paper to non-empty PDF bytes starting with standard PDF magic number."""
        paper = PaperGenerator().generate_paper(sample_blueprint, sample_syllabus)
        renderer = PDFRenderer()

        pdf_bytes = renderer.render_pdf(paper)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 2000
        # Check standard PDF header magic bytes
        assert pdf_bytes.startswith(b"%PDF-")

    def test_render_pdf_to_file(
        self,
        tmp_path: Path,
        sample_syllabus: SyllabusDefinition,
        sample_blueprint: PaperBlueprintConfig,
    ):
        """Should write PDF file to destination path on disk."""
        paper = PaperGenerator().generate_paper(sample_blueprint, sample_syllabus)
        renderer = PDFRenderer()

        out_file = tmp_path / "test_exam_paper.pdf"
        pdf_bytes = renderer.render_pdf(paper, output_path=out_file)

        assert out_file.exists()
        assert out_file.stat().st_size == len(pdf_bytes)
        assert out_file.stat().st_size > 2000
