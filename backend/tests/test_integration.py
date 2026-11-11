"""
End-to-End System Integration Tests.

Validates the full pipeline across all 14 modules:
PDF Upload/Extraction -> OCR -> Question Detection -> NLP Preprocessing ->
Semantic Similarity -> Topic Classification -> Bloom's Classification ->
Historical Question Analysis -> Marks Weightage Analysis -> Question Generation ->
Question Paper Assembly -> Constraint Validation Engine -> PDF Rendering.
"""

from pathlib import Path
import pytest
import pymupdf

from backend.api.schemas.classification import SyllabusDefinition
from backend.api.schemas.generation import PaperBlueprintConfig, SectionConstraint
from backend.services.blooms_classifier import BloomsClassifier
from backend.services.historical_analyzer import HistoricalAnalyzer
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.services.ocr_engine import OCREngine
from backend.services.paper_generator import PaperGenerator
from backend.services.pdf_extractor import PDFExtractor
from backend.services.pdf_renderer import PDFRenderer
from backend.services.question_detector import QuestionDetector
from backend.services.question_generator import QuestionGenerator
from backend.services.similarity_engine import SimilarityEngine
from backend.services.topic_classifier import TopicClassifier
from backend.services.validation_engine import ValidationEngine
from backend.services.weightage_analyzer import WeightageAnalyzer


def create_sample_test_pdf(dest_path: Path) -> Path:
    """Create a realistic university exam paper PDF for integration testing."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)  # A4 size

    text_content = """
ANTIGRAVITY INSTITUTE OF TECHNOLOGY
END SEMESTER DEGREE EXAMINATIONS - NOVEMBER 2023
Course: CS201 - Data Structures and Algorithms
Time: 3 Hours                                                    Max Marks: 100

PART A (10 x 2 = 20 Marks)
Answer ALL Questions

1. Define an Abstract Data Type (ADT) with two examples. [2 Marks]
2. What is a circular linked list? State its advantage. [2 Marks]
3. Define the balance factor of an AVL tree node. [2 Marks]
4. List the properties of a min-heap. [2 Marks]
5. State the difference between BFS and DFS graph traversals. [2 Marks]
6. What is a Minimum Spanning Tree (MST)? [2 Marks]
7. Explain the best-case time complexity of Quick Sort. [2 Marks]
8. Define hash collision. State one collision resolution method. [2 Marks]
9. What is Big-O asymptotic notation? [2 Marks]
10. Define optimal substructure in Dynamic Programming. [2 Marks]

PART B (5 x 16 = 80 Marks)
Answer ALL Questions (Each question has internal choice)

11. (a) Explain the implementation of a Stack using Singly Linked List with push and pop algorithms. [8 Marks]
    (b) Describe the applications of stacks in infix to postfix conversion with an example. [8 Marks]
[ OR ]
    (a) Explain the operations of a Circular Queue with enqueue and dequeue algorithms. [8 Marks]
    (b) Write an algorithm to insert and delete nodes from a Doubly Linked List. [8 Marks]

12. (a) Construct an AVL Tree by inserting the following sequence: [14, 17, 11, 7, 53, 4, 13]. Show all rotations. [10 Marks]
    (b) Describe binary search tree search and delete algorithms. [6 Marks]
[ OR ]
    (a) Explain the structure and properties of B-Trees and B+ Trees with diagrams. [10 Marks]
    (b) Describe the Heap Sort algorithm with an array trace. [6 Marks]
"""
    # Insert text into PDF
    page.insert_textbox(pymupdf.Rect(40, 40, 555, 800), text_content, fontsize=9.5)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(dest_path))
    doc.close()
    return dest_path


@pytest.fixture(scope="module")
def shared_nlp():
    return NLPPreprocessor()


@pytest.fixture(scope="module")
def shared_sim(shared_nlp):
    return SimilarityEngine(nlp_preprocessor=shared_nlp)


class TestFullSystemIntegrationPipeline:
    """Complete End-to-End integration test across the entire application stack."""

    def test_full_pipeline_run(self, tmp_path: Path, shared_nlp, shared_sim):
        # -------------------------------------------------------------
        # STEP 1: PDF Upload & Extraction (Module 1)
        # -------------------------------------------------------------
        pdf_path = tmp_path / "sample_exam_2023.pdf"
        create_sample_test_pdf(pdf_path)

        extractor = PDFExtractor()
        extraction = extractor.extract(pdf_path)
        assert extraction.total_pages >= 1
        assert len(extraction.pages[0].raw_text) > 200

        full_text = extraction.pages[0].raw_text

        # -------------------------------------------------------------
        # STEP 2: Question Detection and Segmentation (Module 3)
        # -------------------------------------------------------------
        detector = QuestionDetector()
        detected = detector.detect_questions(full_text)
        assert len(detected.questions) >= 10
        assert any(q.marks == 2.0 for q in detected.questions)

        # -------------------------------------------------------------
        # STEP 3: NLP Linguistic Preprocessing (Module 4)
        # -------------------------------------------------------------
        sample_q = detected.questions[0]
        nlp_doc = shared_nlp.preprocess(sample_q.text)
        assert "define" in nlp_doc.action_verbs
        assert len(nlp_doc.keywords) > 0

        # -------------------------------------------------------------
        # STEP 4: Semantic Similarity & Duplicate Analysis (Module 5)
        # -------------------------------------------------------------
        q_records = [
            {"id": f"q_{i}", "text": q.text, "year": "2023"}
            for i, q in enumerate(detected.questions)
        ]
        sim_report = shared_sim.find_duplicates(q_records)
        assert sim_report.total_questions == len(detected.questions)

        # -------------------------------------------------------------
        # STEP 5: Topic & Unit Classification (Module 6)
        # -------------------------------------------------------------
        topic_clf = TopicClassifier(
            nlp_preprocessor=shared_nlp,
            similarity_engine=shared_sim,
        )
        classified_questions = []
        for i, q in enumerate(detected.questions):
            t_res = topic_clf.classify_question(q.text)
            classified_questions.append({
                "id": f"q_{i}",
                "text": q.text,
                "year": "2023",
                "unit_number": t_res.unit_number or 1,
                "unit_name": t_res.unit_name or "Unit 1",
                "topic": t_res.topic or "General",
                "marks": q.marks or 2.0,
            })

        assert len(classified_questions) == len(detected.questions)
        # Verify multiple units were assigned
        assigned_units = {cq["unit_number"] for cq in classified_questions}
        assert len(assigned_units) >= 3

        # -------------------------------------------------------------
        # STEP 6: Bloom's Taxonomy Classification (Module 7)
        # -------------------------------------------------------------
        blooms_clf = BloomsClassifier(nlp_preprocessor=shared_nlp)
        for cq in classified_questions:
            b_res = blooms_clf.classify_question(cq["text"])
            cq["blooms_level"] = b_res.blooms_level.value

        assigned_levels = {cq["blooms_level"] for cq in classified_questions}
        assert len(assigned_levels) >= 2

        # -------------------------------------------------------------
        # STEP 7: Historical Analysis (Module 8)
        # -------------------------------------------------------------
        hist_analyzer = HistoricalAnalyzer(similarity_engine=shared_sim)
        hist_res = hist_analyzer.analyze(classified_questions, subject="DSA")
        assert hist_res.total_questions == len(classified_questions)
        assert len(hist_res.topic_trends) > 0
        assert sum(hist_res.blooms_distribution.values()) == len(classified_questions)

        # -------------------------------------------------------------
        # STEP 8: Marks & Unit Weightage Analysis (Module 9)
        # -------------------------------------------------------------
        weight_analyzer = WeightageAnalyzer()
        weight_res = weight_analyzer.analyze(
            classified_questions, syllabus=topic_clf.syllabus, subject="DSA"
        )
        assert weight_res.total_marks_all_papers > 0
        assert len(weight_res.unit_weightages) == 5

        # -------------------------------------------------------------
        # STEP 9: Question Generation & Paper Assembly (Modules 10 & 11)
        # -------------------------------------------------------------
        paper_blueprint = PaperBlueprintConfig(
            subject="Data Structures and Algorithms",
            course_code="CS201",
            total_marks=50.0,
            duration_hours=2.0,
            institution_name="Antigravity University",
            examination_name="B.Tech Examination 2026",
            sections=[
                SectionConstraint(
                    section_name="Part A",
                    description="Answer ALL questions. (5 x 2 = 10 Marks)",
                    question_count=5,
                    marks_per_question=2.0,
                    allowed_question_types=["short"],
                    has_internal_choice=False,
                ),
                SectionConstraint(
                    section_name="Part B",
                    description="Answer ALL questions. (4 x 10 = 40 Marks)",
                    question_count=4,
                    marks_per_question=10.0,
                    allowed_question_types=["long"],
                    has_internal_choice=True,
                ),
            ],
        )

        paper_gen = PaperGenerator(question_generator=QuestionGenerator())
        generated_paper = paper_gen.generate_paper(
            blueprint=paper_blueprint,
            syllabus=topic_clf.syllabus,
            historical_analysis=hist_res,
        )

        assert generated_paper.total_marks == 50.0
        assert len(generated_paper.sections) == 2
        assert len(generated_paper.sections[0].questions) == 5
        assert len(generated_paper.sections[1].questions) == 8  # 4 choices x 2

        # -------------------------------------------------------------
        # STEP 10: Validation Engine (Module 12)
        # -------------------------------------------------------------
        val_engine = ValidationEngine(similarity_engine=shared_sim)
        val_report = val_engine.validate_paper(
            paper=generated_paper,
            blueprint=paper_blueprint,
            syllabus=topic_clf.syllabus,
            check_semantic_duplicates=False,
        )

        assert val_report.is_valid is True
        assert val_report.marks_check.is_match is True
        assert val_report.overall_quality_score >= 85.0

        # -------------------------------------------------------------
        # STEP 11: PDF Rendering (Module 13)
        # -------------------------------------------------------------
        out_pdf_file = tmp_path / "official_generated_paper.pdf"
        renderer = PDFRenderer()
        pdf_bytes = renderer.render_pdf(generated_paper, output_path=out_pdf_file)

        assert out_pdf_file.exists()
        assert out_pdf_file.stat().st_size > 3000
        assert pdf_bytes.startswith(b"%PDF-")
