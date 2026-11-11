"""
Unit and Integration Tests for Module 3: Question Detection and Segmentation Service.

Tests:
- Detection of various numbering formats (1., Q1., Question 1:, 1))
- Section boundary identification (Section A, PART B, Module 1)
- Sub-part decomposition (a, b, c and i, ii, iii)
- Alternative/OR question grouping and shared or_group_id pairing
- Marks extraction and summation across sub-parts
- Question classification heuristics (short, medium, long, numerical, mcq)
- Boilerplate header filtering
- End-to-End integration from PDF Extractor (Module 1) to Question Detector (Module 3)
"""

from pathlib import Path
import pytest

from backend.api.schemas.question import SegmentationResult
from backend.services.pdf_extractor import PDFExtractor
from backend.services.question_detector import QuestionDetector


@pytest.fixture
def detector() -> QuestionDetector:
    return QuestionDetector()


@pytest.fixture
def sample_standard_exam_text() -> str:
    """Standard multi-section exam text with various question types."""
    return """
    ABC UNIVERSITY
    END SEMESTER EXAMINATION - DECEMBER 2023
    COURSE: DATA STRUCTURES & ALGORITHMS
    MAXIMUM MARKS: 100
    TIME ALLOWED: 3 HOURS
    Instructions: Answer all questions from Section A and any two from Section B.

    SECTION A

    1. Define Abstract Data Type (ADT). Give two examples. [5 Marks]
    2. What is the time complexity of Quick Sort in the worst case? [3 Marks]
    3. Differentiate between B-Tree and B+ Tree. (5M)

    SECTION B

    4. (a) Explain Dijkstra's shortest path algorithm with an example. [10 Marks]
       (b) Calculate the minimum spanning tree using Kruskal's algorithm for the given graph. [10 Marks]

    OR

    5. Explain Prim's algorithm for finding Minimum Spanning Tree and derive its time complexity. [20 Marks]

    6. Which of the following is not an in-place sorting algorithm?
       (A) Insertion Sort
       (B) Merge Sort
       (C) Quick Sort
       (D) Selection Sort [2 Marks]
    """


class TestQuestionDetector:
    """Tests for the QuestionDetector service."""

    def test_segment_standard_exam(self, detector: QuestionDetector, sample_standard_exam_text: str):
        """Should detect questions, correct sections, marks, and OR pairing."""
        res = detector.segment_text(sample_standard_exam_text)

        assert isinstance(res, SegmentationResult)
        assert res.total_questions == 6
        assert "Section A" in res.sections_detected
        assert "Section B" in res.sections_detected

        # Check Question 1
        q1 = res.questions[0]
        assert q1.question_number == "1"
        assert "Abstract Data Type" in q1.raw_text
        assert q1.marks == 5
        assert q1.section == "Section A"
        assert q1.question_type == "medium"

        # Check Question 2
        q2 = res.questions[1]
        assert q2.question_number == "2"
        assert q2.marks == 3
        assert q2.question_type == "short"

        # Check Question 3 with (5M) format
        q3 = res.questions[2]
        assert q3.marks == 5
        assert "B-Tree" in q3.raw_text

        # Check Question 4 (Section B, sub-parts)
        q4 = res.questions[3]
        assert q4.question_number == "4"
        assert q4.section == "Section B"
        assert len(q4.sub_parts) == 2
        assert q4.sub_parts[0].part_label == "a"
        assert q4.sub_parts[0].marks == 10
        assert q4.sub_parts[1].part_label == "b"
        assert q4.sub_parts[1].marks == 10
        assert q4.marks == 20  # Sum of sub-parts

        # Check OR pairing between Q4 and Q5
        q5 = res.questions[4]
        assert q5.question_number == "5"
        assert q4.is_or_choice is True
        assert q5.is_or_choice is True
        assert q4.or_group_id is not None
        assert q4.or_group_id == q5.or_group_id
        assert q5.marks == 20
        assert q5.question_type == "long"

        # Check MCQ detection on Q6
        q6 = res.questions[5]
        assert q6.question_number == "6"
        assert q6.question_type == "mcq"

    def test_standalone_question_labels(self, detector: QuestionDetector):
        """Should recognize standalone question labels like 'Q1' followed by text."""
        raw = """
        SECTION 1
        Q1
        Explain the principles of Object Oriented Programming. [10 Marks]

        Q2
        What is polymorphism? (5 Marks)
        """
        res = detector.segment_text(raw)
        assert res.total_questions == 2
        assert res.questions[0].question_number == "1"
        assert "Object Oriented" in res.questions[0].raw_text
        assert res.questions[0].marks == 10

    def test_numerical_question_type(self, detector: QuestionDetector):
        """Should classify computational/calculation problems as numerical."""
        raw = "1. Calculate the determinant and solve for x in the matrix. [8 Marks]"
        res = detector.segment_text(raw)
        assert len(res.questions) == 1
        assert res.questions[0].question_type == "numerical"

    def test_boilerplate_filtered(self, detector: QuestionDetector):
        """Boilerplate instructions should not create spurious questions."""
        raw = """
        UNIVERSITY EXAMINATION 2024
        ROLL NO: ________________
        Time Allowed: 3 Hours
        Candidates are required to answer in their own words.

        1. State the Master Theorem. [4 Marks]
        """
        res = detector.segment_text(raw)
        assert res.total_questions == 1
        assert res.questions[0].question_number == "1"

    def test_segment_multiple_pages(self, detector: QuestionDetector):
        """Should segment text collected across multiple pages."""
        pages = [
            (1, "SECTION A\n1. What is an array? [2 Marks]\n2. What is a stack? [2 Marks]"),
            (2, "SECTION B\n3. Describe Merge Sort in detail. [10 Marks]"),
        ]
        res = detector.segment_pages(pages)
        assert res.total_questions == 3
        assert res.questions[0].page_number == 1
        assert res.questions[2].page_number == 2
        assert res.total_marks_detected == 14


class TestPDFToSegmentationIntegration:
    """Integration test: Module 1 (PDF Extractor) -> Module 3 (Question Detector)."""

    def test_extract_pdf_and_segment(self, sample_digital_pdf: Path):
        """Digital PDF should be extracted by Module 1 and segmented by Module 3."""
        extractor = PDFExtractor()
        pdf_res = extractor.extract_pdf(sample_digital_pdf)

        pages_text = [(p.page_number, p.raw_text) for p in pdf_res.pages]
        detector = QuestionDetector()
        seg_res = detector.segment_pages(pages_text)

        assert seg_res.total_questions >= 4
        q_texts = [q.raw_text for q in seg_res.questions]
        assert any("Abstract Data Type" in t for t in q_texts)
        assert any("Linked List" in t for t in q_texts)
        assert any("Binary Tree" in t for t in q_texts)
        assert any("Dijkstra" in t for t in q_texts)
