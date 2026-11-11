"""
Unit and Integration Tests for the Advanced 6-Module OCR, Structuring,
and DSA Classification Pipeline.

Tests:
- Module 1: Math & Layout-Aware OCR Engine
- Module 2: Gemini / Rule-Based OCR Correction & Flagging
- Module 3: Question Paper Structuring with Pydantic Schemas
- Module 4: DSA Topic Classification (Strict Categories, Algorithmic Concepts)
- Module 5: Validation Engine (Numbering gaps, missing subquestions, marks arithmetic, math syntax)
- Module 6: Pipeline Orchestration and Database Persistence
"""

import uuid
import pytest
import numpy as np

from backend.services.math_ocr_engine import (
    MathOCREngine,
    MathOCRLine,
    MathOCRPageResult,
    MathRegion,
)
from backend.services.ocr_corrector import (
    GeminiOCRCorrector,
    OCRCorrectionResult,
)
from backend.services.question_structurer import (
    QuestionStructurer,
    StructuredExamPaper,
    StructuredQuestion,
    StructuredSubQuestion,
)
from backend.services.dsa_classifier import (
    DSACategory,
    DSATopicClassifier,
    DSATopicClassificationResult,
)
from backend.services.pipeline_validator import (
    PipelineValidator,
    ValidationIssue,
    ValidationSummary,
)
from backend.services.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineExecutionResult,
)
from backend.models.question import Question


# ============================================================================
# MODULE 1 TESTS: OCR & Mathematical Expression Recognition
# ============================================================================

class TestModule1MathOCR:
    """Tests for Module 1 mathematical detection and LaTeX conversion."""

    def test_asymptotic_notation_conversion_to_latex(self):
        engine = MathOCREngine()
        text = "Show that the worst-case running time of QuickSort is O(n^2) and best-case is O(n log n)."
        converted, latex_list = engine._convert_formula_to_latex(text)
        assert "$O(n^{2})$" in latex_list or "$O(n^2)$" in latex_list or "$O(n \\log n)$" in latex_list
        assert "$" in converted

    def test_recurrence_relation_conversion_to_latex(self):
        engine = MathOCREngine()
        text = "Solve the recurrence relation T(n) = 2T(n/2) + O(n) using Master Theorem."
        converted, latex_list = engine._convert_formula_to_latex(text)
        assert any("T(n)" in item for item in latex_list)
        assert "$" in converted

    def test_unicode_operators_conversion(self):
        engine = MathOCREngine()
        text = "For all keys x \u2264 y, where weight \u2265 0 and alpha \u03b1 \u2260 0."
        converted, _ = engine._convert_formula_to_latex(text)
        assert r"\le" in converted
        assert r"\ge" in converted
        assert r"\ne" in converted

    def test_detect_math_in_line_flag(self):
        engine = MathOCREngine()
        has_math, converted, snippets = engine._detect_and_convert_math_in_line(
            text="Calculate the height h \u2264 log(n) for an AVL tree with 2^k nodes.",
            bbox=(10.0, 20.0, 200.0, 35.0),
            img_np=None,
            line_idx=0,
            page_number=1,
        )
        assert has_math is True
        assert len(snippets) > 0


# ============================================================================
# MODULE 2 TESTS: OCR Correction with Verification Flagging
# ============================================================================

class TestModule2OCRCorrection:
    """Tests for Module 2 OCR correction, math preservation, and flagging."""

    def test_corrects_spelling_and_broken_word_spacing(self):
        corrector = GeminiOCRCorrector()
        raw = "Explain teh al gorithm for binary search in a sorted array."
        result = corrector.correct_ocr_text(raw_ocr_text=raw)
        assert "algorithm" in result.corrected_text
        assert "the" in result.corrected_text
        assert result.raw_ocr_text == raw

    def test_preserves_latex_expressions_intact(self):
        corrector = GeminiOCRCorrector()
        raw = "Analyze the time complexity $O(n \\log n)$ for merge sort."
        result = corrector.correct_ocr_text(raw_ocr_text=raw, latex_expressions=["$O(n \\log n)$"])
        assert "$O(n \\log n)$" in result.corrected_text or "$O(n \\log n)$" in result.latex_expressions
        assert result.is_flagged_for_verification is False

    def test_flags_unmatched_parentheses_for_verification(self):
        corrector = GeminiOCRCorrector()
        raw = "Given an array A[1..n with unbalanced opening bracket."
        result = corrector.correct_ocr_text(raw_ocr_text=raw)
        assert result.is_flagged_for_verification is True
        assert any("bracket" in r.lower() for r in result.flag_reasons)

    def test_flags_odd_latex_dollar_delimiters(self):
        corrector = GeminiOCRCorrector()
        raw = "Calculate the recurrence $T(n) = 2T(n/2) + n without closing dollar."
        result = corrector.correct_ocr_text(raw_ocr_text=raw)
        assert result.is_flagged_for_verification is True
        assert any("dollar" in r.lower() for r in result.flag_reasons)


# ============================================================================
# MODULE 3 TESTS: Question Structuring with Pydantic Schemas
# ============================================================================

class TestModule3QuestionStructuring:
    """Tests for Module 3 hierarchical question structuring."""

    def test_hierarchical_question_parsing(self):
        structurer = QuestionStructurer()
        sample_exam = (
            "END SEMESTER EXAMINATION 2024\n"
            "DATA STRUCTURES AND ALGORITHMS\n"
            "SECTION A\n"
            "1. Answer the following short questions: [10 marks]\n"
            "(a) Define an AVL tree and give the balance factor definition. [5 marks]\n"
            "(b) Explain collision resolution using quadratic probing. [5 marks]\n"
            "2. State and prove Master Theorem for divide and conquer recurrences. [10 marks]\n"
        )
        paper = structurer.structure_paper(sample_exam, default_year="2024")
        assert isinstance(paper, StructuredExamPaper)
        assert paper.examination_year == "2024"
        assert len(paper.questions) == 2

        # Check Question 1 hierarchy
        q1 = paper.questions[0]
        assert q1.question_number == "1"
        assert q1.marks == 10
        assert len(q1.subquestions) == 2
        assert q1.subquestions[0].subquestion_id == "a"
        assert q1.subquestions[0].marks == 5
        assert q1.subquestions[1].subquestion_id == "b"
        assert q1.subquestions[1].marks == 5

        # Check Question 2
        q2 = paper.questions[1]
        assert q2.question_number == "2"
        assert q2.marks == 10

    def test_marks_extraction_and_rollup(self):
        structurer = QuestionStructurer()
        sample = (
            "1. Linear Data Structures\n"
            "(a) Implement Queue using two Stacks. [4 marks]\n"
            "(b) What is circular queue? [6 marks]\n"
        )
        paper = structurer.structure_paper(sample)
        q1 = paper.questions[0]
        assert q1.marks == 10  # 4 + 6 rollup


# ============================================================================
# MODULE 4 TESTS: DSA Topic Classification
# ============================================================================

class TestModule4DSAClassification:
    """Tests for Module 4 DSA topic categorization based on algorithmic concepts."""

    def test_classify_graphs_with_greedy_secondary(self):
        classifier = DSATopicClassifier()
        q_text = "Apply Dijkstra's algorithm to find the single-source shortest path in a directed graph with positive weights."
        res = classifier.classify_question(q_text)
        assert res.primary_topic == DSACategory.GRAPHS
        assert res.secondary_topic == DSACategory.GREEDY_ALGORITHMS
        assert res.confidence >= 0.70

    def test_classify_dynamic_programming(self):
        classifier = DSATopicClassifier()
        q_text = "Given an array of item weights and values, formulate the 0/1 Knapsack problem using dynamic programming memoization."
        res = classifier.classify_question(q_text)
        assert res.primary_topic == DSACategory.DYNAMIC_PROGRAMMING
        assert res.confidence >= 0.70

    def test_classify_divide_and_conquer(self):
        classifier = DSATopicClassifier()
        q_text = "Solve the recurrence relation T(n) = 2T(n/2) + O(n) using Master Theorem."
        res = classifier.classify_question(q_text, latex_expressions=["$T(n) = 2T(n/2) + O(n)$"])
        assert res.primary_topic == DSACategory.DIVIDE_AND_CONQUER
        assert res.confidence >= 0.65

    def test_classify_binary_search_tree(self):
        classifier = DSATopicClassifier()
        q_text = "Construct an AVL tree by inserting keys [10, 20, 30, 40, 50] and perform LL and RR rotations."
        res = classifier.classify_question(q_text)
        assert res.primary_topic == DSACategory.BST
        assert res.confidence >= 0.60

    def test_classify_unclassified_for_irrelevant_question(self):
        classifier = DSATopicClassifier()
        q_text = "Explain the laws of thermodynamics and calculate the entropy change in a closed cylinder."
        res = classifier.classify_question(q_text)
        assert res.primary_topic == DSACategory.UNCLASSIFIED
        assert res.confidence < 0.40


# ============================================================================
# MODULE 5 TESTS: Validation Engine
# ============================================================================

class TestModule5ValidationEngine:
    """Tests for Module 5 validation rules."""

    def test_detects_question_numbering_gaps(self):
        validator = PipelineValidator()
        paper = StructuredExamPaper(
            subject="Algorithms",
            examination_year="2023",
            questions=[
                StructuredQuestion(question_number="1", question_text="Question 1 text"),
                StructuredQuestion(question_number="3", question_text="Question 3 text"),
            ],
        )
        summary = validator.validate_paper(paper)
        assert summary.numbering_discontinuities > 0
        assert any(i.category == "numbering" and "Missing question number 2" in i.message for i in summary.issues)

    def test_detects_missing_subquestion_in_sequence(self):
        validator = PipelineValidator()
        paper = StructuredExamPaper(
            subject="Algorithms",
            examination_year="2023",
            questions=[
                StructuredQuestion(
                    question_number="1",
                    question_text="Main Question",
                    subquestions=[
                        StructuredSubQuestion(subquestion_id="a", text="Part a"),
                        StructuredSubQuestion(subquestion_id="c", text="Part c without b"),
                    ],
                )
            ],
        )
        summary = validator.validate_paper(paper)
        assert summary.missing_subquestions_count > 0
        assert any(i.category == "subquestions" and "'b'" in i.message for i in summary.issues)

    def test_detects_marks_arithmetic_discrepancy(self):
        validator = PipelineValidator()
        paper = StructuredExamPaper(
            subject="Algorithms",
            examination_year="2023",
            questions=[
                StructuredQuestion(
                    question_number="1",
                    question_text="Main Question",
                    marks=10,  # Parent claims 10 marks
                    subquestions=[
                        StructuredSubQuestion(subquestion_id="a", text="Part a", marks=4),
                        StructuredSubQuestion(subquestion_id="b", text="Part b", marks=4),  # Sum is only 8!
                    ],
                )
            ],
        )
        summary = validator.validate_paper(paper)
        assert summary.marks_discrepancies_count > 0
        assert any(i.category == "marks" and "mismatch" in i.message for i in summary.issues)

    def test_detects_suspicious_mathematical_expressions(self):
        validator = PipelineValidator()
        paper = StructuredExamPaper(
            subject="Algorithms",
            examination_year="2023",
            questions=[
                StructuredQuestion(
                    question_number="1",
                    question_text="Question with broken math",
                    latex_expressions=["$O(n \\log n", "$T(n) = \\frac{n}{2)$"],
                )
            ],
        )
        summary = validator.validate_paper(paper)
        assert summary.suspicious_math_count > 0


# ============================================================================
# MODULE 6 TESTS: Orchestrator & Persistence
# ============================================================================

class TestModule6PipelineOrchestrator:
    """Tests for Module 6 full pipeline flow and database storage."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_persistence(self, db_session):
        # Create a sample text file acting as paper input
        sample_exam_text = (
            "END TERM EXAMINATION 2024\n"
            "DATA STRUCTURES AND ALGORITHMS\n"
            "SECTION A\n"
            "1. Discuss asymptotic analysis: [10 marks]\n"
            "(a) Compare O(n log n) with O(n^2). [5 marks]\n"
            "(b) Solve recurrence T(n) = 2T(n/2) + n. [5 marks]\n"
            "2. Explain Dijkstra algorithm on a weighted graph. [10 marks]\n"
        )
        temp_file = db_session.bind.url.database
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w+", suffix=".txt", delete=False) as tf:
            tf.write(sample_exam_text)
            tf.flush()
            tmp_path = tf.name

        orchestrator = PipelineOrchestrator()
        
        # Run pipeline
        # math_ocr mock to return sample text
        class MockMathOCR:
            def extract_from_image(self, img, page_number=1):
                return MathOCRPageResult(
                    page_number=page_number,
                    raw_text=sample_exam_text,
                    text_with_latex=sample_exam_text,
                    latex_expressions=["$O(n \\log n)$", "$O(n^2)$", "$T(n) = 2T(n/2) + n$"],
                    average_confidence=0.92,
                )
            def extract_from_pdf(self, path):
                page_res = self.extract_from_image(path, page_number=1)
                return MathOCRPaperResult(
                    total_pages=1,
                    pages=[page_res],
                    all_latex_expressions=page_res.latex_expressions,
                    average_confidence=page_res.average_confidence,
                )

        orchestrator.math_ocr = MockMathOCR()

        result = await orchestrator.run_pipeline(
            file_path=tmp_path,
            subject="Data Structures and Algorithms",
            examination_year="2024",
            session=db_session,
        )

        assert isinstance(result, PipelineExecutionResult)
        assert result.paper_id is not None
        assert len(result.saved_questions) == 2

        # Verify DB records
        q1_dict = result.saved_questions[0]
        assert q1_dict["question_number"] == "1"
        assert q1_dict["marks"] == 10
        assert q1_dict["subquestions_count"] == 2
        assert len(q1_dict["latex_expressions"]) > 0

        q2_dict = result.saved_questions[1]
        assert q2_dict["question_number"] == "2"
        assert q2_dict["dsa_topic"] in ("Graphs", "Greedy Algorithms")
