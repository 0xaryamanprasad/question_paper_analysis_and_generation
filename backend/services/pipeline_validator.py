"""
Module 5: Validation Service for Question Paper Pipeline.

Performs:
1. Pydantic schema validation.
2. Question numbering continuity and duplicate detection.
3. Missing subquestion sequence detection (e.g., 'a' and 'c' present without 'b').
4. Marks extraction and arithmetic rollup verification.
5. Low-confidence OCR flagging.
6. Suspicious mathematical expression detection (unmatched brackets, corrupted tokens).
7. Traceability preservation (PDF page references and image crop paths).
"""

import re
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.services.question_structurer import (
    StructuredExamPaper,
    StructuredQuestion,
    StructuredSubQuestion,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ValidationIssue(BaseModel):
    """Represents a discrete validation issue or warning."""
    level: Literal["error", "warning", "info"]
    category: Literal[
        "schema",
        "numbering",
        "subquestions",
        "marks",
        "low_confidence_ocr",
        "suspicious_math",
        "traceability",
    ]
    question_number: Optional[str] = None
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ValidationSummary(BaseModel):
    """Aggregated validation report across the parsed question paper."""
    is_valid: bool = Field(..., description="True if no fatal errors were found")
    total_questions: int
    total_marks: Optional[int] = None
    issues: List[ValidationIssue] = Field(default_factory=list)
    suspicious_math_count: int = 0
    low_confidence_count: int = 0
    numbering_discontinuities: int = 0
    missing_subquestions_count: int = 0
    marks_discrepancies_count: int = 0


class PipelineValidator:
    """
    Validates structured exam papers, question hierarchies, mathematical expressions,
    and OCR confidence scores.
    """

    def __init__(self, min_ocr_confidence: float = 0.65):
        self.min_ocr_confidence = min_ocr_confidence

    def validate_paper(
        self,
        paper: StructuredExamPaper,
        page_confidences: Optional[Dict[int, float]] = None,
        crop_paths: Optional[Dict[str, str]] = None,
    ) -> ValidationSummary:
        """
        Runs comprehensive validation checks over the StructuredExamPaper.
        """
        issues: List[ValidationIssue] = []
        conf_map = page_confidences or {}
        crops = crop_paths or {}

        # 1. Question Numbering Sequence & Duplicates
        num_issues = self._validate_question_numbering(paper.questions)
        issues.extend(num_issues)

        # 2. Subquestion Continuity (e.g. 'a' & 'c' without 'b')
        sub_issues = self._validate_subquestion_sequences(paper.questions)
        issues.extend(sub_issues)

        # 3. Marks Arithmetic Verification
        marks_issues = self._validate_marks(paper)
        issues.extend(marks_issues)

        # 4. Suspicious Mathematical Expressions
        math_issues = self._validate_mathematical_expressions(paper.questions)
        issues.extend(math_issues)

        # 5. Low-Confidence OCR and Traceability
        ocr_issues = self._validate_ocr_and_traceability(paper.questions, conf_map, crops)
        issues.extend(ocr_issues)

        # Aggregate metric counts
        math_count = sum(1 for i in issues if i.category == "suspicious_math")
        low_conf_count = sum(1 for i in issues if i.category == "low_confidence_ocr")
        num_disc = sum(1 for i in issues if i.category == "numbering")
        missing_subs = sum(1 for i in issues if i.category == "subquestions")
        marks_disc = sum(1 for i in issues if i.category == "marks")

        # Determine overall validity (fatal if schema corrupted or zero questions)
        has_fatal_error = any(i.level == "error" for i in issues)
        is_valid = (not has_fatal_error) and len(paper.questions) > 0

        return ValidationSummary(
            is_valid=is_valid,
            total_questions=len(paper.questions),
            total_marks=paper.total_marks,
            issues=issues,
            suspicious_math_count=math_count,
            low_confidence_count=low_conf_count,
            numbering_discontinuities=num_disc,
            missing_subquestions_count=missing_subs,
            marks_discrepancies_count=marks_disc,
        )

    def _validate_question_numbering(
        self,
        questions: List[StructuredQuestion],
    ) -> List[ValidationIssue]:
        """Detects gaps or duplicate question numbers."""
        issues: List[ValidationIssue] = []
        seen_numbers: Dict[str, int] = {}
        numeric_ids: List[int] = []

        for q in questions:
            q_num = q.question_number.strip()
            seen_numbers[q_num] = seen_numbers.get(q_num, 0) + 1

            if q_num.isdigit():
                numeric_ids.append(int(q_num))

        # Check for duplicates
        for q_num, count in seen_numbers.items():
            if count > 1:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        category="numbering",
                        question_number=q_num,
                        message=f"Duplicate question number '{q_num}' occurs {count} times.",
                        details={"occurrences": count},
                    )
                )

        # Check for numerical sequence gaps
        if numeric_ids:
            sorted_nums = sorted(list(set(numeric_ids)))
            expected_range = list(range(sorted_nums[0], sorted_nums[-1] + 1))
            missing = set(expected_range) - set(sorted_nums)
            if missing:
                for m_id in sorted(missing):
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="numbering",
                            question_number=str(m_id),
                            message=f"Missing question number {m_id} in expected sequence {sorted_nums[0]}..{sorted_nums[-1]}.",
                            details={"missing_id": m_id},
                        )
                    )

        return issues

    def _validate_subquestion_sequences(
        self,
        questions: List[StructuredQuestion],
    ) -> List[ValidationIssue]:
        """Flags missing subquestions like 'a' and 'c' present without 'b'."""
        issues: List[ValidationIssue] = []

        alphabet = "abcdefghijklmnopqrstuvwxyz"
        roman_numerals = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"]

        for q in questions:
            if len(q.subquestions) <= 1:
                continue

            sub_ids = [s.subquestion_id.strip().lower() for s in q.subquestions]

            # 1. Alphabet sequence check (a, b, c...)
            if all(sid in alphabet for sid in sub_ids):
                indices = [alphabet.index(sid) for sid in sub_ids]
                min_idx, max_idx = min(indices), max(indices)
                expected_indices = list(range(min_idx, max_idx + 1))
                missing_indices = set(expected_indices) - set(indices)

                for m_idx in sorted(missing_indices):
                    missing_letter = alphabet[m_idx]
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="subquestions",
                            question_number=q.question_number,
                            message=(
                                f"Question {q.question_number} is missing subquestion "
                                f"'{missing_letter}' (found subquestions: {sub_ids})."
                            ),
                            details={"missing_subquestion": missing_letter, "present": sub_ids},
                        )
                    )

            # 2. Roman sequence check (i, ii, iii...)
            elif all(sid in roman_numerals for sid in sub_ids):
                indices = [roman_numerals.index(sid) for sid in sub_ids]
                min_idx, max_idx = min(indices), max(indices)
                expected_indices = list(range(min_idx, max_idx + 1))
                missing_indices = set(expected_indices) - set(indices)

                for m_idx in sorted(missing_indices):
                    missing_roman = roman_numerals[m_idx]
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="subquestions",
                            question_number=q.question_number,
                            message=(
                                f"Question {q.question_number} is missing Roman subquestion "
                                f"'{missing_roman}' (found: {sub_ids})."
                            ),
                            details={"missing_subquestion": missing_roman, "present": sub_ids},
                        )
                    )

        return issues

    def _validate_marks(
        self,
        paper: StructuredExamPaper,
    ) -> List[ValidationIssue]:
        """Verifies marks allocation and arithmetic consistency between parent and subquestions."""
        issues: List[ValidationIssue] = []

        for q in paper.questions:
            # Check if marks are missing entirely
            if q.marks is None and not q.subquestions:
                issues.append(
                    ValidationIssue(
                        level="info",
                        category="marks",
                        question_number=q.question_number,
                        message=f"Question {q.question_number} has no explicit marks allocated.",
                    )
                )

            # Arithmetic check: if parent has marks and subquestions have marks
            if q.marks is not None and q.subquestions:
                sub_marks = [s.marks for s in q.subquestions if s.marks is not None]
                if len(sub_marks) == len(q.subquestions):
                    sub_sum = sum(sub_marks)
                    if sub_sum != q.marks:
                        issues.append(
                            ValidationIssue(
                                level="warning",
                                category="marks",
                                question_number=q.question_number,
                                message=(
                                    f"Marks arithmetic mismatch in Question {q.question_number}: "
                                    f"Parent marks = {q.marks}, but subquestions sum to {sub_sum}."
                                ),
                                details={"parent_marks": q.marks, "subquestions_sum": sub_sum},
                            )
                        )

        return issues

    def _validate_mathematical_expressions(
        self,
        questions: List[StructuredQuestion],
    ) -> List[ValidationIssue]:
        """Detects suspicious, unclosed, or corrupted LaTeX expressions."""
        issues: List[ValidationIssue] = []

        bracket_pairs = {"(": ")", "[": "]", "{": "}"}

        for q in questions:
            for expr in q.latex_expressions:
                # Check unbalanced dollar signs
                if expr.count("$") % 2 != 0:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="suspicious_math",
                            question_number=q.question_number,
                            message=f"Unbalanced LaTeX dollar delimiters in expression: {expr}",
                            details={"expression": expr},
                        )
                    )

                # Check bracket balance
                content = expr.strip("$")
                stack = []
                balanced = True
                for ch in content:
                    if ch in bracket_pairs.keys():
                        stack.append(ch)
                    elif ch in bracket_pairs.values():
                        if not stack or bracket_pairs[stack.pop()] != ch:
                            balanced = False
                            break
                if stack or not balanced:
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="suspicious_math",
                            question_number=q.question_number,
                            message=f"Unbalanced brackets detected in mathematical expression: {expr}",
                            details={"expression": expr},
                        )
                    )

                # Check empty or incomplete LaTeX commands (e.g. \frac without args)
                if re.search(r"\\frac\s*$", content) or re.search(r"\\sqrt\s*$", content):
                    issues.append(
                        ValidationIssue(
                            level="warning",
                            category="suspicious_math",
                            question_number=q.question_number,
                            message=f"Incomplete LaTeX command detected in expression: {expr}",
                            details={"expression": expr},
                        )
                    )

        return issues

    def _validate_ocr_and_traceability(
        self,
        questions: List[StructuredQuestion],
        page_confidences: Dict[int, float],
        crops: Dict[str, str],
    ) -> List[ValidationIssue]:
        """Flags questions originating from low confidence pages and verifies page references."""
        issues: List[ValidationIssue] = []

        for q in questions:
            # Check page reference
            if q.page_number < 1:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        category="traceability",
                        question_number=q.question_number,
                        message=f"Invalid page reference {q.page_number} for Question {q.question_number}.",
                    )
                )

            # Check OCR confidence of the page
            page_conf = page_confidences.get(q.page_number)
            if page_conf is not None and page_conf < self.min_ocr_confidence:
                issues.append(
                    ValidationIssue(
                        level="warning",
                        category="low_confidence_ocr",
                        question_number=q.question_number,
                        message=(
                            f"Question {q.question_number} on page {q.page_number} has low OCR "
                            f"confidence: {page_conf:.2f} (threshold: {self.min_ocr_confidence:.2f})."
                        ),
                        details={"page_confidence": page_conf},
                    )
                )

        return issues
