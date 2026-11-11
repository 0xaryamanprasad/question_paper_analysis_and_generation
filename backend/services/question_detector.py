"""
Module 3: Question Detection and Segmentation Service.

Identifies section boundaries, questions, sub-parts, and marks allocations
from raw exam paper text (from either digital extraction or OCR output).
Handles multi-format numbering, choice/OR questions, and varied marking patterns.
"""

import re
import uuid
from typing import Any

from backend.api.schemas.question import (
    SegmentedQuestion,
    SegmentationResult,
    SubPart,
)
from backend.utils.logger import get_logger
from backend.utils.text_utils import extract_marks_from_text, normalize_question_text

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Compiled Regex Patterns
# ---------------------------------------------------------------------------

# Section demarcations (e.g., "SECTION A", "PART - II", "GROUP 1", "MODULE III")
SECTION_PATTERN = re.compile(
    r"^(?:(?:SECTION|PART|GROUP|MODULE|UNIT)\s*[-:]*\s*([A-Z0-9]+|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b))"
    r"|(?:(?:SECTION|PART|GROUP|MODULE|UNIT)\s*[-–]\s*([A-Z0-9]+))",
    re.IGNORECASE | re.MULTILINE,
)

# Question number starts (e.g., "1.", "Q1.", "Question 1:", "1)", "Q. 2 -", "(1)", "[1]")
QUESTION_START_PATTERN = re.compile(
    r"^(?:Q(?:uestion|\.)?\s*|\(\s*|\[\s*)?(\d{1,3})\s*[\.\:\)\-\]\s]\s*(.+)$",
    re.IGNORECASE,
)

# Secondary question format: standalone question labels (e.g., "Q1", "Question 2", "1.", "(1)")
QUESTION_STANDALONE_PATTERN = re.compile(
    r"^(?:(?:Q(?:uestion|\.)?\s*(\d{1,3})\s*[\.\:\)\-]?)|(?:\(?(\d{1,3})[\.\:\)\-]))\s*$",
    re.IGNORECASE,
)

# Sub-part patterns: strictly matches single letters (a-h) or roman numerals (i-x)
# Examples: "(a) Text", "a) Text", "a. Text", "(i) Text", "ii. Text"
_PART_IDENTIFIER = r"(?:[a-h]|(?:i{1,3}|iv|vi{0,3}|ix|x))"
SUB_PART_PATTERN = re.compile(
    rf"(?:^|\n|\s+)(?:\(({_PART_IDENTIFIER})\)|({_PART_IDENTIFIER})[\.\)])\s+(.*?)"
    rf"(?=(?:\s+(?:\({_PART_IDENTIFIER}\)|{_PART_IDENTIFIER}[\.\)]))|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Choice / OR separator line (e.g., "OR", "[OR]", "--- OR ---", "-- OR --")
OR_PATTERN = re.compile(
    r"^\s*(?:[-=*~_]{2,}\s*)?(?:OR|\[OR\]|\(OR\))(?:\s*[-=*~_]{2,})?\s*$",
    re.IGNORECASE,
)

# Boilerplate / Header patterns to ignore as questions
BOILERPLATE_PATTERN = re.compile(
    r"(?:university|institute|college|examination|roll\s*no|reg(?:istration)?\s*no|"
    r"semester|time\s*allowed|max(?:imum)?\s*marks|total\s*marks|duration|"
    r"answer\s+all|answer\s+any|note\s*:|instructions?\s*:|figures\s+to\s+the\s+right|"
    r"assume\s+suitable|candidates?\s+are\s+required)",
    re.IGNORECASE,
)

# Multiple choice question indicators (e.g., (A) (B) (C) (D) options)
MCQ_PATTERN = re.compile(
    r"(?:\([A-D]\)|[A-D][\.\)])\s+.*?(?:\([B-D]\)|[B-D][\.\)])",
    re.DOTALL,
)


class QuestionDetector:
    """
    Robust rule-based parser that segments unstructured examination text into
    discrete questions, sub-questions, and associated metadata.
    """

    def __init__(self, default_section_name: str = "Section A"):
        self.default_section = default_section_name

    def _is_boilerplate(self, line: str) -> bool:
        """Check if a line represents header metadata or exam instructions."""
        clean = line.strip()
        if not clean:
            return True
        # Date stamps, watermarks, barcode stamps like 217-22/05/2025--12
        if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", clean) or re.search(r"^\d{2,4}[-/]\d{2}[-/]\d{4}", clean):
            return True
        if BOILERPLATE_PATTERN.search(clean) and len(clean) < 150:
            return True
        return False

    def _infer_question_type(self, text: str, marks: int | None) -> str:
        """
        Infer question classification type: 'short', 'medium', 'long', 'mcq', or 'numerical'.
        Follows examination tier conventions based on marks, MCQ choices, and problem phrasing.
        """
        if MCQ_PATTERN.search(text):
            return "mcq"

        # Tier classification by marks
        if marks is not None:
            if marks <= 3:
                return "short"
            elif marks >= 15:
                return "long"

        # Check for numerical / computational cues for medium marks or unspecified
        numerical_keywords = [
            "calculate", "compute", "find the value", "solve for",
            "evaluate the expression", "solve the following equation",
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in numerical_keywords):
            return "numerical"

        if marks is not None:
            if marks >= 10:
                return "long"
            else:
                return "medium"

        # Fallback based on text length
        return "short" if len(text) < 120 else "medium"

    def _parse_sub_parts(self, text: str) -> list[SubPart]:
        """Extract sub-question parts (a, b, c or i, ii, iii) if present."""
        matches = SUB_PART_PATTERN.findall(text)
        sub_parts: list[SubPart] = []

        for m in matches:
            label = (m[0] or m[1]).strip()
            sub_text = m[2].strip()
            if not sub_text:
                continue

            sub_marks = extract_marks_from_text(sub_text)
            sub_parts.append(
                SubPart(
                    part_label=label,
                    text=normalize_question_text(sub_text),
                    marks=sub_marks,
                )
            )

        return sub_parts

    def segment_text(
        self,
        raw_text: str,
        page_number: int | None = None,
    ) -> SegmentationResult:
        """
        Segment raw document text into questions and sections.

        Args:
            raw_text: Raw multi-line string extracted from PDF or OCR.
            page_number: Optional 1-based page number for tracking.

        Returns:
            SegmentationResult with ordered questions, marks, and section info.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        current_section = self.default_section
        sections_found: list[str] = [current_section]

        raw_question_blocks: list[dict[str, Any]] = []
        current_q: dict[str, Any] | None = None
        pending_or = False
        unsegmented: list[str] = []

        for line in lines:
            # 1. Check for Section header
            sec_match = SECTION_PATTERN.search(line)
            if sec_match:
                matched_sec = sec_match.group(1) or sec_match.group(2)
                current_section = f"Section {matched_sec.upper()}"
                if current_section not in sections_found:
                    sections_found.append(current_section)
                logger.debug("section_detected", section=current_section, line=line)
                continue

            # 2. Check for OR separator
            if OR_PATTERN.match(line):
                pending_or = True
                logger.debug("or_choice_detected", line=line)
                continue

            # 3. Check for Question start
            q_match = QUESTION_START_PATTERN.match(line)
            if q_match:
                q_num = q_match.group(1)
                q_content = q_match.group(2)

                # Save previously accumulating question
                if current_q is not None:
                    raw_question_blocks.append(current_q)

                current_q = {
                    "question_number": q_num,
                    "lines": [q_content],
                    "section": current_section,
                    "is_or_choice": pending_or,
                    "page_number": page_number,
                }
                pending_or = False
                continue

            # 4. Check for standalone question numbering line (e.g. "Q1", "Question 2", "1.")
            q_std_match = QUESTION_STANDALONE_PATTERN.match(line)
            if q_std_match:
                q_num = q_std_match.group(1) or q_std_match.group(2)
                if current_q is not None:
                    raw_question_blocks.append(current_q)

                current_q = {
                    "question_number": q_num,
                    "lines": [],
                    "section": current_section,
                    "is_or_choice": pending_or,
                    "page_number": page_number,
                }
                pending_or = False
                continue

            # 5. Append line to current question or discard as boilerplate
            if current_q is not None:
                current_q["lines"].append(line)
            else:
                if not self._is_boilerplate(line):
                    unsegmented.append(line)

        # Append last question
        if current_q is not None:
            raw_question_blocks.append(current_q)

        # Build structured SegmentedQuestion objects
        questions: list[SegmentedQuestion] = []
        total_marks_detected = 0

        # Pair OR questions with a shared or_group_id
        for i, q_dict in enumerate(raw_question_blocks):
            lines_list = [l.strip() for l in q_dict["lines"] if l.strip()]
            joined_lines = "\n".join(lines_list)
            sub_parts = self._parse_sub_parts(joined_lines)

            if not sub_parts:
                # Reflow broken line wraps into a single coherent, flowing sentence
                full_text = " ".join(lines_list)
                full_text = re.sub(r"(\b[a-zA-Z]{2,})-\s+([a-zA-Z]{2,}\b)", r"\1\2", full_text)
                full_text = re.sub(r"\s+", " ", full_text).strip()
            else:
                full_text = joined_lines.strip()

            cleaned = normalize_question_text(full_text)
            marks = extract_marks_from_text(full_text)

            # If question has multiple sub-parts with individual marks, total is the sum
            if len(sub_parts) > 1:
                part_marks = [p.marks for p in sub_parts if p.marks is not None]
                if len(part_marks) == len(sub_parts):
                    marks = sum(part_marks)
                elif marks is None and part_marks:
                    marks = sum(part_marks)
            elif marks is None and sub_parts:
                part_marks = [p.marks for p in sub_parts if p.marks is not None]
                if part_marks:
                    marks = sum(part_marks)

            q_type = self._infer_question_type(full_text, marks)

            if marks is not None:
                total_marks_detected += marks

            # Assign OR group id
            or_group_id = None
            is_or = q_dict["is_or_choice"]
            if is_or and i > 0:
                # Link with preceding question
                prev_q = questions[i - 1]
                shared_id = prev_q.or_group_id or str(uuid.uuid4())[:8]
                prev_q.or_group_id = shared_id
                prev_q.is_or_choice = True
                or_group_id = shared_id

            question = SegmentedQuestion(
                question_number=q_dict["question_number"],
                raw_text=full_text,
                cleaned_text=cleaned,
                marks=marks,
                section=q_dict["section"],
                question_type=q_type,
                sub_parts=sub_parts,
                is_or_choice=is_or,
                or_group_id=or_group_id,
                page_number=q_dict["page_number"],
            )
            questions.append(question)

        logger.info(
            "segmentation_completed",
            total_questions=len(questions),
            total_marks=total_marks_detected,
            sections=sections_found,
        )

        return SegmentationResult(
            total_questions=len(questions),
            total_marks_detected=total_marks_detected,
            sections_detected=sections_found,
            questions=questions,
            unsegmented_blocks=unsegmented,
        )

    def segment_pages(
        self,
        pages_text: list[tuple[int, str]],
    ) -> SegmentationResult:
        """
        Segment a multi-page exam paper across all pages.

        Args:
            pages_text: List of (page_number, text_content) tuples.

        Returns:
            Aggregated SegmentationResult across the entire document.
        """
        all_questions: list[SegmentedQuestion] = []
        total_marks = 0
        all_sections: list[str] = []
        all_unsegmented: list[str] = []

        # Join text preserving page tracking
        for page_num, text in pages_text:
            page_res = self.segment_text(text, page_number=page_num)
            all_questions.extend(page_res.questions)
            total_marks += page_res.total_marks_detected
            for s in page_res.sections_detected:
                if s not in all_sections:
                    all_sections.append(s)
            all_unsegmented.extend(page_res.unsegmented_blocks)

        return SegmentationResult(
            total_questions=len(all_questions),
            total_marks_detected=total_marks,
            sections_detected=all_sections,
            questions=all_questions,
            unsegmented_blocks=all_unsegmented,
        )

    # Convenience alias
    detect_questions = segment_text
