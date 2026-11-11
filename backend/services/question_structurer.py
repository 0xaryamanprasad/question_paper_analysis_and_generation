"""
Module 3: Question Paper Structuring Service with Pydantic Schemas.

Extracts:
- Subject
- Examination year
- Question number
- Subquestion number
- Question text
- Mathematical expressions in LaTeX
- Marks
- Section
- Page number

Preserves the original question hierarchy (Main question -> Subquestions).
Utilizes Gemini structured JSON outputs with Pydantic schemas, with a robust
deterministic hierarchical parser fallback.
"""

import json
import re
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class StructuredSubQuestion(BaseModel):
    """Represents a subquestion within a parent question (e.g., 1(a), 2(i))."""
    subquestion_id: str = Field(..., description="Subquestion identifier (e.g., 'a', 'b', 'i', 'ii')")
    text: str = Field(..., description="Subquestion text content")
    latex_expressions: List[str] = Field(default_factory=list, description="LaTeX expressions inside subquestion")
    marks: Optional[int] = Field(default=None, description="Marks allocated to this subquestion")


class StructuredQuestion(BaseModel):
    """Represents a full question, containing subquestions if present."""
    question_number: str = Field(..., description="Main question number (e.g., '1', '2', '3')")
    subquestion_number: Optional[str] = Field(default=None, description="Direct subquestion number if flat")
    section: Optional[str] = Field(default=None, description="Section or Part (e.g., 'Section A', 'Part I')")
    question_text: str = Field(..., description="Question prompt / main body text")
    latex_expressions: List[str] = Field(default_factory=list, description="All LaTeX formulas in question")
    marks: Optional[int] = Field(default=None, description="Total marks allocated to this question")
    page_number: int = Field(default=1, description="Page number where the question originates")
    subquestions: List[StructuredSubQuestion] = Field(
        default_factory=list, description="Child subquestions preserving original hierarchy"
    )
    raw_ocr_text: str = Field(default="", description="Original raw OCR text")
    corrected_text: str = Field(default="", description="Cleaned and corrected question text")


class StructuredExamPaper(BaseModel):
    """Hierarchical schema representing the entire structured examination paper."""
    subject: str = Field(default="Data Structures and Algorithms", description="Subject name")
    examination_year: str = Field(default="Unknown", description="Year of the examination")
    total_marks: Optional[int] = Field(default=None, description="Total maximum marks for the paper")
    sections: List[str] = Field(default_factory=list, description="List of detected sections")
    questions: List[StructuredQuestion] = Field(
        default_factory=list, description="Hierarchical list of questions"
    )


class QuestionStructurer:
    """
    Structures question papers into validated Pydantic schemas using Gemini
    or a fallback regex-based hierarchical parser.
    """

    STRUCTURING_PROMPT = """You are an expert exam parser. Convert the following examination paper text into structured JSON matching the provided schema.

REQUIRED EXTRACTIONS:
- Subject (e.g., "Data Structures and Algorithms")
- Examination year (e.g., "2023", "2024")
- Total marks
- Sections (e.g., ["Section A", "Section B"])
- Questions list:
  - question_number (e.g., "1", "2")
  - section (e.g., "Section A")
  - question_text (main question statement)
  - latex_expressions (list of LaTeX strings like ["$O(n \\log n)$"])
  - marks (integer or null)
  - page_number (integer)
  - subquestions (list of child subquestions with subquestion_id like 'a', 'b', text, latex_expressions, marks)
  - raw_ocr_text and corrected_text

Preserve the original hierarchy: if Question 1 has parts (a) and (b), nest them inside the subquestions list.
"""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("gemini_structurer_init_failed", error=str(e))

    def structure_paper(
        self,
        full_text: str,
        page_texts: Optional[dict[int, str]] = None,
        default_subject: str = "Data Structures and Algorithms",
        default_year: str = "2024",
    ) -> StructuredExamPaper:
        """
        Structures paper text into a validated StructuredExamPaper model.
        """
        page_dict = page_texts or {1: full_text}

        # 1. Try Gemini structured outputs
        if self.client and full_text.strip():
            try:
                return self._call_gemini_structurer(full_text, default_subject, default_year)
            except Exception as err:
                logger.warning(
                    "gemini_structuring_failed_fallback_to_hierarchical_parser",
                    error=str(err),
                )

        # 2. Deterministic hierarchical parser fallback
        return self._hierarchical_parse(full_text, page_dict, default_subject, default_year)

    def _call_gemini_structurer(
        self,
        full_text: str,
        default_subject: str,
        default_year: str,
    ) -> StructuredExamPaper:
        """Calls Gemini API requesting structured JSON conforming to StructuredExamPaper."""
        prompt = (
            f"{self.STRUCTURING_PROMPT}\n\n"
            f"EXAM PAPER TEXT:\n```\n{full_text}\n```\n"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )

        data = json.loads(response.text.strip())
        # Enforce subject/year defaults if missing
        if not data.get("subject"):
            data["subject"] = default_subject
        if not data.get("examination_year"):
            data["examination_year"] = default_year

        # Validate with Pydantic
        return StructuredExamPaper.model_validate(data)

    def _hierarchical_parse(
        self,
        full_text: str,
        page_dict: dict[int, str],
        default_subject: str,
        default_year: str,
    ) -> StructuredExamPaper:
        """
        High-fidelity deterministic parser extracting exam hierarchy:
        Sections -> Questions -> Subquestions -> Marks -> LaTeX formulas.
        """
        raw_lines = [l.strip() for l in full_text.splitlines() if l.strip()]

        # Filter out headers, administrative metadata, watermarks, and barcode stamps
        lines: List[str] = []
        for l in raw_lines:
            # Watermark / stamp / date lines (e.g., 217-22/05/2025--12)
            if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", l) or re.search(r"^\d{2,4}[-/]\d{2}[-/]\d{4}", l):
                continue
            # Boilerplate headers
            if re.search(
                r"^(Registration No|Total Number of Pages|Sub_Code|Course:|BRANCH|Time:|Max Marks|Q\.Code|"
                r"Answer Question No|The figures in the right hand margin indicate marks)",
                l,
                re.IGNORECASE,
            ):
                continue
            lines.append(l)

        # Metadata extraction
        subject = default_subject
        year = default_year
        sections_found: List[str] = []

        # Find subject & year from header
        header_text = " ".join(lines[:15])
        year_match = re.search(r"\b(20\d\d)(?:-\d{2,4})?\b", header_text)
        if year_match:
            year = year_match.group(1)

        sub_match = re.search(
            r"(?:SUBJECT:\s*)?(Mathematics(?:\s*[-–]\s*[IVX]+)?|Data Structures(?: and Algorithms)?|"
            r"Design and Analysis of Algorithms|Algorithms|CS\d{3,4}|Calculus|Linear Algebra|Physics)",
            header_text,
            re.IGNORECASE,
        )
        if sub_match:
            subject = sub_match.group(1).strip()

        questions: List[StructuredQuestion] = []
        current_section: Optional[str] = None
        current_q: Optional[StructuredQuestion] = None
        current_sub: Optional[StructuredSubQuestion] = None

        section_pat = re.compile(r"^(SECTION|PART)\s*[-–:]*\s*([A-Z0-9]+|\b(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\b)", re.IGNORECASE)
        q_start_pat = re.compile(
            r"^(?:(?:Q(?:uestion|\.)?\s*(\d{1,3}))|(?:\(?(\d{1,3})[\.\:\)\-]))(?:\s+(.*))?$", re.IGNORECASE
        )
        sub_pat = re.compile(
            r"^(?:\(([a-zA-Z]|[ivxlcdm]+)\)|([a-zA-Z])[\.\)])(?:\s+(.*))?$", re.IGNORECASE
        )

        for line in lines:
            # 1. Section match
            sec_m = section_pat.match(line)
            if sec_m:
                current_section = f"{sec_m.group(1).title()} {sec_m.group(2).upper()}"
                if current_section not in sections_found:
                    sections_found.append(current_section)
                continue

            # 2. Main question match
            q_m = q_start_pat.match(line)
            # Only treat as new main question if not looking like a subquestion
            if q_m and not sub_pat.match(line):
                # Save previous subquestion if active
                if current_q and current_sub:
                    current_q.subquestions.append(current_sub)
                    current_sub = None

                # Save previous question
                if current_q:
                    self._finalize_question(current_q, page_dict)
                    questions.append(current_q)

                q_num = q_m.group(1) or q_m.group(2)
                q_rest = (q_m.group(3) or "").strip()
                marks = self._extract_marks(line)

                current_q = StructuredQuestion(
                    question_number=q_num,
                    section=current_section,
                    question_text=q_rest,
                    marks=marks,
                    page_number=self._find_page_for_text(line, page_dict),
                    raw_ocr_text=line,
                    corrected_text=line,
                )
                continue

            # 3. Subquestion match
            sub_m = sub_pat.match(line)
            if sub_m and current_q is not None:
                if current_sub:
                    current_q.subquestions.append(current_sub)

                sub_id = (sub_m.group(1) or sub_m.group(2) or "").lower()
                sub_rest = (sub_m.group(3) or "").strip()
                sub_marks = self._extract_marks(line)

                current_sub = StructuredSubQuestion(
                    subquestion_id=sub_id,
                    text=sub_rest,
                    marks=sub_marks,
                    latex_expressions=self._extract_latex(sub_rest),
                )
                continue

            # 4. Continuation line
            if current_sub:
                current_sub.text = (current_sub.text + " " + line).strip() if current_sub.text else line
                if not current_sub.marks:
                    current_sub.marks = self._extract_marks(line)
                current_sub.latex_expressions = self._extract_latex(current_sub.text)
            elif current_q:
                current_q.question_text = (current_q.question_text + " " + line).strip() if current_q.question_text else line
                current_q.raw_ocr_text += "\n" + line
                current_q.corrected_text += "\n" + line
                if not current_q.marks:
                    current_q.marks = self._extract_marks(line)

        # Flush remaining question/subquestion
        if current_q:
            if current_sub:
                current_q.subquestions.append(current_sub)
            self._finalize_question(current_q, page_dict)
            questions.append(current_q)

        # Compute total marks
        total_marks = sum(q.marks for q in questions if q.marks is not None) or None

        return StructuredExamPaper(
            subject=subject,
            examination_year=year,
            total_marks=total_marks,
            sections=sections_found,
            questions=questions,
        )

    def _finalize_question(self, q: StructuredQuestion, page_dict: dict[int, str]):
        """Finalizes latex extraction and subquestion marks rollup."""
        all_latex = self._extract_latex(q.question_text)
        for sub in q.subquestions:
            all_latex.extend(sub.latex_expressions)
        q.latex_expressions = list(dict.fromkeys(all_latex))

        # Roll up marks if parent has none but subquestions do
        if q.marks is None and q.subquestions:
            sub_marks = [s.marks for s in q.subquestions if s.marks is not None]
            if sub_marks:
                q.marks = sum(sub_marks)

    def _extract_marks(self, text: str) -> Optional[int]:
        """Extracts mark value (e.g. '[10]', '(5 marks)', '(2 x 10)', '[6M]')."""
        # Multiplication format like (2 x 10) or (6 x 8) or (8 x 2)
        multi_m = re.search(r"\(\s*(\d+)\s*[xX*]\s*(\d+)\s*\)", text)
        if multi_m:
            try:
                v1 = int(multi_m.group(1))
                v2 = int(multi_m.group(2))
                return v1 * v2
            except Exception:
                pass

        m = re.search(r"[\[\(](\d+)\s*(?:marks?|m|pts?)?[\]\)]", text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 100:
                return val
        return None

    def _extract_latex(self, text: str) -> List[str]:
        """Collects all $...$ or $$...$$ LaTeX expressions, or converts raw math tokens to LaTeX."""
        matches = re.findall(r"\$\$?[^$]+\$\$?", text)
        results = list(matches)

        # Detect asymptotic notation if not enclosed in $...$
        for m in re.finditer(r"\b[OT\u0398\u03a9]\s*\(\s*[^)]+\s*\)", text):
            raw = m.group(0).strip()
            candidate = f"${raw}$"
            if candidate not in results and f"${raw}" not in "".join(results):
                results.append(candidate)

        # Detect recurrence relations
        for m in re.finditer(r"\bT\s*\(\s*n\s*\)\s*=\s*[0-9a-zA-Z+\-*/^()\s]+", text):
            raw = m.group(0).strip()
            candidate = f"${raw}$"
            if candidate not in results and f"${raw}" not in "".join(results):
                results.append(candidate)

        return list(dict.fromkeys(results))

    def _find_page_for_text(self, text_snippet: str, page_dict: dict[int, str]) -> int:
        """Finds which page number contained this snippet."""
        clean = text_snippet[:30].strip()
        for p_num, content in page_dict.items():
            if clean in content:
                return p_num
        return 1
