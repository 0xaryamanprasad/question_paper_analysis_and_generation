"""
PDF Generation Module (Module 13).

Generates professional, camera-ready examination question paper PDFs using ReportLab.
Features:
- Formal academic institution and examination header.
- Course code, subject name, exam duration, and maximum marks.
- Section divisions (Part A, Part B, Part C) with instructions.
- Clear Question Numbering, Sub-parts formatting, and Marks alignment.
- Centered, stylized '(OR)' dividers for internal choices.
- Running footer with dynamic page numbering ('Page X of Y').
"""

import io
from pathlib import Path
import re
from typing import Optional, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from backend.api.schemas.generation import GeneratedPaperResult
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def clean_math_text(text: str) -> str:
    """
    Sanitize and convert mathematical expressions and LaTeX notation
    into clean, camera-ready academic typography for ReportLab Paragraphs.
    """
    if not text:
        return ""

    text = str(text).replace("\r\n", "\n").replace("\r", "\n")

    # Escape standalone ampersands
    text = re.sub(r"&(?!(amp|lt|gt|quot|apos|#\d+|[a-zA-Z]+);)", "&amp;", text)

    # Fractions: \frac{num}{den} -> (num)/(den)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)

    # Square roots: \sqrt{x} -> √(x)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", text)

    # Common LaTeX math symbols & operators to HTML entities
    replacements = [
        (r"\\iint", "&#8748;"),
        (r"\\oint", "&#8750;"),
        (r"\\int", "&#8747;"),
        (r"\\nabla", "&#8711;"),
        (r"\\times", "&times;"),
        (r"\\cdot", "&middot;"),
        (r"\\partial", "&#8706;"),
        (r"\\infty", "&infin;"),
        (r"\\pm", "&plusmn;"),
        (r"\\pi", "&pi;"),
        (r"\\theta", "&theta;"),
        (r"\\lambda", "&lambda;"),
        (r"\\alpha", "&alpha;"),
        (r"\\beta", "&beta;"),
        (r"\\gamma", "&gamma;"),
        (r"\\sum", "&sum;"),
        (r"\\vec\{([a-zA-Z])\}", r"<b>\1&#8407;</b>"),
        (r"\\hat\{([a-zA-Z])\}", r"<b>\1&#770;</b>"),
        (r"\\cos", "cos"),
        (r"\\sin", "sin"),
        (r"\\tan", "tan"),
        (r"\\log", "log"),
        (r"\\ln", "ln"),
        (r"\\exp", "exp"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    # Superscripts: ^{n} or ^n -> <sup>n</sup>
    text = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", text)
    text = re.sub(r"\^([0-9a-zA-Z+\-]+)", r"<sup>\1</sup>", text)

    # Subscripts: _{n} or _n -> <sub>n</sub>
    text = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", text)
    text = re.sub(r"_([0-9a-zA-Z+\-]+)", r"<sub>\1</sub>", text)

    # Prime notation y'' -> y″, y' -> y′
    text = text.replace("''", "&#8243;").replace("'", "&#8242;")

    # Strip raw LaTeX fences $ ... $ and $$ ... $$
    text = text.replace("$$", "").replace("$", "")

    # Escape angle brackets not belonging to safe formatting tags
    allowed_tags = r"(?:/?(?:b|i|u|font|sup|sub|br)(?:\s+[^>]*?)?)"
    text = re.sub(r"<(?!" + allowed_tags + r">)", "&lt;", text)

    # Convert line breaks to <br/>
    text = text.replace("\n", "<br/>")

    return text.strip()


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to calculate and render total page count ('Page X of Y')
    and running footer on every page.
    """

    footer_left_text = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#718096"))
        self.setLineWidth(0.5)

        # Draw footer line
        self.line(40, 36, letter[0] - 40, 36)

        # Authentic academic footer (Course code and examination name, never AI labels)
        footer_left = self.footer_left_text or ""
        footer_right = f"Page {self._pageNumber} of {page_count}"

        self.setFillColor(colors.HexColor("#4A5568"))
        if footer_left:
            self.drawString(40, 24, footer_left)
        self.drawRightString(letter[0] - 40, 24, footer_right)

        self.restoreState()


def make_numbered_canvas(footer_text: str):
    """Create a NumberedCanvas subclass with custom academic running footer."""
    class CustomNumberedCanvas(NumberedCanvas):
        footer_left_text = footer_text
    return CustomNumberedCanvas


class PDFRenderer:
    """
    Renders academic question papers to PDF format.
    """

    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Create typography styles for academic paper layout."""
        self.styles.add(
            ParagraphStyle(
                name="InstitutionHeader",
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                alignment=1,  # Center
                textColor=colors.HexColor("#111827"),
                spaceAfter=1,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="InstitutionSub",
                fontName="Helvetica",
                fontSize=8.5,
                leading=11,
                alignment=1,  # Center
                textColor=colors.HexColor("#4B5563"),
                spaceAfter=3,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="ExamTitle",
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=13,
                alignment=1,
                textColor=colors.HexColor("#1F2937"),
                spaceAfter=5,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetaLabel",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetaLabelRight",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                alignment=2,  # Right
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MetaValue",
                fontName="Helvetica",
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#374151"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="GeneralInstructions",
                fontName="Helvetica",
                fontSize=7.5,
                leading=10.5,
                textColor=colors.HexColor("#4B5563"),
                spaceBefore=2,
                spaceAfter=6,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeader",
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=13,
                alignment=1,
                textColor=colors.HexColor("#111827"),
                spaceBefore=7,
                spaceAfter=2,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionDesc",
                fontName="Helvetica-Oblique",
                fontSize=8.5,
                leading=11,
                alignment=1,
                textColor=colors.HexColor("#4B5563"),
                spaceAfter=5,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="QuestionPrompt",
                fontName="Helvetica",
                fontSize=9,
                leading=13,
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubQuestionPrompt",
                fontName="Helvetica",
                fontSize=9,
                leading=13,
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="QuestionNumber",
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=13,
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="MarksRight",
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=13,
                alignment=2,  # Right
                textColor=colors.HexColor("#111827"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="OrDivider",
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                alignment=1,
                textColor=colors.HexColor("#4B5563"),
                spaceBefore=3,
                spaceAfter=3,
            )
        )

    def render_pdf(
        self,
        paper: GeneratedPaperResult,
        output_path: Optional[Union[Path, str]] = None,
    ) -> bytes:
        """
        Render a GeneratedPaperResult to PDF bytes or file.

        Args:
            paper: GeneratedPaperResult instance.
            output_path: Optional destination file path.

        Returns:
            Raw PDF bytes.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            leftMargin=38,
            rightMargin=38,
            topMargin=36,
            bottomMargin=44,
        )

        story = []

        # 1. Registration Number & Paper Code Header Bar
        qp_code = f"S{abs(hash(str(paper.id or '1'))) % 899 + 101}"
        reg_table_data = [
            [
                Paragraph("<b>Reg. No.:</b>&nbsp;&nbsp;[&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;]", self.styles["MetaLabel"]),
                Paragraph(f"<b>Q.P. Code:</b>&nbsp;{qp_code}", self.styles["MetaLabelRight"]),
            ]
        ]
        reg_table = Table(reg_table_data, colWidths=[380, 155])
        reg_table.setStyle(
            TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(reg_table)

        # 2. Authentic Institution & Examination Headers
        inst_name = (paper.institution_name or "").strip()
        if not inst_name or "antigravity" in inst_name.lower():
            inst_name = "NATIONAL INSTITUTE OF TECHNOLOGY"
        else:
            inst_name = inst_name.upper()

        exam_title = (paper.examination_name or "B.Tech Degree End Semester Examinations").upper()
        story.append(Paragraph(inst_name, self.styles["InstitutionHeader"]))
        story.append(Paragraph("Autonomous Institution &middot; Affiliated to State Technological University", self.styles["InstitutionSub"]))
        story.append(Paragraph(exam_title, self.styles["ExamTitle"]))
        story.append(
            HRFlowable(
                width="100%",
                thickness=1.2,
                color=colors.HexColor("#1A202C"),
                spaceBefore=1,
                spaceAfter=5,
            )
        )

        # 3. Metadata Table (Subject, Code, Duration, Max Marks)
        course_str = f"{paper.course_code} - {paper.subject}" if paper.course_code else paper.subject
        meta_data = [
            [
                Paragraph("<b>Course:</b>", self.styles["MetaLabel"]),
                Paragraph(course_str, self.styles["MetaValue"]),
                Paragraph("<b>Max Marks:</b>", self.styles["MetaLabel"]),
                Paragraph(f"{int(paper.total_marks)}", self.styles["MetaValue"]),
            ],
            [
                Paragraph("<b>Duration:</b>", self.styles["MetaLabel"]),
                Paragraph(f"{paper.duration_hours:g} Hours", self.styles["MetaValue"]),
                Paragraph("<b>Semester / Exam:</b>", self.styles["MetaLabel"]),
                Paragraph("Regular / End-Term", self.styles["MetaValue"]),
            ],
        ]

        meta_table = Table(
            meta_data,
            colWidths=[55, 275, 85, 120],
        )
        meta_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(meta_table)

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#CBD5E0"),
                spaceBefore=4,
                spaceAfter=5,
            )
        )

        # 4. Standard Academic Instructions Block
        instructions_text = (
            "<b>General Instructions:</b><br/>"
            "1. Answer ALL questions from Part A (Compulsory).<br/>"
            "2. Answer questions from subsequent sections according to choice constraints.<br/>"
            "3. The figures in the right-hand margin indicate full marks for the respective questions.<br/>"
            "4. Mathematical tables and non-programmable scientific calculators are permitted. State all assumptions clearly."
        )
        story.append(Paragraph(instructions_text, self.styles["GeneralInstructions"]))

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=colors.HexColor("#E2E8F0"),
                spaceBefore=2,
                spaceAfter=6,
            )
        )

        # 5. Sections and Questions
        for sec_idx, sec in enumerate(paper.sections):
            story.append(Paragraph(sec.section_name.upper(), self.styles["SectionHeader"]))

            desc = sec.description or (
                f"Answer any {sec.questions_to_attempt} questions. Each question carries {int(sec.questions[0].marks if sec.questions else 0)} marks."
                if sec.questions_to_attempt
                else f"Answer ALL questions. ({int(sec.total_section_marks)} Marks)"
            )
            story.append(Paragraph(desc, self.styles["SectionDesc"]))
            story.append(Spacer(1, 3))

            # Build questions table for this section
            q_table_data = []

            for q_idx, q in enumerate(sec.questions):
                # Check if this is an OR question alternative
                is_or_choice = "(OR)" in (q.question_number or "")

                if is_or_choice:
                    q_table_data.append(
                        [
                            "",
                            Paragraph("<b>[ &mdash; OR &mdash; ]</b>", self.styles["OrDivider"]),
                            "",
                        ]
                    )

                # Format question with subparts or single prompt
                if q.sub_parts and len(q.sub_parts) > 0:
                    raw_stem = (q.question_text or "").strip()
                    # Only render stem if it provides a genuine introductory instruction
                    has_distinct_stem = (
                        bool(raw_stem)
                        and not raw_stem.lower().startswith("(a)")
                        and not raw_stem.lower().startswith("a)")
                        and len(raw_stem) < 200
                        and not any(sp.text[:25] in raw_stem for sp in q.sub_parts if sp.text)
                    )

                    q_num_p = Paragraph(f"<b>{q.question_number}.</b>" if not is_or_choice else "<b>&nbsp;</b>", self.styles["QuestionNumber"])

                    if has_distinct_stem:
                        stem_p = Paragraph(clean_math_text(raw_stem), self.styles["QuestionPrompt"])
                        marks_p = Paragraph(f"[{int(q.marks)}]", self.styles["MarksRight"])
                        q_table_data.append([q_num_p, stem_p, marks_p])
                        sub_start_num = False
                    else:
                        sub_start_num = True

                    for sp_idx, sp in enumerate(q.sub_parts):
                        sub_lbl = sp.label if sp.label else chr(ord('a') + sp_idx)
                        sp_marks_val = sp.marks if sp.marks is not None else (q.marks / len(q.sub_parts))
                        sp_marks_str = f"[{int(sp_marks_val) if float(sp_marks_val).is_integer() else sp_marks_val}]"

                        row_num_p = (
                            q_num_p
                            if (sub_start_num and sp_idx == 0)
                            else Paragraph("<b>&nbsp;</b>", self.styles["QuestionNumber"])
                        )

                        sp_prompt = f"({sub_lbl})&nbsp;&nbsp;{clean_math_text(sp.text)}"
                        q_table_data.append([
                            row_num_p,
                            Paragraph(sp_prompt, self.styles["SubQuestionPrompt"]),
                            Paragraph(sp_marks_str, self.styles["MarksRight"]),
                        ])
                else:
                    # Single question prompt
                    prompt_p = Paragraph(clean_math_text(q.question_text), self.styles["QuestionPrompt"])
                    q_num_p = Paragraph(f"<b>{q.question_number}.</b>" if not is_or_choice else "<b>&nbsp;</b>", self.styles["QuestionNumber"])
                    marks_p = Paragraph(f"[{int(q.marks)}]", self.styles["MarksRight"])
                    q_table_data.append([q_num_p, prompt_p, marks_p])

            if q_table_data:
                q_table = Table(
                    q_table_data,
                    colWidths=[40, 445, 50],
                )
                q_table.setStyle(
                    TableStyle(
                        [
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                            ("LEFTPADDING", (0, 0), (-1, -1), 2),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ]
                    )
                )
                story.append(KeepTogether(q_table))

            if sec_idx < len(paper.sections) - 1:
                story.append(Spacer(1, 8))

        # Dynamic academic running footer on every page
        footer_label = f"Sub Code: {paper.course_code} | {paper.subject}" if paper.course_code else (paper.subject or "Examination Paper")
        doc.build(story, canvasmaker=make_numbered_canvas(footer_label))
        pdf_bytes = buffer.getvalue()
        buffer.close()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "wb") as f:
                f.write(pdf_bytes)
            logger.info("pdf_saved_to_file", path=str(out_p), bytes_len=len(pdf_bytes))

        return pdf_bytes
