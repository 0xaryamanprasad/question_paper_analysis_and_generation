"""
FastAPI Routes for AI Question Generation, Paper Generation, Validation, and PDF Export.
"""

from collections import defaultdict
import io
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.schemas.classification import SyllabusDefinition
from backend.api.schemas.generation import (
    GeneratedPaperResult,
    GeneratedQuestionItem,
    PaperBlueprintConfig,
)
from backend.api.schemas.validation import ValidationReport
from backend.database import get_db
from backend.models.generated_paper import GeneratedPaper
from backend.models.paper import Paper
from backend.models.question import Question, QuestionClassification
from backend.services.paper_generator import PaperGenerator
from backend.services.pdf_renderer import PDFRenderer
from backend.services.question_generator import QuestionGenerator
from backend.services.topic_classifier import TopicClassifier, resolve_syllabus_path
from backend.services.validation_engine import ValidationEngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/generation", tags=["Generation"])

# Singletons
_question_generator: Optional[QuestionGenerator] = None
_paper_generator: Optional[PaperGenerator] = None
_validation_engine: Optional[ValidationEngine] = None
_pdf_renderer: Optional[PDFRenderer] = None


def get_question_generator() -> QuestionGenerator:
    global _question_generator
    if _question_generator is None:
        _question_generator = QuestionGenerator()
    return _question_generator


def get_paper_generator() -> PaperGenerator:
    global _paper_generator
    if _paper_generator is None:
        _paper_generator = PaperGenerator(question_generator=get_question_generator())
    return _paper_generator


def get_validation_engine() -> ValidationEngine:
    global _validation_engine
    if _validation_engine is None:
        _validation_engine = ValidationEngine()
    return _validation_engine


def get_pdf_renderer() -> PDFRenderer:
    global _pdf_renderer
    if _pdf_renderer is None:
        _pdf_renderer = PDFRenderer()
    return _pdf_renderer


class GenerateQuestionRequest(BaseModel):
    """Payload to generate a single question."""
    subject: str = "Data Structures and Algorithms"
    unit_number: int = 1
    unit_name: str = "Introduction to Data Structures"
    topic: str = "Stacks and applications"
    keywords: List[str] = Field(default_factory=lambda: ["stack", "LIFO"])
    blooms_level: str = "Understand"
    marks: float = 5.0
    difficulty: str = "medium"
    question_type: str = "medium"
    avoid_questions: Optional[List[str]] = None


class GeneratePaperRequest(BaseModel):
    """Payload to generate an entire examination paper."""
    blueprint: PaperBlueprintConfig
    syllabus: Optional[SyllabusDefinition] = None


class ValidatePaperRequest(BaseModel):
    """Payload to validate an existing or generated paper."""
    paper: GeneratedPaperResult
    blueprint: PaperBlueprintConfig
    syllabus: Optional[SyllabusDefinition] = None


@router.post(
    "/generate-question",
    response_model=GeneratedQuestionItem,
    summary="Generate a single question using Generative AI",
)
async def generate_single_question(
    payload: GenerateQuestionRequest,
):
    """Generate a single exam question matching specified topic, marks, and cognitive level."""
    gen = get_question_generator()
    return gen.generate_question(
        subject=payload.subject,
        unit_number=payload.unit_number,
        unit_name=payload.unit_name,
        topic=payload.topic,
        keywords=payload.keywords,
        blooms_level=payload.blooms_level,
        marks=payload.marks,
        difficulty=payload.difficulty,
        question_type=payload.question_type,
        avoid_questions=payload.avoid_questions,
    )


@router.post(
    "/generate-paper",
    response_model=GeneratedPaperResult,
    summary="Generate a complete examination question paper from blueprint",
)
async def generate_paper(
    payload: GeneratePaperRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a full university question paper meeting all section, unit, marks,
    and cognitive distribution constraints. Grounded directly in uploaded PDF papers.
    """
    paper_gen = get_paper_generator()
    blueprint = payload.blueprint
    subject = (blueprint.subject or "").strip()

    # Sanitize institution and exam headers (remove artificial AI placeholders)
    if not blueprint.institution_name or "antigravity" in blueprint.institution_name.lower():
        blueprint.institution_name = "National Institute of Technology"
    if not blueprint.examination_name or "antigravity" in blueprint.examination_name.lower():
        blueprint.examination_name = "B.Tech Degree End Semester Examinations"

    # Step 1: Check database for uploaded question papers
    query = (
        select(Question, Paper.subject, Paper.year)
        .join(Paper, Question.paper_id == Paper.id)
        .options(selectinload(Question.classification))
    )
    res = await db.execute(query)
    all_q_rows = res.all()

    # Identify subjects present in the uploaded corpus
    db_subjects = list(dict.fromkeys(row[1] for row in all_q_rows if row[1]))

    # Check if blueprint subject matches an uploaded corpus
    matching_rows = [row for row in all_q_rows if row[1] and subject.lower() in row[1].lower()]

    # If user left blueprint subject as default DSA, but uploaded papers are for a different subject (e.g. Mathematics-II)
    if not matching_rows and db_subjects and ("data structure" in subject.lower() or "dsa" in subject.lower()):
        target_subject = db_subjects[0]
        logger.info(
            "auto_aligned_subject_with_uploaded_corpus",
            old_subject=subject,
            new_subject=target_subject,
        )
        subject = target_subject
        blueprint.subject = target_subject
        matching_rows = [row for row in all_q_rows if row[1] == target_subject]
    else:
        target_subject = subject

    # Step 2: Extract real exemplar questions and topics from uploaded papers
    exemplars_by_unit: Dict[int, List[str]] = defaultdict(list)
    questions_data: List[Dict[str, Any]] = []

    for q, p_sub, p_year in matching_rows:
        unit_num = 1
        topic_name = "General"
        if q.classification:
            if q.classification.unit and q.classification.unit.isdigit():
                unit_num = int(q.classification.unit)
            if q.classification.topic:
                topic_name = q.classification.topic

        # If question has sub_parts, extract each sub-question prompt
        if q.sub_parts and isinstance(q.sub_parts, list):
            for sp in q.sub_parts:
                sp_text = sp.get("text", "").strip() if isinstance(sp, dict) else str(sp).strip()
                if sp_text and len(sp_text) > 6 and not re.search(r"^\d{2,4}[-/]\d{2}[-/]\d{4}", sp_text):
                    exemplars_by_unit[unit_num].append(sp_text)
                    questions_data.append({
                        "id": f"{q.id}_{sp.get('label', '')}",
                        "paper_id": str(q.paper_id),
                        "year": str(p_year) if p_year else "2023",
                        "text": sp_text,
                        "unit_number": unit_num,
                        "unit_name": f"Unit {unit_num}",
                        "topic": topic_name,
                        "marks": float(sp.get("marks") or 2.0),
                        "blooms_level": "Apply",
                    })
        else:
            q_text = (q.cleaned_text or q.raw_text or "").strip()
            if q_text and len(q_text) > 6 and not re.search(r"^\d{2,4}[-/]\d{2}[-/]\d{4}", q_text):
                exemplars_by_unit[unit_num].append(q_text)
                questions_data.append({
                    "id": str(q.id),
                    "paper_id": str(q.paper_id),
                    "year": str(p_year) if p_year else "2023",
                    "text": q_text,
                    "unit_number": unit_num,
                    "unit_name": f"Unit {unit_num}",
                    "topic": topic_name,
                    "marks": float(q.marks or 5.0),
                    "blooms_level": "Understand",
                })

    # Step 3: Resolve authentic subject syllabus
    syllabus = payload.syllabus
    if syllabus is None:
        syllabus_path = resolve_syllabus_path(target_subject)
        if "sample_syllabus" not in str(syllabus_path) and syllabus_path.exists():
            with open(syllabus_path, "r", encoding="utf-8") as f:
                syllabus = SyllabusDefinition(**json.load(f))
        else:
            # Fall back to TopicClassifier syllabus for target subject
            topic_clf = TopicClassifier(syllabus=syllabus_path)
            syllabus = topic_clf.syllabus

    # Step 4: Run historical analysis to get hot repeated topics from uploaded PDF
    hist_analysis = None
    if questions_data:
        from backend.api.routes.analysis import get_historical_analyzer
        hist_analysis = get_historical_analyzer().analyze(questions_data, subject=target_subject)

    # Step 5: Generate paper grounded in uploaded PDFs
    try:
        paper = paper_gen.generate_paper(
            blueprint=blueprint,
            syllabus=syllabus,
            historical_analysis=hist_analysis,
            exemplar_questions=exemplars_by_unit,
        )

        # Step 6: Persist generated paper to database
        try:
            gen_paper_model = GeneratedPaper(
                id=uuid.uuid4(),
                subject=target_subject,
                title=f"{target_subject} {blueprint.examination_name or 'Examination Paper'}",
                generation_config=blueprint.model_dump(),
                selected_questions=[s.model_dump() for s in paper.sections],
                status="completed",
            )
            db.add(gen_paper_model)
            await db.commit()
        except Exception as db_err:
            logger.warning("failed_to_persist_generated_paper", error=str(db_err))

        return paper

    except Exception as e:
        logger.error("paper_generation_route_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate question paper: {str(e)}",
        ) from e


@router.post(
    "/validate-paper",
    response_model=ValidationReport,
    summary="Independently validate constraints of a generated paper",
)
async def validate_paper(
    payload: ValidatePaperRequest,
):
    """Validate marks totals, question counts, unit coverage, and absence of duplicate questions."""
    val_engine = get_validation_engine()

    syllabus = payload.syllabus
    if syllabus is None:
        syllabus_path = resolve_syllabus_path(payload.blueprint.subject)
        if syllabus_path.exists():
            with open(syllabus_path, "r", encoding="utf-8") as f:
                syllabus = SyllabusDefinition(**json.load(f))
        else:
            topic_clf = TopicClassifier()
            syllabus = topic_clf.syllabus

    report = val_engine.validate_paper(
        paper=payload.paper,
        blueprint=payload.blueprint,
        syllabus=syllabus,
        check_semantic_duplicates=True,
    )
    return report


@router.post(
    "/render-pdf",
    summary="Render generated question paper into downloadable PDF",
    response_class=Response,
)
async def render_paper_pdf(
    paper: GeneratedPaperResult,
):
    """
    Render a camera-ready PDF document from a GeneratedPaperResult.
    Returns PDF stream for in-browser viewing or download.
    """
    renderer = get_pdf_renderer()
    try:
        pdf_bytes = renderer.render_pdf(paper)
        clean_subject = paper.subject.replace(" ", "_").lower()
        filename = f"{clean_subject}_exam_paper.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except Exception as e:
        logger.error("pdf_render_route_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render PDF: {str(e)}",
        ) from e
