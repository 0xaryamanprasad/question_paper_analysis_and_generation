"""
FastAPI Routes for PDF Paper Upload, Extraction, and Question Management.
"""

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.schemas.paper import PageResponse, PaperListResponse, PaperResponse
from backend.api.schemas.question import QuestionResponse
from backend.config import get_settings
from backend.database import get_db
from backend.models.paper import Page, Paper
from backend.models.question import Question, QuestionClassification
from backend.services.nlp_preprocessor import NLPPreprocessor
from backend.services.ocr_engine import OCREngine
from backend.services.pdf_extractor import PDFExtractor
from backend.services.question_detector import QuestionDetector
from backend.services.question_structurer import QuestionStructurer
from backend.utils.file_utils import sanitize_filename, validate_file_extension
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/papers", tags=["Papers"])

# Lazy-loaded service instances
_pdf_extractor: Optional[PDFExtractor] = None
_ocr_engine: Optional[OCREngine] = None
_question_detector: Optional[QuestionDetector] = None
_nlp_preprocessor: Optional[NLPPreprocessor] = None
_question_structurer: Optional[QuestionStructurer] = None


def get_pdf_extractor() -> PDFExtractor:
    global _pdf_extractor
    if _pdf_extractor is None:
        _pdf_extractor = PDFExtractor()
    return _pdf_extractor


def get_ocr_engine() -> OCREngine:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def get_question_detector() -> QuestionDetector:
    global _question_detector
    if _question_detector is None:
        _question_detector = QuestionDetector()
    return _question_detector


def get_nlp_preprocessor() -> NLPPreprocessor:
    global _nlp_preprocessor
    if _nlp_preprocessor is None:
        _nlp_preprocessor = NLPPreprocessor()
    return _nlp_preprocessor


def get_question_structurer() -> QuestionStructurer:
    global _question_structurer
    if _question_structurer is None:
        _question_structurer = QuestionStructurer()
    return _question_structurer
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


def get_question_detector() -> QuestionDetector:
    global _question_detector
    if _question_detector is None:
        _question_detector = QuestionDetector()
    return _question_detector


def get_nlp_preprocessor() -> NLPPreprocessor:
    global _nlp_preprocessor
    if _nlp_preprocessor is None:
        _nlp_preprocessor = NLPPreprocessor()
    return _nlp_preprocessor


@router.post(
    "/upload",
    response_model=PaperResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process past-year question paper PDF",
)
async def upload_paper(
    file: UploadFile = File(..., description="PDF file of previous-year exam paper"),
    subject: str = Form(..., description="Academic course or subject title"),
    year: Optional[str] = Form(None, description="Examination year (e.g. '2023')"),
    semester: Optional[str] = Form(None, description="Semester or exam session"),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a question paper PDF, extract page text (using PyMuPDF for digital pages
    and PaddleOCR PP-OCRv5 for scanned images), segment into questions, and persist.
    """
    settings = get_settings()

    if not validate_file_extension(file.filename, {".pdf"}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files (.pdf) are supported.",
        )

    clean_filename = sanitize_filename(file.filename)
    paper_uuid = uuid.uuid4()
    saved_filename = f"{paper_uuid}_{clean_filename}"
    saved_path = settings.upload_path / saved_filename

    try:
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error("file_save_failed", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file.",
        ) from e

    file_size = saved_path.stat().st_size

    # Create initial database record
    paper = Paper(
        id=paper_uuid,
        subject=subject,
        year=year or "2023",
        semester=semester,
        exam_type="End-Term",
        original_filename=clean_filename,
        file_path=str(saved_path),
        total_pages=1,
        extraction_status="processing",
    )
    db.add(paper)
    await db.commit()
    await db.refresh(paper)

    try:
        pdf_extractor = get_pdf_extractor()
        ocr_engine = get_ocr_engine()
        question_structurer = get_question_structurer()
        nlp_preprocessor = get_nlp_preprocessor()

        # Step 1: Extract Pages (force_render_all_images=True so all formulas/diagrams can be read by OCR)
        extraction_result = pdf_extractor.extract(saved_path, force_render_all_images=True)
        paper.total_pages = extraction_result.total_pages

        combined_text_parts = []
        is_any_page_scanned = False
        page_dict: dict[int, str] = {}

        for page_res in extraction_result.pages:
            page_text = page_res.raw_text
            page_type = page_res.page_type

            # If page is scanned, execute authentic OCR pipeline
            if page_type == "scanned" and page_res.image_path:
                is_any_page_scanned = True
                try:
                    ocr_res = ocr_engine.process_image(page_res.image_path)
                    page_text = ocr_res.raw_text or getattr(ocr_res, "extracted_text", "")
                except Exception as ocr_err:
                    logger.warning("ocr_page_fallback", page=page_res.page_number, error=str(ocr_err))
                    page_text = page_res.raw_text or ""

            # If digital page has embedded raster images (e.g. diagrams or figures),
            # run OCR on the rendered image and append novel image-only annotations.
            elif page_type == "digital" and page_res.image_path:
                layout_meta = page_res.layout_data or {}
                has_embedded_images = layout_meta.get("image_count", 0) > 0
                if has_embedded_images:
                    try:
                        ocr_res = ocr_engine.process_image(page_res.image_path)
                        ocr_text = ocr_res.raw_text or ""
                        if ocr_text.strip():
                            page_text_lower = page_text.lower()
                            ocr_extra_lines = []
                            for ocr_line in ocr_text.splitlines():
                                stripped = ocr_line.strip()
                                # Only append novel text lines that are not already present in the digital text
                                if len(stripped) > 4 and stripped.lower() not in page_text_lower:
                                    ocr_extra_lines.append(stripped)
                            if ocr_extra_lines:
                                page_text = page_text.rstrip() + "\n\n" + "\n".join(ocr_extra_lines)
                                logger.info(
                                    "digital_page_ocr_merged",
                                    page=page_res.page_number,
                                    extra_lines=len(ocr_extra_lines),
                                )
                    except Exception as ocr_err:
                        logger.warning(
                            "digital_page_ocr_fallback",
                            page=page_res.page_number,
                            error=str(ocr_err),
                        )

            # Structure entire page text using spaCy linguistic NLP pipeline
            try:
                structured_page = nlp_preprocessor.structure_page_text(page_text)
                if structured_page.strip():
                    page_text = structured_page
            except Exception as struct_err:
                logger.warning(
                    "nlp_page_structuring_fallback",
                    page=page_res.page_number,
                    error=str(struct_err),
                )

            combined_text_parts.append(page_text)
            page_dict[page_res.page_number] = page_text

            page_model = Page(
                id=uuid.uuid4(),
                paper_id=paper.id,
                page_number=page_res.page_number,
                page_type=page_type,
                raw_text=page_text,
                image_path=str(page_res.image_path) if page_res.image_path else None,
            )
            db.add(page_model)

        full_extracted_text = "\n\n".join(combined_text_parts)

        # Step 2: Structure Paper using QuestionStructurer (Layout-aware, hierarchical, watermark-filtered)
        structured_paper = question_structurer.structure_paper(
            full_text=full_extracted_text,
            page_texts=page_dict,
            default_subject=subject,
            default_year=year or "2023",
        )

        from backend.api.routes.analysis import get_blooms_classifier, get_topic_classifier

        topic_clf = get_topic_classifier(subject=subject)
        blooms_clf = get_blooms_classifier()

        detected_questions = structured_paper.questions
        for sq in detected_questions:
            # Build sub_parts payload matching QuestionResponse & frontend expectations
            sub_parts_json = [
                {
                    "label": sub.subquestion_id,
                    "text": sub.text,
                    "marks": sub.marks,
                    "latex_expressions": sub.latex_expressions,
                }
                for sub in sq.subquestions
            ]

            # Primary prompt text
            q_text = sq.question_text.strip()
            if not q_text:
                if sq.subquestions:
                    q_text = f"Question {sq.question_number} ({len(sq.subquestions)} sub-questions)"
                else:
                    q_text = f"Question {sq.question_number}"

            nlp_doc = nlp_preprocessor.preprocess(q_text)

            # Build full text for topic classification
            combined_clf_text = q_text
            if sq.subquestions:
                combined_clf_text += "\n" + "\n".join(s.text for s in sq.subquestions)

            t_res = topic_clf.classify_question(combined_clf_text)
            b_res = blooms_clf.classify_question(combined_clf_text)

            unit_no = t_res.unit_number or 1
            topic_name = t_res.topic or "General"
            blooms_level = b_res.blooms_level.value

            q_id = uuid.uuid4()
            q_model = Question(
                id=q_id,
                paper_id=paper.id,
                question_number=sq.question_number,
                raw_text=sq.raw_ocr_text or q_text,
                cleaned_text=nlp_doc.normalized_text or q_text,
                latex_expressions=sq.latex_expressions,
                marks=int(sq.marks) if sq.marks else None,
                section=sq.section,
                question_type="Long Answer" if (sq.marks and sq.marks > 5) else "Short Answer",
                sub_parts=sub_parts_json if sub_parts_json else None,
                dsa_topic=topic_name,
                dsa_confidence=t_res.confidence,
                original_page_number=sq.page_number,
            )
            db.add(q_model)

            clf_model = QuestionClassification(
                id=uuid.uuid4(),
                question_id=q_id,
                unit=str(unit_no),
                topic=topic_name,
                topic_confidence=t_res.confidence,
                blooms_level=blooms_level,
                blooms_confidence=b_res.confidence,
                keywords=t_res.matched_keywords,
            )
            db.add(clf_model)

        paper.extraction_status = "completed"
        await db.commit()
        await db.refresh(paper)

        logger.info(
            "paper_processing_completed",
            paper_id=str(paper.id),
            questions_detected=len(detected_questions),
            is_scanned=is_any_page_scanned,
        )

        return PaperResponse.model_validate(paper)

    except Exception as e:
        logger.error("paper_processing_pipeline_failed", paper_id=str(paper.id), error=str(e))
        paper.extraction_status = "failed"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during PDF processing pipeline: {str(e)}",
        ) from e


@router.get(
    "",
    response_model=PaperListResponse,
    summary="List all uploaded question papers",
)
async def list_papers(
    subject: Optional[str] = None,
    year: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve list of all uploaded papers with optional filtering."""
    query = select(Paper)
    if subject:
        query = query.where(Paper.subject.ilike(f"%{subject}%"))
    if year:
        query = query.where(Paper.year == year)

    query = query.order_by(Paper.created_at.desc())
    result = await db.execute(query)
    papers = result.scalars().all()

    return PaperListResponse(
        total=len(papers),
        papers=[PaperResponse.model_validate(p) for p in papers],
    )


@router.delete(
    "/reset",
    summary="Reset corpus, delete papers, and clear context for a subject or all subjects",
)
@router.post(
    "/reset",
    summary="Reset corpus, delete papers, and clear context for a subject or all subjects",
)
async def reset_corpus(
    subject: Optional[str] = None,
    all_data: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Completely reset the learned question papers and context.
    Deletes associated files and cascades deletion of pages, questions,
    and classifications so the platform can start fresh with 0 papers.
    """
    settings = get_settings()
    query = select(Paper)
    if subject and not all_data:
        clean_sub = subject.strip()
        query = query.where(Paper.subject.ilike(f"%{clean_sub}%"))

    result = await db.execute(query)
    papers_to_delete = result.scalars().all()
    count = len(papers_to_delete)

    for paper in papers_to_delete:
        if paper.file_path:
            p = Path(paper.file_path)
            try:
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("failed_to_delete_paper_file", path=str(p), error=str(e))
        try:
            for rendered_img in settings.upload_path.glob(f"*{paper.id}*"):
                rendered_img.unlink(missing_ok=True)
            doc_stem = Path(paper.original_filename).stem if paper.original_filename else ""
            if doc_stem:
                for rendered_img in settings.upload_path.glob(f"{doc_stem}_page_*.png"):
                    rendered_img.unlink(missing_ok=True)
        except Exception:
            pass

        await db.delete(paper)

    await db.commit()
    logger.info("corpus_reset_completed", subject=subject, deleted_papers=count)

    return {
        "status": "success",
        "message": f"Successfully cleared {count} paper(s) and reset intelligence context.",
        "deleted_count": count,
        "subject": subject or "all",
    }


@router.get(
    "/{paper_id}",
    response_model=PaperResponse,
    summary="Get paper details by ID",
)
async def get_paper(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve specific paper record by its UUID."""
    query = select(Paper).where(Paper.id == paper_id).options(selectinload(Paper.pages))
    result = await db.execute(query)
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")

    return PaperResponse.model_validate(paper)


@router.get(
    "/{paper_id}/questions",
    response_model=List[QuestionResponse],
    summary="Get all questions extracted from a paper",
)
async def get_paper_questions(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve structured questions extracted from a given paper."""
    query = (
        select(Question)
        .where(Question.paper_id == paper_id)
        .order_by(Question.created_at.asc())
    )
    result = await db.execute(query)
    questions = result.scalars().all()

    return [QuestionResponse.model_validate(q) for q in questions]


@router.get(
    "/{paper_id}/pages",
    response_model=List[PageResponse],
    summary="Get all pages with structured text extracted from a paper",
)
async def get_paper_pages(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve all extracted pages with their NLP-structured text."""
    query = (
        select(Page)
        .where(Page.paper_id == paper_id)
        .order_by(Page.page_number.asc())
    )
    result = await db.execute(query)
    pages = result.scalars().all()
    return [PageResponse.model_validate(p) for p in pages]


@router.get(
    "/{paper_id}/text",
    summary="Get full document text assembled across all pages in structured format",
)
async def get_paper_full_text(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the unified, NLP-structured text across all pages of a paper."""
    query = (
        select(Page)
        .where(Page.paper_id == paper_id)
        .order_by(Page.page_number.asc())
    )
    result = await db.execute(query)
    pages = result.scalars().all()
    if not pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No extracted pages found for this paper.",
        )

    structured_pages = [p.raw_text or "" for p in pages if (p.raw_text or "").strip()]
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(structured_pages)

    return {
        "paper_id": str(paper_id),
        "total_pages": len(pages),
        "full_structured_text": full_text,
    }


@router.delete(
    "/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a question paper and associated files",
)
async def delete_paper(
    paper_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a paper record, cascading to pages and questions."""
    query = select(Paper).where(Paper.id == paper_id)
    result = await db.execute(query)
    paper = result.scalar_one_or_none()

    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found.")

    # Remove physical file if it exists
    if paper.file_path:
        p = Path(paper.file_path)
        if p.exists():
            p.unlink(missing_ok=True)

    await db.delete(paper)
    await db.commit()
    logger.info("paper_deleted", paper_id=str(paper_id))
    return None
