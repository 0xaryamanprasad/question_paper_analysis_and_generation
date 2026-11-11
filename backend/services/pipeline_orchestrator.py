"""
Module 6: Pipeline Orchestrator and Database Persistence Service.

Connects and executes Modules 1 through 5:
1. Ingests PDF or images -> runs PaddleOCR with layout detection and LaTeX formula extraction.
2. Performs OCR correction using Gemini multimodal model (or robust offline fallback), flagging uncertain tokens.
3. Structures content into hierarchical Pydantic schema (Paper -> Sections -> Questions -> Subquestions).
4. Classifies each question into predefined DSA categories based on algorithmic concepts.
5. Runs rigorous validation (question numbering gaps, missing subquestions, marks math, low confidence OCR, suspicious LaTeX).
6. Persists each question into PostgreSQL (or SQLite local fallback) with all required fields.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

import backend.database as db_module
from backend.models.paper import Paper, Page
from backend.models.question import Question, QuestionClassification
from backend.services.math_ocr_engine import MathOCREngine, MathOCRPaperResult
from backend.services.ocr_corrector import GeminiOCRCorrector, OCRCorrectionResult
from backend.services.question_structurer import QuestionStructurer, StructuredExamPaper, StructuredQuestion
from backend.services.dsa_classifier import DSATopicClassifier, DSATopicClassificationResult
from backend.services.pipeline_validator import PipelineValidator, ValidationSummary
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineExecutionResult:
    """Encapsulates the end-to-end output of the advanced pipeline."""
    def __init__(
        self,
        paper_id: uuid.UUID,
        structured_paper: StructuredExamPaper,
        validation_summary: ValidationSummary,
        ocr_result: MathOCRPaperResult,
        saved_questions: List[Dict[str, Any]],
    ):
        self.paper_id = paper_id
        self.structured_paper = structured_paper
        self.validation_summary = validation_summary
        self.ocr_result = ocr_result
        self.saved_questions = saved_questions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": str(self.paper_id),
            "subject": self.structured_paper.subject,
            "examination_year": self.structured_paper.examination_year,
            "total_marks": self.structured_paper.total_marks,
            "total_questions": len(self.saved_questions),
            "is_valid": self.validation_summary.is_valid,
            "validation_issues_count": len(self.validation_summary.issues),
            "validation_summary": self.validation_summary.model_dump(),
            "questions": self.saved_questions,
        }


class PipelineOrchestrator:
    """
    Coordinates Modules 1-5 and persists final question records into the database.
    """

    def __init__(
        self,
        math_ocr: Optional[MathOCREngine] = None,
        ocr_corrector: Optional[GeminiOCRCorrector] = None,
        structurer: Optional[QuestionStructurer] = None,
        dsa_classifier: Optional[DSATopicClassifier] = None,
        validator: Optional[PipelineValidator] = None,
    ):
        self.math_ocr = math_ocr or MathOCREngine()
        self.ocr_corrector = ocr_corrector or GeminiOCRCorrector()
        self.structurer = structurer or QuestionStructurer()
        self.dsa_classifier = dsa_classifier or DSATopicClassifier()
        self.validator = validator or PipelineValidator()

    async def run_pipeline(
        self,
        file_path: str | Path,
        subject: str = "Data Structures and Algorithms",
        examination_year: str = "2024",
        session: Optional[AsyncSession] = None,
    ) -> PipelineExecutionResult:
        """
        Executes complete pipeline on a PDF or image file and stores questions in the database.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")

        logger.info("pipeline_started", file=str(file_path))

        # --- MODULE 1: OCR & Math Detection ---
        if file_path.suffix.lower() == ".pdf":
            ocr_paper_res = self.math_ocr.extract_from_pdf(file_path)
        else:
            page_res = self.math_ocr.extract_from_image(file_path, page_number=1)
            ocr_paper_res = MathOCRPaperResult(
                total_pages=1,
                pages=[page_res],
                all_latex_expressions=page_res.latex_expressions,
                average_confidence=page_res.average_confidence,
            )

        logger.info(
            "module_1_ocr_complete",
            total_pages=ocr_paper_res.total_pages,
            avg_conf=ocr_paper_res.average_confidence,
            latex_count=len(ocr_paper_res.all_latex_expressions),
        )

        # Build page text dictionary and page confidences
        page_dict: Dict[int, str] = {}
        page_conf_map: Dict[int, float] = {}
        for p in ocr_paper_res.pages:
            page_dict[p.page_number] = p.text_with_latex or p.raw_text
            page_conf_map[p.page_number] = p.average_confidence

        full_raw_text = "\n\n".join(page_dict.values())

        # --- MODULE 2: OCR Correction ---
        corrected_pages: Dict[int, OCRCorrectionResult] = {}
        for p in ocr_paper_res.pages:
            corr = self.ocr_corrector.correct_ocr_text(
                raw_ocr_text=p.text_with_latex or p.raw_text,
                latex_expressions=p.latex_expressions,
                image_path=p.image_path,
            )
            corrected_pages[p.page_number] = corr

        full_corrected_text = "\n\n".join(c.corrected_text for c in corrected_pages.values())
        logger.info("module_2_ocr_correction_complete")

        # --- MODULE 3: Question Structuring ---
        structured_paper = self.structurer.structure_paper(
            full_text=full_corrected_text,
            page_texts={p: c.corrected_text for p, c in corrected_pages.items()},
            default_subject=subject,
            default_year=examination_year,
        )
        logger.info("module_3_structuring_complete", questions_count=len(structured_paper.questions))

        # --- MODULE 4: DSA Topic Classification ---
        # Classify each question individually including subquestions context
        classifications: Dict[str, DSATopicClassificationResult] = {}
        for q in structured_paper.questions:
            full_q_text = q.question_text
            if q.subquestions:
                sub_texts = [f"({s.subquestion_id}) {s.text}" for s in q.subquestions]
                full_q_text += "\n" + "\n".join(sub_texts)

            cls_res = self.dsa_classifier.classify_question(
                question_text=full_q_text,
                latex_expressions=q.latex_expressions,
            )
            classifications[q.question_number] = cls_res

        logger.info("module_4_dsa_classification_complete")

        # --- MODULE 5: Validation ---
        validation_summary = self.validator.validate_paper(
            paper=structured_paper,
            page_confidences=page_conf_map,
        )
        logger.info(
            "module_5_validation_complete",
            is_valid=validation_summary.is_valid,
            issues_count=len(validation_summary.issues),
        )

        # --- MODULE 6: Database Persistence ---
        saved_records = []
        paper_id = uuid.uuid4()

        if session is not None:
            paper_id, saved_records = await self._persist_to_db(
                session=session,
                paper_id=paper_id,
                file_path=file_path,
                structured_paper=structured_paper,
                ocr_paper_res=ocr_paper_res,
                classifications=classifications,
                corrected_pages=corrected_pages,
                validation_summary=validation_summary,
            )
        else:
            async with db_module.async_session_factory() as local_session:
                paper_id, saved_records = await self._persist_to_db(
                    session=local_session,
                    paper_id=paper_id,
                    file_path=file_path,
                    structured_paper=structured_paper,
                    ocr_paper_res=ocr_paper_res,
                    classifications=classifications,
                    corrected_pages=corrected_pages,
                    validation_summary=validation_summary,
                )
                await local_session.commit()

        logger.info("module_6_persistence_complete", paper_id=str(paper_id))

        return PipelineExecutionResult(
            paper_id=paper_id,
            structured_paper=structured_paper,
            validation_summary=validation_summary,
            ocr_result=ocr_paper_res,
            saved_questions=saved_records,
        )

    async def _persist_to_db(
        self,
        session: AsyncSession,
        paper_id: uuid.UUID,
        file_path: Path,
        structured_paper: StructuredExamPaper,
        ocr_paper_res: MathOCRPaperResult,
        classifications: Dict[str, DSATopicClassificationResult],
        corrected_pages: Dict[int, OCRCorrectionResult],
        validation_summary: ValidationSummary,
    ) -> tuple[uuid.UUID, List[Dict[str, Any]]]:
        """Inserts Paper, Pages, Questions, and Classifications into database."""
        # 1. Create Paper record
        paper_record = Paper(
            id=paper_id,
            subject=structured_paper.subject,
            year=structured_paper.examination_year,
            original_filename=file_path.name,
            file_path=str(file_path),
            total_pages=ocr_paper_res.total_pages,
            extraction_status="completed",
        )
        session.add(paper_record)
        await session.flush()

        # 2. Create Page records
        page_id_map: Dict[int, uuid.UUID] = {}
        for p in ocr_paper_res.pages:
            p_id = uuid.uuid4()
            page_record = Page(
                id=p_id,
                paper_id=paper_id,
                page_number=p.page_number,
                page_type="scanned" if p.image_path else "digital",
                raw_text=p.raw_text,
                image_path=p.image_path,
                layout_data={
                    "structured_text": p.text_with_latex,
                    "ocr_confidence": p.average_confidence,
                },
            )
            session.add(page_record)
            page_id_map[p.page_number] = p_id

        await session.flush()

        # Build issue map for question numbers
        q_issue_map: Dict[str, List[str]] = {}
        for issue in validation_summary.issues:
            if issue.question_number:
                q_issue_map.setdefault(issue.question_number, []).append(
                    f"[{issue.level.upper()}] {issue.category}: {issue.message}"
                )

        # 3. Create Question records
        saved_question_dicts: List[Dict[str, Any]] = []

        for q in structured_paper.questions:
            cls_info = classifications.get(
                q.question_number,
                DSATopicClassificationResult(
                    primary_topic="Unclassified",
                    confidence=0.0,
                    reasoning="Unclassified",
                ),
            )
            corr_page = corrected_pages.get(q.page_number)
            page_ocr_conf = page_conf_map_lookup = (
                ocr_paper_res.pages[q.page_number - 1].average_confidence
                if 1 <= q.page_number <= len(ocr_paper_res.pages)
                else 0.8
            )

            is_flagged = (corr_page.is_flagged_for_verification if corr_page else False) or (
                q.question_number in q_issue_map
            )
            flags = list(q_issue_map.get(q.question_number, []))
            if corr_page and corr_page.flag_reasons:
                flags.extend(corr_page.flag_reasons)

            q_id = uuid.uuid4()
            question_record = Question(
                id=q_id,
                paper_id=paper_id,
                page_id=page_id_map.get(q.page_number),
                question_number=q.question_number,
                subquestion_number=q.subquestion_number,
                raw_text=q.raw_ocr_text or q.question_text,
                cleaned_text=q.corrected_text or q.question_text,
                latex_expressions=q.latex_expressions,
                marks=q.marks,
                section=q.section,
                dsa_topic=cls_info.primary_topic.value,
                dsa_subtopic=cls_info.subtopic,
                dsa_secondary_topic=(
                    cls_info.secondary_topic.value if cls_info.secondary_topic else None
                ),
                dsa_confidence=cls_info.confidence,
                ocr_confidence=page_ocr_conf,
                is_flagged_for_verification=is_flagged,
                verification_flags=flags,
                original_page_number=q.page_number,
                sub_parts=[s.model_dump() for s in q.subquestions] if q.subquestions else None,
            )
            session.add(question_record)

            # Also create standard QuestionClassification
            q_class_record = QuestionClassification(
                id=uuid.uuid4(),
                question_id=q_id,
                unit=q.section,
                topic=cls_info.primary_topic.value,
                topic_confidence=cls_info.confidence,
                blooms_level="Apply",
                blooms_confidence=0.85,
                keywords={"subtopic": cls_info.subtopic, "reasoning": cls_info.reasoning},
            )
            session.add(q_class_record)

            saved_question_dicts.append({
                "id": str(q_id),
                "question_number": q.question_number,
                "section": q.section,
                "marks": q.marks,
                "dsa_topic": cls_info.primary_topic.value,
                "subtopic": cls_info.subtopic,
                "dsa_secondary_topic": (
                    cls_info.secondary_topic.value if cls_info.secondary_topic else None
                ),
                "dsa_confidence": cls_info.confidence,
                "ocr_confidence": page_ocr_conf,
                "is_flagged_for_verification": is_flagged,
                "latex_expressions": q.latex_expressions,
                "subquestions_count": len(q.subquestions),
                "original_page_number": q.page_number,
                "cleaned_text": q.corrected_text or q.question_text,
            })

        await session.flush()
        return paper_id, saved_question_dicts
