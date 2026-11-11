"""
FastAPI Routes for Topic, Bloom's Taxonomy, Similarity, and Historical Analytics.
"""

from typing import Any, Dict, List, Optional
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.schemas.analysis import (
    HistoricalAnalysisResult,
    WeightageAnalysisResult,
)
from backend.api.schemas.classification import (
    BloomsClassificationResult,
    TopicClassificationResult,
)
from backend.api.schemas.similarity import SimilarityReport
from backend.database import get_db
from backend.models.paper import Paper
from backend.models.question import Question, QuestionClassification
from backend.services.blooms_classifier import BloomsClassifier
from backend.services.historical_analyzer import HistoricalAnalyzer
from backend.services.similarity_engine import SimilarityEngine
from backend.services.topic_classifier import TopicClassifier
from backend.services.weightage_analyzer import WeightageAnalyzer
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Service singletons
_topic_classifier: Optional[TopicClassifier] = None
_blooms_classifier: Optional[BloomsClassifier] = None
_similarity_engine: Optional[SimilarityEngine] = None
_historical_analyzer: Optional[HistoricalAnalyzer] = None
_weightage_analyzer: Optional[WeightageAnalyzer] = None


def get_similarity_engine() -> SimilarityEngine:
    global _similarity_engine
    if _similarity_engine is None:
        _similarity_engine = SimilarityEngine()
    return _similarity_engine


_topic_classifiers: Dict[str, TopicClassifier] = {}


def get_topic_classifier(subject: Optional[str] = None) -> TopicClassifier:
    from backend.services.topic_classifier import resolve_syllabus_path
    key = "default"
    if subject:
        sub_lower = subject.lower()
        if any(kw in sub_lower for kw in ["os", "operating system"]):
            key = "os"
        elif any(kw in sub_lower for kw in ["coa", "computer organization", "architecture", "microprocessor", "pipelining", "hardware"]):
            key = "coa"
        elif any(kw in sub_lower for kw in ["math", "calculus", "differential", "algebra", "integral"]):
            key = "mathematics"
        elif any(kw in sub_lower for kw in ["dbms", "database", "sql"]):
            key = "dbms"
        elif any(kw in sub_lower for kw in ["dsa", "data structure", "algorithm"]):
            key = "dsa"
        else:
            key = sub_lower
    if key not in _topic_classifiers:
        s_path = resolve_syllabus_path(subject)
        _topic_classifiers[key] = TopicClassifier(
            syllabus=s_path,
            similarity_engine=get_similarity_engine(),
        )
    return _topic_classifiers[key]


def get_blooms_classifier() -> BloomsClassifier:
    global _blooms_classifier
    if _blooms_classifier is None:
        _blooms_classifier = BloomsClassifier()
    return _blooms_classifier


def get_historical_analyzer() -> HistoricalAnalyzer:
    global _historical_analyzer
    if _historical_analyzer is None:
        _historical_analyzer = HistoricalAnalyzer(similarity_engine=get_similarity_engine())
    return _historical_analyzer


def get_weightage_analyzer() -> WeightageAnalyzer:
    global _weightage_analyzer
    if _weightage_analyzer is None:
        _weightage_analyzer = WeightageAnalyzer()
    return _weightage_analyzer


class ClassifyQuestionRequest(BaseModel):
    """Request payload for classifying an arbitrary question."""
    question_text: str = Field(..., description="Prompt text of the question to classify")
    question_id: Optional[str] = None
    subject: Optional[str] = Field(None, description="Optional subject or 'auto' for automatic domain detection")


class QuestionClassificationBundle(BaseModel):
    """Bundled result containing both Topic and Bloom's classifications."""
    topic_classification: TopicClassificationResult
    blooms_classification: BloomsClassificationResult


@router.post(
    "/classify-question",
    response_model=QuestionClassificationBundle,
    summary="Classify a question into syllabus topic and Bloom's cognitive level",
)
async def classify_question(
    payload: ClassifyQuestionRequest,
):
    """Run dual topic and Bloom's classification on question text with domain detection."""
    target_subject = payload.subject
    is_auto = not target_subject or target_subject.lower() in ["auto", "all", "detect"]

    # 1. First priority: Try Gemini LLM for deep academic topic and domain classification
    from backend.services.topic_classifier import (
        classify_domain_and_topic_with_gemini,
        detect_subject_domain,
    )
    from backend.api.schemas.classification import TopicCandidate, TopicClassificationResult

    blooms_clf = get_blooms_classifier()
    blooms_res = blooms_clf.classify_question(payload.question_text, question_id=payload.question_id)

    if is_auto:
        gemini_result = classify_domain_and_topic_with_gemini(payload.question_text)
        if gemini_result and gemini_result.get("topic"):
            cand = TopicCandidate(
                unit_number=int(gemini_result.get("unit_number", 1)),
                unit_name=str(gemini_result.get("unit_name", "Unit 1")),
                topic=str(gemini_result.get("topic", "General")),
                confidence=float(gemini_result.get("confidence", 0.96)),
                matched_keywords=gemini_result.get("matched_keywords", []),
                method="gemini_llm",
            )
            topic_res = TopicClassificationResult(
                question_id=payload.question_id,
                question_text=payload.question_text,
                subject=gemini_result.get("subject", "Computer Organization and Architecture"),
                course_code=gemini_result.get("course_code", "CS204"),
                unit_number=int(gemini_result.get("unit_number", 1)),
                unit_name=str(gemini_result.get("unit_name", "Unit 1")),
                topic=str(gemini_result.get("topic", "General")),
                confidence=float(gemini_result.get("confidence", 0.96)),
                matched_keywords=gemini_result.get("matched_keywords", []),
                classification_method="gemini_llm",
                top_candidates=[cand],
            )
            return QuestionClassificationBundle(
                topic_classification=topic_res,
                blooms_classification=blooms_res,
            )

    # 2. Local Multi-tier Heuristic + Dense Semantic Vector Classifier Fallback
    if is_auto:
        detected = detect_subject_domain(payload.question_text)
        if detected:
            target_subject = detected

    topic_clf = get_topic_classifier(target_subject)
    topic_res = topic_clf.classify_question(payload.question_text, question_id=payload.question_id)
    if not topic_res.subject:
        topic_res.subject = topic_clf.syllabus.subject
    if not topic_res.course_code:
        topic_res.course_code = topic_clf.syllabus.course_code

    return QuestionClassificationBundle(
        topic_classification=topic_res,
        blooms_classification=blooms_res,
    )


@router.get(
    "/historical",
    response_model=HistoricalAnalysisResult,
    summary="Get multi-year historical analysis and trends for a subject",
)
async def get_historical_analysis(
    subject: str = Query(..., description="Subject or course name"),
    db: AsyncSession = Depends(get_db),
):
    """
    Query all historical questions for a subject, classify unclassified questions,
    and compute topic recurrence, longitudinal trends, and Bloom's distribution.
    """
    # 1. Fetch all papers for this subject
    query = (
        select(Question, Paper.year, Paper.subject)
        .join(Paper, Question.paper_id == Paper.id)
        .where(Paper.subject.ilike(f"%{subject}%"))
        .options(selectinload(Question.classification))
    )
    result = await db.execute(query)
    rows = result.all()

    if not rows:
        # Return empty structured report
        return get_historical_analyzer().analyze([], subject=subject)

    topic_clf = get_topic_classifier(subject=subject)
    blooms_clf = get_blooms_classifier()

    questions_for_analysis: List[Dict[str, Any]] = []

    for q, p_year, p_subject in rows:
        sub_items = q.sub_parts if isinstance(q.sub_parts, list) and len(q.sub_parts) > 0 else []
        if sub_items:
            marks_per_sp = float(q.marks) / len(sub_items) if q.marks else 2.0
            for sp in sub_items:
                sp_text = sp.get("text", "").strip() if isinstance(sp, dict) else str(sp).strip()
                if not sp_text or len(sp_text) < 4:
                    continue
                if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", sp_text):
                    continue
                sp_marks = float(sp.get("marks") or marks_per_sp) if isinstance(sp, dict) else marks_per_sp
                sp_t_res = topic_clf.classify_question(sp_text)
                sp_b_res = blooms_clf.classify_question(sp_text)

                questions_for_analysis.append(
                    {
                        "id": f"{q.id}_{sp.get('label', '')}" if isinstance(sp, dict) else str(q.id),
                        "paper_id": str(q.paper_id),
                        "year": str(p_year) if p_year else "Unknown",
                        "text": sp_text,
                        "unit_number": sp_t_res.unit_number or 1,
                        "unit_name": f"Unit {sp_t_res.unit_number or 1}",
                        "topic": sp_t_res.topic or "General",
                        "marks": sp_marks,
                        "blooms_level": sp_b_res.blooms_level.value,
                    }
                )
        else:
            q_text = (q.cleaned_text or q.raw_text or "").strip()
            if not q_text or len(q_text) < 4:
                continue
            if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", q_text):
                continue

            # Check if already classified and topic belongs to subject domain
            is_stale_dsa_on_math = (
                "math" in (p_subject or subject).lower()
                and q.classification
                and any(kw in (q.classification.topic or "").lower() for kw in ["sort", "tree", "graph", "adt", "stack", "queue", "hash"])
            )
            if q.classification and q.classification.topic and not is_stale_dsa_on_math:
                unit_no = 1
                if q.classification.unit and q.classification.unit.isdigit():
                    unit_no = int(q.classification.unit)
                topic = q.classification.topic
                blooms = q.classification.blooms_level or "Understand"
            else:
                # Classify dynamically with subject-specific classifier
                t_res = topic_clf.classify_question(q_text, question_id=str(q.id))
                b_res = blooms_clf.classify_question(q_text, question_id=str(q.id))

                unit_no = t_res.unit_number or 1
                topic = t_res.topic or "General"
                blooms = b_res.blooms_level.value

                # Save classification back to DB for future caching
                if q.classification:
                    q.classification.unit = str(unit_no)
                    q.classification.topic = topic
                    q.classification.topic_confidence = t_res.confidence
                    q.classification.blooms_level = blooms
                    q.classification.blooms_confidence = b_res.confidence
                    q.classification.keywords = t_res.matched_keywords
                else:
                    clf_model = QuestionClassification(
                        id=uuid.uuid4(),
                        question_id=q.id,
                        unit=str(unit_no),
                        topic=topic,
                        topic_confidence=t_res.confidence,
                        blooms_level=blooms,
                        blooms_confidence=b_res.confidence,
                        keywords=t_res.matched_keywords,
                    )
                    db.add(clf_model)

            questions_for_analysis.append(
                {
                    "id": str(q.id),
                    "paper_id": str(q.paper_id),
                    "year": str(p_year) if p_year else "Unknown",
                    "text": q_text,
                    "unit_number": unit_no,
                    "unit_name": f"Unit {unit_no}",
                    "topic": topic,
                    "marks": float(q.marks or 0.0),
                    "blooms_level": blooms,
                }
            )

    await db.commit()

    analyzer = get_historical_analyzer()
    return analyzer.analyze(questions_for_analysis, subject=subject)


@router.get(
    "/weightage",
    response_model=WeightageAnalysisResult,
    summary="Get marks and unit weightage analysis for a subject",
)
async def get_weightage_analysis(
    subject: str = Query(..., description="Subject or course name"),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute unit-wise and topic-wise marks allocations, percentages,
    and blueprint deviations across past-year papers.
    """
    query = (
        select(Question, Paper.year, Paper.subject)
        .join(Paper, Question.paper_id == Paper.id)
        .where(Paper.subject.ilike(f"%{subject}%"))
        .options(selectinload(Question.classification))
    )
    result = await db.execute(query)
    rows = result.all()

    if not rows:
        return get_weightage_analyzer().analyze([], subject=subject)

    topic_clf = get_topic_classifier(subject=subject)

    questions_data: List[Dict[str, Any]] = []
    for q, p_year, p_subject in rows:
        sub_items = q.sub_parts if isinstance(q.sub_parts, list) and len(q.sub_parts) > 0 else []
        if sub_items:
            marks_per_sp = float(q.marks) / len(sub_items) if q.marks else 2.0
            for sp in sub_items:
                sp_text = sp.get("text", "").strip() if isinstance(sp, dict) else str(sp).strip()
                if not sp_text or len(sp_text) < 4:
                    continue
                if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", sp_text):
                    continue
                sp_marks = float(sp.get("marks") or marks_per_sp) if isinstance(sp, dict) else marks_per_sp
                sp_t_res = topic_clf.classify_question(sp_text)

                questions_data.append(
                    {
                        "id": f"{q.id}_{sp.get('label', '')}" if isinstance(sp, dict) else str(q.id),
                        "paper_id": str(q.paper_id),
                        "year": str(p_year) if p_year else "Unknown",
                        "unit_number": sp_t_res.unit_number or 1,
                        "unit_name": f"Unit {sp_t_res.unit_number or 1}",
                        "topic": sp_t_res.topic or "General",
                        "marks": sp_marks,
                    }
                )
        else:
            q_text = (q.cleaned_text or q.raw_text or "").strip()
            if not q_text or len(q_text) < 4:
                continue
            if re.search(r"\b\d{2,4}[-/]\d{2}[-/]\d{2,4}\b", q_text):
                continue

            is_stale_dsa_on_math = (
                "math" in (p_subject or subject).lower()
                and q.classification
                and any(kw in (q.classification.topic or "").lower() for kw in ["sort", "tree", "graph", "adt", "stack", "queue", "hash"])
            )
            if q.classification and q.classification.unit and not is_stale_dsa_on_math:
                unit_no = int(q.classification.unit) if q.classification.unit.isdigit() else 1
                topic = q.classification.topic or "General"
            else:
                t_res = topic_clf.classify_question(q_text)
                unit_no = t_res.unit_number or 1
                topic = t_res.topic or "General"

            questions_data.append(
                {
                    "id": str(q.id),
                    "paper_id": str(q.paper_id),
                    "year": str(p_year) if p_year else "Unknown",
                    "unit_number": unit_no,
                    "unit_name": f"Unit {unit_no}",
                    "topic": topic,
                    "marks": float(q.marks or 0.0),
                }
            )

    analyzer = get_weightage_analyzer()
    return analyzer.analyze(
        questions=questions_data,
        syllabus=topic_clf.syllabus,
        subject=subject,
    )


@router.post(
    "/duplicates",
    response_model=SimilarityReport,
    summary="Detect duplicates and recurring clusters in questions",
)
async def detect_duplicates(
    questions: List[Dict[str, Any]],
):
    """Run pairwise similarity and connected-component clustering on a set of questions."""
    sim_engine = get_similarity_engine()
    return sim_engine.find_duplicates(questions)


@router.delete(
    "/reset",
    summary="Reset corpus intelligence and clear stored papers",
)
@router.post(
    "/reset",
    summary="Reset corpus intelligence and clear stored papers",
)
async def reset_intelligence(
    subject: Optional[str] = None,
    all_data: bool = False,
    db: AsyncSession = Depends(get_db),
):
    """
    Reset intelligence context and remove papers/questions.
    Allows the platform to learn fresh for another subject.
    """
    from backend.api.routes.papers import reset_corpus
    return await reset_corpus(subject=subject, all_data=all_data, db=db)

