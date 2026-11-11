"""
Pydantic Schemas for Question Generation and Paper Generation (Modules 10 & 11).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SubPartItem(BaseModel):
    """Sub-part of a composite exam question."""
    label: str = Field(..., description="Sub-part label (e.g. 'a', 'b', 'i', 'ii')")
    text: str = Field(..., description="Prompt text of the sub-part")
    marks: float = Field(..., ge=1.0, description="Marks allocated to this sub-part")


class GeneratedQuestionItem(BaseModel):
    """An individual generated question candidate."""
    id: Optional[str] = None
    question_number: Optional[str] = None
    question_text: str = Field(..., description="Complete question prompt text")
    sub_parts: List[SubPartItem] = Field(default_factory=list)
    marks: float = Field(..., ge=1.0, description="Total marks for this question")
    unit_number: int = Field(..., ge=1, description="Syllabus unit number")
    unit_name: Optional[str] = None
    topic: str = Field(..., description="Syllabus topic covered")
    blooms_level: str = Field(..., description="Bloom's taxonomy cognitive level (L1 to L6)")
    difficulty: str = Field("medium", description="'easy', 'medium', or 'hard'")
    question_type: str = Field("medium", description="'short', 'medium', 'long', 'numerical', 'mcq'")
    marking_scheme: List[str] = Field(default_factory=list, description="Step-by-step marking rubrics")
    sample_answer: Optional[str] = Field(None, description="Model solution or brief answer outline")
    or_choice_alternative: Optional[str] = Field(None, description="Optional ID of alternative OR choice question")


class SectionConstraint(BaseModel):
    """Constraints defining a specific section of an examination paper."""
    section_name: str = Field(..., description="Section label (e.g. 'Section A', 'Part B')")
    description: Optional[str] = None
    question_count: int = Field(..., ge=1, description="Number of questions in this section")
    questions_to_attempt: Optional[int] = Field(None, description="Number of questions student must answer")
    marks_per_question: float = Field(..., ge=1.0, description="Marks for each question")
    allowed_question_types: List[str] = Field(
        default=["short", "medium"], description="Allowed types in this section"
    )
    unit_coverage: Optional[List[int]] = Field(None, description="Specific units to cover, or None for all")
    has_internal_choice: bool = Field(False, description="Whether questions have internal OR choice")


class PaperBlueprintConfig(BaseModel):
    """Complete specification and configuration constraints for generating an exam paper."""
    subject: str = Field(..., description="Course / subject title")
    course_code: Optional[str] = Field(None, description="Course code (e.g. CS201)")
    total_marks: float = Field(..., ge=1.0, description="Total marks for the paper (e.g. 100.0, 50.0, or 25.0)")
    duration_hours: float = Field(3.0, ge=0.5, le=5.0, description="Examination duration in hours")
    institution_name: str = Field("National Institute of Technology", description="Institution header")
    examination_name: str = Field("B.Tech Degree End Semester Examinations", description="Examination title")
    sections: List[SectionConstraint] = Field(..., min_length=1, description="List of paper sections")
    unit_distribution_targets: Optional[Dict[int, float]] = Field(
        None, description="Target percentage per unit (e.g. {1: 20.0, 2: 20.0, ...})"
    )
    blooms_distribution_targets: Optional[Dict[str, float]] = Field(
        None, description="Target percentage per Bloom's level"
    )
    incorporate_historical_trends: bool = Field(True, description="Prioritize high-recurrence historical topics")
    avoid_recent_duplicates: bool = Field(True, description="Avoid questions identical to recent prior papers")


class PaperSectionResult(BaseModel):
    """An assembled section in a generated question paper."""
    section_name: str
    description: Optional[str] = None
    questions_to_attempt: Optional[int] = None
    total_section_marks: float
    questions: List[GeneratedQuestionItem]


class GeneratedPaperResult(BaseModel):
    """A complete generated examination question paper."""
    id: str
    subject: str
    course_code: Optional[str] = None
    institution_name: str
    examination_name: str
    total_marks: float
    duration_hours: float
    sections: List[PaperSectionResult]
    total_questions: int
    unit_coverage_summary: Dict[int, float] = Field(default_factory=dict, description="Unit -> marks allocated")
    blooms_coverage_summary: Dict[str, float] = Field(default_factory=dict, description="Bloom's level -> marks")
    created_at: str
