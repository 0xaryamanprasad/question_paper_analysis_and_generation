"""
Pydantic Schemas for Topic, Unit, and Bloom's Taxonomy Classification.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BloomsLevelEnum(str, Enum):
    """Standard Bloom's Revised Taxonomy Cognitive Levels (L1 to L6)."""
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


class UnitDefinition(BaseModel):
    """Schema for a single syllabus unit."""
    unit_number: int = Field(..., description="Unit number (1-indexed)")
    unit_name: str = Field(..., description="Descriptive name of the unit")
    topics: List[str] = Field(default_factory=list, description="List of syllabus topics in this unit")
    keywords: List[str] = Field(default_factory=list, description="Domain keywords associated with this unit")


class SyllabusDefinition(BaseModel):
    """Schema for a complete subject syllabus."""
    subject: str = Field(..., description="Subject or course name")
    course_code: Optional[str] = Field(None, description="Course code (e.g. CS201)")
    units: List[UnitDefinition] = Field(..., description="List of units comprising the syllabus")


class TopicCandidate(BaseModel):
    """Candidate topic match with associated confidence and tier scores."""
    unit_number: int
    unit_name: str
    topic: str
    confidence: float
    matched_keywords: List[str] = Field(default_factory=list)
    method: str = Field("hybrid", description="Classification method: keyword, tfidf, semantic, or hybrid")


class TopicClassificationResult(BaseModel):
    """Result of classifying a question into syllabus unit and topic."""
    question_id: Optional[str] = None
    question_text: str
    subject: Optional[str] = None
    course_code: Optional[str] = None
    unit_number: Optional[int] = None
    unit_name: Optional[str] = None
    topic: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    matched_keywords: List[str] = Field(default_factory=list)
    classification_method: str = "hybrid"
    top_candidates: List[TopicCandidate] = Field(default_factory=list)


class BloomsClassificationResult(BaseModel):
    """Result of classifying a question into Bloom's taxonomy cognitive level."""
    question_id: Optional[str] = None
    question_text: str
    blooms_level: BloomsLevelEnum
    level_code: str = Field(..., description="Bloom's Level Code (L1, L2, L3, L4, L5, L6)")
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    action_verbs: List[str] = Field(default_factory=list, description="Action verbs found in the question")
    matched_keywords: List[str] = Field(default_factory=list, description="Bloom's indicator words detected")
    rationale: Optional[str] = None
