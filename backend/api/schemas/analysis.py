"""
Pydantic Schemas for Historical Analysis and Marks/Unit Weightage Analytics (Modules 8 & 9).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TopicTrend(BaseModel):
    """Trend statistics for an individual syllabus topic."""
    topic: str
    unit_number: int
    unit_name: str
    total_occurrences: int
    yearly_counts: Dict[str, int] = Field(default_factory=dict, description="Year -> occurrence count")
    trend_direction: str = Field(..., description="'increasing', 'decreasing', 'stable', or 'intermittent'")
    average_marks: float = 0.0
    importance_score: float = Field(0.0, description="Normalized score 0.0-1.0 based on frequency and marks")


class RecurrenceGroup(BaseModel):
    """Cluster of recurring or semantically equivalent questions across years."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    canonical_question: str
    question_ids: List[str]
    years_appeared: List[str]
    occurrence_count: int
    topic: Optional[str] = None
    unit_number: Optional[int] = None
    average_similarity: float = 1.0


class HistoricalAnalysisResult(BaseModel):
    """Comprehensive historical question analysis across uploaded past-year papers."""
    subject: str
    total_papers_analyzed: int
    years_covered: List[str]
    total_questions: int
    topic_frequency: Dict[str, Dict[str, int]] = Field(
        default_factory=dict, description="topic -> {year: count, 'total': count}"
    )
    topic_trends: List[TopicTrend] = Field(default_factory=list)
    top_repeated_topics: List[str] = Field(default_factory=list)
    blooms_distribution: Dict[str, int] = Field(default_factory=dict, description="Bloom level -> count")
    blooms_percentage: Dict[str, float] = Field(default_factory=dict, description="Bloom level -> percentage")
    recurring_questions: List[RecurrenceGroup] = Field(default_factory=list)
    overall_repetition_rate: float = Field(
        0.0, ge=0.0, le=100.0, description="Percentage of questions that recur across years"
    )


class UnitWeightage(BaseModel):
    """Marks and percentage weightage for a syllabus unit."""
    unit_number: int
    unit_name: str
    total_marks: float
    average_marks_per_paper: float
    weightage_percentage: float = Field(..., description="Percentage of total exam marks allocated to this unit")
    expected_percentage: Optional[float] = None
    deviation_percentage: Optional[float] = None
    topics_covered: List[str] = Field(default_factory=list)


class TopicWeightage(BaseModel):
    """Marks weightage for a specific topic."""
    topic: str
    unit_number: int
    total_marks: float
    average_marks_per_paper: float
    weightage_percentage: float


class WeightageAnalysisResult(BaseModel):
    """Complete examination marks and unit weightage analytics report."""
    subject: str
    total_papers: int
    total_marks_all_papers: float
    average_paper_total_marks: float
    unit_weightages: List[UnitWeightage] = Field(default_factory=list)
    topic_weightages: List[TopicWeightage] = Field(default_factory=list)
    marks_by_question_type: Dict[str, float] = Field(
        default_factory=dict, description="'short' (<=2m), 'medium' (3-7m), 'long' (>=8m) -> total marks"
    )
    most_weighty_units: List[int] = Field(default_factory=list)
    underrepresented_units: List[int] = Field(default_factory=list)
