"""
Pydantic Schemas for Question Paper Validation Engine (Module 12).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MarksCheckResult(BaseModel):
    """Detailed breakdown of marks verification."""
    target_total_marks: float
    calculated_total_marks: float
    is_match: bool
    section_marks_summary: Dict[str, float] = Field(default_factory=dict)
    subpart_discrepancies: List[str] = Field(default_factory=list)


class DuplicateCheckResult(BaseModel):
    """Result of independent semantic duplicate detection within generated paper."""
    duplicate_pairs_found: int = 0
    highest_similarity: float = 0.0
    violating_pairs: List[Dict[str, Any]] = Field(default_factory=list)


class UnitCoverageCheckResult(BaseModel):
    """Result of verifying syllabus unit representation."""
    total_syllabus_units: int
    covered_units: List[int]
    missing_units: List[int] = Field(default_factory=list)
    max_deviation_percentage: float = 0.0
    unit_marks_breakdown: Dict[int, float] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """Comprehensive independent validation report for a generated examination paper."""
    paper_id: str
    is_valid: bool
    overall_quality_score: float = Field(..., ge=0.0, le=100.0, description="Overall compliance score out of 100")
    passed_checks: List[str] = Field(default_factory=list)
    failed_checks: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    marks_check: MarksCheckResult
    duplicate_check: DuplicateCheckResult
    unit_coverage_check: UnitCoverageCheckResult
    blooms_distribution: Dict[str, float] = Field(default_factory=dict)
