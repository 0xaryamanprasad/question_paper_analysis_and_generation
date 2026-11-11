"""
Pydantic Schemas for Semantic Similarity and Duplicate Detection.

Defines schemas for pairwise similarity matches, duplicate clusters,
and semantic query results.
"""

from typing import Literal
from pydantic import BaseModel, Field


class SimilarityMatch(BaseModel):
    """Represents a pairwise semantic similarity comparison between two questions."""
    question_a: str = Field(..., description="First question text")
    question_b: str = Field(..., description="Second question text")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)")
    relation_type: Literal["duplicate", "near_duplicate", "related", "distinct"] = Field(
        ..., description="Classification of the relationship based on threshold"
    )
    question_a_id: str | None = Field(default=None, description="Optional ID of first question")
    question_b_id: str | None = Field(default=None, description="Optional ID of second question")


class DuplicateCluster(BaseModel):
    """A cluster of recurring or duplicate questions representing the same concept."""
    cluster_id: str = Field(..., description="Unique cluster identifier")
    canonical_question: str = Field(..., description="Representative question of the cluster")
    member_questions: list[str] = Field(..., description="List of questions in this cluster")
    member_ids: list[str] = Field(default_factory=list, description="IDs of questions in this cluster")
    size: int = Field(..., description="Number of questions in the cluster")
    years_spanned: list[str] = Field(default_factory=list, description="Exam years where this question recurred")


class SimilarityReport(BaseModel):
    """Comprehensive duplicate and similarity report across questions."""
    total_questions: int
    total_pairs_evaluated: int
    duplicates_count: int
    near_duplicates_count: int
    matches: list[SimilarityMatch]
    clusters: list[DuplicateCluster]
