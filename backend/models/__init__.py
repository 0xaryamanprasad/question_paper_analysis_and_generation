"""
SQLAlchemy ORM Models Package.

Re-exports all models for convenient imports and ensures all models
are registered with the Base metadata for Alembic migrations.
"""

from backend.models.paper import Paper, Page
from backend.models.question import (
    Question,
    QuestionClassification,
    QuestionEmbedding,
    SimilarityPair,
)
from backend.models.analysis import HistoricalAnalysis, Syllabus
from backend.models.generated_paper import GeneratedPaper, GeneratedQuestion

__all__ = [
    "Paper",
    "Page",
    "Question",
    "QuestionClassification",
    "QuestionEmbedding",
    "SimilarityPair",
    "HistoricalAnalysis",
    "Syllabus",
    "GeneratedPaper",
    "GeneratedQuestion",
]
