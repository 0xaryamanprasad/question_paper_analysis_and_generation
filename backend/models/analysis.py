"""
SQLAlchemy Models for Analysis Results.

Stores syllabus definitions and computed historical analysis data.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from backend.database import Base


class Syllabus(Base):
    """
    Represents a syllabus definition for a subject.

    The units field is a JSONB array of unit objects, each containing:
    - unit_number: int
    - unit_name: str
    - topics: list of topic strings
    - keywords: list of keyword strings

    Example:
        {
            "units": [
                {
                    "unit_number": 1,
                    "unit_name": "Introduction to Data Structures",
                    "topics": ["Arrays", "Linked Lists", "Stacks", "Queues"],
                    "keywords": ["array", "linked list", "stack", "queue", "LIFO", "FIFO"]
                }
            ]
        }

    Attributes:
        id: Unique identifier.
        subject: Subject name.
        course_code: Course code (e.g., "CS201").
        units: JSONB containing the syllabus unit structure.
        created_at: When the syllabus was defined.
    """

    __tablename__ = "syllabi"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(String(255), nullable=False, index=True)
    course_code = Column(String(50), nullable=True)
    units = Column(JSONB, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<Syllabus(id={self.id}, subject='{self.subject}', code='{self.course_code}')>"


class HistoricalAnalysis(Base):
    """
    Stores precomputed historical analysis results for a subject.

    This table acts as a cache for expensive analysis computations.
    It is recomputed when new papers are uploaded for the subject.

    Attributes:
        id: Unique identifier.
        subject: Subject name this analysis covers.
        topic_frequency: Topic → year → count mapping.
        marks_distribution: Unit → average marks mapping.
        blooms_distribution: Bloom's level → percentage mapping.
        recurrence_data: Questions that recur across multiple years.
        computed_at: When this analysis was last computed.
    """

    __tablename__ = "historical_analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(String(255), nullable=False, index=True)
    topic_frequency = Column(JSONB, nullable=True)
    marks_distribution = Column(JSONB, nullable=True)
    blooms_distribution = Column(JSONB, nullable=True)
    recurrence_data = Column(JSONB, nullable=True)
    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<HistoricalAnalysis(id={self.id}, subject='{self.subject}')>"
