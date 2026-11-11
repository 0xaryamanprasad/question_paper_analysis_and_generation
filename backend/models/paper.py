"""
SQLAlchemy Models for Papers and Pages.

Represents uploaded question papers and their individual pages.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from backend.database import Base


class Paper(Base):
    """
    Represents an uploaded question paper PDF.

    Attributes:
        id: Unique identifier.
        subject: Subject name (e.g., "Data Structures").
        year: Examination year (e.g., "2023").
        semester: Semester (e.g., "Fall", "Spring", "1", "2").
        exam_type: Type of exam (e.g., "Mid-Term", "End-Term", "Supplementary").
        original_filename: Original name of the uploaded file.
        file_path: Path where the file is stored on disk.
        total_pages: Number of pages in the PDF.
        extraction_status: Current processing status.
        created_at: Timestamp when the record was created.
        updated_at: Timestamp of last update.
    """

    __tablename__ = "papers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject = Column(String(255), nullable=False, index=True)
    year = Column(String(10), nullable=False, index=True)
    semester = Column(String(50), nullable=True)
    exam_type = Column(String(50), nullable=True)
    original_filename = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=False)
    total_pages = Column(Integer, nullable=True)
    extraction_status = Column(
        SAEnum(
            "pending", "processing", "completed", "failed",
            name="extraction_status_enum",
        ),
        nullable=False,
        default="pending",
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    pages = relationship("Page", back_populates="paper", cascade="all, delete-orphan")
    questions = relationship("Question", back_populates="paper", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Paper(id={self.id}, subject='{self.subject}', year='{self.year}')>"


class Page(Base):
    """
    Represents a single page within a question paper.

    Attributes:
        id: Unique identifier.
        paper_id: Foreign key to the parent Paper.
        page_number: 1-based page number.
        page_type: Whether the page is digital text or scanned image.
        raw_text: Extracted raw text content.
        image_path: Path to the saved page image (for scanned pages).
        layout_data: Structured layout information from PP-StructureV3.
    """

    __tablename__ = "pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer, nullable=False)
    page_type = Column(
        SAEnum("digital", "scanned", name="page_type_enum"),
        nullable=False,
    )
    raw_text = Column(Text, nullable=True)
    image_path = Column(String(1000), nullable=True)
    layout_data = Column(JSONB, nullable=True)

    # Relationships
    paper = relationship("Paper", back_populates="pages")
    questions = relationship("Question", back_populates="page", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Page(id={self.id}, paper_id={self.paper_id}, page={self.page_number})>"
