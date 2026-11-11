"""
SQLAlchemy Models for Questions, Embeddings, and Classifications.

Represents segmented questions with their NLP analysis results.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship

from backend.database import Base


class Question(Base):
    """
    Represents a single question extracted from a paper.

    Attributes:
        id: Unique identifier.
        paper_id: Foreign key to the parent Paper.
        page_id: Foreign key to the Page where this question was found.
        question_number: Display question number (e.g., "1", "2a", "3.ii").
        raw_text: Original question text as extracted.
        cleaned_text: Preprocessed/normalized text.
        marks: Marks allocated to this question.
        section: Section of the paper (e.g., "A", "B", "Part I").
        question_type: Type of question.
        sub_parts: JSON array of sub-question objects.
    """

    __tablename__ = "questions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    paper_id = Column(
        UUID(as_uuid=True),
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_id = Column(
        UUID(as_uuid=True),
        ForeignKey("pages.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    question_number = Column(String(20), nullable=True)
    subquestion_number = Column(String(20), nullable=True)
    raw_text = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=True)
    latex_expressions = Column(JSON, nullable=True)
    marks = Column(Integer, nullable=True)
    section = Column(String(50), nullable=True)
    question_type = Column(String(50), nullable=True)
    sub_parts = Column(JSONB, nullable=True)
    dsa_topic = Column(String(100), nullable=True, index=True)
    dsa_subtopic = Column(String(100), nullable=True)
    dsa_secondary_topic = Column(String(100), nullable=True)
    dsa_confidence = Column(Float, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    is_flagged_for_verification = Column(Boolean, default=False, nullable=False)
    verification_flags = Column(JSON, nullable=True)
    crop_image_path = Column(String(500), nullable=True)
    original_page_number = Column(Integer, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    paper = relationship("Paper", back_populates="questions")
    page = relationship("Page", back_populates="questions")
    embedding = relationship(
        "QuestionEmbedding",
        back_populates="question",
        uselist=False,
        cascade="all, delete-orphan",
    )
    classification = relationship(
        "QuestionClassification",
        back_populates="question",
        uselist=False,
        cascade="all, delete-orphan",
    )
    similarity_pairs_a = relationship(
        "SimilarityPair",
        foreign_keys="SimilarityPair.question_a_id",
        back_populates="question_a",
        cascade="all, delete-orphan",
    )
    similarity_pairs_b = relationship(
        "SimilarityPair",
        foreign_keys="SimilarityPair.question_b_id",
        back_populates="question_b",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Question(id={self.id}, paper_id={self.paper_id}, "
            f"q_no='{self.question_number}', marks={self.marks})>"
        )


class QuestionEmbedding(Base):
    """
    Stores the sentence embedding vector for a question.

    The embedding is stored as raw bytes (float32 array serialized).
    This avoids requiring the pgvector extension while keeping
    the data close to the question record.

    Attributes:
        id: Unique identifier.
        question_id: Foreign key to the Question.
        embedding_vector: Serialized float32 embedding (bytea).
        model_name: Name of the model used to generate the embedding.
        computed_at: When the embedding was last computed.
    """

    __tablename__ = "question_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(
        UUID(as_uuid=True),
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    embedding_vector = Column(LargeBinary, nullable=False)
    model_name = Column(String(255), nullable=False)
    computed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    question = relationship("Question", back_populates="embedding")

    def __repr__(self) -> str:
        return f"<QuestionEmbedding(question_id={self.question_id}, model='{self.model_name}')>"


class QuestionClassification(Base):
    """
    Stores topic and Bloom's taxonomy classification for a question.

    Attributes:
        id: Unique identifier.
        question_id: Foreign key to the Question.
        unit: Syllabus unit number/name.
        topic: Specific topic within the unit.
        topic_confidence: Confidence score for topic classification.
        blooms_level: Bloom's taxonomy cognitive level.
        blooms_confidence: Confidence score for Bloom's classification.
        keywords: Extracted keywords relevant to classification.
    """

    __tablename__ = "question_classifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(
        UUID(as_uuid=True),
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    unit = Column(String(100), nullable=True)
    topic = Column(String(255), nullable=True)
    topic_confidence = Column(Float, nullable=True)
    blooms_level = Column(String(50), nullable=True)
    blooms_confidence = Column(Float, nullable=True)
    keywords = Column(JSONB, nullable=True)

    # Relationships
    question = relationship("Question", back_populates="classification")

    def __repr__(self) -> str:
        return (
            f"<QuestionClassification(question_id={self.question_id}, "
            f"unit='{self.unit}', blooms='{self.blooms_level}')>"
        )


class SimilarityPair(Base):
    """
    Stores pairwise similarity scores between questions.

    Only pairs exceeding the similarity threshold are stored.

    Attributes:
        id: Unique identifier.
        question_a_id: First question in the pair.
        question_b_id: Second question in the pair.
        similarity_score: Cosine similarity score (0.0–1.0).
        relation_type: Classification of the similarity.
    """

    __tablename__ = "similarity_pairs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_a_id = Column(
        UUID(as_uuid=True),
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_b_id = Column(
        UUID(as_uuid=True),
        ForeignKey("questions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    similarity_score = Column(Float, nullable=False)
    relation_type = Column(String(50), nullable=False)

    # Relationships
    question_a = relationship(
        "Question", foreign_keys=[question_a_id], back_populates="similarity_pairs_a",
    )
    question_b = relationship(
        "Question", foreign_keys=[question_b_id], back_populates="similarity_pairs_b",
    )

    def __repr__(self) -> str:
        return (
            f"<SimilarityPair(a={self.question_a_id}, b={self.question_b_id}, "
            f"score={self.similarity_score:.3f})>"
        )
