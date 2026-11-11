"""
Pydantic Schemas for Questions and Segmentation.

Defines schemas for segmented questions, sub-parts, sections,
and question extraction requests/responses.
"""

import uuid
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class SubPart(BaseModel):
    """Represents a sub-question part (e.g., 1(a), 1(b))."""
    part_label: str = Field(..., description="Part identifier, e.g., 'a', 'b', 'i'")
    text: str = Field(..., description="Text of the sub-part")
    marks: int | None = Field(default=None, description="Marks allocated to this specific sub-part")

    @property
    def label(self) -> str:
        return self.part_label


class SegmentedQuestion(BaseModel):
    """Represents a fully segmented question extracted from an exam paper."""
    question_number: str = Field(..., description="Normalized question number, e.g., '1', '2a', 'Q3'")
    raw_text: str = Field(..., description="Original question text as extracted")
    cleaned_text: str = Field(default="", description="Normalized question text")
    marks: int | None = Field(default=None, description="Allocated marks for the question")
    section: str = Field(default="Default", description="Section name, e.g., 'Section A', 'Part I'")
    question_type: str = Field(
        default="medium",
        description="Question type: 'short', 'medium', 'long', 'numerical', 'mcq'",
    )
    sub_parts: list[SubPart] = Field(
        default_factory=list, description="List of parsed sub-parts if question has sub-items"
    )
    is_or_choice: bool = Field(
        default=False, description="True if this is part of an OR/alternative question pair"
    )
    or_group_id: str | None = Field(
        default=None, description="Shared ID linking questions that are alternatives to each other"
    )
    page_number: int | None = Field(default=None, description="1-based page number where question begins")

    @property
    def text(self) -> str:
        """Convenience property returning raw_text."""
        return self.raw_text


class SegmentationResult(BaseModel):
    """Result of question segmentation on a question paper document."""
    total_questions: int
    total_marks_detected: int
    sections_detected: list[str]
    questions: list[SegmentedQuestion]
    unsegmented_blocks: list[str] = Field(
        default_factory=list, description="Blocks identified as instructions or non-question text"
    )


class QuestionResponse(BaseModel):
    """Schema for Question API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    paper_id: uuid.UUID
    page_id: uuid.UUID | None = None
    question_number: str | None = None
    raw_text: str
    cleaned_text: str | None = None
    marks: int | None = None
    section: str | None = None
    question_type: str | None = None
    sub_parts: list[dict[str, Any]] | None = None
