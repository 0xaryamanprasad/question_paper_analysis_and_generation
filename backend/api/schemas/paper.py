"""
Pydantic Schemas for Papers and PDF Extraction.

Defines request/response data models for paper upload,
page extraction, and extraction status.
"""

import uuid
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class TextBlock(BaseModel):
    """Represents a text block within a page layout."""
    bbox: tuple[float, float, float, float] = Field(
        ..., description="(x0, y0, x1, y1) bounding box coordinates"
    )
    text: str = Field(..., description="Block text content")
    block_type: int = Field(0, description="0 for text, 1 for image")
    block_num: int = Field(0, description="Block index on the page")


class ExtractedPage(BaseModel):
    """Represents extraction results for a single page."""
    page_number: int = Field(..., description="1-based page number")
    page_type: Literal["digital", "scanned"] = Field(
        ..., description="Detected page type: digital text or scanned image"
    )
    raw_text: str = Field(default="", description="Extracted text content")
    image_path: str | None = Field(
        default=None, description="Path to rendered page image if scanned or needed"
    )
    char_count: int = Field(default=0, description="Number of text characters on page")
    word_count: int = Field(default=0, description="Number of words on page")
    blocks: list[TextBlock] = Field(
        default_factory=list, description="Extracted text blocks with bounding boxes"
    )
    layout_data: dict[str, Any] | None = Field(
        default=None, description="Additional layout information"
    )


class PDFExtractionResult(BaseModel):
    """Overall result of extracting content from a PDF document."""
    original_filename: str
    total_pages: int
    digital_pages_count: int
    scanned_pages_count: int
    pages: list[ExtractedPage]
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_fully_digital: bool = Field(default=False)
    is_fully_scanned: bool = Field(default=False)


class PaperCreate(BaseModel):
    """Schema for creating a paper record."""
    subject: str = Field(..., min_length=1, max_length=255)
    year: str = Field(..., min_length=2, max_length=10)
    semester: str | None = Field(default=None, max_length=50)
    exam_type: str | None = Field(default=None, max_length=50)


class PaperResponse(BaseModel):
    """Schema for paper API responses."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    subject: str
    year: str
    semester: str | None = None
    exam_type: str | None = None
    original_filename: str
    file_path: str
    total_pages: int | None = None
    extraction_status: str
    created_at: datetime
    updated_at: datetime


class PageResponse(BaseModel):
    """Schema for individual page API response."""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    paper_id: uuid.UUID
    page_number: int
    page_type: str
    raw_text: str | None = None
    image_path: str | None = None


class PaperListResponse(BaseModel):
    """Schema for paginated/listed paper API responses."""
    total: int
    papers: list[PaperResponse]
