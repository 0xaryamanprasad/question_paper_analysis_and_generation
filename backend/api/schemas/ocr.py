"""
Pydantic Schemas for OCR Results and Layout Analysis.

Defines data models for recognized text lines, bounding boxes,
confidence metrics, and layout elements from OCR processing.
"""

from pydantic import BaseModel, Field


class OCRLine(BaseModel):
    """Represents a recognized line of text from OCR."""
    text: str = Field(..., description="Recognized text string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score between 0.0 and 1.0")
    bbox: tuple[float, float, float, float] = Field(
        ..., description="(x0, y0, x1, y1) axis-aligned bounding box"
    )
    polygon: list[list[float]] = Field(
        default_factory=list, description="4-point oriented bounding polygon [[x,y], ...]"
    )


class OCRResult(BaseModel):
    """Aggregated result of OCR processing on a document page or image."""
    raw_text: str = Field(..., description="Full combined text in reading order")
    structured_text: str | None = Field(default=None, description="NLP-structured and reflowed text for the entire page")
    lines: list[OCRLine] = Field(default_factory=list, description="Extracted text lines with coordinates")
    average_confidence: float = Field(default=0.0, description="Mean confidence score of recognized lines")
    line_count: int = Field(default=0, description="Total number of recognized lines")
    preprocessing_applied: list[str] = Field(
        default_factory=list, description="List of image preprocessing operations applied"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing duration in milliseconds")

    @property
    def extracted_text(self) -> str:
        """Convenience property for raw_text."""
        return self.raw_text

    @property
    def text(self) -> str:
        """Convenience property for raw_text."""
        return self.raw_text
