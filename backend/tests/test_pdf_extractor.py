"""
Unit Tests for Module 1: PDF Upload and Extraction Service.

Tests extraction on:
- Digital PDFs (native text with paragraphs and questions)
- Scanned-style PDFs (image-based pages with zero/low text)
- Hybrid PDFs (mixture of digital and scanned pages)
- Corrupt and encrypted PDFs
- Rendered image validation (dimensions, DPI, disk persistence)
"""

import io
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pytest
from PIL import Image

from backend.api.schemas.paper import PDFExtractionResult
from backend.services.pdf_extractor import PDFExtractionError, PDFExtractor


@pytest.fixture
def sample_scanned_pdf(tmp_path: Path) -> Path:
    """Create a PDF simulating a scanned document (image inserted into page)."""
    pdf_path = tmp_path / "scanned_paper.pdf"
    doc = fitz.open()

    # Create a dummy image
    img = Image.new("RGB", (800, 1100), color=(245, 245, 245))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    p = doc.new_page(width=595, height=842)
    rect = fitz.Rect(0, 0, 595, 842)
    p.insert_image(rect, stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_hybrid_pdf(tmp_path: Path) -> Path:
    """Create a PDF with one digital page and one scanned page."""
    pdf_path = tmp_path / "hybrid_paper.pdf"
    doc = fitz.open()

    # Page 1: Digital
    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "Mid-Term Examination: Computer Networks\n"
        "Answer all questions. Each question carries equal marks.\n"
        "Question 1: Explain the OSI 7-layer architecture in detail.",
        fontsize=12,
    )

    # Page 2: Scanned image
    img = Image.new("RGB", (600, 800), color=(250, 250, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    p2 = doc.new_page(width=595, height=842)
    p2.insert_image(fitz.Rect(0, 0, 595, 842), stream=buf.getvalue())

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestPDFExtractor:
    """Tests for PDFExtractor service."""

    def test_extract_digital_pdf(self, sample_digital_pdf: Path, tmp_path: Path):
        """Should accurately extract text, blocks, and classify as digital."""
        extractor = PDFExtractor(
            page_image_output_dir=tmp_path / "rendered",
            render_dpi=150,
        )
        result = extractor.extract_pdf(sample_digital_pdf)

        assert isinstance(result, PDFExtractionResult)
        assert result.total_pages == 2
        assert result.digital_pages_count == 2
        assert result.scanned_pages_count == 0
        assert result.is_fully_digital is True
        assert result.is_fully_scanned is False

        # Inspect page 1
        p1 = result.pages[0]
        assert p1.page_number == 1
        assert p1.page_type == "digital"
        assert "Abstract Data Type" in p1.raw_text
        assert p1.char_count > 50
        assert p1.word_count > 10
        assert len(p1.blocks) > 0
        assert p1.image_path is None  # Not rendered by default for digital pages

        # Inspect page 2
        p2 = result.pages[1]
        assert p2.page_number == 2
        assert p2.page_type == "digital"
        assert "Dijkstra" in p2.raw_text

    def test_extract_scanned_pdf(self, sample_scanned_pdf: Path, tmp_path: Path):
        """Should detect scanned page and render high-resolution PNG image."""
        render_dir = tmp_path / "rendered"
        extractor = PDFExtractor(
            page_image_output_dir=render_dir,
            render_dpi=150,
        )
        result = extractor.extract_pdf(sample_scanned_pdf)

        assert result.total_pages == 1
        assert result.digital_pages_count == 0
        assert result.scanned_pages_count == 1
        assert result.is_fully_scanned is True

        p1 = result.pages[0]
        assert p1.page_type == "scanned"
        assert p1.char_count < 50
        assert p1.image_path is not None

        # Verify rendered image file exists on disk
        img_path = Path(p1.image_path)
        assert img_path.exists()
        assert img_path.is_file()
        assert img_path.suffix.lower() == ".png"

        # Verify image properties
        with Image.open(img_path) as img:
            assert img.width > 0
            assert img.height > 0

    def test_extract_hybrid_pdf(self, sample_hybrid_pdf: Path, tmp_path: Path):
        """Should distinguish digital from scanned in a mixed document."""
        extractor = PDFExtractor(page_image_output_dir=tmp_path / "rendered")
        result = extractor.extract_pdf(sample_hybrid_pdf)

        assert result.total_pages == 2
        assert result.digital_pages_count == 1
        assert result.scanned_pages_count == 1
        assert result.is_fully_digital is False
        assert result.is_fully_scanned is False

        assert result.pages[0].page_type == "digital"
        assert result.pages[1].page_type == "scanned"
        assert result.pages[1].image_path is not None

    def test_force_render_all_images(self, sample_digital_pdf: Path, tmp_path: Path):
        """Should render images even for digital pages if force_render is requested."""
        extractor = PDFExtractor(page_image_output_dir=tmp_path / "rendered")
        result = extractor.extract_pdf(sample_digital_pdf, force_render_all_images=True)

        for p in result.pages:
            assert p.image_path is not None
            assert Path(p.image_path).exists()

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError when PDF does not exist."""
        extractor = PDFExtractor()
        with pytest.raises(FileNotFoundError):
            extractor.extract_pdf(tmp_path / "non_existent.pdf")

    def test_invalid_corrupt_pdf(self, tmp_path: Path):
        """Should raise PDFExtractionError on invalid PDF byte sequence."""
        corrupt_file = tmp_path / "corrupt.pdf"
        corrupt_file.write_bytes(b"%PDF-1.4\nCorrupt header and damaged EOF")

        extractor = PDFExtractor()
        with pytest.raises(PDFExtractionError):
            extractor.extract_pdf(corrupt_file)

    def test_empty_pdf(self, tmp_path: Path):
        """Should raise PDFExtractionError on 0-page document."""
        empty_pdf = tmp_path / "empty.pdf"
        # Standard PDF with 0 pages in the Pages tree
        empty_pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
            b"xref\n0 3\n"
            b"0000000000 65535 f \n"
            b"0000000010 00000 n \n"
            b"0000000060 00000 n \n"
            b"trailer\n<< /Root 1 0 R /Size 3 >>\n"
            b"startxref\n115\n%%EOF\n"
        )
        empty_pdf.write_bytes(empty_pdf_bytes)

        extractor = PDFExtractor()
        with pytest.raises(PDFExtractionError, match="0 pages"):
            extractor.extract_pdf(empty_pdf)
