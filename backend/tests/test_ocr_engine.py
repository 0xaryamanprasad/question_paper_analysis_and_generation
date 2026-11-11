"""
Unit and Integration Tests for Module 2: OCR and Image Preprocessing Engine.

Tests:
- Image preprocessing stages (deskew, denoise, CLAHE, binarization)
- Reading-order sorting algorithm (multiline & multi-column layouts)
- Authentic PaddleOCR inference on synthetic question images
- Confidence threshold filtering
- Batch processing across multiple page images
- Integration between PDFExtractor (Module 1) and OCREngine (Module 2)
"""

from pathlib import Path

import cv2
import fitz
import numpy as np
import pytest

from backend.api.schemas.ocr import OCRLine, OCRResult
from backend.services.ocr_engine import OCREngine, OCREngineError
from backend.services.pdf_extractor import PDFExtractor


@pytest.fixture
def sample_question_image() -> np.ndarray:
    """Create a synthetic high-contrast document image with exam questions."""
    # 300x800 white canvas
    img = np.full((300, 800, 3), 255, dtype=np.uint8)

    # Line 1: Header
    cv2.putText(img, "PART A: Answer All Questions", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    # Line 2: Question 1
    cv2.putText(img, "1. Define Abstract Data Type (ADT). [5 Marks]", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    # Line 3: Question 2
    cv2.putText(img, "2. Explain Binary Search Tree with example.", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    return img


@pytest.fixture
def sample_scanned_pdf_with_text(tmp_path: Path) -> Path:
    """Create a PDF with a rasterized image containing question text."""
    pdf_path = tmp_path / "scanned_exam_sheet.pdf"
    doc = fitz.open()

    # Create image with questions
    img = np.full((1200, 900, 3), 255, dtype=np.uint8)
    cv2.putText(img, "MID-TERM EXAMINATION 2023", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    cv2.putText(img, "Q1. What is an AVL Tree?", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Q2. Explain Heap Sort algorithm.", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    img_path = tmp_path / "temp_page.png"
    cv2.imwrite(str(img_path), img)

    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(0, 0, 595, 842), filename=str(img_path))

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


class TestOCREngine:
    """Tests for the OCREngine service."""

    def test_engine_initialization(self):
        """Engine should initialize with default configuration."""
        engine = OCREngine(lang="en", confidence_threshold=0.6)
        assert engine.lang == "en"
        assert engine.confidence_threshold == 0.6
        assert engine.use_angle_cls is True

    def test_preprocessing_pipeline(self, sample_question_image: np.ndarray):
        """Preprocessing pipeline should execute and record applied operations."""
        engine = OCREngine()
        processed, applied = engine.preprocess(
            sample_question_image,
            apply_deskew=True,
            apply_denoise=True,
            apply_contrast=True,
            apply_binarize=False,
        )

        assert isinstance(processed, np.ndarray)
        assert processed.shape[:2] == sample_question_image.shape[:2]
        assert "deskew" in applied
        assert "denoise" in applied
        assert "contrast_enhancement" in applied
        assert "binarize" not in applied

    def test_reading_order_sort(self):
        """Should order lines top-to-bottom and left-to-right within the same row."""
        engine = OCREngine(reading_order_line_tolerance=15.0)

        # Create out-of-order lines
        l_row2 = OCRLine(text="Line 2", confidence=0.9, bbox=(50.0, 100.0, 300.0, 120.0))
        l_row1_right = OCRLine(text="Right Header", confidence=0.95, bbox=(400.0, 20.0, 600.0, 40.0))
        l_row1_left = OCRLine(text="Left Header", confidence=0.95, bbox=(50.0, 22.0, 250.0, 42.0))
        l_row3 = OCRLine(text="Line 3", confidence=0.88, bbox=(50.0, 180.0, 300.0, 200.0))

        sorted_lines = engine._sort_reading_order([l_row2, l_row1_right, l_row3, l_row1_left])

        assert len(sorted_lines) == 4
        assert sorted_lines[0].text == "Left Header"
        assert sorted_lines[1].text == "Right Header"
        assert sorted_lines[2].text == "Line 2"
        assert sorted_lines[3].text == "Line 3"

    def test_process_synthetic_image(self, sample_question_image: np.ndarray):
        """Should detect and recognize text lines with high confidence."""
        engine = OCREngine(confidence_threshold=0.5)
        result = engine.process_image(sample_question_image)

        assert isinstance(result, OCRResult)
        assert result.line_count >= 2
        assert result.average_confidence > 0.8
        assert result.processing_time_ms > 0

        # Check content recognition
        raw = result.raw_text
        assert "Answer" in raw or "Questions" in raw or "PART" in raw
        assert "Abstract Data Type" in raw or "ADT" in raw
        assert "Binary Search Tree" in raw

    def test_confidence_threshold_filtering(self, sample_question_image: np.ndarray):
        """Lines below the confidence threshold should be filtered out."""
        # Unreasonably high threshold
        strict_engine = OCREngine(confidence_threshold=0.9999)
        result = strict_engine.process_image(sample_question_image)

        # Most or all lines should be filtered
        assert result.line_count == 0 or result.average_confidence >= 0.9999

    def test_process_pages_batch(self, sample_question_image: np.ndarray):
        """Should process a list of page images sequentially."""
        engine = OCREngine()
        results = engine.process_pages([sample_question_image, sample_question_image])

        assert len(results) == 2
        assert all(isinstance(r, OCRResult) for r in results)
        assert results[0].line_count > 0
        assert results[1].line_count > 0

    def test_invalid_input_type(self):
        """Should raise ValueError when input is neither path nor array."""
        engine = OCREngine()
        with pytest.raises(ValueError, match="Unsupported image input type"):
            engine.process_image(12345)  # type: ignore

    def test_file_not_found(self, tmp_path: Path):
        """Should raise FileNotFoundError when image file does not exist."""
        engine = OCREngine()
        with pytest.raises(FileNotFoundError):
            engine.process_image(tmp_path / "non_existent.png")

    def test_multi_column_reading_order_dejumble(self):
        """
        Verify that a 2-column examination layout is read sequentially:
        Header -> Column 1 (top to bottom) -> Column 2 (top to bottom).
        """
        engine = OCREngine()
        two_col_lines = [
            # Full-width header spanning both columns
            OCRLine(text="MID-TERM EXAMINATION 2023", confidence=0.98, bbox=(50.0, 30.0, 750.0, 55.0)),
            OCRLine(text="PART A: Answer All Questions", confidence=0.95, bbox=(50.0, 65.0, 750.0, 85.0)),
            # Left column (x: 50..350)
            OCRLine(text="1. Explain Quick Sort algorithm.", confidence=0.9, bbox=(50.0, 110.0, 350.0, 130.0)),
            OCRLine(text="with an example and trace.", confidence=0.9, bbox=(50.0, 140.0, 350.0, 160.0)),
            OCRLine(text="2. What is an AVL Tree?", confidence=0.9, bbox=(50.0, 190.0, 350.0, 210.0)),
            # Right column (x: 450..750)
            OCRLine(text="3. Define B-Tree and its properties.", confidence=0.9, bbox=(450.0, 110.0, 750.0, 130.0)),
            OCRLine(text="with node splitting diagrams.", confidence=0.9, bbox=(450.0, 140.0, 750.0, 160.0)),
            OCRLine(text="4. Explain Heap Sort algorithm.", confidence=0.9, bbox=(450.0, 190.0, 750.0, 210.0)),
        ]

        sorted_lines = engine._sort_reading_order(two_col_lines)
        texts = [l.text for l in sorted_lines]

        # Verify header is first
        assert texts[0] == "MID-TERM EXAMINATION 2023"
        assert texts[1] == "PART A: Answer All Questions"

        # Verify Column 1 questions appear before Column 2 questions
        q1_idx = next(i for i, t in enumerate(texts) if "Quick Sort" in t)
        q2_idx = next(i for i, t in enumerate(texts) if "AVL Tree" in t)
        q3_idx = next(i for i, t in enumerate(texts) if "B-Tree" in t)
        q4_idx = next(i for i, t in enumerate(texts) if "Heap Sort" in t)

        assert q1_idx < q2_idx < q3_idx < q4_idx

    def test_words_on_same_line_joined_with_space(self):
        """
        Verify that word tokens detected on the same horizontal line
        are reconstructed into a single coherent sentence with spaces.
        """
        engine = OCREngine()
        word_boxes = [
            OCRLine(text="1.", confidence=0.9, bbox=(50.0, 100.0, 70.0, 120.0)),
            OCRLine(text="Define", confidence=0.9, bbox=(80.0, 101.0, 145.0, 121.0)),
            OCRLine(text="Abstract", confidence=0.9, bbox=(155.0, 100.0, 230.0, 120.0)),
            OCRLine(text="Data", confidence=0.9, bbox=(240.0, 102.0, 290.0, 122.0)),
            OCRLine(text="Type", confidence=0.9, bbox=(300.0, 100.0, 350.0, 120.0)),
            OCRLine(text="[5 Marks]", confidence=0.9, bbox=(550.0, 100.0, 650.0, 120.0)),
        ]

        text = engine._build_reading_text(word_boxes)
        assert text == "1. Define Abstract Data Type [5 Marks]"

    def test_shifted_exponents_stay_in_sentence(self):
        """
        Mathematical superscripts/subscripts with slight vertical offset
        must remain in correct left-to-right sequence within their line.
        """
        engine = OCREngine()
        words_with_superscript = [
            OCRLine(text="Prove", confidence=0.9, bbox=(50.0, 100.0, 120.0, 140.0)),
            OCRLine(text="that", confidence=0.9, bbox=(130.0, 105.0, 180.0, 145.0)),
            OCRLine(text="O(n^2)", confidence=0.9, bbox=(190.0, 88.0, 260.0, 128.0)),  # shifted up
            OCRLine(text="is", confidence=0.9, bbox=(270.0, 105.0, 290.0, 145.0)),
            OCRLine(text="optimal.", confidence=0.9, bbox=(300.0, 105.0, 380.0, 145.0)),
        ]

        sorted_lines = engine._sort_reading_order(words_with_superscript)
        assert [l.text for l in sorted_lines] == ["Prove", "that", "O(n^2)", "is", "optimal."]
        text = engine._build_reading_text(sorted_lines)
        assert text == "Prove that O(n^2) is optimal."


class TestPDFAndOCRIntegration:
    """Integration tests combining Module 1 (PDF Extractor) and Module 2 (OCR Engine)."""

    def test_extract_and_ocr_scanned_pdf(
        self,
        sample_scanned_pdf_with_text: Path,
        tmp_path: Path,
    ):
        """
        End-to-End integration test:
        1. PDFExtractor ingests scanned PDF and renders high-DPI page images.
        2. OCREngine ingests rendered page image and performs full OCR.
        3. Extracted text matches actual document contents accurately.
        """
        # Step 1: Extract PDF
        extractor = PDFExtractor(
            page_image_output_dir=tmp_path / "rendered",
            render_dpi=200,
        )
        pdf_res = extractor.extract_pdf(sample_scanned_pdf_with_text)

        assert pdf_res.total_pages == 1
        assert pdf_res.scanned_pages_count == 1
        scanned_page = pdf_res.pages[0]
        assert scanned_page.page_type == "scanned"
        assert scanned_page.image_path is not None
        assert Path(scanned_page.image_path).exists()

        # Step 2: Run OCR on the extracted page image
        ocr_engine = OCREngine(confidence_threshold=0.5)
        ocr_res = ocr_engine.process_image(scanned_page.image_path)

        assert isinstance(ocr_res, OCRResult)
        assert ocr_res.line_count >= 2
        assert ocr_res.average_confidence > 0.75

        # Step 3: Validate extracted text matches questions
        combined_text = ocr_res.raw_text
        assert "EXAMINATION" in combined_text or "MID-TERM" in combined_text
        assert "AVL Tree" in combined_text
        assert "Heap Sort" in combined_text
