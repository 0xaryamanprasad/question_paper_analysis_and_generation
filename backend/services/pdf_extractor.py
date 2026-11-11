"""
Module 1: PDF Upload and Extraction Service.

Extracts text, metadata, and structural layout from uploaded PDF documents.
Accurately detects whether each page is digital-born text or a scanned document,
and renders scanned pages to high-DPI images for downstream OCR processing.
"""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from backend.api.schemas.paper import ExtractedPage, PDFExtractionResult, TextBlock
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class PDFExtractionError(Exception):
    """Raised when PDF extraction encounters an unrecoverable failure."""
    pass


class PDFExtractor:
    """
    High-performance PDF extractor utilizing PyMuPDF (fitz).

    Features:
    - Fast digital text extraction with bounding box coordinates.
    - Robust heuristic classification of digital vs. scanned pages.
    - High-resolution (300 DPI) page rasterization for scanned content.
    - Detection of encrypted/corrupt files with informative exceptions.
    """

    def __init__(
        self,
        min_digital_char_threshold: int = 50,
        render_dpi: int = 300,
        page_image_output_dir: Path | str | None = None,
    ):
        """
        Initialize the PDFExtractor.

        Args:
            min_digital_char_threshold: Minimum characters required on a page
                to classify it as a digital text page. Pages below this threshold
                that contain images or raster content are classified as scanned.
            render_dpi: Target DPI for rendering scanned pages to images (default 300).
            page_image_output_dir: Directory where rendered page images will be stored.
                Defaults to <upload_path>/rendered_pages.
        """
        self.min_digital_char_threshold = min_digital_char_threshold
        self.render_dpi = render_dpi

        settings = get_settings()
        if page_image_output_dir is None:
            self.output_dir = settings.upload_path / "rendered_pages"
        else:
            self.output_dir = Path(page_image_output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def classify_and_extract_page(
        self,
        page: fitz.Page,
        page_number: int,
        doc_stem: str,
        force_render_image: bool = False,
    ) -> ExtractedPage:
        """
        Extract text, examine layout, and classify a single PDF page.

        Args:
            page: PyMuPDF Page object.
            page_number: 1-based page number.
            doc_stem: Unique identifier/stem for the parent document.
            force_render_image: Whether to render an image even for digital pages.

        Returns:
            ExtractedPage with extracted text, blocks, and classification.
        """
        raw_text = page.get_text("text").strip()
        char_count = len(raw_text)
        words = raw_text.split()
        word_count = len(words)

        # Extract text blocks with coordinates (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        text_blocks: list[TextBlock] = []
        for b in raw_blocks:
            if len(b) >= 7:
                text_blocks.append(
                    TextBlock(
                        bbox=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                        text=str(b[4]).strip(),
                        block_num=int(b[5]),
                        block_type=int(b[6]),
                    )
                )

        images = page.get_images(full=True)
        has_images = len(images) > 0

        # Also check for vector drawings (common for LaTeX math radicals, fraction bars, brackets)
        try:
            drawings = page.get_drawings()
            has_vector_graphics = len(drawings) > 5
        except Exception:
            has_vector_graphics = False

        # Heuristic classification:
        # 1. If text >= threshold, page has extractable digital text.
        # 2. If text < threshold and images are present, page is scanned.
        # 3. If text is near-empty and no XObject images exist, it could be a
        #    full-page background vector/bitmap or blank.
        is_scanned = (char_count < self.min_digital_char_threshold) and (has_images or char_count == 0)
        page_type = "scanned" if is_scanned else "digital"

        image_path: str | None = None
        # Render high-resolution image for scanned pages, digital pages with embedded
        # images/LaTeX vector graphics, or when forced.
        if is_scanned or has_images or has_vector_graphics or force_render_image:
            image_path = str(self._render_page_to_image(page, page_number, doc_stem))

        logger.debug(
            "page_processed",
            page_number=page_number,
            page_type=page_type,
            char_count=char_count,
            word_count=word_count,
            images_found=len(images),
            rendered_image=image_path is not None,
        )

        return ExtractedPage(
            page_number=page_number,
            page_type=page_type,
            raw_text=raw_text,
            image_path=image_path,
            char_count=char_count,
            word_count=word_count,
            blocks=text_blocks,
            layout_data={
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "rotation": int(page.rotation),
                "image_count": len(images),
            },
        )

    def _render_page_to_image(
        self,
        page: fitz.Page,
        page_number: int,
        doc_stem: str,
    ) -> Path:
        """
        Render a PDF page to a PNG image at configured DPI.

        Standard PDF resolution is 72 DPI. Matrix zoom = target_dpi / 72.
        """
        zoom = self.render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        image_filename = f"{doc_stem}_page_{page_number:03d}.png"
        image_path = self.output_dir / image_filename
        pix.save(str(image_path))

        logger.info(
            "page_rendered_to_image",
            page_number=page_number,
            image_path=str(image_path),
            resolution=(pix.width, pix.height),
            dpi=self.render_dpi,
        )
        return image_path

    def extract_pdf(
        self,
        pdf_path: str | Path,
        force_render_all_images: bool = False,
    ) -> PDFExtractionResult:
        """
        Extract content and classify all pages of a PDF file.

        Args:
            pdf_path: Path to the target PDF file.
            force_render_all_images: When True, renders images for digital pages too.

        Returns:
            PDFExtractionResult with metadata and page-by-page extractions.

        Raises:
            FileNotFoundError: If the PDF path does not exist.
            PDFExtractionError: If the document is corrupted or password-protected.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        logger.info("pdf_extraction_started", file_path=str(path))

        try:
            doc = fitz.open(str(path))
        except Exception as e:
            logger.error("pdf_open_failed", path=str(path), error=str(e))
            raise PDFExtractionError(f"Failed to open PDF document: {e}") from e

        try:
            if doc.is_encrypted:
                logger.error("pdf_encrypted", path=str(path))
                raise PDFExtractionError("PDF document is encrypted/password-protected.")

            total_pages = len(doc)
            if total_pages == 0:
                logger.warning("pdf_empty", path=str(path))
                raise PDFExtractionError("PDF document has 0 pages.")

            extracted_pages: list[ExtractedPage] = []
            digital_count = 0
            scanned_count = 0
            doc_stem = path.stem

            for page_index in range(total_pages):
                page_number = page_index + 1
                page = doc.load_page(page_index)

                page_res = self.classify_and_extract_page(
                    page=page,
                    page_number=page_number,
                    doc_stem=doc_stem,
                    force_render_image=force_render_all_images,
                )

                extracted_pages.append(page_res)
                if page_res.page_type == "digital":
                    digital_count += 1
                else:
                    scanned_count += 1

            meta: dict[str, Any] = {
                "format": doc.metadata.get("format", ""),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "file_size_bytes": path.stat().st_size,
            }

            result = PDFExtractionResult(
                original_filename=path.name,
                total_pages=total_pages,
                digital_pages_count=digital_count,
                scanned_pages_count=scanned_count,
                pages=extracted_pages,
                metadata=meta,
                is_fully_digital=(digital_count == total_pages),
                is_fully_scanned=(scanned_count == total_pages),
            )

            logger.info(
                "pdf_extraction_completed",
                total_pages=total_pages,
                digital_pages=digital_count,
                scanned_pages=scanned_count,
            )
            return result

        finally:
            doc.close()

    # Convenience alias
    extract = extract_pdf
