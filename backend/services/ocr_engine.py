"""
Module 2: OCR and Image Preprocessing Engine.

Performs document image preprocessing (deskew, denoise, CLAHE contrast enhancement,
binarization) and optical character recognition utilizing PaddleOCR PP-OCR models.
Extracts bounding boxes, oriented polygons, confidence scores, and reading-order text.
"""

import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.api.schemas.ocr import OCRLine, OCRResult
from backend.config import get_settings
from backend.utils.image_utils import (
    binarize,
    deskew,
    denoise,
    enhance_contrast,
    load_image,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class OCREngineError(Exception):
    """Raised when OCR processing encounters an unrecoverable error."""
    pass


class OCREngine:
    """
    Production-grade OCR Engine combining OpenCV preprocessing with PaddleOCR.

    Features:
    - Preprocessing pipeline with configurable stages.
    - Angle classification and rotation correction.
    - Reading order sorting (top-to-bottom, left-to-right).
    - Confidence scoring and coordinate normalization.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        confidence_threshold: float = 0.5,
        reading_order_line_tolerance: float = 12.0,
    ):
        """
        Initialize the OCR Engine.

        Args:
            lang: Language code for OCR models (default: "en").
            use_angle_cls: Whether to use orientation angle classifier.
            confidence_threshold: Minimum confidence to include a detected line.
            reading_order_line_tolerance: Vertical distance (pixels) to group
                words into the same horizontal line during reading-order sort.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.confidence_threshold = confidence_threshold
        self.reading_order_line_tolerance = reading_order_line_tolerance
        self._page_structurer = None

        # Ensure environment variables for Paddle cache point to local workspace
        settings = get_settings()
        workspace_root = Path(__file__).resolve().parent.parent.parent
        self.paddle_cache = workspace_root / ".paddle_cache"
        if self.paddle_cache.exists():
            os.environ["HOME"] = str(self.paddle_cache)
            os.environ["PADDLE_HOME"] = str(self.paddle_cache)
            os.environ["PADDLEX_HOME"] = str(self.paddle_cache)

        self._ocr = None
        logger.info(
            "ocr_engine_initialized",
            lang=self.lang,
            use_angle_cls=self.use_angle_cls,
            confidence_threshold=self.confidence_threshold,
        )

    @property
    def ocr_client(self):
        """Lazy-load the PaddleOCR client on first inference call."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR

                ocr_kwargs: dict[str, Any] = {
                    "use_angle_cls": self.use_angle_cls,
                    "lang": self.lang,
                    "show_log": False,
                }

                # If local cached models exist, supply paths directly to avoid network/fs calls
                det_dir = self.paddle_cache / ".paddleocr" / "whl" / "det" / "en" / "en_PP-OCRv3_det_infer"
                rec_dir = self.paddle_cache / ".paddleocr" / "whl" / "rec" / "en" / "en_PP-OCRv4_rec_infer"
                cls_dir = self.paddle_cache / ".paddleocr" / "whl" / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"

                if det_dir.exists():
                    ocr_kwargs["det_model_dir"] = str(det_dir)
                if rec_dir.exists():
                    ocr_kwargs["rec_model_dir"] = str(rec_dir)
                if cls_dir.exists():
                    ocr_kwargs["cls_model_dir"] = str(cls_dir)

                self._ocr = PaddleOCR(**ocr_kwargs)
                logger.info("paddleocr_client_loaded", kwargs=ocr_kwargs)
            except Exception as e:
                logger.error("paddleocr_init_failed", error=str(e))
                raise OCREngineError(f"Failed to initialize PaddleOCR: {e}") from e
        return self._ocr

    @property
    def page_structurer(self):
        """Lazy-load the NLP page structurer."""
        if self._page_structurer is None:
            try:
                from backend.services.nlp_preprocessor import NLPPageStructurer
                self._page_structurer = NLPPageStructurer()
            except Exception as e:
                logger.warning("failed_to_load_nlp_structurer", error=str(e))
                self._page_structurer = None
        return self._page_structurer

    def preprocess(
        self,
        image: np.ndarray,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_contrast: bool = True,
        apply_binarize: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Apply a series of OpenCV preprocessing steps to improve OCR accuracy.

        Args:
            image: Input image (BGR or grayscale).
            apply_deskew: Whether to correct page rotation skew.
            apply_denoise: Whether to apply non-local means denoising.
            apply_contrast: Whether to enhance contrast with CLAHE.
            apply_binarize: Whether to apply adaptive thresholding.
                (Note: PaddleOCR often performs better on grayscale/color
                than binary images, so binarize defaults to False).

        Returns:
            Tuple of (preprocessed_image, list_of_applied_operations).
        """
        processed = image.copy()
        applied: list[str] = []

        if apply_deskew:
            processed = deskew(processed)
            applied.append("deskew")

        if apply_denoise:
            processed = denoise(processed, strength=7)
            applied.append("denoise")

        if apply_contrast:
            processed = enhance_contrast(processed, clip_limit=2.0)
            applied.append("contrast_enhancement")

        if apply_binarize:
            processed = binarize(processed, method="adaptive")
            applied.append("binarize")

        return processed, applied

    def _sort_single_block_reading_order(
        self,
        lines: list[OCRLine],
        tolerance: float | None = None,
    ) -> list[OCRLine]:
        """
        Sort a single coherent block or column of OCR lines into reading order.

        Uses adaptive vertical overlap clustering:
        - Dynamically derives tolerance from median box height.
        - Measures vertical overlap and vertical center proximity.
        - Prevents merging of horizontally overlapping boxes (vertically stacked lines).
        - Sorts rows top-to-bottom and boxes within each row left-to-right.
        """
        if not lines:
            return []
        if len(lines) == 1:
            return list(lines)

        heights = [l.bbox[3] - l.bbox[1] for l in lines if (l.bbox[3] - l.bbox[1]) > 0]
        med_h = float(np.median(heights)) if heights else 20.0
        tol = max(6.0, min(med_h * 0.55, (tolerance or self.reading_order_line_tolerance) * 1.5))

        # Sort primarily by vertical center, then x0
        by_y = sorted(lines, key=lambda l: ((l.bbox[1] + l.bbox[3]) / 2.0, l.bbox[0]))

        rows: list[list[OCRLine]] = []
        for line in by_y:
            y_mid = (line.bbox[1] + line.bbox[3]) / 2.0
            line_h = max(1.0, line.bbox[3] - line.bbox[1])
            placed = False

            for row in rows:
                row_y_mid = sum((r.bbox[1] + r.bbox[3]) / 2.0 for r in row) / len(row)
                row_min_y = min(r.bbox[1] for r in row)
                row_max_y = max(r.bbox[3] for r in row)
                row_avg_h = sum(r.bbox[3] - r.bbox[1] for r in row) / len(row)

                v_overlap = max(0.0, min(line.bbox[3], row_max_y) - max(line.bbox[1], row_min_y))
                min_h = min(line_h, row_avg_h)

                # Check horizontal collision (boxes stacked vertically should not share a row)
                h_collision = any(
                    (max(0.0, min(line.bbox[2], r.bbox[2]) - max(line.bbox[0], r.bbox[0]))
                     / max(1.0, min(line.bbox[2] - line.bbox[0], r.bbox[2] - r.bbox[0]))) > 0.45
                    for r in row
                )

                if not h_collision and (v_overlap >= 0.35 * min_h or abs(y_mid - row_y_mid) <= tol):
                    row.append(line)
                    placed = True
                    break

            if not placed:
                rows.append([line])

        # Sort rows top-to-bottom by average vertical center
        rows.sort(key=lambda r: sum((l.bbox[1] + l.bbox[3]) / 2.0 for l in r) / len(r))

        sorted_lines: list[OCRLine] = []
        for row in rows:
            # Sort each row left-to-right by x0
            row.sort(key=lambda l: l.bbox[0])
            sorted_lines.extend(row)

        return sorted_lines

    def _sort_reading_order(self, lines: list[OCRLine]) -> list[OCRLine]:
        """
        Sort OCR lines in natural reading order: top-to-bottom, left-to-right,
        with intelligent multi-column layout detection and de-jumbling.

        If a multi-column exam layout is detected (e.g. two parallel columns of
        questions with a vertical gutter):
        - Spanning header blocks are read first.
        - Column 1 is read top-to-bottom.
        - Column 2 is read top-to-bottom.
        - Spanning footer blocks are read last.
        """
        if not lines or len(lines) <= 1:
            return list(lines)

        x_min = min(l.bbox[0] for l in lines)
        x_max = max(l.bbox[2] for l in lines)
        y_min = min(l.bbox[1] for l in lines)
        y_max = max(l.bbox[3] for l in lines)
        total_w = x_max - x_min
        total_h = y_max - y_min

        # Check for multi-column layout if page is wide enough and has sufficient boxes
        if len(lines) >= 4 and total_w > 200:
            headers: list[OCRLine] = []
            footers: list[OCRLine] = []
            body: list[OCRLine] = []

            for l in lines:
                w = l.bbox[2] - l.bbox[0]
                is_spanning = (w > 0.55 * total_w) or (
                    l.bbox[0] < x_min + 0.35 * total_w and l.bbox[2] > x_min + 0.65 * total_w
                )
                if is_spanning and l.bbox[1] < y_min + 0.35 * total_h:
                    headers.append(l)
                elif is_spanning and l.bbox[3] > y_max - 0.20 * total_h:
                    footers.append(l)
                else:
                    body.append(l)

            if len(body) >= 4:
                cand_x = np.linspace(x_min + 0.35 * total_w, x_min + 0.65 * total_w, 21)
                best_split = None
                min_cross = len(body)

                for cx in cand_x:
                    cross = sum(1 for l in body if l.bbox[0] < cx - 10 and l.bbox[2] > cx + 10)
                    if cross < min_cross:
                        min_cross = cross
                        best_split = cx

                if best_split is not None and min_cross <= max(1, 0.05 * len(body)):
                    left_col = [l for l in body if l.bbox[2] <= best_split + 10]
                    right_col = [l for l in body if l.bbox[0] >= best_split - 10]

                    if len(left_col) >= 2 and len(right_col) >= 2:
                        left_y0 = min(l.bbox[1] for l in left_col)
                        left_y1 = max(l.bbox[3] for l in left_col)
                        right_y0 = min(l.bbox[1] for l in right_col)
                        right_y1 = max(l.bbox[3] for l in right_col)
                        v_overlap = max(0.0, min(left_y1, right_y1) - max(left_y0, right_y0))
                        body_span = max(left_y1, right_y1) - min(left_y0, right_y0)

                        if v_overlap >= 0.30 * max(1.0, body_span):
                            # Confirmed 2-column examination layout
                            return (
                                self._sort_single_block_reading_order(headers)
                                + self._sort_single_block_reading_order(left_col)
                                + self._sort_single_block_reading_order(right_col)
                                + self._sort_single_block_reading_order(footers)
                            )

        return self._sort_single_block_reading_order(lines)

    def _build_reading_text(self, sorted_lines: list[OCRLine]) -> str:
        """
        Reconstruct coherent, sequential text from ordered OCR lines.

        Words/boxes on the same horizontal line are joined with spaces.
        Distinct lines are joined with newlines.
        De-hyphenates split words across adjacent boxes on the same line.
        """
        if not sorted_lines:
            return ""
        if len(sorted_lines) == 1:
            return sorted_lines[0].text.strip()

        heights = [l.bbox[3] - l.bbox[1] for l in sorted_lines if (l.bbox[3] - l.bbox[1]) > 0]
        med_h = float(np.median(heights)) if heights else 20.0
        line_tol = max(6.0, min(med_h * 0.55, self.reading_order_line_tolerance * 1.5))

        output_lines: list[str] = []
        current_tokens: list[str] = []
        prev_box: OCRLine | None = None

        for curr in sorted_lines:
            curr_text = curr.text.strip()
            if not curr_text:
                continue
            if prev_box is None:
                current_tokens.append(curr_text)
                prev_box = curr
                continue

            v_overlap = max(0.0, min(prev_box.bbox[3], curr.bbox[3]) - max(prev_box.bbox[1], curr.bbox[1]))
            min_h = max(1.0, min(prev_box.bbox[3] - prev_box.bbox[1], curr.bbox[3] - curr.bbox[1]))
            y_mid_prev = (prev_box.bbox[1] + prev_box.bbox[3]) / 2.0
            y_mid_curr = (curr.bbox[1] + curr.bbox[3]) / 2.0

            is_x_forward = curr.bbox[0] >= prev_box.bbox[0] - 15.0
            is_same_row = (
                is_x_forward and (
                    (v_overlap / min_h >= 0.35) or
                    (abs(y_mid_prev - y_mid_curr) <= line_tol)
                )
            )

            if is_same_row:
                # Check for de-hyphenation across words on the same line
                if current_tokens and current_tokens[-1].endswith("-") and curr_text and curr_text[0].isalpha():
                    current_tokens[-1] = current_tokens[-1][:-1] + curr_text
                else:
                    current_tokens.append(curr_text)
            else:
                line_str = " ".join(current_tokens)
                if line_str:
                    output_lines.append(line_str)
                current_tokens = [curr_text]

            prev_box = curr

        if current_tokens:
            line_str = " ".join(current_tokens)
            if line_str:
                output_lines.append(line_str)

        return "\n".join(output_lines)

    def process_image(
        self,
        image_input: str | Path | np.ndarray,
        preprocess: bool = True,
        apply_deskew: bool = True,
        apply_denoise: bool = True,
        apply_contrast: bool = True,
        apply_binarize: bool = False,
    ) -> OCRResult:
        """
        Execute full OCR on an image.

        Args:
            image_input: File path or NumPy image array (BGR).
            preprocess: Whether to apply preprocessing prior to OCR.
            apply_deskew: If preprocessing, apply deskew.
            apply_denoise: If preprocessing, apply denoising.
            apply_contrast: If preprocessing, apply CLAHE contrast.
            apply_binarize: If preprocessing, apply adaptive binarization.

        Returns:
            OCRResult containing recognized lines, bounding boxes, and metadata.

        Raises:
            FileNotFoundError: If image file path does not exist.
            OCREngineError: If inference fails.
        """
        start_time = time.perf_counter()

        # Load image if path provided
        if isinstance(image_input, (str, Path)):
            img = load_image(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        applied_ops: list[str] = []
        if preprocess:
            img, applied_ops = self.preprocess(
                img,
                apply_deskew=apply_deskew,
                apply_denoise=apply_denoise,
                apply_contrast=apply_contrast,
                apply_binarize=apply_binarize,
            )

        try:
            # PaddleOCR returns a list of results, one per image
            raw_res = self.ocr_client.ocr(img, cls=self.use_angle_cls)
        except Exception as e:
            logger.error("ocr_inference_failed", error=str(e))
            raise OCREngineError(f"PaddleOCR inference failed: {e}") from e

        ocr_lines: list[OCRLine] = []

        # PaddleOCR result structure: list of [[box, (text, score)], ...]
        if raw_res and len(raw_res) > 0 and raw_res[0] is not None:
            for item in raw_res[0]:
                poly = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = item[1]  # (text_str, confidence_float)
                text = str(text_info[0]).strip()
                score = float(text_info[1])

                if score < self.confidence_threshold or not text:
                    continue

                # Compute axis-aligned bbox from polygon
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

                polygon = [[float(pt[0]), float(pt[1])] for pt in poly]

                ocr_lines.append(
                    OCRLine(
                        text=text,
                        confidence=round(score, 4),
                        bbox=bbox,
                        polygon=polygon,
                    )
                )

        # Sort into reading order
        sorted_lines = self._sort_reading_order(ocr_lines)
        combined_text = self._build_reading_text(sorted_lines)

        # Structure entire page text using NLP pipeline
        structured_text = combined_text
        if self.page_structurer is not None and combined_text.strip():
            try:
                structured_res = self.page_structurer.structure_page(combined_text)
                if structured_res.strip():
                    structured_text = structured_res
            except Exception as nlp_err:
                logger.warning("nlp_page_structuring_failed", error=str(nlp_err))

        total_lines = len(sorted_lines)
        avg_confidence = (
            round(sum(line.confidence for line in sorted_lines) / total_lines, 4)
            if total_lines > 0
            else 0.0
        )
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        logger.info(
            "ocr_processing_completed",
            lines_detected=total_lines,
            avg_confidence=avg_confidence,
            duration_ms=duration_ms,
        )

        return OCRResult(
            raw_text=structured_text,
            structured_text=structured_text,
            lines=sorted_lines,
            average_confidence=avg_confidence,
            line_count=total_lines,
            preprocessing_applied=applied_ops,
            processing_time_ms=duration_ms,
        )

    def process_pages(
        self,
        page_images: list[str | Path | np.ndarray],
    ) -> list[OCRResult]:
        """
        Process multiple page images sequentially.

        Args:
            page_images: List of image paths or image arrays.

        Returns:
            List of OCRResult objects in the same order as input pages.
        """
        results: list[OCRResult] = []
        for idx, page_img in enumerate(page_images):
            logger.info("processing_page_ocr", page_index=idx + 1, total=len(page_images))
            res = self.process_image(page_img)
            results.append(res)
        return results

    # Convenience alias for process_image
    extract = process_image
