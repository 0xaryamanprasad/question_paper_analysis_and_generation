"""
Module 1: Advanced OCR and Mathematical Expression Recognition Engine.

Features:
1. Accepts PDF documents or image files.
2. Extracts individual pages and page numbers.
3. Performs layout-aware OCR utilizing PaddleOCR.
4. Preserves page numbers, text lines, and precise bounding boxes.
5. Detects mathematical expressions, equation regions, and formula layouts.
6. Recognizes formula regions using Mathpix OCR API (if configured) or
   built-in mathematical parsing.
7. Preserves mathematical expressions in standardized LaTeX format ($...$, $$...$$).
"""

import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
from pydantic import BaseModel, Field

from backend.api.schemas.ocr import OCRLine, OCRResult
from backend.config import get_settings
from backend.services.ocr_engine import OCREngine
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MathRegion(BaseModel):
    """Represents a detected mathematical equation or formula region."""
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="(x0, y0, x1, y1) bounding box of the formula"
    )
    raw_text: str = Field(..., description="Raw OCR text in this region")
    latex: str = Field(..., description="Preserved LaTeX expression")
    is_block: bool = Field(default=False, description="Whether formula is a display/block equation")
    confidence: float = Field(default=0.85, description="Confidence of the formula extraction")
    crop_path: Optional[str] = Field(default=None, description="Path to cropped equation image")


class MathOCRLine(BaseModel):
    """Represents an OCR text line enriched with math detection."""
    text: str = Field(..., description="Original OCR text line")
    text_with_latex: str = Field(..., description="Text line with inline LaTeX ($...$)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: Tuple[float, float, float, float]
    polygon: List[List[float]] = Field(default_factory=list)
    has_math: bool = Field(default=False)
    latex_snippets: List[str] = Field(default_factory=list)


class MathOCRPageResult(BaseModel):
    """OCR and Math recognition result for a single document page."""
    page_number: int = Field(..., ge=1, description="1-based page number")
    raw_text: str = Field(..., description="Raw text from OCR in reading order")
    text_with_latex: str = Field(..., description="Text enriched with LaTeX formulas")
    latex_expressions: List[str] = Field(default_factory=list, description="All detected LaTeX expressions")
    lines: List[MathOCRLine] = Field(default_factory=list)
    math_regions: List[MathRegion] = Field(default_factory=list)
    average_confidence: float = Field(default=0.0)
    image_path: Optional[str] = Field(default=None)


class MathOCRPaperResult(BaseModel):
    """Complete OCR extraction result across all document pages."""
    total_pages: int
    pages: List[MathOCRPageResult]
    all_latex_expressions: List[str] = Field(default_factory=list)
    average_confidence: float = Field(default=0.0)


class MathOCREngine:
    """
    Advanced OCR engine with mathematical expression detection and LaTeX preservation.
    Combines PaddleOCR with Mathpix or rule-based formula extraction.
    """

    # Comprehensive mathematical symbols and notation regexes
    MATH_INDICATOR_PATTERNS = [
        re.compile(r"\b[OT\u0398\u03a9]\s*\(\s*[^)]+\s*\)"),  # O(n), T(n), Theta(n), Omega(n)
        re.compile(r"T\s*\(\s*n\s*\)\s*=\s*.*"),  # Recurrence relations: T(n) = 2T(n/2) + ...
        re.compile(r"[a-zA-Z0-9_]+\s*[\^]\s*(\{[^}]+\}|[a-zA-Z0-9+\-]+)"),  # Exponents: x^2, 2^n
        re.compile(r"[a-zA-Z]\s*_\s*(\{[^}]+\}|[a-zA-Z0-9+\-]+)"),  # Subscripts: a_i, x_n
        re.compile(r"(\b[a-zA-Z]\b|\d+)\s*[/]\s*(\b[a-zA-Z]\b|\d+)"),  # Fractions: n/2, 1/2
        re.compile(r"(\u2264|<=|\u2265|>=|\u2260|!=|\u2208|\u2211|\u220f|\u221a|\u222b|\u221e)"),  # Unicode math
        re.compile(r"\b(log|ln|lg|sin|cos|tan|det|gcd|max|min)\b_?\{?[0-9a-zA-Z]*\}?"),  # Math functions
        re.compile(r"\b[a-z]\s*=\s*[0-9a-zA-Z+\-*/^()]+\s*$"),  # Assignment / equation
        re.compile(r"\$[^$]+\$"),  # Pre-existing inline LaTeX
        re.compile(r"\$\$[^$]+\$\$"),  # Pre-existing block LaTeX
    ]

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        mathpix_app_id: Optional[str] = None,
        mathpix_app_key: Optional[str] = None,
        crops_output_dir: Optional[Path] = None,
    ):
        settings = get_settings()
        self.ocr = ocr_engine or OCREngine()
        self.mathpix_app_id = mathpix_app_id or settings.mathpix_app_id
        self.mathpix_app_key = mathpix_app_key or settings.mathpix_app_key
        
        self.crops_dir = crops_output_dir or (settings.upload_path / "math_crops")
        self.crops_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_pdf(self, pdf_path: str | Path) -> MathOCRPaperResult:
        """
        Accepts a PDF file, extracts each page, performs OCR, detects math,
        and preserves expressions in LaTeX.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        page_results: List[MathOCRPageResult] = []
        all_latex: List[str] = []
        conf_scores: List[float] = []

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                page_num = page_index + 1
                
                # Render high-resolution 300 DPI page image
                pix = page.get_pixmap(dpi=300)
                page_img_path = self.crops_dir.parent / f"page_{page_num}_{pdf_path.stem}.png"
                pix.save(str(page_img_path))

                # Process page image
                page_res = self.extract_from_image(
                    image_input=str(page_img_path),
                    page_number=page_num,
                )
                page_results.append(page_res)
                all_latex.extend(page_res.latex_expressions)
                if page_res.average_confidence > 0:
                    conf_scores.append(page_res.average_confidence)
        finally:
            doc.close()

        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        return MathOCRPaperResult(
            total_pages=len(page_results),
            pages=page_results,
            all_latex_expressions=list(dict.fromkeys(all_latex)),
            average_confidence=round(avg_conf, 4),
        )

    def extract_from_image(
        self,
        image_input: str | Path | np.ndarray,
        page_number: int = 1,
    ) -> MathOCRPageResult:
        """
        Processes a single page image: runs PaddleOCR, preserves bounding boxes,
        identifies math regions, applies LaTeX conversion (Mathpix or local),
        and returns enriched structured output.
        """
        # Run PaddleOCR
        if hasattr(self.ocr, "process_image"):
            ocr_result: OCRResult = self.ocr.process_image(image_input)
        elif hasattr(self.ocr, "extract"):
            ocr_result = self.ocr.extract(image_input)
        else:
            raise AttributeError("OCR engine missing process_image or extract method")
        
        # Load image array for cropping math regions if needed
        img_np: Optional[np.ndarray] = None
        img_path_str: Optional[str] = None
        if isinstance(image_input, (str, Path)):
            img_path_str = str(image_input)
            img_np = cv2.imread(img_path_str)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input

        lines_with_math: List[MathOCRLine] = []
        math_regions: List[MathRegion] = []
        page_latex: List[str] = []

        for idx, line in enumerate(ocr_result.lines):
            raw_line_text = line.text.strip()
            detected_math, converted_line, latex_snippets = self._detect_and_convert_math_in_line(
                text=raw_line_text,
                bbox=line.bbox,
                img_np=img_np,
                line_idx=idx,
                page_number=page_number,
            )

            if detected_math:
                page_latex.extend(latex_snippets)
                # Create math region
                crop_file = self._save_crop(img_np, line.bbox, page_number, idx)
                region = MathRegion(
                    bbox=line.bbox,
                    raw_text=raw_line_text,
                    latex="; ".join(latex_snippets) if latex_snippets else converted_line,
                    is_block=self._is_block_equation(raw_line_text),
                    confidence=line.confidence,
                    crop_path=crop_file,
                )
                math_regions.append(region)

            lines_with_math.append(
                MathOCRLine(
                    text=raw_line_text,
                    text_with_latex=converted_line,
                    confidence=line.confidence,
                    bbox=line.bbox,
                    polygon=line.polygon,
                    has_math=detected_math,
                    latex_snippets=latex_snippets,
                )
            )

        full_raw_text = "\n".join(l.text for l in lines_with_math)
        full_latex_text = "\n".join(l.text_with_latex for l in lines_with_math)

        return MathOCRPageResult(
            page_number=page_number,
            raw_text=full_raw_text,
            text_with_latex=full_latex_text,
            latex_expressions=list(dict.fromkeys(page_latex)),
            lines=lines_with_math,
            math_regions=math_regions,
            average_confidence=ocr_result.average_confidence,
            image_path=img_path_str,
        )

    def _detect_and_convert_math_in_line(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        img_np: Optional[np.ndarray],
        line_idx: int,
        page_number: int,
    ) -> Tuple[bool, str, List[str]]:
        """
        Detects if a text line contains mathematical notation.
        Converts math segments to LaTeX ($...$).
        """
        has_math = any(pattern.search(text) for pattern in self.MATH_INDICATOR_PATTERNS)
        if not has_math:
            return False, text, []

        # If Mathpix is configured and enabled, attempt Mathpix API for high fidelity
        if self.mathpix_app_id and self.mathpix_app_key and img_np is not None:
            mathpix_latex = self._call_mathpix_ocr(img_np, bbox)
            if mathpix_latex:
                return True, f"${mathpix_latex}$", [f"${mathpix_latex}$"]

        # Otherwise, perform robust local formula recognition & LaTeX conversion
        converted_text, latex_snippets = self._convert_formula_to_latex(text)
        return True, converted_text, latex_snippets

    def _convert_formula_to_latex(self, text: str) -> Tuple[str, List[str]]:
        """
        Converts algorithmic and mathematical text tokens into standard LaTeX.
        Preserves existing LaTeX expressions intact.
        """
        extracted_latex: List[str] = []
        out_text = text

        # 1. Preserve Big-O, Omega, Theta notations
        def repl_asymptotic(m):
            raw = m.group(0).strip()
            # Normalize symbol
            sym = raw[0]
            if sym in ("\u0398", "\u03f4"):
                sym = r"\Theta"
            elif sym in ("\u03a9", "\u03c9"):
                sym = r"\Omega"
            inner = raw[raw.find("(") + 1 : raw.rfind(")")].strip()
            # Clean inner expressions: log n, n^2, etc.
            inner = re.sub(r"\blog\b", r"\\log ", inner)
            inner = re.sub(r"(\w+)\^(\d+)", r"\1^{\2}", inner)
            latex = f"${sym}({inner})$"
            extracted_latex.append(latex)
            return latex

        asymp_pat = re.compile(r"\b[OT\u0398\u03a9]\s*\(\s*[^)]+\s*\)")
        out_text = asymp_pat.sub(repl_asymptotic, out_text)

        # 2. Recurrence relations: T(n) = ...
        def repl_recurrence(m):
            raw = m.group(0).strip()
            conv = raw
            conv = re.sub(r"T\s*\(\s*([^)]+)\s*\)", r"T(\1)", conv)
            conv = re.sub(r"(\w+)/(\w+)", r"\\frac{\1}{\2}", conv)
            conv = re.sub(r"\bO\s*\(([^)]+)\)", r"O(\1)", conv)
            latex = f"${conv}$"
            extracted_latex.append(latex)
            return latex

        rec_pat = re.compile(r"T\s*\(\s*n\s*\)\s*=\s*[0-9a-zA-Z+\-*/^()\s]+")
        out_text = rec_pat.sub(repl_recurrence, out_text)

        # 3. Logarithms with explicit bases (e.g., log2(n) -> \log_2(n))
        def repl_log_base(m):
            base = m.group(1)
            arg = m.group(2)
            latex = f"$\\log_{{{base}}}({arg})$"
            extracted_latex.append(latex)
            return latex

        log_pat = re.compile(r"\blog(\d+)\s*\(([^)]+)\)")
        out_text = log_pat.sub(repl_log_base, out_text)

        # 4. Unicode comparisons and operators to LaTeX
        operator_map = {
            "\u2264": r"\le",
            "<=": r"\le",
            "\u2265": r"\ge",
            ">=": r"\ge",
            "\u2260": r"\ne",
            "!=": r"\ne",
            "\u2208": r"\in",
            "\u2211": r"\sum",
            "\u220f": r"\prod",
            "\u221a": r"\sqrt",
            "\u221e": r"\infty",
            "\u03b1": r"\alpha",
            "\u03b2": r"\beta",
            "\u03b8": r"\theta",
            "\u03bb": r"\lambda",
        }
        for uni_op, lat_op in operator_map.items():
            if uni_op in out_text and lat_op not in out_text:
                out_text = out_text.replace(uni_op, f" {lat_op} ")

        # 5. Standard powers: 2^n, x^2, n^k
        def repl_exp(m):
            base = m.group(1)
            exp = m.group(2).strip("{}")
            latex = f"${base}^{{{exp}}}$"
            extracted_latex.append(latex)
            return latex

        exp_pat = re.compile(r"\b([a-zA-Z0-9]+)\^([a-zA-Z0-9+\-]+)\b")
        out_text = exp_pat.sub(repl_exp, out_text)

        # 6. Collect any remaining $...$ expressions
        all_dollars = re.findall(r"\$[^$]+\$", out_text)
        for d in all_dollars:
            if d not in extracted_latex:
                extracted_latex.append(d)

        return out_text, extracted_latex

    def _is_block_equation(self, text: str) -> bool:
        """Determines whether a line appears to be a standalone equation."""
        stripped = text.strip()
        if stripped.startswith("$$") and stripped.endswith("$$"):
            return True
        if re.match(r"^T\s*\(\s*n\s*\)\s*=", stripped):
            return True
        if re.match(r"^[a-zA-Z]\s*=\s*.*[+\-*/]", stripped) and len(stripped.split()) <= 6:
            return True
        return False

    def _save_crop(
        self,
        img: Optional[np.ndarray],
        bbox: Tuple[float, float, float, float],
        page_num: int,
        line_idx: int,
    ) -> Optional[str]:
        """Crops the formula region and saves it to disk."""
        if img is None:
            return None
        try:
            h, w = img.shape[:2]
            x0, y0, x1, y1 = bbox
            # Add small padding
            pad = 4
            px0 = max(0, int(x0) - pad)
            py0 = max(0, int(y0) - pad)
            px1 = min(w, int(x1) + pad)
            py1 = min(h, int(y1) + pad)

            if px1 <= px0 or py1 <= py0:
                return None

            crop = img[py0:py1, px0:px1]
            out_filename = f"crop_p{page_num}_l{line_idx}.png"
            out_path = self.crops_dir / out_filename
            cv2.imwrite(str(out_path), crop)
            return str(out_path)
        except Exception as e:
            logger.warning("crop_save_failed", error=str(e), bbox=bbox)
            return None

    def _call_mathpix_ocr(
        self,
        img: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[str]:
        """Calls Mathpix OCR API for equation recognition if credentials are configured."""
        if not (self.mathpix_app_id and self.mathpix_app_key):
            return None
        try:
            import base64
            h, w = img.shape[:2]
            x0, y0, x1, y1 = [int(v) for v in bbox]
            crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
            _, buf = cv2.imencode(".png", crop)
            b64_str = base64.b64encode(buf).decode("utf-8")
            data_uri = f"data:image/png;base64,{b64_str}"

            req_body = json.dumps({
                "src": data_uri,
                "formats": ["latex_styled", "text"],
                "data_options": {"include_latex": True},
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.mathpix.com/v3/text",
                data=req_body,
                headers={
                    "content-type": "application/json",
                    "app_id": self.mathpix_app_id,
                    "app_key": self.mathpix_app_key,
                },
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("latex_styled") or result.get("text")
        except Exception as e:
            logger.warning("mathpix_call_failed", error=str(e))
            return None
