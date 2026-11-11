"""
Module 2: Multimodal OCR Correction Service with Google Gemini.

Enforces:
1. Passes original image crops, raw OCR output, and mathematical OCR output to Gemini.
2. Corrects spelling, spacing, punctuation, and character confusion (e.g. 'rn' vs 'm').
3. Preserves all mathematical expressions, superscripts, subscripts, symbols, and pseudocode.
4. Never invents missing content or hallucinates questions.
5. Never modifies the mathematical meaning or asymptotic bounds of any expression.
6. Flags unconfident or ambiguous expressions for manual verification.
7. Maintains both original OCR text and corrected text side-by-side.
"""

import json
import re
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class OCRCorrectionResult(BaseModel):
    """Result of OCR correction on a block of text or page."""
    raw_ocr_text: str = Field(..., description="Original raw OCR text before correction")
    corrected_text: str = Field(..., description="Polished text with errors resolved")
    latex_expressions: List[str] = Field(default_factory=list, description="Preserved LaTeX formulas")
    is_flagged_for_verification: bool = Field(
        default=False, description="Flagged when expressions cannot be recovered confidently"
    )
    flag_reasons: List[str] = Field(default_factory=list, description="Explanations for verification flags")
    corrections_made: List[str] = Field(default_factory=list, description="Summary of corrected errors")


class GeminiOCRCorrector:
    """
    Multimodal OCR correction engine utilizing Google Gemini 2.5 Flash
    with rule-based fallback and strict hallucination prevention.
    """

    CORRECTION_SYSTEM_PROMPT = """You are an expert OCR correction system specialized in Computer Science and Data Structures & Algorithms exam papers.

STRICT INSTRUCTIONS:
1. Fix spelling errors, broken word spacing (e.g., "al gorithm" -> "algorithm"), punctuation, and character recognition errors (e.g., "rn" read as "m", "1" vs "l").
2. PRESERVE ALL MATHEMATICAL AND ALGORITHMIC EXPRESSIONS:
   - Keep LaTeX expressions inside $...$ or $$...$$ intact.
   - Do not change variable names, constants, or complexity bounds (e.g., O(n log n), T(n)=2T(n/2)+n).
   - Preserve pseudocode indentation, keywords, and operators.
3. NEVER INVENT MISSING CONTENT:
   - Do not complete truncated sentences or hallucinate missing question parts.
   - Only correct what is present in the OCR / image.
4. NEVER MODIFY THE MATHEMATICAL MEANING of an expression.
5. IF UNCONFIDENT: If an equation, symbol, or word is too blurred/corrupted to be recovered with certainty, DO NOT GUESS. Keep the original text and set "is_flagged_for_verification" to true with a descriptive "flag_reasons" entry.
6. Maintain exact equivalence between raw text and corrected text.

Return your response as a valid JSON object matching this schema:
{
  "corrected_text": "...",
  "latex_expressions": ["$...$", ...],
  "is_flagged_for_verification": false,
  "flag_reasons": [],
  "corrections_made": ["Fixed spacing in 'al gorithm'", ...]
}
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model_name or settings.gemini_model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("gemini_client_init_failed", error=str(e))

    def correct_ocr_text(
        self,
        raw_ocr_text: str,
        latex_expressions: Optional[List[str]] = None,
        image_path: Optional[str | Path] = None,
    ) -> OCRCorrectionResult:
        """
        Corrects raw OCR text using Gemini multimodal model if available,
        or deterministic rule-based corrector if offline or API unavailable.
        """
        latex_list = latex_expressions or []
        
        # 1. Attempt Gemini correction
        if self.client and raw_ocr_text.strip():
            try:
                return self._call_gemini_correction(
                    raw_text=raw_ocr_text,
                    latex_list=latex_list,
                    image_path=image_path,
                )
            except Exception as err:
                logger.warning(
                    "gemini_correction_failed_fallback_to_rules",
                    error=str(err),
                )

        # 2. Deterministic offline fallback
        return self._rule_based_correction(raw_ocr_text, latex_list)

    def _call_gemini_correction(
        self,
        raw_text: str,
        latex_list: List[str],
        image_path: Optional[str | Path],
    ) -> OCRCorrectionResult:
        """Invokes Gemini 2.5 Flash with multimodal inputs and structured JSON response."""
        contents = []

        # Attach image if available
        if image_path and Path(image_path).exists():
            try:
                from google.genai import types
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                mime = "image/png" if str(image_path).endswith(".png") else "image/jpeg"
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            except Exception as e:
                logger.warning("gemini_image_attach_failed", error=str(e))

        prompt_text = (
            f"Here is the raw OCR text extracted from the document:\n\n"
            f"```\n{raw_text}\n```\n\n"
            f"Detected LaTeX expressions:\n{json.dumps(latex_list, indent=2)}\n\n"
            f"Please clean and correct the OCR text following all instructions."
        )
        contents.append(prompt_text)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={
                "system_instruction": self.CORRECTION_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "temperature": 0.1,  # Low temperature for strict adherence to facts
            },
        )

        response_text = response.text.strip()
        data = json.loads(response_text)

        # Merge latex lists
        gemini_latex = data.get("latex_expressions", [])
        combined_latex = list(dict.fromkeys(latex_list + gemini_latex))

        return OCRCorrectionResult(
            raw_ocr_text=raw_text,
            corrected_text=data.get("corrected_text", raw_text),
            latex_expressions=combined_latex,
            is_flagged_for_verification=bool(data.get("is_flagged_for_verification", False)),
            flag_reasons=data.get("flag_reasons", []),
            corrections_made=data.get("corrections_made", []),
        )

    def _rule_based_correction(
        self,
        raw_text: str,
        latex_list: List[str],
    ) -> OCRCorrectionResult:
        """
        High-precision rule-based corrector.
        Handles common OCR artifacts, whitespace spacing, punctuation spacing,
        and flags malformed expressions.
        """
        corrections: List[str] = []
        flag_reasons: List[str] = []
        is_flagged = False

        corrected = raw_text

        # 1. Broken word joiner (e.g., "al gorithm" -> "algorithm")
        common_cs_splits = [
            (r"\bal\s+gorithm\b", "algorithm"),
            (r"\bdata\s+struc\s*ture\b", "data structure"),
            (r"\blinked\s+list\b", "linked list"),
            (r"\bbinary\s+search\b", "binary search"),
            (r"\bcom\s*plex\s*ity\b", "complexity"),
            (r"\bre\s*cur\s*rence\b", "recurrence"),
            (r"\bad\s*ja\s*cency\b", "adjacency"),
            (r"\btra\s*ver\s*sal\b", "traversal"),
            (r"\bfunc\s*tion\b", "function"),
            (r"\bpseudo\s*code\b", "pseudocode"),
        ]
        for pattern, replacement in common_cs_splits:
            if re.search(pattern, corrected, re.IGNORECASE):
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                corrections.append(f"Fixed split word: {replacement}")

        # 2. Common character confusion in English text
        word_typos = [
            (r"\bteh\b", "the"),
            (r"\brecursionn\b", "recursion"),
            (r"\bquene\b", "queue"),
            (r"\bvertice\b", "vertex"),
        ]
        for typo, fix in word_typos:
            if re.search(typo, corrected, re.IGNORECASE):
                corrected = re.sub(typo, fix, corrected, flags=re.IGNORECASE)
                corrections.append(f"Fixed typo: {fix}")

        # 3. Punctuation spacing normalization (don't alter LaTeX dollars)
        corrected = re.sub(r"\s+([,.:;?!])", r"\1", corrected)
        corrected = re.sub(r"([,.:;?!])(?=[A-Za-z])", r"\1 ", corrected)

        # 4. Check for ambiguous or corrupted mathematical expressions
        # Unmatched parentheses/brackets
        open_parens = corrected.count("(")
        close_parens = corrected.count(")")
        if open_parens != close_parens:
            is_flagged = True
            flag_reasons.append(
                f"Unmatched parentheses detected: {open_parens} '(' vs {close_parens} ')'"
            )

        open_brackets = corrected.count("[")
        close_brackets = corrected.count("]")
        if open_brackets != close_brackets:
            is_flagged = True
            flag_reasons.append(
                f"Unmatched square brackets detected: {open_brackets} '[' vs {close_brackets} ']'"
            )

        # Unbalanced LaTeX dollar signs
        dollar_count = corrected.count("$")
        if dollar_count % 2 != 0:
            is_flagged = True
            flag_reasons.append(
                f"Odd count of LaTeX dollar signs ($): {dollar_count}. Expression may be malformed."
            )

        # Corrupted OCR tokens (e.g. unrecognizable replacement characters or illegal control characters)
        if "\ufffd" in corrected or any((ord(c) < 32 and c not in "\r\n\t") for c in corrected):
            is_flagged = True
            flag_reasons.append("Unrecognized OCR characters or corrupted control characters detected.")

        return OCRCorrectionResult(
            raw_ocr_text=raw_text,
            corrected_text=corrected,
            latex_expressions=latex_list,
            is_flagged_for_verification=is_flagged,
            flag_reasons=flag_reasons,
            corrections_made=corrections,
        )
