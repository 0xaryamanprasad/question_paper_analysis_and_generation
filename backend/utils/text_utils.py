"""
Text Cleaning Utility Functions.

Provides helpers for text normalization, OCR artifact correction,
and other text-processing utilities used across multiple modules.
"""

import re
import unicodedata

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Common OCR misreadings and their corrections
OCR_CORRECTIONS: dict[str, str] = {
    "|": "l",
    "0": "O",  # Only applied in word context, not numbers
    "rn": "m",
    "vv": "w",
    "cI": "d",
    "Iist": "list",
    "cornputer": "computer",
    "rnernory": "memory",
}

# Patterns for cleaning
MULTIPLE_SPACES = re.compile(r"[^\S\n]+")
MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
LEADING_TRAILING_WHITESPACE = re.compile(r"^[ \t]+|[ \t]+$", re.MULTILINE)


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters to their canonical forms.

    Converts fancy quotes, dashes, and other Unicode variants
    to their ASCII equivalents where possible.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    # Normalize to NFKC form (compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", text)

    # Replace common Unicode variants with ASCII equivalents
    replacements = {
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201C": '"',  # Left double quotation mark
        "\u201D": '"',  # Right double quotation mark
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "...",  # Ellipsis
        "\u00A0": " ",  # Non-breaking space
        "\u200B": "",   # Zero-width space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def clean_whitespace(text: str) -> str:
    """
    Normalize whitespace: collapse multiple spaces and newlines.

    Args:
        text: Input text.

    Returns:
        Text with normalized whitespace.
    """
    text = LEADING_TRAILING_WHITESPACE.sub("", text)
    text = MULTIPLE_SPACES.sub(" ", text)
    text = MULTIPLE_NEWLINES.sub("\n\n", text)
    return text.strip()


def fix_ocr_artifacts(text: str) -> str:
    """
    Attempt to fix common OCR recognition errors.

    This is a conservative correction pass. Only well-known
    substitutions are applied to avoid introducing new errors.

    Args:
        text: OCR-produced text.

    Returns:
        Text with common OCR artifacts corrected.
    """
    # Fix ligature issues
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("ﬀ", "ff")
    text = text.replace("ﬃ", "ffi")
    text = text.replace("ﬄ", "ffl")

    # Fix common character confusions (conservative)
    # Replace pipe with lowercase L only when surrounded by letters
    text = re.sub(r"(?<=[a-zA-Z])\|(?=[a-zA-Z])", "l", text)

    # Fix 'rn' → 'm' only in known words
    known_rn_fixes = {
        "cornputer": "computer",
        "rnernory": "memory",
        "inforrnation": "information",
        "cornrnunication": "communication",
        "prograrnrning": "programming",
        "algorithrn": "algorithm",
        "environrnent": "environment",
        "irnplernentation": "implementation",
        "requirernent": "requirement",
        "rnanagernent": "management",
    }
    for wrong, correct in known_rn_fixes.items():
        text = text.replace(wrong, correct)

    return text


def normalize_question_text(text: str) -> str:
    """
    Apply the full text normalization pipeline for a question.

    Steps: Unicode normalization → OCR fix → de-hyphenation → whitespace cleanup.

    Args:
        text: Raw question text.

    Returns:
        Normalized question text.
    """
    text = normalize_unicode(text)
    text = fix_ocr_artifacts(text)
    # Rejoin words broken across line breaks by trailing hyphens
    text = re.sub(r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)", r"\1\2", text)
    text = clean_whitespace(text)
    return text


def extract_marks_from_text(text: str) -> int | None:
    """
    Extract marks/points value from question text.

    Looks for patterns like "[5 marks]", "(5M)", "5 Marks", etc.

    Args:
        text: Question text.

    Returns:
        Extracted marks as integer, or None if not found.
    """
    patterns = [
        r"\[(\d+)\s*(?:marks?|pts?|points?)\]",
        r"\((\d+)\s*(?:marks?|pts?|points?|M)\)",
        r"(\d+)\s*(?:marks?|Marks?|MARKS?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            marks = int(match.group(1))
            # Sanity check: marks should be between 1 and 50
            if 1 <= marks <= 50:
                return marks
    return None
