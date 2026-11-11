"""
Tests for utility modules: text_utils, file_utils, image_utils.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.utils.text_utils import (
    normalize_unicode,
    clean_whitespace,
    fix_ocr_artifacts,
    normalize_question_text,
    extract_marks_from_text,
)
from backend.utils.image_utils import (
    to_grayscale,
    denoise,
    binarize,
    enhance_contrast,
    deskew,
    preprocess_for_ocr,
)
from backend.utils.file_utils import (
    generate_unique_filename,
    save_upload_file,
    read_file_bytes,
    cleanup_file,
)


class TestTextUtils:
    """Tests for text utility functions."""

    def test_normalize_unicode_fancy_quotes(self):
        """Should replace smart quotes with ASCII equivalents."""
        text = "\u201CHello\u201D \u2018world\u2019"
        result = normalize_unicode(text)
        assert result == '"Hello" \'world\''

    def test_normalize_unicode_dashes(self):
        """Should replace en-dash and em-dash with hyphens."""
        text = "A\u2013B\u2014C"
        result = normalize_unicode(text)
        assert result == "A-B-C"

    def test_normalize_unicode_nbsp(self):
        """Should replace non-breaking spaces with regular spaces."""
        text = "hello\u00A0world"
        result = normalize_unicode(text)
        assert result == "hello world"

    def test_clean_whitespace(self):
        """Should collapse multiple spaces and newlines."""
        text = "  hello    world  \n\n\n\n  test  "
        result = clean_whitespace(text)
        assert result == "hello world\n\ntest"

    def test_fix_ocr_artifacts_ligatures(self):
        """Should fix ligature characters."""
        text = "ﬁnd the ﬂow of ﬀects"
        result = fix_ocr_artifacts(text)
        assert "fi" in result
        assert "fl" in result
        assert "ff" in result

    def test_fix_ocr_artifacts_known_words(self):
        """Should fix known OCR misreadings."""
        assert "computer" in fix_ocr_artifacts("cornputer")
        assert "memory" in fix_ocr_artifacts("rnernory")
        assert "algorithm" in fix_ocr_artifacts("algorithrn")

    def test_fix_ocr_pipe_in_words(self):
        """Should replace pipe with 'l' when between letters."""
        text = "he|lo wor|d"
        result = fix_ocr_artifacts(text)
        assert result == "hello world"

    def test_normalize_question_text_full_pipeline(self):
        """Full normalization pipeline should clean text."""
        text = "  \u201CDeﬁne\u201D the cornputer\u00A0archithrn   "
        result = normalize_question_text(text)
        assert "\u201C" not in result
        assert "\u00A0" not in result
        assert "fi" in result  # ligature fixed

    def test_extract_marks_bracket(self):
        """Should extract marks from [N marks] format."""
        assert extract_marks_from_text("Explain BST. [5 marks]") == 5
        assert extract_marks_from_text("Define ADT [10 Marks]") == 10

    def test_extract_marks_parenthesis(self):
        """Should extract marks from (NM) format."""
        assert extract_marks_from_text("Explain BST. (5M)") == 5
        assert extract_marks_from_text("Define ADT (10 marks)") == 10

    def test_extract_marks_inline(self):
        """Should extract marks from inline format."""
        assert extract_marks_from_text("Explain BST. 5 marks") == 5

    def test_extract_marks_none(self):
        """Should return None when no marks found."""
        assert extract_marks_from_text("Explain BST.") is None

    def test_extract_marks_rejects_invalid(self):
        """Should reject marks outside 1-50 range."""
        assert extract_marks_from_text("Page 200 marks") is None


class TestImageUtils:
    """Tests for image utility functions."""

    @pytest.fixture
    def sample_bgr_image(self):
        """Create a sample BGR image for testing."""
        return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_gray_image(self):
        """Create a sample grayscale image for testing."""
        return np.random.randint(0, 255, (100, 200), dtype=np.uint8)

    def test_to_grayscale_from_bgr(self, sample_bgr_image):
        """Should convert BGR to grayscale."""
        gray = to_grayscale(sample_bgr_image)
        assert len(gray.shape) == 2
        assert gray.shape == (100, 200)

    def test_to_grayscale_already_gray(self, sample_gray_image):
        """Should return same image if already grayscale."""
        gray = to_grayscale(sample_gray_image)
        assert np.array_equal(gray, sample_gray_image)

    def test_denoise(self, sample_bgr_image):
        """Denoising should return image of same shape."""
        result = denoise(sample_bgr_image)
        assert result.shape == (100, 200)

    def test_binarize_adaptive(self, sample_gray_image):
        """Adaptive binarization should produce binary image."""
        result = binarize(sample_gray_image, method="adaptive")
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_binarize_otsu(self, sample_gray_image):
        """Otsu binarization should produce binary image."""
        result = binarize(sample_gray_image, method="otsu")
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_binarize_invalid_method(self, sample_gray_image):
        """Should raise ValueError for unknown method."""
        with pytest.raises(ValueError, match="Unknown binarization method"):
            binarize(sample_gray_image, method="unknown")

    def test_enhance_contrast(self, sample_gray_image):
        """Contrast enhancement should return image of same shape."""
        result = enhance_contrast(sample_gray_image)
        assert result.shape == sample_gray_image.shape

    def test_deskew(self, sample_bgr_image):
        """Deskew should return image of same shape."""
        result = deskew(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape

    def test_preprocess_for_ocr(self, sample_bgr_image):
        """Full preprocessing pipeline should return 2D image."""
        result = preprocess_for_ocr(sample_bgr_image)
        assert len(result.shape) == 2

    def test_load_image_not_found(self):
        """Should raise FileNotFoundError for missing image."""
        from backend.utils.image_utils import load_image
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.png")


class TestFileUtils:
    """Tests for file utility functions."""

    def test_generate_unique_filename(self):
        """Should generate unique filename with correct extension."""
        name = generate_unique_filename("test_paper.pdf")
        assert name.endswith(".pdf")
        assert len(name) == 36  # 32 hex chars + '.pdf'

    def test_generate_unique_filename_different_extension(self):
        """Should preserve the original extension."""
        name = generate_unique_filename("image.PNG")
        assert name.endswith(".png")

    @pytest.mark.asyncio
    async def test_save_and_read_file(self, tmp_path, monkeypatch):
        """Should save a file and read it back."""
        monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))

        # Need to clear settings cache since we changed env
        from backend.config import get_settings
        get_settings.cache_clear()

        content = b"Hello, this is test PDF content"
        saved_path = await save_upload_file(content, "test.pdf")

        assert saved_path.exists()
        read_content = await read_file_bytes(saved_path)
        assert read_content == content

        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            await read_file_bytes(Path("/nonexistent/file.pdf"))

    def test_cleanup_file(self, tmp_path):
        """Should delete file and return True."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        assert cleanup_file(test_file) is True
        assert not test_file.exists()

    def test_cleanup_nonexistent_file(self):
        """Should return False for missing file."""
        assert cleanup_file(Path("/nonexistent/file.txt")) is False
