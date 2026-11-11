"""
Image Preprocessing Utility Functions.

Provides helpers for image preprocessing operations used by the OCR engine:
deskewing, denoising, binarization, and contrast enhancement.
"""

from pathlib import Path

import cv2
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Load an image from disk.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a numpy array (BGR).

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be decoded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    logger.debug("image_loaded", path=str(path), shape=image.shape)
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply non-local means denoising to a grayscale image.

    Args:
        image: Grayscale image.
        strength: Denoising strength (higher = more smoothing).

    Returns:
        Denoised image.
    """
    gray = to_grayscale(image)
    denoised = cv2.fastNlMeansDenoising(gray, h=strength)
    logger.debug("image_denoised", strength=strength)
    return denoised


def binarize(
    image: np.ndarray,
    method: str = "adaptive",
    block_size: int = 11,
    constant: int = 2,
) -> np.ndarray:
    """
    Binarize a grayscale image using thresholding.

    Args:
        image: Grayscale image.
        method: "adaptive" for adaptive Gaussian thresholding,
                "otsu" for Otsu's method.
        block_size: Block size for adaptive thresholding (must be odd).
        constant: Constant subtracted from the mean in adaptive method.

    Returns:
        Binarized image.
    """
    gray = to_grayscale(image)

    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, constant,
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    logger.debug("image_binarized", method=method)
    return binary


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Grayscale image.
        clip_limit: Threshold for contrast limiting.
        tile_size: Size of the grid for histogram equalization.

    Returns:
        Contrast-enhanced image.
    """
    gray = to_grayscale(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    logger.debug("contrast_enhanced", clip_limit=clip_limit)
    return enhanced


def deskew(image: np.ndarray) -> np.ndarray:
    """
    Correct skew in a document image.

    Uses Hough line transform to detect the dominant angle
    and rotates the image to correct it.

    Args:
        image: Input image (BGR or grayscale).

    Returns:
        Deskewed image.
    """
    gray = to_grayscale(image)
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10,
    )

    if lines is None:
        logger.debug("deskew_no_lines_detected")
        return image

    # Calculate the dominant angle
    angles = []
    for line in lines:
        coords = line.flatten()
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (within ±15°)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        logger.debug("deskew_no_suitable_angles")
        return image

    median_angle = np.median(angles)
    logger.debug("deskew_angle", angle=float(median_angle))

    # Rotate image to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Apply the full preprocessing pipeline for OCR.

    Steps: deskew → denoise → enhance contrast → binarize.

    Args:
        image: Input image (BGR).

    Returns:
        Preprocessed image ready for OCR.
    """
    result = deskew(image)
    result = denoise(result)
    result = enhance_contrast(result)
    result = binarize(result, method="adaptive")
    logger.info("image_preprocessing_complete")
    return result
