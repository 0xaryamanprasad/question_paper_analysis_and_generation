"""
File I/O Utility Functions.

Provides helpers for file operations: saving uploads, reading files,
generating unique filenames, and cleaning up temporary files.
"""

import uuid
from pathlib import Path

import aiofiles

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename preserving the original extension.

    Args:
        original_filename: The original name of the uploaded file.

    Returns:
        A unique filename with UUID prefix.
    """
    extension = Path(original_filename).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    return unique_name


async def save_upload_file(content: bytes, original_filename: str) -> Path:
    """
    Save uploaded file content to the uploads directory.

    Args:
        content: Raw file bytes.
        original_filename: The original name of the uploaded file.

    Returns:
        Path to the saved file.

    Raises:
        IOError: If the file cannot be written.
    """
    settings = get_settings()
    unique_name = generate_unique_filename(original_filename)
    file_path = settings.upload_path / unique_name

    try:
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        logger.info(
            "file_saved",
            original_filename=original_filename,
            saved_path=str(file_path),
            size_bytes=len(content),
        )
        return file_path
    except IOError as e:
        logger.error(
            "file_save_failed",
            original_filename=original_filename,
            error=str(e),
        )
        raise


async def read_file_bytes(file_path: Path) -> bytes:
    """
    Read file contents as bytes.

    Args:
        file_path: Path to the file to read.

    Returns:
        File contents as bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiofiles.open(file_path, "rb") as f:
        return await f.read()


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing path separators and unsafe characters."""
    import re
    clean = Path(filename).name
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    return clean or "unnamed_file"


def validate_file_extension(filename: str, allowed_extensions: set[str]) -> bool:
    """Check whether filename extension matches an allowed set."""
    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def cleanup_file(file_path: Path) -> bool:
    """
    Delete a file if it exists.

    Args:
        file_path: Path to the file to delete.

    Returns:
        True if the file was deleted, False if it didn't exist.
    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.info("file_deleted", path=str(file_path))
            return True
        return False
    except OSError as e:
        logger.error("file_delete_failed", path=str(file_path), error=str(e))
        return False
