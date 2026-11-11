"""
Shared Test Fixtures.

Provides reusable test fixtures for database sessions,
FastAPI test client, and sample data objects.
"""

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.database import Base, get_db, get_db_session
from backend.main import app


# ---------------------------------------------------------------------------
# Event loop fixture – one loop for the entire test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    """Create a single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# In-memory SQLite engine for fast, isolated tests
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_qpaper.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
test_session_factory = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False,
)


@pytest_asyncio.fixture(autouse=True)
async def setup_database():
    """Create all tables before each test and drop them after."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide a database session for tests."""
    async with test_session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP test client with the test database session.
    """
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Sample data factories
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_paper_data() -> dict:
    """Return sample paper metadata for testing."""
    return {
        "subject": "Data Structures and Algorithms",
        "year": "2023",
        "semester": "Fall",
        "exam_type": "End-Term",
        "original_filename": "DS_2023_EndTerm.pdf",
        "file_path": "/tmp/test_uploads/DS_2023_EndTerm.pdf",
        "total_pages": 4,
        "extraction_status": "pending",
    }


@pytest.fixture
def sample_question_data() -> dict:
    """Return sample question data for testing."""
    return {
        "question_number": "1",
        "raw_text": "Define the term Abstract Data Type (ADT). Give two examples.",
        "cleaned_text": "define term abstract data type adt give two examples",
        "marks": 5,
        "section": "A",
        "question_type": "short",
        "sub_parts": None,
    }


@pytest.fixture
def sample_digital_pdf(tmp_path: Path) -> Path:
    """Create a multi-page digital PDF with known text content."""
    import fitz
    pdf_path = tmp_path / "digital_paper.pdf"
    doc = fitz.open()

    p1 = doc.new_page(width=595, height=842)
    p1.insert_text(
        (50, 72),
        "UNIVERSITY EXAMINATION 2023\n"
        "DEPARTMENT OF COMPUTER SCIENCE\n"
        "COURSE: DATA STRUCTURES AND ALGORITHMS (CS201)\n\n"
        "SECTION A (Answer all questions) [20 Marks]\n\n"
        "1. Define the term Abstract Data Type (ADT). Give two examples. [5 Marks]\n"
        "2. Differentiate between Singly Linked List and Doubly Linked List. [5 Marks]\n",
        fontsize=11,
    )

    p2 = doc.new_page(width=595, height=842)
    p2.insert_text(
        (50, 72),
        "SECTION B (Answer any two questions) [20 Marks]\n\n"
        "3. Explain the algorithm for Inorder, Preorder and Postorder traversal of a Binary Tree. [10 Marks]\n"
        "4. Describe Dijkstra's shortest path algorithm with an example graph. [10 Marks]\n",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_questions_for_similarity() -> list[dict]:
    """Return pairs of similar and dissimilar questions for testing."""
    return [
        {
            "text": "Explain the concept of a binary search tree.",
            "expected_similar_to": 1,
        },
        {
            "text": "Describe BST and its properties with an example.",
            "expected_similar_to": 0,
        },
        {
            "text": "What is the time complexity of quicksort algorithm?",
            "expected_similar_to": None,
        },
    ]
