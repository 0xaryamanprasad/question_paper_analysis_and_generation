"""
Tests for SQLAlchemy ORM models.
"""

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import (
    Paper, Page, Question, QuestionEmbedding, QuestionClassification,
    SimilarityPair, Syllabus, HistoricalAnalysis,
    GeneratedPaper, GeneratedQuestion,
)


class TestPaperModel:
    """Tests for the Paper and Page models."""

    @pytest.mark.asyncio
    async def test_create_paper(self, db_session: AsyncSession, sample_paper_data):
        """Should create a paper record with all fields."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        assert paper.id is not None
        assert paper.subject == "Data Structures and Algorithms"
        assert paper.year == "2023"
        assert paper.extraction_status == "pending"
        assert paper.created_at is not None

    @pytest.mark.asyncio
    async def test_create_page_with_paper(self, db_session: AsyncSession, sample_paper_data):
        """Should create a page linked to a paper."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        page = Page(
            paper_id=paper.id,
            page_number=1,
            page_type="digital",
            raw_text="Sample extracted text from page 1.",
        )
        db_session.add(page)
        await db_session.flush()

        assert page.id is not None
        assert page.paper_id == paper.id
        assert page.page_type == "digital"

    @pytest.mark.asyncio
    async def test_paper_page_relationship(self, db_session: AsyncSession, sample_paper_data):
        """Paper should have a pages relationship."""
        paper = Paper(**sample_paper_data)
        page1 = Page(paper_id=None, page_number=1, page_type="digital", raw_text="Page 1")
        page2 = Page(paper_id=None, page_number=2, page_type="scanned", image_path="/img/p2.png")
        paper.pages = [page1, page2]

        db_session.add(paper)
        await db_session.flush()

        result = await db_session.execute(
            select(Page).where(Page.paper_id == paper.id)
        )
        pages = result.scalars().all()
        assert len(pages) == 2


class TestQuestionModel:
    """Tests for Question and related models."""

    @pytest.mark.asyncio
    async def test_create_question(
        self, db_session: AsyncSession, sample_paper_data, sample_question_data
    ):
        """Should create a question linked to a paper."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        question = Question(paper_id=paper.id, **sample_question_data)
        db_session.add(question)
        await db_session.flush()

        assert question.id is not None
        assert question.paper_id == paper.id
        assert question.marks == 5

    @pytest.mark.asyncio
    async def test_create_question_embedding(
        self, db_session: AsyncSession, sample_paper_data, sample_question_data
    ):
        """Should create an embedding record for a question."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        question = Question(paper_id=paper.id, **sample_question_data)
        db_session.add(question)
        await db_session.flush()

        # Simulate a 384-dim embedding as bytes
        import numpy as np
        fake_embedding = np.random.randn(384).astype(np.float32).tobytes()

        embedding = QuestionEmbedding(
            question_id=question.id,
            embedding_vector=fake_embedding,
            model_name="all-MiniLM-L6-v2",
        )
        db_session.add(embedding)
        await db_session.flush()

        assert embedding.id is not None
        assert embedding.question_id == question.id
        assert len(embedding.embedding_vector) == 384 * 4  # float32 = 4 bytes each

    @pytest.mark.asyncio
    async def test_create_question_classification(
        self, db_session: AsyncSession, sample_paper_data, sample_question_data
    ):
        """Should create a classification record for a question."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        question = Question(paper_id=paper.id, **sample_question_data)
        db_session.add(question)
        await db_session.flush()

        classification = QuestionClassification(
            question_id=question.id,
            unit="Unit 1",
            topic="Abstract Data Types",
            topic_confidence=0.92,
            blooms_level="Remember",
            blooms_confidence=0.88,
            keywords=["ADT", "abstract", "data type"],
        )
        db_session.add(classification)
        await db_session.flush()

        assert classification.id is not None
        assert classification.blooms_level == "Remember"
        assert classification.topic_confidence == 0.92

    @pytest.mark.asyncio
    async def test_create_similarity_pair(
        self, db_session: AsyncSession, sample_paper_data
    ):
        """Should create a similarity pair between two questions."""
        paper = Paper(**sample_paper_data)
        db_session.add(paper)
        await db_session.flush()

        q1 = Question(
            paper_id=paper.id, raw_text="Explain BST.", marks=5, question_number="1",
        )
        q2 = Question(
            paper_id=paper.id, raw_text="Describe binary search tree.", marks=5,
            question_number="2",
        )
        db_session.add_all([q1, q2])
        await db_session.flush()

        pair = SimilarityPair(
            question_a_id=q1.id,
            question_b_id=q2.id,
            similarity_score=0.91,
            relation_type="near_duplicate",
        )
        db_session.add(pair)
        await db_session.flush()

        assert pair.similarity_score == 0.91
        assert pair.relation_type == "near_duplicate"


class TestAnalysisModels:
    """Tests for Syllabus and HistoricalAnalysis models."""

    @pytest.mark.asyncio
    async def test_create_syllabus(self, db_session: AsyncSession):
        """Should create a syllabus with JSONB units."""
        syllabus = Syllabus(
            subject="Data Structures",
            course_code="CS201",
            units=[
                {
                    "unit_number": 1,
                    "unit_name": "Introduction",
                    "topics": ["Arrays", "Linked Lists"],
                    "keywords": ["array", "linked list"],
                }
            ],
        )
        db_session.add(syllabus)
        await db_session.flush()

        assert syllabus.id is not None
        assert len(syllabus.units) == 1
        assert syllabus.units[0]["unit_name"] == "Introduction"

    @pytest.mark.asyncio
    async def test_create_historical_analysis(self, db_session: AsyncSession):
        """Should create a historical analysis with JSONB data."""
        analysis = HistoricalAnalysis(
            subject="Data Structures",
            topic_frequency={"Arrays": {"2021": 3, "2022": 5}},
            marks_distribution={"Unit 1": 25.5, "Unit 2": 30.0},
            blooms_distribution={"Remember": 20, "Understand": 30, "Apply": 50},
            recurrence_data={"q1_hash": {"count": 3, "years": ["2021", "2022", "2023"]}},
        )
        db_session.add(analysis)
        await db_session.flush()

        assert analysis.id is not None
        assert analysis.topic_frequency["Arrays"]["2022"] == 5


class TestGeneratedPaperModel:
    """Tests for GeneratedPaper and GeneratedQuestion models."""

    @pytest.mark.asyncio
    async def test_create_generated_paper(self, db_session: AsyncSession):
        """Should create a generated paper with config."""
        gen_paper = GeneratedPaper(
            subject="Data Structures",
            title="End-Term Examination 2024",
            generation_config={
                "total_marks": 100,
                "sections": [{"name": "A", "marks": 20}],
            },
            status="draft",
        )
        db_session.add(gen_paper)
        await db_session.flush()

        assert gen_paper.id is not None
        assert gen_paper.status == "draft"

    @pytest.mark.asyncio
    async def test_create_generated_question(self, db_session: AsyncSession):
        """Should create generated questions linked to a paper."""
        gen_paper = GeneratedPaper(
            subject="Data Structures",
            title="Test Paper",
            generation_config={"total_marks": 100},
            status="draft",
        )
        db_session.add(gen_paper)
        await db_session.flush()

        gen_q = GeneratedQuestion(
            generated_paper_id=gen_paper.id,
            question_text="Define an array. State its advantages.",
            marks=5,
            section="A",
            unit="Unit 1",
            blooms_level="Remember",
            source="generated",
        )
        db_session.add(gen_q)
        await db_session.flush()

        assert gen_q.id is not None
        assert gen_q.generated_paper_id == gen_paper.id
        assert gen_q.source == "generated"

    @pytest.mark.asyncio
    async def test_generated_paper_question_relationship(self, db_session: AsyncSession):
        """Generated paper should cascade delete its questions."""
        gen_paper = GeneratedPaper(
            subject="Data Structures",
            title="Test Paper",
            generation_config={"total_marks": 100},
            status="draft",
        )
        gen_q1 = GeneratedQuestion(
            question_text="Q1 text", marks=5, source="generated",
        )
        gen_q2 = GeneratedQuestion(
            question_text="Q2 text", marks=10, source="pool",
        )
        gen_paper.generated_questions = [gen_q1, gen_q2]

        db_session.add(gen_paper)
        await db_session.flush()

        result = await db_session.execute(
            select(GeneratedQuestion).where(
                GeneratedQuestion.generated_paper_id == gen_paper.id
            )
        )
        questions = result.scalars().all()
        assert len(questions) == 2
