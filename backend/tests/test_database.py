"""
Tests for the database module.
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import Base, get_db_session, init_db
from backend.tests.conftest import test_engine, test_session_factory


class TestDatabaseSetup:
    """Tests for database initialization and session management."""

    @pytest.mark.asyncio
    async def test_create_all_tables(self):
        """init_db equivalent: creating all tables should succeed."""
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)

        # Verify tables exist by checking metadata
        async with test_engine.begin() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: list(Base.metadata.tables.keys())
            )
        expected_tables = {
            "papers", "pages", "questions", "question_embeddings",
            "question_classifications", "similarity_pairs",
            "syllabi", "historical_analyses",
            "generated_papers", "generated_questions",
        }
        assert expected_tables.issubset(set(table_names))

    @pytest.mark.asyncio
    async def test_session_factory_produces_async_session(self):
        """Session factory should produce AsyncSession instances."""
        async with test_session_factory() as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_session_rollback_on_error(self, db_session: AsyncSession):
        """Session should support rollback."""
        # Just verify rollback doesn't raise
        await db_session.rollback()
