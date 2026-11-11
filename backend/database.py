"""
Database Connection and Session Management.

Provides the async SQLAlchemy engine, session factory, and base model class.
Uses asyncpg for async operations with PostgreSQL.
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


def create_engine():
    """
    Create the async SQLAlchemy engine.

    Returns:
        An AsyncEngine instance configured from settings.
    """
    settings = get_settings()
    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    logger.info("database_engine_created", url=settings.database_url.split("@")[-1])
    return engine


# Global engine and session factory
engine = create_engine()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db_session() -> AsyncSession:
    """
    Dependency that provides an async database session.

    Yields an AsyncSession and ensures it is closed after use.
    Intended for use with FastAPI's Depends().

    Yields:
        An AsyncSession instance.
    """
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Convenience alias for FastAPI Depends
get_db = get_db_session


async def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Attempts connection to configured DATABASE_URL (PostgreSQL).
    If connection is unavailable (e.g. PostgreSQL container not started yet),
    automatically activates local SQLite fallback (qpaper_local.db) for reliable persistence.
    """
    global engine, async_session_factory
    settings = get_settings()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_tables_created", dialect="postgresql")
        return
    except (OSError, ConnectionRefusedError, Exception) as e:
        logger.warning(
            "postgresql_unavailable_falling_back_to_sqlite",
            error=str(e),
            note="PostgreSQL is not reachable at localhost:5432. Activating local SQLite database (qpaper_local.db).",
        )

    # Initialize local SQLite fallback
    try:
        sqlite_engine = create_async_engine(
            "sqlite+aiosqlite:///./qpaper_local.db",
            echo=settings.debug,
        )
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Ensure newly added Question columns exist in existing SQLite databases
            def _migrate_sqlite_columns(sync_conn):
                from sqlalchemy import text
                res = sync_conn.execute(text("PRAGMA table_info(questions);")).fetchall()
                existing_cols = {r[1] for r in res}
                new_cols = [
                    ("subquestion_number", "VARCHAR(20)"),
                    ("latex_expressions", "JSON"),
                    ("dsa_topic", "VARCHAR(100)"),
                    ("dsa_subtopic", "VARCHAR(100)"),
                    ("dsa_secondary_topic", "VARCHAR(100)"),
                    ("dsa_confidence", "FLOAT"),
                    ("ocr_confidence", "FLOAT"),
                    ("is_flagged_for_verification", "BOOLEAN DEFAULT 0"),
                    ("verification_flags", "JSON"),
                    ("crop_image_path", "VARCHAR(500)"),
                    ("original_page_number", "INTEGER"),
                ]
                for col_name, col_type in new_cols:
                    if col_name not in existing_cols:
                        sync_conn.execute(text(f"ALTER TABLE questions ADD COLUMN {col_name} {col_type};"))
            await conn.run_sync(_migrate_sqlite_columns)

        engine = sqlite_engine
        async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("sqlite_fallback_active", path="qpaper_local.db")
    except Exception as sq_err:
        logger.error("sqlite_fallback_failed", error=str(sq_err))


async def close_db() -> None:
    """Dispose of the database engine and connection pool."""
    await engine.dispose()
    logger.info("database_engine_disposed")
