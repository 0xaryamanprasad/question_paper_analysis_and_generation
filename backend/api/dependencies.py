"""
Shared API Dependencies.

Provides reusable dependency functions for FastAPI route handlers.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db_session


async def get_db() -> AsyncSession:
    """
    Database session dependency for route handlers.

    Usage in routes:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async for session in get_db_session():
        yield session
