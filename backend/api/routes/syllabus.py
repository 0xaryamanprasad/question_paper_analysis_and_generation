"""
FastAPI Routes for Syllabus Management.
"""

from typing import List, Optional
import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.schemas.classification import SyllabusDefinition
from backend.database import get_db
from backend.models.analysis import Syllabus
from backend.services.topic_classifier import DEFAULT_SYLLABUS_PATH
from backend.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/syllabus", tags=["Syllabus"])


@router.get(
    "/default",
    response_model=SyllabusDefinition,
    summary="Get the default syllabus definition",
)
async def get_default_syllabus():
    """Retrieve the sample Data Structures and Algorithms syllabus."""
    if not DEFAULT_SYLLABUS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Default syllabus file not found.",
        )
    with open(DEFAULT_SYLLABUS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SyllabusDefinition(**data)


@router.get(
    "",
    response_model=List[SyllabusDefinition],
    summary="List all syllabi stored in the database",
)
async def list_syllabi(
    db: AsyncSession = Depends(get_db),
):
    """List all user-defined syllabi."""
    query = select(Syllabus).order_by(Syllabus.created_at.desc())
    result = await db.execute(query)
    rows = result.scalars().all()

    syllabi = []
    for r in rows:
        syllabi.append(
            SyllabusDefinition(
                subject=r.subject,
                course_code=r.course_code,
                units=r.units,
            )
        )

    # If database is empty, return default syllabus as initial element
    if not syllabi and DEFAULT_SYLLABUS_PATH.exists():
        with open(DEFAULT_SYLLABUS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        syllabi.append(SyllabusDefinition(**data))

    return syllabi


@router.post(
    "",
    response_model=SyllabusDefinition,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update a subject syllabus",
)
async def create_syllabus(
    payload: SyllabusDefinition,
    db: AsyncSession = Depends(get_db),
):
    """Store a custom syllabus definition with units, topics, and keywords."""
    units_json = [u.model_dump() for u in payload.units]

    record = Syllabus(
        subject=payload.subject,
        course_code=payload.course_code,
        units=units_json,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    logger.info("syllabus_created", subject=payload.subject, units_count=len(payload.units))
    return payload
