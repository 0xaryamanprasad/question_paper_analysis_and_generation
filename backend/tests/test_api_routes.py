"""
Integration Tests for FastAPI Routes & Endpoints.
"""

import io
import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """GET /health should return 200 with healthy status."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "version" in data


@pytest.mark.asyncio
async def test_get_default_syllabus():
    """GET /api/v1/syllabus/default should return the 5-unit sample syllabus."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.get("/api/v1/syllabus/default")
        assert res.status_code == 200
        data = res.json()
        assert data["subject"] == "Data Structures and Algorithms"
        assert len(data["units"]) == 5


@pytest.mark.asyncio
async def test_classify_question_endpoint():
    """POST /api/v1/analysis/classify-question should return dual topic & Bloom's classification."""
    payload = {
        "question_text": "Explain the time complexity of Quick Sort with the partition subroutine.",
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.post("/api/v1/analysis/classify-question", json=payload)
        assert res.status_code == 200
        data = res.json()

        topic_clf = data["topic_classification"]
        assert topic_clf["unit_number"] in [4, 5]
        assert "Quick Sort" in topic_clf["topic"] or "Algorithm" in (topic_clf["unit_name"] or "")

        blooms_clf = data["blooms_classification"]
        assert blooms_clf["blooms_level"] in ["Understand", "Analyze"]


@pytest.mark.asyncio
async def test_generate_single_question_endpoint():
    """POST /api/v1/generation/generate-question should generate question item."""
    payload = {
        "subject": "Data Structures and Algorithms",
        "unit_number": 2,
        "unit_name": "Trees",
        "topic": "AVL Trees",
        "keywords": ["AVL", "rotation"],
        "blooms_level": "Apply",
        "marks": 5.0,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        res = await ac.post("/api/v1/generation/generate-question", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["unit_number"] == 2
        assert data["topic"] == "AVL Trees"
        assert data["marks"] == 5.0
        assert len(data["question_text"]) > 10


@pytest.mark.asyncio
async def test_generate_validate_and_render_paper_pipeline(client: AsyncClient):
    """End-to-End API workflow: Generate Paper -> Validate Paper -> Render & Download PDF."""
    blueprint_payload = {
        "subject": "Data Structures and Algorithms",
        "course_code": "CS201",
        "total_marks": 30.0,
        "duration_hours": 1.5,
        "institution_name": "Antigravity University",
        "examination_name": "Class Assessment Test",
        "sections": [
            {
                "section_name": "Section A",
                "question_count": 5,
                "marks_per_question": 2.0,
                "allowed_question_types": ["short"],
                "has_internal_choice": False,
            },
            {
                "section_name": "Section B",
                "question_count": 2,
                "marks_per_question": 10.0,
                "allowed_question_types": ["long"],
                "has_internal_choice": True,
            },
        ],
    }

    # 1. Generate Paper
    gen_res = await client.post(
        "/api/v1/generation/generate-paper",
        json={"blueprint": blueprint_payload},
    )
    assert gen_res.status_code == 200
    paper_data = gen_res.json()
    assert paper_data["total_marks"] == 30.0
    assert len(paper_data["sections"]) == 2

    # 2. Validate Paper
    val_res = await client.post(
        "/api/v1/generation/validate-paper",
        json={"paper": paper_data, "blueprint": blueprint_payload},
    )
    assert val_res.status_code == 200
    val_report = val_res.json()
    assert val_report["marks_check"]["is_match"] is True
    assert val_report["is_valid"] is True

    # 3. Render PDF
    pdf_res = await client.post(
        "/api/v1/generation/render-pdf",
        json=paper_data,
    )
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert pdf_res.content.startswith(b"%PDF-")
    assert len(pdf_res.content) > 1000
