"""
Tests for the FastAPI application and health endpoint.
"""

import pytest


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Health check should return 200 with status and version."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    @pytest.mark.asyncio
    async def test_openapi_schema(self, client):
        """OpenAPI schema should be available."""
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "info" in schema
        assert schema["info"]["title"] == "Question Paper Analysis & Generation System"
