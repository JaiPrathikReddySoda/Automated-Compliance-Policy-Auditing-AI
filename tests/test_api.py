from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncClient:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


def test_health_check(client: TestClient) -> None:
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ask_question(async_client: AsyncClient) -> None:
    """Test the ask endpoint."""
    response = await async_client.post(
        "/api/v1/ask",
        json={"question": "What is GDPR?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "metadata" in data


@pytest.mark.asyncio
async def test_upload_document(async_client: AsyncClient) -> None:
    """Test the upload endpoint."""
    # Test URL upload
    response = await async_client.post(
        "/api/v1/upload",
        json={"url": "https://gdpr.eu/what-is-gdpr/"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "chunks_indexed" in data
    
    # Test file upload
    files = {"file": ("test.txt", "This is a test document.")}
    response = await async_client.post(
        "/api/v1/upload",
        files=files,
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "chunks_indexed" in data


@pytest.mark.asyncio
async def test_upload_invalid(async_client: AsyncClient) -> None:
    """Test invalid upload requests."""
    # Test missing file and URL
    response = await async_client.post("/api/v1/upload")
    assert response.status_code == 400
    
    # Test invalid URL
    response = await async_client.post(
        "/api/v1/upload",
        json={"url": "invalid-url"},
    )
    assert response.status_code == 422 