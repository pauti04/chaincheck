"""Tests for the FastAPI server."""

from fastapi.testclient import TestClient

from chaincheck import server


def test_health() -> None:
    """The health endpoint reports ok."""
    client = TestClient(server.app)
    assert client.get("/health").json()["status"] == "ok"


def test_check_endpoint(monkeypatch) -> None:
    """The check endpoint returns a request ID header and detection JSON."""
    monkeypatch.setattr(server, "detect", _fake_detect)
    client = TestClient(server.app)
    response = client.post("/check", json={"response": "R", "methods": ["nli"]})
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    assert response.json()["response"] == "R"


def test_batch_endpoint(monkeypatch) -> None:
    """The batch endpoint returns a list of detection results."""
    monkeypatch.setattr(server, "detect", _fake_detect)
    client = TestClient(server.app)
    response = client.post("/batch", json={"inputs": [{"response": "R", "methods": ["nli"]}]})
    assert response.status_code == 200
    assert response.json()[0]["response"] == "R"


async def _fake_detect(response: str, context: str = "", prompt: str = "", methods=None):
    del context, prompt, methods
    from chaincheck.detect import DetectionResult

    return DetectionResult(response=response, claims=["R"], aggregate_score=0.1, risk_level="low")
