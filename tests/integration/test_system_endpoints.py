from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ok_endpoint(client: TestClient) -> None:
    response = client.get("/ok")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_info_endpoint(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"]
    assert body["langgraph_py_version"]
    assert body["flags"]["agents"] is True
    assert body["flags"]["assistants"] is True
    assert body["flags"]["protocol_v2"] is True
    assert isinstance(body["metadata"], dict)


def test_metrics_endpoint_prometheus_default(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "agentseek_api_info" in response.text


def test_metrics_endpoint_json_format(client: TestClient) -> None:
    response = client.get("/metrics?format=json")
    assert response.status_code == 200
    body = response.json()
    assert body["app_name"] == "AgentSeek API"
    assert body["checks"]["database"] == "ok"
