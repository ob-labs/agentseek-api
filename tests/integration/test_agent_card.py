from fastapi.testclient import TestClient

from agentseek_api.services.langgraph_service import get_langgraph_service


def test_agent_card_endpoint_returns_assistant_shaped_card(
    client: TestClient,
    monkeypatch,
) -> None:
    assistant = client.post(
        "/assistants",
        json={
            "name": "Stress Agent",
            "description": "Deterministic agent card coverage",
            "graph_id": "stress_test",
        },
    )
    assistant.raise_for_status()
    assistant_id = assistant.json()["assistant_id"]

    entry = get_langgraph_service().get_entry("stress_test")
    monkeypatch.setattr(
        entry,
        "input_schema",
        {
            "type": "object",
            "properties": {"messages": {"type": "array"}},
            "required": ["messages"],
        },
    )

    response = client.get(f"/.well-known/agent-card.json?assistant_id={assistant_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "Stress Agent"
    assert body["description"] == "Deterministic agent card coverage"
    assert body["url"].endswith(f"/a2a/{assistant_id}")
