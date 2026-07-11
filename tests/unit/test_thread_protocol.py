import pytest

from agentseek_api.settings import settings
from agentseek_api.services import thread_protocol as protocol
from agentseek_api.services.thread_protocol import ThreadProtocolEventBroker


def test_thread_protocol_broker_prunes_old_events_per_thread() -> None:
    broker = ThreadProtocolEventBroker(max_events_per_thread=2)

    broker.publish("thread-1", {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": {"n": 1}}})
    broker.publish("thread-1", {"method": "values", "params": {"namespace": [], "timestamp": 2, "data": {"n": 2}}})
    broker.publish("thread-1", {"method": "values", "params": {"namespace": [], "timestamp": 3, "data": {"n": 3}}})

    assert [event["seq"] for event in broker._events["thread-1"]] == [2, 3]


def test_thread_protocol_broker_discards_stale_idle_threads() -> None:
    broker = ThreadProtocolEventBroker(max_events_per_thread=4, max_idle_threads=1)

    broker.run_started("thread-1")
    broker.publish("thread-1", {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": {"n": 1}}})
    broker.run_finished("thread-1")

    broker.run_started("thread-2")
    broker.publish("thread-2", {"method": "values", "params": {"namespace": [], "timestamp": 2, "data": {"n": 2}}})
    broker.run_finished("thread-2")

    assert "thread-1" not in broker._events
    assert "thread-1" not in broker._signals
    assert "thread-1" not in broker._next_seq
    assert "thread-1" not in broker._active_runs


@pytest.mark.asyncio
async def test_thread_protocol_broker_apublish_waits_for_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    persisted: list[tuple[str, int]] = []

    async def fake_persist_thread_stream_event(thread_id: str, event: dict) -> None:
        persisted.append((thread_id, event["seq"]))

    monkeypatch.setattr(
        "agentseek_api.services.stream_persistence.persist_thread_stream_event",
        fake_persist_thread_stream_event,
    )

    event = await broker.apublish(
        "thread-1",
        {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": {"n": 1}}},
    )

    assert event["seq"] == 1
    assert persisted == [("thread-1", 1)]


@pytest.mark.asyncio
async def test_apublish_thread_event_uses_atomic_redis_append(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(settings, "EXECUTOR_BACKEND", "redis")
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    async def fake_append(thread_id: str, payload: dict) -> tuple[int, dict]:
        assert thread_id == "thread-1"
        return 9, {"type": "event", "event_id": "thread-1:9", "seq": 9, **payload}

    async def unexpected_next_seq(_thread_id: str) -> int:
        raise AssertionError("Redis sequence allocation must be part of the append")

    monkeypatch.setattr("agentseek_api.services.stream_persistence.append_redis_thread_stream_event", fake_append)
    monkeypatch.setattr("agentseek_api.services.stream_persistence.next_thread_stream_seq", unexpected_next_seq)

    event = await protocol._apublish_thread_event(
        "thread-1",
        {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": {"ok": True}}},
    )

    assert event["seq"] == 9
    assert event["event_id"] == "thread-1:9"
    assert broker._events["thread-1"] == [event]


@pytest.mark.asyncio
async def test_thread_protocol_stream_filters_channels_namespaces_depth_and_since() -> None:
    broker = ThreadProtocolEventBroker()
    broker.run_started("thread-1")
    broker.publish(
        "thread-1",
        {"method": "input.requested", "params": {"namespace": ["root", "leaf"], "timestamp": 1, "data": {"ok": True}}},
    )
    broker.publish(
        "thread-1",
        {"method": "values", "params": {"namespace": ["root"], "timestamp": 2, "data": {"step": 1}}},
    )
    broker.run_finished("thread-1")

    events = [
        event
        async for event in broker.stream(
            "thread-1",
            channels=["input"],
            namespaces=[["root"]],
            depth=2,
            since=0,
        )
    ]
    assert [event["method"] for event in events] == ["input.requested"]

    replay = [
        event
        async for event in broker.stream(
            "thread-1",
            channels=["values"],
            namespaces=None,
            depth=1,
            since=1,
        )
    ]
    assert [event["seq"] for event in replay] == [2]


@pytest.mark.asyncio
async def test_thread_protocol_stream_does_not_lose_events_published_during_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = ThreadProtocolEventBroker()
    broker.run_started("thread-1")
    broker.publish(
        "thread-1",
        {"method": "values", "params": {"namespace": [], "timestamp": 1, "data": {"step": 1}}},
    )

    signal = broker._signals["thread-1"]
    original_clear = signal.clear
    injected = {"done": False}

    def clear_with_injected_event() -> None:
        if not injected["done"]:
            injected["done"] = True
            broker.publish(
                "thread-1",
                {"method": "values", "params": {"namespace": [], "timestamp": 2, "data": {"step": 2}}},
            )
            broker.run_finished("thread-1")
        original_clear()

    monkeypatch.setattr(signal, "clear", clear_with_injected_event)

    events = [
        event
        async for event in broker.stream(
            "thread-1",
            channels=["values"],
            namespaces=None,
            depth=None,
            since=0,
        )
    ]

    assert [event["seq"] for event in events] == [1, 2]


def test_publish_helpers_emit_expected_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    protocol.publish_lifecycle_event("thread-1", event="started", graph_name="default", error="boom")
    protocol.publish_tool_event(
        "thread-1",
        tool_event="tool-started",
        tool_call_id="call-1",
        tool_name="search",
        node="tool_node",
        input_payload={"q": "weather"},
        output_payload={"result": "sunny"},
        error_message="warning",
    )
    protocol.publish_values_event("thread-1", values={"answer": 42})
    protocol.publish_input_requested("thread-1", interrupt_id="interrupt-1", payload="Provide value:")
    protocol.publish_message_chunk("thread-1", message_id="msg-1", role="ai", text="hel")
    protocol.publish_message_chunk_delta("thread-1", text="lo")
    protocol.publish_message_finish("thread-1")

    methods = [event["method"] for event in broker._events["thread-1"]]
    assert methods == [
        "lifecycle",
        "tools",
        "values",
        "input.requested",
        "messages",
        "messages",
        "messages",
        "messages",
        "messages",
        "messages",
    ]
    assert broker._events["thread-1"][1]["params"]["node"] == "tool_node"
    assert broker._events["thread-1"][2]["params"]["data"] == {"answer": 42}
    assert broker._events["thread-1"][4]["params"]["data"]["event"] == "message-start"
    assert broker._events["thread-1"][6]["params"]["data"]["delta"] == {"type": "text-delta", "text": "hel"}


def test_publish_message_transcript_handles_text_tool_calls_and_unknown_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    protocol.publish_message_transcript(
        "thread-1",
        run_id="run-1",
        messages=[
            {"type": "UnknownMessage", "content": "skip"},
            {
                "type": "AIMessage",
                "content": "hello",
                "tool_calls": [{"id": "tool-1", "name": "search", "args": {"q": "weather"}}, "skip"],
            },
            {"type": "HumanMessage", "content": "hi"},
        ],
    )

    events = broker._events["thread-1"]
    assert events[0]["params"]["data"] == {"event": "message-start", "role": "ai", "id": "run-1:1"}
    assert events[1]["params"]["data"]["content"] == {"type": "text", "text": "hello"}
    assert events[3]["params"]["data"]["content"]["type"] == "tool_call"
    assert events[6]["params"]["data"] == {"event": "message-start", "role": "human", "id": "run-1:2"}
    assert events[-1]["params"]["data"] == {"event": "message-finish"}


def test_publish_message_transcript_handles_structured_content_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    protocol.publish_message_transcript(
        "thread-1",
        run_id="run-1",
        messages=[
            {
                "type": "AIMessage",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]},
                ],
            }
        ],
    )

    block_starts = [
        event["params"]["data"]
        for event in broker._events["thread-1"]
        if event["params"]["data"]["event"] == "content-block-start"
    ]
    assert block_starts[0]["content"] == {"type": "text", "text": "hello"}
    assert block_starts[1]["content"]["type"] == "reasoning"


def test_publish_message_transcript_does_not_duplicate_structured_tool_call_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    protocol.publish_message_transcript(
        "thread-1",
        run_id="run-1",
        messages=[
            {
                "type": "AIMessage",
                "content": [
                    {"type": "tool_call", "id": "tool-1", "name": "search", "args": {"q": "weather"}},
                ],
                "tool_calls": [{"id": "tool-1", "name": "search", "args": {"q": "weather"}}],
            }
        ],
    )

    block_starts = [
        event["params"]["data"]["content"]
        for event in broker._events["thread-1"]
        if event["params"]["data"]["event"] == "content-block-start"
    ]
    assert block_starts == [{"type": "tool_call", "id": "tool-1", "name": "search", "args": {"q": "weather"}}]


def test_message_chunk_finish_order_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = ThreadProtocolEventBroker()
    monkeypatch.setattr(protocol, "thread_protocol_broker", broker)

    protocol.publish_message_chunk("thread-1", message_id="msg-1", role="ai", text="first")
    protocol.publish_message_chunk("thread-1", message_id="msg-2", role="ai", text="second")
    protocol.publish_message_finish("thread-1")
    protocol.publish_message_finish("thread-1")

    finish_events = [event["params"]["data"]["event"] for event in broker._events["thread-1"][-4:]]
    assert finish_events == [
        "content-block-finish",
        "message-finish",
        "content-block-finish",
        "message-finish",
    ]
