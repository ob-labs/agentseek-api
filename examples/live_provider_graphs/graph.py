from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessageChunk, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import NOT_PROVIDED
from langgraph.types import interrupt


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def _stream_to_message(model: Any, messages: list[Any]) -> BaseMessageChunk:
    full_message: BaseMessageChunk | None = None
    async for chunk in model.astream(messages):
        full_message = chunk if full_message is None else full_message + chunk
    if full_message is None:
        raise RuntimeError("Model returned no streamed chunks.")
    return full_message


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)


def _coerce_memory_value(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"text": value}
    return {"text": json.dumps(value)}


def _build_stream_graph(*, model: Any, graph_name: str, checkpointer: Any | None = None):
    async def call_model(state: MessagesState) -> dict[str, list[BaseMessageChunk]]:
        response = await _stream_to_message(model, state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(name=graph_name, checkpointer=checkpointer)


class StoreMemoryState(TypedDict, total=False):
    memory_key: str
    memory_value: dict[str, Any]
    output: dict[str, Any]


def _build_store_graph(*, model: Any, graph_name: str, checkpointer: Any | None = None, store: Any | None = None):
    async def persist_memory(state: StoreMemoryState) -> StoreMemoryState:
        if store is None:
            raise RuntimeError("Live provider store graph requires an injected store.")

        memory_key = state.get("memory_key") or "memory"
        memory_value = _coerce_memory_value(state.get("memory_value"))
        prompt = (
            "Summarize the following JSON object in one sentence and preserve any exact proper names. "
            f"JSON: {json.dumps(memory_value, sort_keys=True)}"
        )
        response = await _stream_to_message(model, [HumanMessage(content=prompt)])
        stored_value = dict(memory_value)
        stored_value["provider_summary"] = _message_text(response).strip()

        namespace = ("graph", "memory")
        await store.aput(namespace, memory_key, stored_value, ttl=NOT_PROVIDED)
        stored = await store.aget(namespace, memory_key, refresh_ttl=False)
        if stored is None:
            raise RuntimeError("Live provider store graph could not reload the stored item.")

        return {
            "memory_key": memory_key,
            "memory_value": stored_value,
            "output": {
                "namespace": list(stored.namespace),
                "key": stored.key,
                "value": dict(stored.value),
            },
        }

    builder: StateGraph[StoreMemoryState] = StateGraph(StoreMemoryState)
    builder.add_node("persist_memory", persist_memory)
    builder.add_edge(START, "persist_memory")
    builder.add_edge("persist_memory", END)
    return builder.compile(name=graph_name, checkpointer=checkpointer, store=store)


class ProviderHitlState(TypedDict, total=False):
    foo: str
    model_prefix: str
    resume_value: str


def _build_hitl_graph(*, model: Any, graph_name: str, checkpointer: Any | None = None):
    async def draft_prefix(state: ProviderHitlState) -> ProviderHitlState:
        response = await _stream_to_message(
            model,
            [
                HumanMessage(
                    content=(
                        "Rewrite the following input as a short, friendly prefix ending with a space. "
                        f"Input: {state.get('foo', '')}"
                    )
                )
            ],
        )
        return {"model_prefix": _message_text(response).strip()}

    def request_value(_state: ProviderHitlState) -> ProviderHitlState:
        value = interrupt("Provide value:")
        return {"resume_value": str(value)}

    async def finalize(state: ProviderHitlState) -> ProviderHitlState:
        resume_value = state.get("resume_value", "")
        response = await _stream_to_message(
            model,
            [
                HumanMessage(
                    content=(
                        "Reply with a brief lead-in of at most six words for the exact token "
                        f"'{resume_value}'. Do not include the token itself."
                    )
                )
            ],
        )
        lead_in = _message_text(response).strip()
        final_text = " ".join(part for part in [state.get("model_prefix", "").strip(), lead_in, resume_value] if part).strip()
        return {
            "foo": final_text,
            "model_prefix": state.get("model_prefix", ""),
            "resume_value": resume_value,
        }

    builder: StateGraph[ProviderHitlState] = StateGraph(ProviderHitlState)
    builder.add_node("draft_prefix", draft_prefix)
    builder.add_node("request_value", request_value)
    builder.add_node("finalize", finalize)
    builder.add_edge(START, "draft_prefix")
    builder.add_edge("draft_prefix", "request_value")
    builder.add_edge("request_value", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(name=graph_name, checkpointer=checkpointer)


def _build_openai_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=_require_env("LIVE_OPENAI_COMPAT_MODEL"),
        api_key=_require_env("LIVE_OPENAI_COMPAT_API_KEY"),
        base_url=_require_env("LIVE_OPENAI_COMPAT_BASE_URL"),
        temperature=0,
        streaming=True,
        stream_usage=False,
        use_responses_api=False,
        max_retries=1,
        timeout=60,
    )


def _build_anthropic_model() -> ChatAnthropic:
    return ChatAnthropic(
        model_name=_require_env("LIVE_ANTHROPIC_COMPAT_MODEL"),
        api_key=_require_env("LIVE_ANTHROPIC_COMPAT_API_KEY"),
        base_url=_require_env("LIVE_ANTHROPIC_COMPAT_BASE_URL"),
        temperature=0,
        streaming=True,
        stream_usage=True,
        max_retries=1,
        timeout=60,
    )


def build_openai_graph(checkpointer=None):
    return _build_stream_graph(
        model=_build_openai_model(),
        graph_name="Live OpenAI-Compatible Stream Graph",
        checkpointer=checkpointer,
    )


def build_openai_store_graph(checkpointer=None, store=None):
    return _build_store_graph(
        model=_build_openai_model(),
        graph_name="Live OpenAI-Compatible Store Graph",
        checkpointer=checkpointer,
        store=store,
    )


def build_openai_hitl_graph(checkpointer=None):
    return _build_hitl_graph(
        model=_build_openai_model(),
        graph_name="Live OpenAI-Compatible HITL Graph",
        checkpointer=checkpointer,
    )


def build_anthropic_graph(checkpointer=None):
    return _build_stream_graph(
        model=_build_anthropic_model(),
        graph_name="Live Anthropic-Compatible Stream Graph",
        checkpointer=checkpointer,
    )


def build_anthropic_store_graph(checkpointer=None, store=None):
    return _build_store_graph(
        model=_build_anthropic_model(),
        graph_name="Live Anthropic-Compatible Store Graph",
        checkpointer=checkpointer,
        store=store,
    )


def build_anthropic_hitl_graph(checkpointer=None):
    return _build_hitl_graph(
        model=_build_anthropic_model(),
        graph_name="Live Anthropic-Compatible HITL Graph",
        checkpointer=checkpointer,
    )
