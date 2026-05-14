from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessageChunk
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph


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


def _build_graph(*, model: Any, graph_name: str, checkpointer: Any | None = None):
    async def call_model(state: MessagesState) -> dict[str, list[BaseMessageChunk]]:
        response = await _stream_to_message(model, state["messages"])
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(name=graph_name, checkpointer=checkpointer)


def build_openai_graph(checkpointer=None):
    model = ChatOpenAI(
        model=_require_env("LIVE_OPENAI_COMPAT_MODEL"),
        api_key=_require_env("LIVE_OPENAI_COMPAT_API_KEY"),
        base_url=_require_env("LIVE_OPENAI_COMPAT_BASE_URL"),
        temperature=0,
        streaming=True,
        stream_usage=True,
        max_retries=1,
        timeout=60,
    )
    return _build_graph(model=model, graph_name="Live OpenAI-Compatible Stream Graph", checkpointer=checkpointer)


def build_anthropic_graph(checkpointer=None):
    model = ChatAnthropic(
        model_name=_require_env("LIVE_ANTHROPIC_COMPAT_MODEL"),
        api_key=_require_env("LIVE_ANTHROPIC_COMPAT_API_KEY"),
        base_url=_require_env("LIVE_ANTHROPIC_COMPAT_BASE_URL"),
        temperature=0,
        streaming=True,
        stream_usage=True,
        max_retries=1,
        timeout=60,
    )
    return _build_graph(model=model, graph_name="Live Anthropic-Compatible Stream Graph", checkpointer=checkpointer)
