from dataclasses import dataclass
from typing import Any

from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.types import Command

from agentseek_api.core.database import db_manager
from agentseek_api.services.langgraph_service import ensure_sync_checkpoint_mode, get_langgraph_service

UNSET = object()


@dataclass
class RunExecutionResult:
    output: dict[str, Any]
    interrupted: bool
    interrupts: list[dict[str, Any]]


async def execute_run(
    *,
    thread_id: str,
    run_id: str,
    payload: dict[str, Any],
    graph_id: str | None = None,
    resume: Any = UNSET,
) -> RunExecutionResult:
    ensure_sync_checkpoint_mode(requested_async=False)
    entry = get_langgraph_service().get_entry(graph_id)
    graph = entry.build_graph(db_manager.get_langgraph_checkpointer())

    config = {
        CONF: {
            "thread_id": thread_id,
            "checkpoint_ns": run_id,
            CONFIG_KEY_CHECKPOINTER: db_manager.get_langgraph_checkpointer(),
        }
    }
    if resume is UNSET:
        invocation = entry.prepare_input(payload)
    else:
        invocation = Command(resume=resume)

    result = await graph.ainvoke(invocation, config)
    output = entry.extract_output(result, payload)
    interrupts = output.get("interrupts", []) if isinstance(output, dict) else []
    interrupted = bool(output.get("interrupted")) if isinstance(output, dict) else False

    checkpointer = db_manager.get_checkpointer()
    await db_manager.run_checkpointer_call(
        checkpointer.save_checkpoint,
        thread_id=thread_id,
        run_id=run_id,
        payload={
            "input": payload,
            "resume": None if resume is UNSET else resume,
            "output": output,
            "graph_id": graph_id or "default",
        },
    )
    return RunExecutionResult(output=output, interrupted=interrupted, interrupts=interrupts)
