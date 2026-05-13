from typing import Any

from agentseek_api.core.database import db_manager
from agentseek_api.services.langgraph_service import ensure_sync_checkpoint_mode, get_langgraph_service


async def execute_run(
    *,
    thread_id: str,
    run_id: str,
    payload: dict[str, Any],
    graph_id: str | None = None,
) -> dict[str, Any]:
    ensure_sync_checkpoint_mode(requested_async=False)
    entry = get_langgraph_service().get_entry(graph_id)
    prepared_input = entry.prepare_input(payload)
    config = {"configurable": {"thread_id": thread_id, "run_id": run_id}}
    result = await entry.graph.ainvoke(prepared_input, config)
    output = entry.extract_output(result, payload)
    checkpointer = db_manager.get_checkpointer()
    await db_manager.run_checkpointer_call(
        checkpointer.save_checkpoint,
        thread_id=thread_id,
        run_id=run_id,
        payload={"input": payload, "output": output, "graph_id": graph_id or "default"},
    )
    return output
