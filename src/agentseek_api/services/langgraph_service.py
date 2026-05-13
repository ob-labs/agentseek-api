from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel


class RunState(TypedDict):
    input: dict[str, Any]
    output: dict[str, Any]


def ensure_sync_checkpoint_mode(*, requested_async: bool) -> None:
    if requested_async:
        raise RuntimeError(
            "OceanBaseCheckpointSaver is sync-oriented in this milestone; async graph execution is not supported yet."
        )


def _echo_node(state: RunState) -> RunState:
    payload = state.get("input", {})
    return {"input": payload, "output": {"echo": payload}}


def _build_echo_graph() -> Pregel:
    builder: StateGraph[RunState] = StateGraph(RunState)
    builder.add_node("echo", _echo_node)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile()


def _echo_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    return {"input": payload}


def _echo_extract(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return result.get("output", {"echo": payload})


@dataclass(frozen=True)
class GraphEntry:
    graph: Pregel
    prepare_input: Callable[[dict[str, Any]], Any]
    extract_output: Callable[[Any, dict[str, Any]], dict[str, Any]]


class LangGraphService:
    def __init__(self) -> None:
        self._registry: dict[str, GraphEntry] = {}
        self.register(
            "default",
            graph=_build_echo_graph(),
            prepare_input=_echo_prepare,
            extract_output=_echo_extract,
        )
        self._register_sample_graphs()

    def _register_sample_graphs(self) -> None:
        try:
            from agentseek_api.services.sample_graphs import build_sample_registry
        except Exception:  # noqa: BLE001
            return
        for graph_id, entry in build_sample_registry().items():
            self.register(graph_id, **entry)

    def register(
        self,
        graph_id: str,
        *,
        graph: Pregel,
        prepare_input: Callable[[dict[str, Any]], Any],
        extract_output: Callable[[Any, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._registry[graph_id] = GraphEntry(
            graph=graph,
            prepare_input=prepare_input,
            extract_output=extract_output,
        )

    def get_entry(self, graph_id: str | None) -> GraphEntry:
        if graph_id and graph_id in self._registry:
            return self._registry[graph_id]
        return self._registry["default"]

    def get_graph(self, graph_id: str | None = None) -> Pregel:
        return self.get_entry(graph_id).graph

    def registered_graph_ids(self) -> list[str]:
        return sorted(self._registry.keys())


_langgraph_service: LangGraphService | None = None


def get_langgraph_service() -> LangGraphService:
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service
