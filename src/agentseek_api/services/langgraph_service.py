import json
import inspect
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, TypedDict

from langgraph.constants import CONF, CONFIG_KEY_CHECKPOINTER
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import Pregel

from agentseek_api.settings import settings

GraphFactory = Callable[..., Any] | Pregel
PrepareInput = Callable[[dict[str, Any]], Any]
ExtractOutput = Callable[[Any, dict[str, Any]], dict[str, Any]]


class RunState(TypedDict):
    input: dict[str, Any]
    output: dict[str, Any]


class GraphManifestError(RuntimeError):
    pass


def ensure_sync_checkpoint_mode(*, requested_async: bool) -> None:
    if requested_async:
        raise RuntimeError(
            "OceanBaseCheckpointSaver is sync-oriented in this milestone; async graph execution is not supported yet."
        )


def _echo_node(state: RunState) -> RunState:
    payload = state.get("input", {})
    return {"input": payload, "output": {"echo": payload}}


def _build_echo_graph(checkpointer: Any | None = None) -> Pregel:
    builder: StateGraph[RunState] = StateGraph(RunState)
    builder.add_node("echo", _echo_node)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)
    return builder.compile(checkpointer=checkpointer)


def _echo_prepare(payload: dict[str, Any]) -> dict[str, Any]:
    return {"input": payload}


def _echo_extract(result: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return result.get("output", {"echo": payload})


def _load_python_file_module(module_ref: str, manifest_path: Path) -> Any:
    file_path = (manifest_path.parent / module_ref).resolve() if not Path(module_ref).is_absolute() else Path(module_ref)
    module_name = f"agentseek_manifest_{abs(hash(file_path))}"
    spec = spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise GraphManifestError(f"Could not load Python module from '{file_path}'.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_module_symbol(*, dotted_path: str, graph_id: str, field_name: str, manifest_path: Path) -> Any:
    if ":" not in dotted_path:
        raise GraphManifestError(
            f"AGENTSEEK_GRAPHS='{manifest_path}' graph '{graph_id}' field '{field_name}' must use 'module.path:symbol'."
        )

    module_name, symbol_name = dotted_path.split(":", maxsplit=1)
    if not module_name or not symbol_name:
        raise GraphManifestError(
            f"AGENTSEEK_GRAPHS='{manifest_path}' graph '{graph_id}' field '{field_name}' must use 'module.path:symbol'."
        )

    try:
        if module_name.endswith(".py") or module_name.startswith(".") or "/" in module_name:
            module = _load_python_file_module(module_name, manifest_path)
        else:
            module = import_module(module_name)
        return getattr(module, symbol_name)
    except Exception as exc:  # noqa: BLE001
        raise GraphManifestError(
            f"AGENTSEEK_GRAPHS='{manifest_path}' graph '{graph_id}' could not load {field_name} '{dotted_path}': {exc}"
        ) from exc


def _apply_manifest_dependencies(payload: dict[str, Any], manifest_path: Path) -> None:
    dependencies = payload.get("dependencies")
    if dependencies is None:
        return
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        raise GraphManifestError(f"AGENTSEEK_GRAPHS='{manifest_path}' top-level 'dependencies' must be an array of strings.")

    for dependency in dependencies:
        if dependency == ".":
            root = manifest_path.parent.resolve()
        else:
            candidate = Path(dependency).expanduser()
            if candidate.is_absolute():
                root = candidate.resolve()
            else:
                root = (manifest_path.parent / candidate).resolve()

        if dependency == "." or root.exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            continue

        if dependency.startswith(".") or "/" in dependency or "\\" in dependency:
            raise GraphManifestError(
                f"AGENTSEEK_GRAPHS='{manifest_path}' dependency '{dependency}' does not exist relative to the manifest."
            )


def _coerce_graph(graph_object: Any, *, checkpointer: Any | None) -> Pregel:
    if isinstance(graph_object, Pregel):
        return graph_object
    compile_graph = getattr(graph_object, "compile", None)
    if callable(compile_graph):
        return compile_graph(checkpointer=checkpointer)
    raise GraphManifestError("Graph definition did not resolve to a compiled graph or compilable StateGraph.")


def _build_factory_config(*, checkpointer: Any | None) -> dict[str, Any]:
    return {
        CONF: {
            CONFIG_KEY_CHECKPOINTER: checkpointer,
        },
        "checkpointer": checkpointer,
    }


def _build_graph_from_definition(graph_definition: GraphFactory, checkpointer: Any | None) -> Pregel:
    if isinstance(graph_definition, Pregel):
        return graph_definition

    if not callable(graph_definition):
        raise GraphManifestError("Graph definition must be a compiled graph or callable.")

    signature = inspect.signature(graph_definition)
    parameters = list(signature.parameters.values())
    if not parameters:
        built = graph_definition()
    elif any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters) or any(
        parameter.name == "checkpointer" for parameter in parameters
    ):
        built = graph_definition(checkpointer=checkpointer)
    else:
        built = graph_definition(_build_factory_config(checkpointer=checkpointer))
    return _coerce_graph(built, checkpointer=checkpointer)


@dataclass
class GraphEntry:
    graph_factory: GraphFactory
    prepare_input: PrepareInput
    extract_output: ExtractOutput

    def build_graph(self, checkpointer: Any | None = None) -> Pregel:
        return _build_graph_from_definition(self.graph_factory, checkpointer)


class LangGraphService:
    def __init__(self, *, manifest_path: str | Path | None = None) -> None:
        self._registry: dict[str, GraphEntry] = {}
        self.register(
            "default",
            graph_factory=_build_echo_graph,
            prepare_input=_echo_prepare,
            extract_output=_echo_extract,
        )
        self._register_sample_graphs()
        self._register_manifest_graphs(manifest_path=manifest_path)

    def _register_sample_graphs(self) -> None:
        try:
            from agentseek_api.services.sample_graphs import build_sample_registry
        except Exception:  # noqa: BLE001
            return
        for graph_id, entry in build_sample_registry().items():
            self.register(graph_id, **entry)

    def _register_manifest_graphs(self, *, manifest_path: str | Path | None) -> None:
        configured_path = Path(manifest_path or settings.AGENTSEEK_GRAPHS).expanduser() if (manifest_path or settings.AGENTSEEK_GRAPHS) else None
        if configured_path is None:
            return
        if not configured_path.exists():
            raise GraphManifestError(f"AGENTSEEK_GRAPHS='{configured_path}' does not exist.")

        try:
            payload = json.loads(configured_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise GraphManifestError(f"AGENTSEEK_GRAPHS='{configured_path}' could not be parsed as JSON: {exc}") from exc

        _apply_manifest_dependencies(payload, configured_path)

        graphs = payload.get("graphs")
        if not isinstance(graphs, dict):
            raise GraphManifestError(f"AGENTSEEK_GRAPHS='{configured_path}' must contain a top-level 'graphs' object.")

        from agentseek_api.services.sample_graphs import _ensure_messages_payload, _extract_messages_output

        for graph_id, config in graphs.items():
            if isinstance(config, str):
                config = {"graph": config}
            if not isinstance(config, dict):
                raise GraphManifestError(
                    f"AGENTSEEK_GRAPHS='{configured_path}' graph '{graph_id}' must be a graph definition string or object."
                )
            graph_factory = _load_module_symbol(
                dotted_path=str(config.get("graph", "")),
                graph_id=graph_id,
                field_name="graph",
                manifest_path=configured_path,
            )
            if not callable(graph_factory) and not isinstance(graph_factory, Pregel):
                raise GraphManifestError(
                    f"AGENTSEEK_GRAPHS='{configured_path}' graph '{graph_id}' graph definition must be callable or a compiled graph."
                )

            prepare_input = _ensure_messages_payload
            if "prepare_input" in config:
                prepare_input = _load_module_symbol(
                    dotted_path=str(config["prepare_input"]),
                    graph_id=graph_id,
                    field_name="prepare_input",
                    manifest_path=configured_path,
                )
                if not callable(prepare_input):
                    raise GraphManifestError(
                        f"AGENTSEEK_GRAPHS='{configured_path}' graph '{graph_id}' prepare_input must be callable."
                    )

            extract_output = _extract_messages_output
            if "extract_output" in config:
                extract_output = _load_module_symbol(
                    dotted_path=str(config["extract_output"]),
                    graph_id=graph_id,
                    field_name="extract_output",
                    manifest_path=configured_path,
                )
                if not callable(extract_output):
                    raise GraphManifestError(
                        f"AGENTSEEK_GRAPHS='{configured_path}' graph '{graph_id}' extract_output must be callable."
                    )

            self.register(
                graph_id,
                graph_factory=graph_factory,
                prepare_input=prepare_input,
                extract_output=extract_output,
            )

    def register(
        self,
        graph_id: str,
        *,
        graph_factory: GraphFactory,
        prepare_input: PrepareInput,
        extract_output: ExtractOutput,
    ) -> None:
        self._registry[graph_id] = GraphEntry(
            graph_factory=graph_factory,
            prepare_input=prepare_input,
            extract_output=extract_output,
        )

    def get_entry(self, graph_id: str | None) -> GraphEntry:
        if graph_id and graph_id in self._registry:
            return self._registry[graph_id]
        return self._registry["default"]

    def get_graph(self, graph_id: str | None = None, *, checkpointer: Any | None = None) -> Pregel:
        return self.get_entry(graph_id).build_graph(checkpointer)

    def registered_graph_ids(self) -> list[str]:
        return sorted(self._registry.keys())


_langgraph_service: LangGraphService | None = None


def get_langgraph_service() -> LangGraphService:
    global _langgraph_service
    if _langgraph_service is None:
        _langgraph_service = LangGraphService()
    return _langgraph_service
