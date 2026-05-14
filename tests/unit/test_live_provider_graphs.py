import importlib.util
from pathlib import Path


def test_build_openai_graph_uses_compatible_streaming_defaults(monkeypatch):
    module_path = Path(__file__).resolve().parents[2] / "examples" / "live_provider_graphs" / "graph.py"
    spec = importlib.util.spec_from_file_location("live_provider_graph", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def astream(self, _messages):
            if False:
                yield None

    class FakeCompiledGraph:
        pass

    class FakeBuilder:
        def add_node(self, *_args, **_kwargs):
            return None

        def add_edge(self, *_args, **_kwargs):
            return None

        def compile(self, **_kwargs):
            return FakeCompiledGraph()

    monkeypatch.setenv("LIVE_OPENAI_COMPAT_MODEL", "test-model")
    monkeypatch.setenv("LIVE_OPENAI_COMPAT_API_KEY", "test-key")
    monkeypatch.setenv("LIVE_OPENAI_COMPAT_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(module, "ChatOpenAI", FakeChatOpenAI)
    monkeypatch.setattr(module, "StateGraph", lambda _state: FakeBuilder())

    graph = module.build_openai_graph()

    assert isinstance(graph, FakeCompiledGraph)
    assert captured["model"] == "test-model"
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["streaming"] is True
    assert captured["stream_usage"] is False
    assert captured["use_responses_api"] is False
