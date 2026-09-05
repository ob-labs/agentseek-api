from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from agentseek_api.services.langgraph_service import LangGraphService


async def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    manifest_path = Path(__file__).resolve().parent / "manifest.json"
    service = LangGraphService(manifest_path=manifest_path)
    entry = service.get_entry("external_hello")
    graph = entry.build_graph()
    prepared = entry.prepare_input({"message": "hello from manifest"})
    result = await graph.ainvoke(prepared)
    output = entry.extract_output(result, {"message": "hello from manifest"})
    assert "external graph heard: hello from manifest" in output["final_text"]
    print(json.dumps(output))


if __name__ == "__main__":
    asyncio.run(main())
