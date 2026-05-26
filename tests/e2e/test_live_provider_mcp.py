from __future__ import annotations

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tests.e2e.live_provider_helpers import provider_capability_enabled, provider_graph_id


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_live_provider_mcp_tool_round_trip(live_provider_base_url: str) -> None:
    if not provider_capability_enabled("mcp"):
        pytest.skip("Live provider MCP coverage is disabled for this backend tier.")

    tool_name = provider_graph_id("stream")

    async with httpx.AsyncClient(headers={"x-user-id": "mcp-live-provider-user"}, timeout=30.0, trust_env=False) as http_client:
        async with streamable_http_client(
            url=f"{live_provider_base_url}/mcp",
            http_client=http_client,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                listed = await session.list_tools()
                assert any(tool.name == tool_name for tool in listed.tools)

                result = await session.call_tool(
                    tool_name,
                    {
                        "message": (
                            "Explain why MCP transport checks matter for live provider tests in exactly two sentences."
                        )
                    },
                )
                assert result.structuredContent["final_text"]
