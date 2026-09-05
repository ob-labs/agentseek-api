from __future__ import annotations

import json
import os
import sys

PROVIDERS = [
    {"provider_kind": "openai", "provider_label": "OpenAI-Compatible"},
    {"provider_kind": "anthropic", "provider_label": "Anthropic-Compatible"},
]

BACKENDS = ["seekdb", "oceanbase", "mysql", "postgresql-metadata"]
BACKEND_CAPABILITIES = {
    "seekdb": "streaming,store,mcp,hitl",
    "oceanbase": "streaming,store,mcp,hitl",
    "mysql": "streaming,hitl",
    "postgresql-metadata": "streaming,mcp",
}
LOCAL_RUNTIME_BACKENDS = {
    "seekdb": "seekdb",
    "oceanbase": "oceanbase",
    "mysql": "mysql",
    "postgresql-metadata": "seekdb",
}


def capability_set_for_backend(backend_name: str) -> str:
    try:
        return BACKEND_CAPABILITIES[backend_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported backend tier: {backend_name}") from exc


def local_backend_env_for_tier(backend_name: str, *, embedded_available: bool) -> dict[str, str]:
    capability_set_for_backend(backend_name)
    runtime_backend = LOCAL_RUNTIME_BACKENDS[backend_name]
    if backend_name == "seekdb" and embedded_available:
        return {
            "SEEKDB_MODE": "embed",
            "SEEKDB_DOCKER_BACKEND": runtime_backend,
        }
    return {
        "SEEKDB_MODE": "docker",
        "SEEKDB_DOCKER_BACKEND": runtime_backend,
    }


def build_matrix(
    *,
    event_name: str,
    run_openai_compatible: bool,
    run_anthropic_compatible: bool,
    backend_tier: str,
) -> dict[str, list[dict[str, str]]]:
    if event_name == "schedule":
        enabled_providers = {"openai", "anthropic"}
        selected_backend = "all"
    else:
        enabled_providers = {
            provider
            for provider, enabled in {
                "openai": run_openai_compatible,
                "anthropic": run_anthropic_compatible,
            }.items()
            if enabled
        }
        if not enabled_providers:
            raise ValueError("At least one provider must be selected for workflow_dispatch.")
        selected_backend = backend_tier
        if selected_backend != "all":
            capability_set_for_backend(selected_backend)

    include: list[dict[str, str]] = []
    for provider in PROVIDERS:
        if provider["provider_kind"] not in enabled_providers:
            continue
        for backend_name in BACKENDS:
            if selected_backend != "all" and backend_name != selected_backend:
                continue
            include.append({**provider, "backend_name": backend_name})

    if not include:
        raise ValueError("Selected live provider matrix is empty.")
    return {"include": include}


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes"}


def main() -> int:
    try:
        matrix = build_matrix(
            event_name=os.getenv("GITHUB_EVENT_NAME", "workflow_dispatch"),
            run_openai_compatible=_parse_bool(os.getenv("RUN_OPENAI_COMPATIBLE", "true")),
            run_anthropic_compatible=_parse_bool(os.getenv("RUN_ANTHROPIC_COMPATIBLE", "true")),
            backend_tier=os.getenv("BACKEND_TIER_INPUT", "all").strip() or "all",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={json.dumps(matrix)}\n")
    else:
        print(json.dumps(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
