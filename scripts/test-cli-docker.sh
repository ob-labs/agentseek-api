#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
umask 077

IMAGE_TAG="${IMAGE_TAG:-agentseek-api-cli-smoke:0.3.0}"
APP_CONTAINER="${APP_CONTAINER:-agentseek-up-8123}"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/agentseek-cli-docker.XXXXXX")"
PROJECT_DIR="$TMP_DIR/source-only-project"
BUNDLE_DIR="$TMP_DIR/build-bundle"
BUNDLE_CONTEXT="$BUNDLE_DIR/context"
CANDIDATE_DIR="$PROJECT_DIR/candidate-dist"
IMAGE_ARCHIVE="$TMP_DIR/image.tar"
HISTORY_FILE="$TMP_DIR/image.history"
BUILD_LOG="$TMP_DIR/build.log"
WHEEL_BUILD_LOG="$TMP_DIR/wheel-build.log"
BUILD_SENTINEL="$(uv run python -c 'import secrets; print(secrets.token_urlsafe(24))')"
IMAGE_OWNED=0
CONTAINER_OWNED=0

container_id() {
  docker container ls --all --filter "name=^/${APP_CONTAINER}$" --format '{{.ID}}'
}

image_id() {
  docker image ls --all --no-trunc --filter "reference=${IMAGE_TAG}" --format '{{.ID}}'
}

cleanup() {
  local status=$?
  local cleanup_failed=0
  local remaining=""
  trap - EXIT
  set +e
  if [[ "$CONTAINER_OWNED" -eq 1 ]]; then
    remaining="$(container_id 2>/dev/null)" || cleanup_failed=1
    if [[ -n "$remaining" ]]; then
      docker rm -f "$APP_CONTAINER" >/dev/null 2>&1 || cleanup_failed=1
    fi
    remaining="$(container_id 2>/dev/null)" || cleanup_failed=1
    if [[ -n "$remaining" ]]; then
      cleanup_failed=1
    fi
  fi
  if [[ "$IMAGE_OWNED" -eq 1 ]]; then
    remaining="$(image_id 2>/dev/null)" || cleanup_failed=1
    if [[ -n "$remaining" ]]; then
      docker image rm -f "$IMAGE_TAG" >/dev/null 2>&1 || cleanup_failed=1
    fi
    remaining="$(image_id 2>/dev/null)" || cleanup_failed=1
    if [[ -n "$remaining" ]]; then
      cleanup_failed=1
    fi
  fi
  rm -rf -- "$TMP_DIR" || cleanup_failed=1
  if [[ -e "$TMP_DIR" ]]; then
    cleanup_failed=1
  fi
  if [[ "$status" -eq 0 && "$cleanup_failed" -ne 0 ]]; then
    echo "Owned container resource cleanup boundary failed." >&2
    exit 1
  fi
  exit "$status"
}

print_logs() {
  docker logs "$APP_CONTAINER" || true
}

trap cleanup EXIT

mkdir -m 700 -p "$PROJECT_DIR" "$CANDIDATE_DIR"

cat >"$PROJECT_DIR/graph.py" <<'PY'
from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph


async def respond(state: MessagesState) -> dict:
    text = state["messages"][-1].content if state["messages"] else ""
    return {"messages": [AIMessage(content=f"external graph heard: {text}")]}


def build_graph(checkpointer=None):
    builder = StateGraph(MessagesState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile(name="Source Only Graph", checkpointer=checkpointer)
PY

cat >"$PROJECT_DIR/auth_backend.py" <<'PY'
from langgraph_sdk import Auth

HeaderAuthBackend = Auth()


@HeaderAuthBackend.authenticate
async def authenticate(headers: dict[bytes, bytes]) -> dict:
    raw = headers.get(b"x-user-id", b"default_user")
    identity = raw.decode() if isinstance(raw, bytes) else str(raw)
    return {"identity": identity}


@HeaderAuthBackend.on.threads.create
async def on_threads_create(ctx: Auth.types.AuthContext, value: dict) -> None:
    value.setdefault("metadata", {})["owner"] = ctx.user.identity


@HeaderAuthBackend.on.threads.read
async def on_threads_read(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.update
async def on_threads_update(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.delete
async def on_threads_delete(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}


@HeaderAuthBackend.on.threads.search
async def on_threads_search(ctx: Auth.types.AuthContext, value: dict) -> dict:
    return {"owner": ctx.user.identity}
PY

cat >"$PROJECT_DIR/application.env" <<EOF
BUILD_BOUNDARY_SENTINEL=$BUILD_SENTINEL
METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek.db
EOF

cat >"$PROJECT_DIR/agentseek.json" <<'JSON'
{
  "dependencies": ["packaging==25.0"],
  "graphs": {"external_hello": "./graph.py:build_graph"},
  "auth": {"path": "./auth_backend.py:HeaderAuthBackend"},
  "env": "application.env",
  "dockerfile_lines": [
    "RUN [\"python\", \"-c\", \"import pathlib; pathlib.Path('/tmp/custom-boundary').write_text('ok')\"]"
  ]
}
JSON

cat >"$PROJECT_DIR/launch.json" <<'JSON'
{
  "dependencies": [],
  "graphs": {"external_hello": "./graph.py:build_graph"},
  "env": "application.env"
}
JSON

if ! uv build --wheel --out-dir "$CANDIDATE_DIR" >"$WHEEL_BUILD_LOG" 2>&1; then
  echo "Candidate wheel build failed." >&2
  exit 1
fi

CANDIDATE_WHEELS=("$CANDIDATE_DIR"/agentseek_api-0.3.0-*.whl)
if [[ "${#CANDIDATE_WHEELS[@]}" -ne 1 || ! -f "${CANDIDATE_WHEELS[0]}" ]]; then
  echo "Candidate wheel selection failed." >&2
  exit 1
fi
CANDIDATE_WHEEL="${CANDIDATE_WHEELS[0]}"
CANDIDATE_SHA256="$(shasum -a 256 "$CANDIDATE_WHEEL" | awk '{print $1}')"

uv run python - "$PROJECT_DIR/agentseek.json" "$BUNDLE_DIR" "$CANDIDATE_WHEEL" "$CANDIDATE_SHA256" >"$TMP_DIR/bundle.log" <<'PY'
from pathlib import Path
import sys

from agentseek_api.cli import main
from agentseek_api.container_build import candidate_runtime_artifact

config, bundle, wheel = (Path(value) for value in sys.argv[1:4])
artifact = candidate_runtime_artifact(wheel, sys.argv[4])
raise SystemExit(
    main(
        ["dockerfile", "--config", str(config), str(bundle)],
        cwd=config.parent,
        runtime_artifact=artifact,
    )
)
PY

if [[ ! -d "$BUNDLE_CONTEXT" || ! -s "$BUNDLE_CONTEXT/Dockerfile" ]]; then
  echo "Private build bundle was not produced." >&2
  exit 1
fi

uv run python - "$BUNDLE_DIR" "$BUILD_SENTINEL" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

bundle = Path(sys.argv[1])
sentinel = sys.argv[2].encode()
context = bundle / "context"
dockerfile_path = context / "Dockerfile"
manifest_path = context / "manifest.v1.json"
inventory_path = bundle / "inventory.json"
if not all(path.is_file() for path in (dockerfile_path, manifest_path, inventory_path)):
    raise SystemExit("Private build bundle metadata was not produced")
dockerfile_bytes = dockerfile_path.read_bytes()
dockerfile = dockerfile_bytes.decode("utf-8")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
if not isinstance(manifest, dict) or not isinstance(inventory, list):
    raise SystemExit("Private build bundle metadata shape failed")
matches = [item for item in inventory if item.get("relative_path") == "Dockerfile"]
if (
    len(matches) != 1
    or matches[0].get("size") != len(dockerfile_bytes)
    or matches[0].get("sha256") != hashlib.sha256(dockerfile_bytes).hexdigest()
):
    raise SystemExit("Dockerfile inventory binding failed")
positions = [
    dockerfile.index("packaging==25.0"),
    dockerfile.index("/tmp/custom-boundary"),
    dockerfile.index("agentseek-api-0.3.0.whl[embedded]"),
    dockerfile.index("COPY manifest.v1.json /opt/agentseek/manifest.v1.json"),
    dockerfile.index('"python", "-m", "pip", "check"'),
    dockerfile.index("importlib.metadata"),
]
if positions != sorted(positions) or len(set(positions)) != len(positions):
    raise SystemExit("Dockerfile install and verification order failed")
for path in context.rglob("*"):
    if path.is_file() and sentinel in path.read_bytes():
        raise SystemExit("Build context sentinel isolation failed")
PY

if ! EXISTING_IMAGE="$(image_id)"; then
  echo "Candidate image ownership query failed." >&2
  exit 1
fi
if [[ -n "$EXISTING_IMAGE" ]]; then
  echo "Candidate image ownership boundary failed." >&2
  exit 1
fi
IMAGE_OWNED=1
if ! docker buildx build --load --file "$BUNDLE_CONTEXT/Dockerfile" --tag "$IMAGE_TAG" "$BUNDLE_CONTEXT" >"$BUILD_LOG" 2>&1; then
  echo "Candidate bundle image build failed." >&2
  exit 1
fi

uv run python - "$IMAGE_TAG" <<'PY'
import json
import subprocess
import sys

expected = {
    "org.agentseek.environment-contract": "preloaded-v1",
    "org.agentseek.runtime-manifest": "/opt/agentseek/manifest.v1.json",
    "org.agentseek.runtime-distribution": "agentseek-api",
    "org.agentseek.runtime-version": "0.3.0",
}
raw = subprocess.run(
    ["docker", "image", "inspect", sys.argv[1]],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
).stdout
image = json.loads(raw)[0]
if any(image["Config"]["Labels"].get(name) != value for name, value in expected.items()):
    raise SystemExit("Image label contract failed")
command = image["Config"]["Cmd"]
if "--environment-mode" not in command or "preloaded-v1" not in command:
    raise SystemExit("Image command mode contract failed")
PY

docker run --rm -i "$IMAGE_TAG" python - <<'PY'
import importlib.metadata
import json
import pathlib

path = pathlib.Path("/opt/agentseek/manifest.v1.json")
raw = path.read_bytes()
document = json.loads(raw)
canonical = (json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
if raw != canonical:
    raise SystemExit("Runtime manifest canonicalization failed")
if document["runtime"] != {
    "contract": "preloaded-v1",
    "distribution": "agentseek-api",
    "version": "0.3.0",
}:
    raise SystemExit("Runtime manifest identity failed")
if importlib.metadata.version("agentseek-api") != "0.3.0":
    raise SystemExit("Runtime distribution version failed")
PY

if ! EXISTING_CONTAINER="$(container_id)"; then
  echo "Application container ownership query failed." >&2
  exit 1
fi
if [[ -n "$EXISTING_CONTAINER" ]]; then
  echo "Application container ownership boundary failed." >&2
  exit 1
fi
CONTAINER_OWNED=1
if ! uv run agentseek-api up \
  --config "$PROJECT_DIR/launch.json" \
  --image "$IMAGE_TAG" \
  --port 8123 \
  --recreate >"$TMP_DIR/up.log" 2>&1; then
  print_logs
  exit 1
fi

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8123/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS "http://127.0.0.1:8123/health" | grep -q '"healthy"'; then
  print_logs
  echo "Source-only app container did not become healthy." >&2
  exit 1
fi

if ! uv run python scripts/verify_docker_api.py --base-url http://127.0.0.1:8123 --mode smoke; then
  print_logs
  exit 1
fi

RUNTIME_RECORD="$(docker exec "$APP_CONTAINER" python -c 'import importlib.metadata,pathlib,agentseek_api; print(importlib.metadata.version("agentseek-api")); print(pathlib.Path(agentseek_api.__file__).resolve())')"
RUNTIME_VERSION="$(printf '%s\n' "$RUNTIME_RECORD" | sed -n '1p')"
RUNTIME_MODULE="$(printf '%s\n' "$RUNTIME_RECORD" | sed -n '2p')"
if [[ "$RUNTIME_VERSION" != "0.3.0" ]]; then
  echo "Running distribution version boundary failed." >&2
  exit 1
fi
if [[ "$RUNTIME_MODULE" != *"site-packages"* || "$RUNTIME_MODULE" == /deps/agent/* ]]; then
  echo "Running distribution path boundary failed." >&2
  exit 1
fi

PROCESS_COMMAND="$(docker container inspect --format '{{json .Path}} {{json .Args}}' "$APP_CONTAINER")"
if [[ "$PROCESS_COMMAND" != *"--environment-mode"* || "$PROCESS_COMMAND" != *"preloaded-v1"* ]]; then
  echo "Running process environment mode boundary failed." >&2
  exit 1
fi

docker image save --output "$IMAGE_ARCHIVE" "$IMAGE_TAG"
docker history --no-trunc --format '{{json .}}' "$IMAGE_TAG" >"$HISTORY_FILE"
chmod 600 "$IMAGE_ARCHIVE" "$HISTORY_FILE"
AGENTSEEK_IMAGE_SCAN_SENTINEL="$BUILD_SENTINEL" \
  uv run python scripts/container_image_archive.py \
  --archive "$IMAGE_ARCHIVE" \
  --history "$HISTORY_FILE"

printf 'candidate wheel sha256: %s\n' "$CANDIDATE_SHA256"
printf 'runtime version: %s\n' "$RUNTIME_VERSION"
printf 'runtime module: %s\n' "$RUNTIME_MODULE"
