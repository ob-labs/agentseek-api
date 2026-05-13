#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEKDB_MODE="${SEEKDB_MODE:-auto}"
SEEKDB_EMBED_CMD="${SEEKDB_EMBED_CMD:-}"
SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-oceanbase/oceanbase-ce:latest}"
SEEKDB_CONTAINER_NAME="${SEEKDB_CONTAINER_NAME:-agentseek-seekdb-test}"
EXAMPLE_API_PORT="${EXAMPLE_API_PORT:-2026}"

export OCEANBASE_HOST="${OCEANBASE_HOST:-127.0.0.1}"
export OCEANBASE_PORT="${OCEANBASE_PORT:-2881}"
export OCEANBASE_USER="${OCEANBASE_USER:-root@test}"
export OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD:-}"
export OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME:-seekdb}"
export SEEKDB_URL="${SEEKDB_URL:-mysql+aiomysql://root%40test:@127.0.0.1:${OCEANBASE_PORT}/${OCEANBASE_DB_NAME}}"
export EXAMPLE_BASE_URL="${EXAMPLE_BASE_URL:-http://127.0.0.1:${EXAMPLE_API_PORT}}"

embed_pid=""
docker_started="false"
server_pid=""

cleanup() {
  if [[ -n "$embed_pid" ]] && kill -0 "$embed_pid" >/dev/null 2>&1; then
    kill "$embed_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$docker_started" == "true" ]]; then
    docker rm -f "$SEEKDB_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" >/dev/null 2>&1; then
    kill "$server_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

wait_for_seekdb() {
  uv run python - <<'PY'
import os
import time
import pymysql

host = os.environ["OCEANBASE_HOST"]
port = int(os.environ["OCEANBASE_PORT"])
user = os.environ["OCEANBASE_USER"]
password = os.environ["OCEANBASE_PASSWORD"]
db_name = os.environ["OCEANBASE_DB_NAME"]

deadline = time.time() + 180
last_error = None
while time.time() < deadline:
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name,
            autocommit=True,
            charset="utf8mb4",
        )
        conn.close()
        print("SeekDB is reachable")
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)

raise SystemExit(f"SeekDB did not become ready: {last_error}")
PY
}

wait_for_api() {
  uv run python - <<'PY'
import os
import time
import httpx

base_url = os.environ["EXAMPLE_BASE_URL"]
deadline = time.time() + 60
last_error = None
while time.time() < deadline:
    try:
        response = httpx.get(f"{base_url}/health", timeout=2.0)
        if response.status_code == 200:
            print("API is reachable")
            raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(1)

raise SystemExit(f"API did not become ready: {last_error}")
PY
}

if [[ "$SEEKDB_MODE" == "auto" ]]; then
  if [[ -n "$SEEKDB_EMBED_CMD" ]]; then
    SEEKDB_MODE="embed"
  elif command -v docker >/dev/null 2>&1; then
    SEEKDB_MODE="docker"
  else
    echo "No backend launcher available."
    echo "Set SEEKDB_EMBED_CMD for embed mode, or install Docker for docker mode."
    exit 1
  fi
fi

if [[ "$SEEKDB_MODE" == "embed" ]]; then
  if [[ -z "$SEEKDB_EMBED_CMD" ]]; then
    echo "SEEKDB_MODE=embed requires SEEKDB_EMBED_CMD."
    exit 1
  fi
  bash -lc "$SEEKDB_EMBED_CMD" >/tmp/agentseek-seekdb-embed.log 2>&1 &
  embed_pid="$!"
elif [[ "$SEEKDB_MODE" == "docker" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed."
    exit 1
  fi
  docker rm -f "$SEEKDB_CONTAINER_NAME" >/dev/null 2>&1 || true
  docker run -d \
    --name "$SEEKDB_CONTAINER_NAME" \
    -e MODE=mini \
    -e OB_SERVER_IP=127.0.0.1 \
    -p "${OCEANBASE_PORT}:2881" \
    "$SEEKDB_DOCKER_IMAGE" >/tmp/agentseek-seekdb-docker.log
  docker_started="true"
else
  echo "Unsupported SEEKDB_MODE: $SEEKDB_MODE"
  exit 1
fi

wait_for_seekdb
uv run python scripts/seekdb_checkpoint_smoke.py

# Start API and validate real HTTP flow end-to-end.
uv run uvicorn agentseek_api.main:app --host 127.0.0.1 --port "${EXAMPLE_API_PORT}" >/tmp/agentseek-api-example.log 2>&1 &
server_pid="$!"
wait_for_api
uv run python tests/e2e/e2e_live_http_flow.py
uv run python tests/e2e/e2e_live_http_multi_graph.py
uv run pytest tests/e2e -q -m e2e
