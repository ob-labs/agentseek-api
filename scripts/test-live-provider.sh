#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEEKDB_MODE="${SEEKDB_MODE:-auto}"
SEEKDB_EMBED_CMD="${SEEKDB_EMBED_CMD:-uv run python scripts/seekdb_embed_launcher.py}"
SEEKDB_DOCKER_BACKEND="${SEEKDB_DOCKER_BACKEND:-seekdb}"
SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-}"
SEEKDB_CONTAINER_NAME="${SEEKDB_CONTAINER_NAME:-agentseek-live-provider-test}"
OCEANBASE_DOCKER_MODE="${OCEANBASE_DOCKER_MODE:-mini}"

export OCEANBASE_HOST="${OCEANBASE_HOST:-127.0.0.1}"
export LIVE_PROVIDER_REQUIRED="${LIVE_PROVIDER_REQUIRED:-1}"

embed_pid=""
docker_started="false"

cleanup() {
  if [[ -n "$embed_pid" ]] && kill -0 "$embed_pid" >/dev/null 2>&1; then
    kill "$embed_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$docker_started" == "true" ]]; then
    docker rm -f "$SEEKDB_CONTAINER_NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

readiness_timeout_seconds() {
  case "${SEEKDB_MODE}:${SEEKDB_DOCKER_BACKEND:-embed}" in
    docker:oceanbase)
      echo 420
      ;;
    *)
      echo 180
      ;;
  esac
}

print_backend_debug() {
  if [[ "$SEEKDB_MODE" == "docker" ]]; then
    echo "=== docker ps ==="
    docker ps -a || true
    echo "=== docker logs ($SEEKDB_CONTAINER_NAME) ==="
    docker logs "$SEEKDB_CONTAINER_NAME" || true
  else
    echo "=== embedded launcher log ==="
    cat /tmp/agentseek-live-provider-embed.log || true
  fi
}

wait_for_seekdb() {
  local timeout_seconds
  timeout_seconds="$(readiness_timeout_seconds)"
  export SEEKDB_READINESS_TIMEOUT_SECONDS="$timeout_seconds"

  if ! uv run python - <<'PY'
import os
import time
import pymysql

host = os.environ["OCEANBASE_HOST"]
port = int(os.environ["OCEANBASE_PORT"])
user = os.environ["OCEANBASE_USER"]
password = os.environ["OCEANBASE_PASSWORD"]
timeout_seconds = float(os.environ["SEEKDB_READINESS_TIMEOUT_SECONDS"])

deadline = time.time() + timeout_seconds
last_error = None
while time.time() < deadline:
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            autocommit=True,
            charset="utf8mb4",
        )
        conn.close()
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)

raise SystemExit(f"SeekDB did not become ready within {timeout_seconds:.0f}s: {last_error}")
PY
  then
    print_backend_debug
    return 1
  fi
}

ensure_database_exists() {
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
            autocommit=True,
            charset="utf8mb4",
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            raise SystemExit(0)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)

raise SystemExit(f"Database bootstrap did not become ready: {last_error}")
PY
}

set_backend_defaults() {
  local encoded_user

  case "$1" in
    embed)
      export OCEANBASE_PORT="${OCEANBASE_PORT:-2881}"
      export OCEANBASE_USER="${OCEANBASE_USER:-root}"
      export OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD:-}"
      export OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME:-seekdb}"
      ;;
    seekdb)
      export SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-oceanbase/seekdb:latest}"
      export OCEANBASE_PORT="${OCEANBASE_PORT:-2881}"
      export OCEANBASE_USER="${OCEANBASE_USER:-root}"
      export OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD:-}"
      export OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME:-seekdb}"
      ;;
    oceanbase)
      export SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-oceanbase/oceanbase-ce:latest}"
      export OCEANBASE_PORT="${OCEANBASE_PORT:-2881}"
      export OCEANBASE_USER="${OCEANBASE_USER:-root@test}"
      export OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD:-}"
      export OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME:-seekdb}"
      ;;
    mysql)
      export SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-mysql:8.4}"
      export OCEANBASE_PORT="${OCEANBASE_PORT:-3306}"
      export OCEANBASE_USER="${OCEANBASE_USER:-root}"
      export OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD:-root}"
      export OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME:-seekdb}"
      ;;
    *)
      echo "Unsupported backend defaults: $1"
      exit 1
      ;;
  esac

  encoded_user="${OCEANBASE_USER//@/%40}"
  export SEEKDB_URL="${SEEKDB_URL:-mysql+aiomysql://${encoded_user}:${OCEANBASE_PASSWORD}@127.0.0.1:${OCEANBASE_PORT}/${OCEANBASE_DB_NAME}}"
}

if [[ "$SEEKDB_MODE" == "auto" ]]; then
  if uv run python -c "import pylibseekdb" >/dev/null 2>&1; then
    SEEKDB_MODE="embed"
  elif command -v docker >/dev/null 2>&1; then
    SEEKDB_MODE="docker"
  else
    echo "No backend launcher available."
    echo "Install embedded mode with 'uv sync --dev --extra embedded', or install Docker for docker mode."
    exit 1
  fi
fi

if [[ "$SEEKDB_MODE" == "embed" ]]; then
  set_backend_defaults embed
  bash -lc "$SEEKDB_EMBED_CMD" >/tmp/agentseek-live-provider-embed.log 2>&1 &
  embed_pid="$!"
elif [[ "$SEEKDB_MODE" == "docker" ]]; then
  set_backend_defaults "$SEEKDB_DOCKER_BACKEND"
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed."
    exit 1
  fi
  docker rm -f "$SEEKDB_CONTAINER_NAME" >/dev/null 2>&1 || true
  case "$SEEKDB_DOCKER_BACKEND" in
    seekdb)
      docker run -d \
        --name "$SEEKDB_CONTAINER_NAME" \
        -p "${OCEANBASE_PORT}:2881" \
        -p 2886:2886 \
        "$SEEKDB_DOCKER_IMAGE" >/tmp/agentseek-live-provider-docker.log
      ;;
    oceanbase)
      docker run -d \
        --name "$SEEKDB_CONTAINER_NAME" \
        -e MODE="${OCEANBASE_DOCKER_MODE}" \
        -p "${OCEANBASE_PORT}:2881" \
        "$SEEKDB_DOCKER_IMAGE" >/tmp/agentseek-live-provider-docker.log
      ;;
    mysql)
      docker run -d \
        --name "$SEEKDB_CONTAINER_NAME" \
        -e MYSQL_ROOT_PASSWORD="${OCEANBASE_PASSWORD}" \
        -e MYSQL_DATABASE="${OCEANBASE_DB_NAME}" \
        -p "${OCEANBASE_PORT}:3306" \
        "$SEEKDB_DOCKER_IMAGE" \
        --character-set-server=utf8mb4 \
        --collation-server=utf8mb4_unicode_ci >/tmp/agentseek-live-provider-docker.log
      ;;
    *)
      echo "Unsupported SEEKDB_DOCKER_BACKEND: $SEEKDB_DOCKER_BACKEND"
      exit 1
      ;;
  esac
  docker_started="true"
else
  echo "Unsupported SEEKDB_MODE: $SEEKDB_MODE"
  exit 1
fi

export LIVE_PROVIDER_CAPABILITIES="${LIVE_PROVIDER_CAPABILITIES:-$(python3 - <<'PY'
import importlib.util
import os
from pathlib import Path

module_path = Path("scripts/build_live_provider_matrix.py").resolve()
spec = importlib.util.spec_from_file_location("build_live_provider_matrix", module_path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
backend_name = os.getenv("LIVE_PROVIDER_MATRIX_BACKEND") or os.getenv("SEEKDB_DOCKER_BACKEND") or "seekdb"
print(module.capability_set_for_backend(backend_name))
PY
)}"

wait_for_seekdb
ensure_database_exists
uv run python scripts/seekdb_checkpoint_smoke.py
uv run pytest \
  tests/integration/test_live_provider_streaming.py \
  tests/e2e/test_live_provider_api.py \
  tests/e2e/test_live_provider_mcp.py \
  -q
