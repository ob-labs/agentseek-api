#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_TAG="${IMAGE_TAG:-agentseek-api-redis-smoke:latest}"
CONFIG_PATH="${CONFIG_PATH:-examples/docker_ci_auth/manifest.json}"
NETWORK_NAME="${NETWORK_NAME:-agentseek-redis-runtime}"
BACKEND_CONTAINER="${BACKEND_CONTAINER:-agentseek-redis-backend}"
REDIS_CONTAINER="${REDIS_CONTAINER:-agentseek-redis}"
REDIS_HOST_PORT="${REDIS_HOST_PORT:-6391}"
API_CONTAINER="${API_CONTAINER:-agentseek-redis-api}"
WORKER_CONTAINER="${WORKER_CONTAINER:-agentseek-redis-worker}"
API_PORT="${API_PORT:-8126}"
SEEKDB_DOCKER_BACKEND="${SEEKDB_DOCKER_BACKEND:-seekdb}"
SEEKDB_DOCKER_IMAGE="${SEEKDB_DOCKER_IMAGE:-}"
OCEANBASE_DOCKER_MODE="${OCEANBASE_DOCKER_MODE:-mini}"
STATE_DIR="${STATE_DIR:-$ROOT_DIR/.tmp/redis-runtime}"
RESUME_STATE_FILE="$STATE_DIR/resume-state.json"
SHUTDOWN_STATE_FILE=""
REDIS_WORKER_LOCK_KEY="${REDIS_WORKER_LOCK_KEY:-agentseek:worker:active}"
WORKER_CONCURRENT_JOBS="${WORKER_CONCURRENT_JOBS:-10}"

cleanup() {
  local status=$?

  trap - EXIT
  stop_worker >/dev/null 2>&1 || true
  docker rm -f "$API_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$BACKEND_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
  if [[ -n "$SHUTDOWN_STATE_FILE" ]]; then
    rm -f "$SHUTDOWN_STATE_FILE" || true
  fi
  exit "$status"
}

print_logs() {
  echo "=== docker ps ==="
  docker ps -a || true
  echo "=== backend logs ==="
  docker logs "$BACKEND_CONTAINER" || true
  echo "=== redis logs ==="
  docker logs "$REDIS_CONTAINER" || true
  echo "=== api logs ==="
  docker logs "$API_CONTAINER" || true
  echo "=== worker logs ==="
  docker logs "$WORKER_CONTAINER" || true
}

trap cleanup EXIT

set_backend_defaults() {
  local encoded_user

  case "$1" in
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
      echo "Unsupported SEEKDB_DOCKER_BACKEND: $1" >&2
      exit 1
      ;;
  esac

  encoded_user="${OCEANBASE_USER//@/%40}"
  export SEEKDB_URL="${SEEKDB_URL:-mysql+aiomysql://${encoded_user}:${OCEANBASE_PASSWORD}@${BACKEND_CONTAINER}:${OCEANBASE_PORT}/${OCEANBASE_DB_NAME}}"
}

wait_for_backend() {
  if ! uv run python - <<'PY'
import os
import time
import pymysql

host = "127.0.0.1"
port = int(os.environ["OCEANBASE_PORT"])
user = os.environ["OCEANBASE_USER"]
password = os.environ["OCEANBASE_PASSWORD"]
deadline = time.time() + 420
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

raise SystemExit(f"backend did not become ready: {last_error}")
PY
  then
    print_logs
    exit 1
  fi
}

wait_for_redis() {
  local attempt

  for ((attempt = 1; attempt <= 30; attempt++)); do
    if docker exec "$REDIS_CONTAINER" redis-cli ping >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "Redis did not become ready" >&2
  print_logs
  return 1
}

ensure_database_exists() {
  uv run python - <<'PY'
import os
import time
import pymysql

host = "127.0.0.1"
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

raise SystemExit(f"database bootstrap failed: {last_error}")
PY
}

wait_for_api() {
  local base_url="$1"

  if ! uv run python - <<'PY'
import os
import time
import httpx

base_url = os.environ["BASE_URL"]
deadline = time.time() + 120
last_error = None

while time.time() < deadline:
    try:
        response = httpx.get(f"{base_url}/health", timeout=2.0, trust_env=False)
        if response.status_code == 200:
            raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(1)

raise SystemExit(f"api did not become ready: {last_error}")
PY
  then
    print_logs
    exit 1
  fi
}

stop_worker() {
  local timeout="${1:-10}"
  local container_id
  local running
  local status

  if container_id="$(docker ps -aq --filter "name=^/${WORKER_CONTAINER}$")"; then
    :
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
  if [[ -z "$container_id" ]]; then
    return 0
  fi

  if running="$(docker inspect --format '{{.State.Running}}' "$container_id")"; then
    :
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
  if [[ "$running" == "true" ]]; then
    if docker stop -t "$timeout" "$WORKER_CONTAINER" >/dev/null; then
      :
    else
      status=$?
      print_logs >&2
      return "$status"
    fi
  fi
  if docker rm -f "$WORKER_CONTAINER" >/dev/null; then
    :
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
  if docker exec "$REDIS_CONTAINER" redis-cli DEL "$REDIS_WORKER_LOCK_KEY" >/dev/null; then
    return 0
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
}

start_worker() {
  local status

  if stop_worker; then
    :
  else
    status=$?
    return "$status"
  fi
  if docker run -d \
    --name "$WORKER_CONTAINER" \
    --network "$NETWORK_NAME" \
    -e EXECUTOR_BACKEND=redis \
    -e REDIS_URL="redis://${REDIS_CONTAINER}:6379/0" \
    -e REDIS_WORKER_LOCK_KEY="${REDIS_WORKER_LOCK_KEY}" \
    -e WORKER_CONCURRENT_JOBS="${WORKER_CONCURRENT_JOBS}" \
    -e SEEKDB_URL="${SEEKDB_URL}" \
    -e OCEANBASE_HOST="${BACKEND_CONTAINER}" \
    -e OCEANBASE_PORT="${OCEANBASE_PORT}" \
    -e OCEANBASE_USER="${OCEANBASE_USER}" \
    -e OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD}" \
    -e OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME}" \
    "$IMAGE_TAG" \
    python -m agentseek_api.cli worker >/dev/null; then
    return 0
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
}

run_probe() {
  local status

  if uv run python scripts/verify_worker_concurrency.py \
    --base-url "$BASE_URL" \
    --redis-url "redis://127.0.0.1:${REDIS_HOST_PORT}/0" \
    "$@"; then
    return 0
  else
    status=$?
    print_logs >&2
    return "$status"
  fi
}

start_backend() {
  docker rm -f "$BACKEND_CONTAINER" >/dev/null 2>&1 || true

  case "$SEEKDB_DOCKER_BACKEND" in
    seekdb)
      docker run -d \
        --name "$BACKEND_CONTAINER" \
        --network "$NETWORK_NAME" \
        -p "${OCEANBASE_PORT}:2881" \
        -p 2886:2886 \
        "$SEEKDB_DOCKER_IMAGE" >/dev/null
      ;;
    oceanbase)
      docker run -d \
        --name "$BACKEND_CONTAINER" \
        --network "$NETWORK_NAME" \
        -e MODE="${OCEANBASE_DOCKER_MODE}" \
        -p "${OCEANBASE_PORT}:2881" \
        "$SEEKDB_DOCKER_IMAGE" >/dev/null
      ;;
    mysql)
      docker run -d \
        --name "$BACKEND_CONTAINER" \
        --network "$NETWORK_NAME" \
        -e MYSQL_ROOT_PASSWORD="${OCEANBASE_PASSWORD}" \
        -e MYSQL_DATABASE="${OCEANBASE_DB_NAME}" \
        -p "${OCEANBASE_PORT}:3306" \
        "$SEEKDB_DOCKER_IMAGE" \
        --character-set-server=utf8mb4 \
        --collation-server=utf8mb4_unicode_ci >/dev/null
      ;;
  esac
}

set_backend_defaults "$SEEKDB_DOCKER_BACKEND"

docker network rm "$NETWORK_NAME" >/dev/null 2>&1 || true
docker network create "$NETWORK_NAME" >/dev/null
mkdir -p "$STATE_DIR"
SHUTDOWN_STATE_FILE="$(mktemp "$STATE_DIR/shutdown-state.XXXXXX")"

uv run agentseek-api build --config "$CONFIG_PATH" -t "$IMAGE_TAG"

start_backend

docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$REDIS_CONTAINER" \
  --network "$NETWORK_NAME" \
  -p "127.0.0.1:${REDIS_HOST_PORT}:6379" \
  redis:7-alpine >/dev/null

wait_for_redis

if ! AGENTSEEK_TEST_REDIS_URL="redis://127.0.0.1:${REDIS_HOST_PORT}/0" \
  uv run pytest \
    tests/integration/test_live_redis_stream_persistence.py \
    tests/integration/test_live_redis_queue.py \
    -q; then
  print_logs
  exit 1
fi

wait_for_backend
ensure_database_exists

docker rm -f "$API_CONTAINER" >/dev/null 2>&1 || true
docker run -d \
  --name "$API_CONTAINER" \
  --network "$NETWORK_NAME" \
  -p "${API_PORT}:2024" \
  -e EXECUTOR_BACKEND=redis \
  -e REDIS_URL="redis://${REDIS_CONTAINER}:6379/0" \
  -e SEEKDB_URL="${SEEKDB_URL}" \
  -e OCEANBASE_HOST="${BACKEND_CONTAINER}" \
  -e OCEANBASE_PORT="${OCEANBASE_PORT}" \
  -e OCEANBASE_USER="${OCEANBASE_USER}" \
  -e OCEANBASE_PASSWORD="${OCEANBASE_PASSWORD}" \
  -e OCEANBASE_DB_NAME="${OCEANBASE_DB_NAME}" \
  "$IMAGE_TAG" >/dev/null

BASE_URL="http://127.0.0.1:${API_PORT}"
export BASE_URL
wait_for_api "$BASE_URL"

start_worker

if ! uv run python scripts/verify_docker_api.py --base-url "$BASE_URL" --mode full; then
  print_logs
  exit 1
fi

if ! uv run python scripts/verify_docker_api.py --base-url "$BASE_URL" --mode resume-seed >"$RESUME_STATE_FILE"; then
  print_logs
  exit 1
fi

start_worker

if ! uv run python scripts/verify_docker_api.py --base-url "$BASE_URL" --mode resume-check --state-file "$RESUME_STATE_FILE"; then
  print_logs
  exit 1
fi

WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode bounded

WORKER_CONCURRENT_JOBS=10
start_worker
run_probe --mode fanout

WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode failure

WORKER_CONCURRENT_JOBS=2
start_worker
run_probe --mode shutdown-seed >"$SHUTDOWN_STATE_FILE"
stop_worker 1
start_worker
run_probe --mode shutdown-check --state-file "$SHUTDOWN_STATE_FILE"
