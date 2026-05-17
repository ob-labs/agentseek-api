#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"
MYSQL_PORT="${MYSQL_PORT:-3306}"
MYSQL_USER="${MYSQL_USER:-root}"
MYSQL_PASSWORD="${MYSQL_PASSWORD:-root}"
MYSQL_DB="${MYSQL_DB:-seekdb}"
EXAMPLE_API_PORT="${EXAMPLE_API_PORT:-2024}"
EXAMPLE_BASE_URL="${EXAMPLE_BASE_URL:-http://127.0.0.1:${EXAMPLE_API_PORT}}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.tmp/cli-dev-samples}"
SERVER_LOG="${SERVER_LOG:-$TMP_DIR/agentseek-dev.log}"
PID_FILE="$TMP_DIR/agentseek-dev.pid"

mkdir -p "$TMP_DIR"

cleanup() {
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
      wait "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}
trap cleanup EXIT

export METADATA_DB_URL="sqlite+aiosqlite:///${TMP_DIR}/metadata.db"
export OCEANBASE_HOST="$MYSQL_HOST"
export OCEANBASE_PORT="$MYSQL_PORT"
export OCEANBASE_USER="$MYSQL_USER"
export OCEANBASE_PASSWORD="$MYSQL_PASSWORD"
export OCEANBASE_DB_NAME="$MYSQL_DB"
export SEEKDB_URL="mysql+aiomysql://${MYSQL_USER}:${MYSQL_PASSWORD}@${MYSQL_HOST}:${MYSQL_PORT}/${MYSQL_DB}"
export EXAMPLE_BASE_URL

uv run python - <<'PY'
import os
import time
import pymysql

host = os.environ["OCEANBASE_HOST"]
port = int(os.environ["OCEANBASE_PORT"])
user = os.environ["OCEANBASE_USER"]
password = os.environ["OCEANBASE_PASSWORD"]
db_name = os.environ["OCEANBASE_DB_NAME"]

deadline = time.time() + 120
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
        with conn.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
        conn.close()
        raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)

raise SystemExit(f"MySQL did not become ready: {last_error}")
PY

uv run agentseek dev \
  --config examples/sample_graphs_manifest.json \
  --host 127.0.0.1 \
  --port "$EXAMPLE_API_PORT" \
  --no-reload \
  >"$SERVER_LOG" 2>&1 &
echo "$!" >"$PID_FILE"

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
            raise SystemExit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(1)

raise SystemExit(f"agentseek dev did not become ready: {last_error}")
PY

uv run python tests/e2e/e2e_live_http_multi_graph.py
