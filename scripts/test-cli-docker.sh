#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_TAG="${IMAGE_TAG:-agentseek-api-cli-smoke:latest}"
DB_CONTAINER="${DB_CONTAINER:-agentseek-cli-mysql}"
APP_CONTAINER="${APP_CONTAINER:-agentseek-up-8123}"
APP_CONTAINER_AUTOBUILD="${APP_CONTAINER_AUTOBUILD:-agentseek-up-8124}"
PG_CONTAINER="${PG_CONTAINER:-agentseek-cli-postgres}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.tmp/cli-docker}"

cleanup() {
  docker rm -f "$APP_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$APP_CONTAINER_AUTOBUILD" >/dev/null 2>&1 || true
  docker rm -f "$DB_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 || true
}

print_logs() {
  docker logs "$APP_CONTAINER" || true
  docker logs "$APP_CONTAINER_AUTOBUILD" || true
  docker logs "$DB_CONTAINER" || true
  docker logs "$PG_CONTAINER" || true
}

trap cleanup EXIT

mkdir -p "$TMP_DIR"

cat >"$TMP_DIR/up.env" <<'EOF'
METADATA_DB_URL=sqlite+aiosqlite:////tmp/agentseek.db
OCEANBASE_HOST=host.docker.internal
OCEANBASE_PORT=3306
OCEANBASE_USER=root
OCEANBASE_PASSWORD=
OCEANBASE_DB_NAME=seekdb
EOF

CONFIG_PATH="examples/docker_ci_auth/manifest.json"

uv run agentseek-api dockerfile --config "$CONFIG_PATH" "$TMP_DIR/agentseek.Dockerfile"
test -s "$TMP_DIR/agentseek.Dockerfile"

uv run agentseek-api build --config "$CONFIG_PATH" -t "$IMAGE_TAG"

docker rm -f "$DB_CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "$DB_CONTAINER" \
  -e MYSQL_ALLOW_EMPTY_PASSWORD=yes \
  -e MYSQL_DATABASE=seekdb \
  -p 3306:3306 \
  mysql:8.4 >/dev/null

docker rm -f "$PG_CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "$PG_CONTAINER" \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=agentseek \
  -p 5432:5432 \
  postgres:16 >/dev/null

for _ in $(seq 1 60); do
  if docker exec "$DB_CONTAINER" mysqladmin ping -h 127.0.0.1 --silent >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker exec "$DB_CONTAINER" mysqladmin ping -h 127.0.0.1 --silent >/dev/null 2>&1; then
  print_logs
  echo "MySQL container did not become ready." >&2
  exit 1
fi

for _ in $(seq 1 60); do
  if docker exec "$PG_CONTAINER" pg_isready -U postgres -d agentseek >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! docker exec "$PG_CONTAINER" pg_isready -U postgres -d agentseek >/dev/null 2>&1; then
  print_logs
  echo "PostgreSQL container did not become ready." >&2
  exit 1
fi

if ! uv run agentseek-api up \
  --config "$CONFIG_PATH" \
  --image "$IMAGE_TAG" \
  --port 8123 \
  --env-file "$TMP_DIR/up.env" \
  --recreate; then
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
  echo "App container did not become healthy." >&2
  exit 1
fi

if ! uv run python scripts/verify_docker_api.py --base-url http://127.0.0.1:8123 --mode full; then
  print_logs
  exit 1
fi

DUPLICATE_STDERR="$TMP_DIR/up-duplicate.stderr"
set +e
uv run agentseek-api up \
  --config "$CONFIG_PATH" \
  --image "$IMAGE_TAG" \
  --port 8123 \
  --env-file "$TMP_DIR/up.env" \
  2>"$DUPLICATE_STDERR"
DUPLICATE_EXIT=$?
set -e

if [[ "$DUPLICATE_EXIT" -eq 0 ]]; then
  print_logs
  echo "Duplicate agentseek-api up unexpectedly succeeded without --recreate." >&2
  exit 1
fi

if ! grep -q "already exists" "$DUPLICATE_STDERR" || ! grep -q -- "--recreate" "$DUPLICATE_STDERR"; then
  print_logs
  cat "$DUPLICATE_STDERR" >&2 || true
  echo "Duplicate agentseek-api up did not emit the expected recreate guidance." >&2
  exit 1
fi

if grep -Eqi "Conflict|already in use by container|Error response from daemon" "$DUPLICATE_STDERR"; then
  print_logs
  cat "$DUPLICATE_STDERR" >&2 || true
  echo "Duplicate agentseek-api up leaked raw Docker conflict output." >&2
  exit 1
fi

if ! uv run agentseek-api up \
  --config "$CONFIG_PATH" \
  --port 8124 \
  --base-image python:3.13-slim-bookworm \
  --env-file "$TMP_DIR/up.env" \
  --postgres-uri postgresql://postgres:postgres@host.docker.internal:5432/agentseek \
  --no-pull \
  --wait \
  --recreate; then
  print_logs
  exit 1
fi

if ! curl -fsS "http://127.0.0.1:8124/health" | grep -q '"healthy"'; then
  print_logs
  echo "Auto-built app container did not become healthy." >&2
  exit 1
fi

if ! uv run python scripts/verify_docker_api.py --base-url http://127.0.0.1:8124 --mode smoke; then
  print_logs
  exit 1
fi
