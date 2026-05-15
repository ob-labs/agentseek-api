#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_TAG="${IMAGE_TAG:-agentseek-api-cli-smoke:latest}"
DB_CONTAINER="${DB_CONTAINER:-agentseek-cli-mysql}"
APP_CONTAINER="${APP_CONTAINER:-agentseek-up-8123}"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/.tmp/cli-docker}"

cleanup() {
  docker rm -f "$APP_CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$DB_CONTAINER" >/dev/null 2>&1 || true
}

print_logs() {
  docker logs "$APP_CONTAINER" || true
  docker logs "$DB_CONTAINER" || true
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

uv run agentseek dockerfile --config examples/external_graph/manifest.json "$TMP_DIR/agentseek.Dockerfile"
test -s "$TMP_DIR/agentseek.Dockerfile"

uv run agentseek build --config examples/external_graph/manifest.json -t "$IMAGE_TAG"

docker rm -f "$DB_CONTAINER" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "$DB_CONTAINER" \
  -e MYSQL_ALLOW_EMPTY_PASSWORD=yes \
  -e MYSQL_DATABASE=seekdb \
  -p 3306:3306 \
  mysql:8.4 >/dev/null

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

uv run agentseek up \
  --config examples/external_graph/manifest.json \
  --image "$IMAGE_TAG" \
  --port 8123 \
  --env-file "$TMP_DIR/up.env" \
  --recreate

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

if ! curl -fsS "http://127.0.0.1:8123/info" | grep -q '"version"'; then
  print_logs
  echo "App info endpoint did not return version metadata." >&2
  exit 1
fi

uv run python - <<'PY'
import json
import urllib.request

base_url = "http://127.0.0.1:8123"
headers = {"Content-Type": "application/json", "x-user-id": "cli-docker-smoke"}

def request(path: str, payload: dict | None = None, method: str = "GET") -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}

assistant = request("/assistants", {"name": "external-smoke", "graph_id": "external_hello"}, method="POST")
thread = request("/threads", {"metadata": {"source": "cli-docker-smoke"}}, method="POST")
run = request(
    f"/threads/{thread['thread_id']}/runs",
    {"assistant_id": assistant["assistant_id"], "input": {"message": "hello-from-docker"}},
    method="POST",
)
waited = request(f"/threads/{thread['thread_id']}/runs/{run['run_id']}/wait")
assert waited["status"] == "success", waited
output = waited["output"]
assert output["final_text"] == "external graph heard: hello-from-docker", output
PY
