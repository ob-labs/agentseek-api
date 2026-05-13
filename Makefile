.PHONY: test test-cov test-e2e test-seekdb

test:
	uv run pytest tests/unit tests/integration -q

test-cov:
	uv run pytest tests/unit tests/integration --cov=src/agentseek_api --cov-report=term-missing --cov-fail-under=90 -q

test-e2e:
	uv run pytest tests/e2e -q -m e2e

test-seekdb:
	bash ./scripts/test-seekdb.sh
