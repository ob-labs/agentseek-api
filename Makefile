.PHONY: test test-cov test-samples test-e2e test-checkpoints test-seekdb

test:
	uv run pytest tests/unit tests/integration -q

test-cov:
	uv run pytest tests/unit tests/integration --cov=src/agentseek_api --cov-report=term-missing --cov-fail-under=90 -q

test-samples:
	uv run python examples/run_sample_graphs.py
	uv run python tests/e2e/e2e_inprocess_flow.py

test-e2e:
	uv run pytest tests/e2e -q -m e2e

test-checkpoints:
	bash ./scripts/test-checkpoints.sh

test-seekdb: test-checkpoints
