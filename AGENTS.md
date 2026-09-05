# Agent Notes

## Live Provider Streaming CI

- Keep real model streaming checks in the manual workflow `.github/workflows/live-provider-streaming.yml`.
- Do not move these provider-backed checks into the always-on `ci.yml` path unless the user explicitly asks for spendier default coverage.
- The manual workflow is the canonical way to prove token streaming against real providers.

### Configuration contract

- OpenAI-compatible:
  - repo variable `OPENAI_COMPAT_MODEL`
  - repo variable `OPENAI_COMPAT_BASE_URL`
  - repo secret `OPENAI_COMPAT_API_KEY`
- Anthropic-compatible:
  - repo variable `ANTHROPIC_COMPAT_MODEL`
  - repo variable `ANTHROPIC_COMPAT_BASE_URL`
  - repo secret `ANTHROPIC_COMPAT_API_KEY`

### Workflow behavior

- `workflow_dispatch` inputs may override the model and base URL for a single run.
- API keys stay in GitHub secrets only; do not add dispatch inputs for secrets.
- The workflow runs `tests/integration/test_live_provider_streaming.py` against manifest-registered graphs in `examples/live_provider_graphs/manifest.json`.
- The proof target is SSE `message_chunk` events from real provider-backed graphs, not just a final successful response.
