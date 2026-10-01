# Durable streaming and SDK compatibility

Run streams use a single persisted, run-scoped cursor. Raw, filtered, and
create-and-stream responses use the same IDs; thread stream IDs belong to a
separate thread log and must not be sent to a run endpoint. Reconnect using the
last received `id` in `Last-Event-ID`. Reconnecting after the terminal record
does not repeat the terminal error or already delivered content.

## Message wire format

The SDK endpoints follow the
[official LangChain streaming API](https://docs.langchain.com/langsmith/streaming)
and [SDK stream types](https://github.com/langchain-ai/langgraph/blob/main/libs/sdk-py/langgraph_sdk/schema.py):

- Request `messages`: receive `messages/metadata`, accumulated
  `messages/partial` message lists, then `messages/complete` lists. Replace the
  prior partial for the same message ID; do not concatenate accumulated text.
- Request `messages-tuple`: receive `messages` events containing
  `[message_chunk, metadata]`. These are deltas. The request mode and SSE event
  name intentionally differ.
- Message IDs are stable within a model invocation. Explicit provider IDs are
  retained; missing IDs are derived from invocation identity and namespace.
  Custom graph adapters must invoke LangChain model callbacks or supply
  `model_invocation_id` metadata for id-less messages.
- AgentSeek protocol-v2 block events remain on the thread protocol endpoint;
  they are not mixed into the SDK accumulated-message stream.

Migration: AgentSeek's old custom `message_chunk` SSE name is not the official
SDK event name and is no longer emitted. Consumers should select one of the
two message modes above. This is an intentional wire compatibility change;
there is no duplicate legacy alias in the same stream.

## Durability and recovery

Inline execution commits state, terminal records, and their cursors together.
Protocol pairs share a transaction even at a batch boundary. Live responses
read the durable logs, so lost or reordered broker notifications cannot skip
committed events. Bounded buffering adds up to the configured flush interval
before visibility; it does not wait for the whole model response.

SQL stores execution generation, lease ownership, dispatch intent, and pending
terminal delivery. Startup/background reconciliation recovers abandoned work;
a stored terminal result is delivered without rerunning the graph. Retrying
graph execution after a crash can repeat graph-side effects: tools requiring
exactly-once behavior need their own idempotency keys.

Before upgrading an existing deployment, stop/drain the old inline workers.
Pre-upgrade pending/running inline rows have no durable generation or resume
command. Startup marks them `error` with `RunInterruptedByUpgrade` and releases
their busy threads; explicitly resubmit them if needed. It does not guess a
resume command or silently repeat unknown side effects. Redis legacy queued
payloads retain their worker-lease recovery path.

Redis terminal delivery has three phases: persist the intended result in SQL,
append both durable stream envelopes, then apply the final SQL state. An SSE
terminal event can therefore appear while GET run temporarily reports
`running` if the last SQL commit must be retried. Reconciliation converges to
the stored result. This is not a cross-store atomic transaction. Redis itself
must be configured for the persistence/replication guarantees your deployment
requires; an accepted write to a non-persistent Redis instance is not a
power-loss guarantee.

Generation-specific terminal cleanup records are stored separately from runs.
Deletion and interrupt/resume cannot discard the only ownership record for
non-expiring Redis delivery markers. Reconciliation removes deleted-run markers
or gives completed-generation markers their normal bounded retention.

Provider-backed proof stays in `.github/workflows/live-provider-streaming.yml`,
not default PR CI. It must demonstrate multiple incremental frames and final
reconstruction, using only repository secrets for credentials. Offline tests
and a green unit suite are not substitutes for that exact-commit provider run.
