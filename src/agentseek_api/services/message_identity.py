"""Model invocation identity, distinct from execution and delivery identities."""
from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage


class MessageIdentityTracker:
    def __init__(self):
        self.identities: dict[tuple, str] = {}

    def resolve(self, *, invocation_id: str, namespace: tuple[str, ...],
                provider_message_id: str | None, message_index: int, complete: bool) -> str:
        if not invocation_id and not provider_message_id:
            raise ValueError("Id-less messages require a model invocation ID from callbacks or events")
        key = (invocation_id, namespace, message_index)
        if key not in self.identities:
            self.identities[key] = provider_message_id or f"message:{uuid5(NAMESPACE_URL, json.dumps(key))}"
        return self.identities[key]


class ModelInvocationCallback(BaseCallbackHandler):
    """Observe the same message object emitted by LangGraph's messages handler.

    Holding its reference until consumption avoids object-ID reuse. This callback
    runs inline before stream consumption and never writes to provider objects.
    """
    run_inline = True

    def __init__(self):
        self.messages: dict[int, tuple[BaseMessage, str, str | None]] = {}

    def on_llm_new_token(self, token, *, chunk=None, run_id, **kwargs):
        message = getattr(chunk, "message", None)
        if isinstance(message, BaseMessage):
            self.messages[id(message)] = (message, str(run_id), message.id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                if isinstance(message, BaseMessage):
                    self.messages[id(message)] = (message, str(run_id), message.id)

    def take(self, message):
        observed = self.messages.pop(id(message), None)
        return observed[1:] if observed is not None and observed[0] is message else None

    def attach(self, config):
        callbacks = config.get("callbacks")
        if callbacks is None or isinstance(callbacks, list):
            config["callbacks"] = [*(callbacks or []), self]
        else:
            manager = callbacks.copy()
            manager.add_handler(self, inherit=True)
            config["callbacks"] = manager
