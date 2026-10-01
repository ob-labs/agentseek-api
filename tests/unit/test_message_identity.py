def test_invocation_identity_keeps_three_chunks_and_completion_together():
    from agentseek_api.services.message_identity import MessageIdentityTracker
    tracker = MessageIdentityTracker()
    def resolve(invocation, namespace=(), index=0, provider=None, complete=False):
        return tracker.resolve(invocation_id=invocation, namespace=namespace,
            provider_message_id=provider, message_index=index, complete=complete)
    first = resolve("call-a")
    assert [resolve("call-a") for _ in range(3)] == [first] * 3
    second = resolve("call-b")
    assert second != first and resolve("call-a", complete=True) == first
    assert resolve("call-a", ("child",)) not in {first, second}
    assert resolve("call-a", index=1) != first
    assert resolve("call-c", provider="provider-message") == "provider-message"
    # A late provider ID cannot split a message already visible to clients.
    assert resolve("call-a", provider="late-id") == first


def test_idless_custom_adapter_must_supply_invocation_identity():
    import pytest
    from agentseek_api.services.message_identity import MessageIdentityTracker
    with pytest.raises(ValueError, match="model invocation ID"):
        MessageIdentityTracker().resolve(invocation_id="", namespace=(),
            provider_message_id=None, message_index=0, complete=False)


def test_completion_callback_preserves_existing_manager_and_message():
    from uuid import uuid4
    from langchain_core.callbacks import CallbackManager
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import LLMResult, ChatGeneration
    from agentseek_api.services.message_identity import ModelInvocationCallback

    callback = ModelInvocationCallback()
    existing = CallbackManager([])
    config = {"callbacks": existing}
    callback.attach(config)
    assert callback not in existing.handlers
    assert callback in config["callbacks"].handlers
    message = AIMessage(content="complete", id="provider-id")
    invocation = uuid4()
    callback.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=invocation)
    assert callback.take(message) == (str(invocation), "provider-id")
    assert callback.take(message) is None
    assert message.id == "provider-id"
