import pytest
from llm.models import CompletionRequest, Message, Tier
from pydantic import ValidationError


def _messages() -> list[Message]:
    return [Message(role="user", content="q")]


def test_request_without_feature_validates() -> None:
    request = CompletionRequest(tier=Tier.FAST, messages=_messages())

    assert request.feature is None


def test_request_accepts_feature_up_to_64_chars() -> None:
    request = CompletionRequest(tier=Tier.FAST, messages=_messages(), feature="g" * 64)

    assert request.feature == "g" * 64


def test_request_rejects_feature_longer_than_64_chars() -> None:
    with pytest.raises(ValidationError):
        CompletionRequest(tier=Tier.FAST, messages=_messages(), feature="g" * 65)
