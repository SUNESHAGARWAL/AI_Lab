import pytest
from llm.config import GatewaySettings


@pytest.fixture
def clean_audit_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.delenv("LLM_AUDIT_TRACE", raising=False)
    monkeypatch.delenv("LLM_AUDIT_PIN_DEEPSEEK", raising=False)
    return monkeypatch


def test_audit_settings_default_off(clean_audit_env: pytest.MonkeyPatch) -> None:
    # `_env_file=None` keeps a developer's local `.env` out of the defaults check.
    settings = GatewaySettings(_env_file=None)

    assert settings.audit_trace is None
    assert settings.audit_pin_deepseek is False


def test_audit_trace_loads_path_from_env(clean_audit_env: pytest.MonkeyPatch) -> None:
    clean_audit_env.setenv("LLM_AUDIT_TRACE", "/tmp/attempts.jsonl")

    assert GatewaySettings(_env_file=None).audit_trace == "/tmp/attempts.jsonl"


def test_audit_trace_loads_stdout_from_env(clean_audit_env: pytest.MonkeyPatch) -> None:
    clean_audit_env.setenv("LLM_AUDIT_TRACE", "stdout")

    assert GatewaySettings(_env_file=None).audit_trace == "stdout"


def test_audit_pin_deepseek_loads_from_env(clean_audit_env: pytest.MonkeyPatch) -> None:
    clean_audit_env.setenv("LLM_AUDIT_PIN_DEEPSEEK", "true")

    assert GatewaySettings(_env_file=None).audit_pin_deepseek is True
