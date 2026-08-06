from typing import Literal

from llm.models import Tier


class BudgetExceeded(Exception):
    """Raised before spending, never after — see `.claude/rules/llm-gateway.md`."""

    def __init__(
        self, scope: Literal["per_request", "per_day"], estimated_tokens: int, ceiling: int
    ) -> None:
        self.scope = scope
        self.estimated_tokens = estimated_tokens
        self.ceiling = ceiling
        super().__init__(
            f"budget exceeded ({scope}): estimated {estimated_tokens} tokens against a "
            f"ceiling of {ceiling}"
        )


class AllProvidersExhausted(Exception):
    """Raised when every provider in a tier's fallback chain has failed. The API
    surfaces this as a 503 with a retry hint — never a 500, never a silent empty answer."""

    def __init__(self, tier: Tier) -> None:
        self.tier = tier
        super().__init__(f"all providers exhausted for tier {tier.value!r}")
