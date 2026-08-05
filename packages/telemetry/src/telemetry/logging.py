import logging

import structlog

_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Structured logger. Never log full prompt/response bodies at INFO — token counts
    and hashes only, per the project's cost/security rules."""
    _configure()
    return structlog.get_logger(name)
