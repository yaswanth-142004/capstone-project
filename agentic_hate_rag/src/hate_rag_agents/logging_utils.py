from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


LOGGER_NAME = "hate_rag_agents"


def setup_app_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved = str(path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == resolved:
            return logger

    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.info("logging_started path=%s", resolved)
    return logger


def get_app_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


@contextmanager
def log_timing(event: str, **fields: Any) -> Iterator[None]:
    logger = get_app_logger()
    start = time.perf_counter()
    logger.info("%s_start %s", event, _format_fields(fields))
    try:
        yield
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("%s_error elapsed_ms=%.2f %s", event, elapsed_ms, _format_fields(fields))
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s_end elapsed_ms=%.2f %s", event, elapsed_ms, _format_fields(fields))


def _format_fields(fields: dict[str, Any]) -> str:
    return " ".join(f"{key}={_clean_value(value)}" for key, value in fields.items())


def _clean_value(value: Any) -> str:
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if len(text) > 220:
        text = text[:217] + "..."
    return text
