from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

import requests


@dataclass(frozen=True)
class OllamaHealth:
    reachable: bool
    model_available: bool
    models: tuple[str, ...] = ()
    error: str = ""


def check_ollama(base_url: str, model: str, timeout: float = 5.0) -> OllamaHealth:
    """Check that the Ollama HTTP API is reachable and the model is present."""
    tags_url = urljoin(base_url.rstrip("/") + "/", "api/tags")
    try:
        response = requests.get(tags_url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return OllamaHealth(
            reachable=False,
            model_available=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    except ValueError as exc:
        return OllamaHealth(
            reachable=False,
            model_available=False,
            error=f"Ollama returned non-JSON from {tags_url}: {exc}",
        )

    models = tuple(
        item.get("name") or item.get("model") or ""
        for item in data.get("models", [])
        if isinstance(item, dict)
    )
    return OllamaHealth(
        reachable=True,
        model_available=model in models,
        models=models,
    )


def format_ollama_error(base_url: str, model: str, health: OllamaHealth | None = None) -> str:
    lines = [
        f"Cannot use Ollama model '{model}' at {base_url}.",
    ]

    if health is not None and health.error:
        lines.append(f"Details: {health.error}")

    if health is not None and health.reachable and not health.model_available:
        available = ", ".join(health.models) if health.models else "none"
        lines.extend(
            [
                f"Ollama is running, but the model is not installed. Available models: {available}",
                "",
                f"Install it with: ollama pull {model}",
            ]
        )
    else:
        lines.extend(
            [
                "Ollama does not appear to be reachable.",
                "",
                "Start it in a separate terminal with: ollama serve",
                f"Then make sure the model exists with: ollama pull {model}",
            ]
        )

    lines.extend(
        [
            "",
            "If you use a different Ollama host, pass it with:",
            "python -m hate_rag_agents.classify --ollama-base-url http://HOST:11434 --model MODEL --text \"...\"",
        ]
    )
    return "\n".join(lines)
