from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class VLLMHealth:
    reachable: bool
    model_available: bool
    models: tuple[str, ...] = ()
    error: str = ""


def check_vllm(base_url: str, model: str, api_key: str = "", timeout: float = 5.0) -> VLLMHealth:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        response = requests.get(f"{base_url.rstrip('/')}/v1/models", headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        return VLLMHealth(
            reachable=False,
            model_available=False,
            error=str(exc),
        )

    try:
        data = response.json()
    except ValueError as exc:
        return VLLMHealth(
            reachable=False,
            model_available=False,
            error=f"Invalid JSON from vLLM endpoint: {exc}",
        )

    models = tuple(
        str(item.get("id") or "").strip()
        for item in data.get("data", [])
        if str(item.get("id") or "").strip()
    )
    return VLLMHealth(
        reachable=True,
        model_available=model in models,
        models=models,
    )


def format_vllm_error(base_url: str, model: str, health: VLLMHealth | None = None) -> str:
    lines = [
        f"Cannot use vLLM model '{model}' at {base_url}.",
    ]
    if health is not None and health.error:
        lines.append(f"Error: {health.error}")
    if health is not None and health.reachable and not health.model_available:
        available = ", ".join(health.models) if health.models else "none"
        lines.extend(
            [
                f"vLLM is running, but the model is not available. Reported models: {available}",
                "Make sure the served model name matches VLLM_MODEL or pass --model explicitly.",
            ]
        )
    else:
        lines.extend(
            [
                "Make sure your vLLM backend or wrapper service is running.",
                "Example classifier usage:",
                'python -m hate_rag_agents.classify --llm-provider vllm --vllm-base-url http://HOST:8090 --model MODEL --text "..."',
            ]
        )
    return "\n".join(lines)
