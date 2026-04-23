from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests
from langchain_ollama import ChatOllama

from .config import AppConfig


@dataclass
class ChatResult:
    content: str


class VLLMChatClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        timeout_seconds: float,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds

    def invoke(self, prompt: str) -> ChatResult:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return ChatResult(content=str(content))


def build_llm(config: AppConfig):
    if config.llm_provider == "vllm":
        return VLLMChatClient(
            model=config.vllm_model,
            base_url=config.vllm_base_url,
            api_key=config.vllm_api_key,
            temperature=config.temperature,
            timeout_seconds=config.vllm_timeout_seconds,
        )
    return ChatOllama(
        model=config.ollama_model,
        base_url=config.ollama_base_url,
        temperature=config.temperature,
        format="json",
        sync_client_kwargs={"timeout": config.ollama_timeout_seconds},
    )
