from __future__ import annotations

import argparse
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


@dataclass(frozen=True)
class ServiceSettings:
    backend_url: str
    forced_model: str
    api_key: str
    request_timeout: int
    cors_origins: list[str]
    host: str
    port: int


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False


def message_to_dict(message: Message) -> dict[str, str]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


def build_settings(args: argparse.Namespace) -> ServiceSettings:
    if args.env_file:
        load_dotenv(args.env_file, override=True)
    else:
        load_dotenv()

    cors_origins = [
        origin.strip()
        for origin in os.getenv("VLLM_SERVICE_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    return ServiceSettings(
        backend_url=os.getenv("VLLM_BACKEND_URL", "http://127.0.0.1:8000").rstrip("/"),
        forced_model=os.getenv("VLLM_MODEL", "google/gemma-2-9b-it"),
        api_key=os.getenv("VLLM_API_KEY", ""),
        request_timeout=int(os.getenv("VLLM_SERVICE_TIMEOUT", "180")),
        cors_origins=cors_origins,
        host=args.host or os.getenv("VLLM_SERVICE_HOST", "0.0.0.0"),
        port=args.port or int(os.getenv("VLLM_SERVICE_PORT", "8090")),
    )


def create_app(settings: ServiceSettings) -> FastAPI:
    app = FastAPI(
        title="vLLM FastAPI Service",
        version="0.1.0",
        description="Small HTTP service for exposing a Gemma model from vLLM through a stable wrapper endpoint.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def vllm_headers() -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if settings.api_key:
            headers["Authorization"] = f"Bearer {settings.api_key}"
        return headers

    def vllm_get(path: str) -> dict[str, Any]:
        try:
            response = requests.get(
                f"{settings.backend_url}{path}",
                headers=vllm_headers(),
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"vLLM request failed: {exc}") from exc

    def vllm_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{settings.backend_url}{path}",
                headers=vllm_headers(),
                json=payload,
                timeout=settings.request_timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise HTTPException(status_code=503, detail=f"vLLM request failed: {exc}") from exc

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "vllm-fastapi-service",
            "vllm_backend_url": settings.backend_url,
            "forced_model": settings.forced_model,
            "endpoints": [
                "GET /health",
                "GET /models",
                "GET /v1/models",
                "POST /v1/chat/completions",
            ],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        data = vllm_get("/v1/models")
        models = [item.get("id", "") for item in data.get("data", [])]
        return {
            "status": "ok",
            "vllm_backend_url": settings.backend_url,
            "forced_model": settings.forced_model,
            "model_available": settings.forced_model in models,
            "models_available": models,
        }

    @app.get("/models")
    def models() -> dict[str, Any]:
        return vllm_get("/v1/models")

    @app.get("/v1/models")
    def openai_models() -> dict[str, Any]:
        return vllm_get("/v1/models")

    @app.post("/v1/chat/completions")
    def openai_chat_completion(request: OpenAIChatRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported by this wrapper yet.")

        payload: dict[str, Any] = {
            "model": settings.forced_model,
            "messages": [message_to_dict(message) for message in request.messages],
            "temperature": request.temperature,
            "stream": False,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        response = vllm_post("/v1/chat/completions", payload)
        choices = response.get("choices", [])
        content = ""
        if choices:
            content = choices[0].get("message", {}).get("content", "")
        created = int(time.time())

        return {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
            "object": response.get("object", "chat.completion"),
            "created": response.get("created", created),
            "model": response.get("model", settings.forced_model),
            "choices": response.get(
                "choices",
                [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            ),
            "usage": response.get("usage", {}),
            "vllm_raw": response,
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose a vLLM-served Gemma model through FastAPI.")
    parser.add_argument("--env-file", help="Optional path to a dedicated env file, such as agentic_hate_rag/.env.vllm.")
    parser.add_argument("--host", help="Override VLLM_SERVICE_HOST.")
    parser.add_argument("--port", type=int, help="Override VLLM_SERVICE_PORT.")
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = build_settings(args)
    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
