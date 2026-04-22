from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Any, Literal

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = "gemma4:26b"
REQUEST_TIMEOUT = int(os.getenv("OLLAMA_SERVICE_TIMEOUT", "180"))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("OLLAMA_SERVICE_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]


app = FastAPI(
    title="Ollama FastAPI Service",
    version="0.1.0",
    description="Small HTTP service for exposing local Ollama models through a separate endpoint.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    model: str | None = None
    system: str | None = None
    temperature: float = 0.0
    format: Literal["", "json"] = ""
    options: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = 0.0
    format: Literal["", "json"] = ""
    options: dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    stream: bool = False


def ollama_get(path: str) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{OLLAMA_BASE_URL}{path}",
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Ollama request failed: {exc}") from exc


def ollama_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}{path}",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise HTTPException(status_code=503, detail=f"Ollama request failed: {exc}") from exc


def message_to_dict(message: Message) -> dict[str, str]:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "ollama-fastapi-service",
        "ollama_base_url": OLLAMA_BASE_URL,
        "forced_model": DEFAULT_MODEL,
        "endpoints": [
            "GET /health",
            "GET /models",
            "POST /ollama/generate",
            "POST /ollama/chat",
            "POST /v1/chat/completions",
        ],
    }


@app.get("/health")
def health() -> dict[str, Any]:
    data = ollama_get("/api/tags")
    return {
        "status": "ok",
        "ollama_base_url": OLLAMA_BASE_URL,
        "forced_model": DEFAULT_MODEL,
        "models_available": len(data.get("models", [])),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    return ollama_get("/api/tags")


@app.post("/ollama/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    options = dict(request.options)
    options.setdefault("temperature", request.temperature)
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "prompt": request.prompt,
        "stream": False,
        "options": options,
    }
    if request.system:
        payload["system"] = request.system
    if request.format:
        payload["format"] = request.format
    return ollama_post("/api/generate", payload)


@app.post("/ollama/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    options = dict(request.options)
    options.setdefault("temperature", request.temperature)
    payload: dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "messages": [message_to_dict(message) for message in request.messages],
        "stream": False,
        "options": options,
    }
    if request.format:
        payload["format"] = request.format
    return ollama_post("/api/chat", payload)


@app.post("/v1/chat/completions")
def openai_chat_completion(request: OpenAIChatRequest) -> dict[str, Any]:
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported by this wrapper yet.")

    options: dict[str, Any] = {"temperature": request.temperature}
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens

    payload = {
        "model": DEFAULT_MODEL,
        "messages": [message_to_dict(message) for message in request.messages],
        "stream": False,
        "options": options,
    }
    response = ollama_post("/api/chat", payload)
    message = response.get("message", {})
    content = message.get("content", "")
    created = int(time.time())

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": response.get("model", DEFAULT_MODEL),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        },
        "ollama_raw": response,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expose local Ollama models through FastAPI.")
    parser.add_argument("--host", default=os.getenv("OLLAMA_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("OLLAMA_SERVICE_PORT", "8088")))
    parser.add_argument("--reload", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        "ollama_fastapi_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
