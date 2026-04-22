# Ollama FastAPI Service

This standalone FastAPI service exposes your local Ollama runtime through a separate HTTP endpoint. It is meant for use with ngrok.

The service always uses:

```text
gemma4:26b
```

Any `model` field sent by a client is ignored. There is no bearer-token authentication.

## Files

- `..\ollama_fastapi_service.py`: standalone FastAPI app.
- `.env.example`: service host, port, Ollama URL, timeout, and CORS settings.

## Install

From `agentic_hate_rag`:

```powershell
python -m pip install -r requirements.txt
```

## Start Ollama

In a separate terminal:

```powershell
ollama serve
```

Make sure the forced model exists locally:

```powershell
ollama pull gemma4:26b
```

## Start The FastAPI Service

From `agentic_hate_rag`:

```powershell
python ..\ollama_fastapi_service.py --host 0.0.0.0 --port 8088
```

Or from the repo root:

```powershell
python .\ollama_fastapi_service.py --host 0.0.0.0 --port 8088
```

Local docs:

```text
http://127.0.0.1:8088/docs
```

## Expose With ngrok

```powershell
ngrok http 8088
```

Your ngrok endpoint:

```text
https://accent-copied-scrabble.ngrok-free.dev
```

Important: the tunnel must point to port `8088`, not Ollama's port `11434`.

Correct:

```text
ngrok http 8088
```

Wrong for this FastAPI service:

```text
ngrok http 11434
```

## Plain Curl Test Commands

Use these in Git Bash, Linux, macOS, WSL, or any shell with standard `curl`.

### Health

```bash
curl https://accent-copied-scrabble.ngrok-free.dev/health
```

### List Ollama Models

```bash
curl https://accent-copied-scrabble.ngrok-free.dev/models
```

### Raw Generate Endpoint

```bash
curl -X POST "https://accent-copied-scrabble.ngrok-free.dev/ollama/generate" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"prompt":"Explain RAG in one paragraph","temperature":0}'
```

### Raw Chat Endpoint

```bash
curl -X POST "https://accent-copied-scrabble.ngrok-free.dev/ollama/chat" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"temperature":0}'
```

### OpenAI-Compatible Chat Endpoint

```bash
curl -X POST "https://accent-copied-scrabble.ngrok-free.dev/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "ngrok-skip-browser-warning: true" \
  -d '{"messages":[{"role":"user","content":"Classify this text as offensive or non-offensive: mee party vallu ila matladatam tappu"}],"temperature":0}'
```

## Single-Line Plain Curl Commands

```bash
curl https://accent-copied-scrabble.ngrok-free.dev/health
```

```bash
curl -X POST "https://accent-copied-scrabble.ngrok-free.dev/ollama/chat" -H "Content-Type: application/json" -H "ngrok-skip-browser-warning: true" -d '{"messages":[{"role":"user","content":"Say hello"}],"temperature":0}'
```

```bash
curl -X POST "https://accent-copied-scrabble.ngrok-free.dev/v1/chat/completions" -H "Content-Type: application/json" -H "ngrok-skip-browser-warning: true" -d '{"messages":[{"role":"user","content":"Say hello"}],"temperature":0}'
```

## Local Test Curls

```bash
curl http://127.0.0.1:8088/health
```

```bash
curl -X POST "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"temperature":0}'
```

## Troubleshooting ngrok Access

Check what port ngrok is forwarding:

```bash
curl http://127.0.0.1:4040/api/tunnels
```

Look for:

```json
"addr": "http://localhost:8088"
```

If it says:

```json
"addr": "http://localhost:11434"
```

then ngrok is exposing Ollama directly instead of this FastAPI service. Stop ngrok and restart it with:

```bash
ngrok http 8088
```

## Environment Variables

```text
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_SERVICE_HOST=0.0.0.0
OLLAMA_SERVICE_PORT=8088
OLLAMA_SERVICE_TIMEOUT=180
OLLAMA_SERVICE_CORS_ORIGINS=*
```

## Notes

- This service does not run the RAG workflow by itself.
- It only forwards model calls to Ollama and returns the response.
- Every model call is forced to `gemma4:26b`.
- Use `python -m hate_rag_agents.classify` when you want the full LangGraph + RAG classifier.
