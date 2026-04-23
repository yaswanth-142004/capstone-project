# vLLM FastAPI Service

This standalone FastAPI service exposes a Gemma model served by vLLM through a separate HTTP endpoint. It is useful when you want a stable local wrapper URL for the classifier or for tunneling with ngrok.

The wrapper always uses the model from:

```text
VLLM_MODEL
```

Any `model` field sent by a client is ignored. Bearer-token auth to the underlying vLLM server is optional and controlled by `VLLM_API_KEY`.

## Files

- `..\vllm_fastapi_service.py`: standalone FastAPI app.
- `.env.vllm.example`: dedicated example env file for the wrapper and classifier.

## Install

From `agentic_hate_rag`:

```powershell
python -m pip install -r requirements.txt
```

## Start vLLM

In a separate terminal, start the OpenAI-compatible vLLM server. Example:

```powershell
python -m vllm.entrypoints.openai.api_server --model google/gemma-2-9b-it --host 127.0.0.1 --port 8000
```

## Prepare The Separate vLLM Env File

Copy:

```text
agentic_hate_rag/.env.vllm.example
```

to:

```text
agentic_hate_rag/.env.vllm
```

Adjust `VLLM_MODEL`, `VLLM_BACKEND_URL`, and `VLLM_SERVICE_PORT` if needed.

## Start The FastAPI Wrapper Service

From the repo root:

```powershell
python .\vllm_fastapi_service.py --env-file .\agentic_hate_rag\.env.vllm
```

Or explicitly override host and port:

```powershell
python .\vllm_fastapi_service.py --env-file .\agentic_hate_rag\.env.vllm --host 0.0.0.0 --port 8090
```

Local docs:

```text
http://127.0.0.1:8090/docs
```

## Health Checks

```bash
curl http://127.0.0.1:8090/health
```

```bash
curl http://127.0.0.1:8090/models
```

```bash
curl -X POST "http://127.0.0.1:8090/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello"}],"temperature":0}'
```

## Use The Classifier With vLLM

Using CLI flags:

```powershell
python -m hate_rag_agents.classify `
  --llm-provider vllm `
  --vllm-base-url http://127.0.0.1:8090 `
  --model google/gemma-2-9b-it `
  --text "mee party vallu ila matladatam tappu"
```

Or by loading values from `.env.vllm` into your environment before running the classifier.

You can also point the classifier at that file directly:

```powershell
python -m hate_rag_agents.classify `
  --env-file .\.env.vllm `
  --text "mee party vallu ila matladatam tappu"
```

## Notes

- The wrapper does not run the RAG workflow by itself.
- It only forwards chat-completion calls to a vLLM backend and forces the configured Gemma model.
- The classifier can also target a direct vLLM OpenAI-compatible endpoint by setting `VLLM_BASE_URL` to that server instead of this wrapper.
