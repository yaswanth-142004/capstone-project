# Agentic Code-Mixed Hate Speech RAG

This project implements a no-fine-tuning hate/offensive speech classifier for Telugu-English and other Indian code-mixed comments.

It uses:

- LangGraph for the multi-agent execution graph.
- LangChain for LLM, tool, and vector store integration.
- Ollama-hosted open-source instruction models such as `llama3.1:8b`, `gemma2:9b`, or an Aya model.
- MuRIL embeddings for multilingual and transliterated Indian text retrieval.
- ChromaDB as the local RAG memory.
- Optional transliteration and syntactic code-mixing checks before reasoning.
- Human-in-the-loop export for low-confidence or suspicious comments.

## Implementation Plan

1. **Project setup**
   - Install the Python dependencies.
   - Run or pull your chosen Ollama instruction model.
   - Keep datasets as CSV/XLSX files with a text column and a label column.

2. **Data ingestion**
   - Load rows from specialized datasets such as DravidianCodeMix, OffMix-3L, or your existing classified CSVs.
   - Normalize text, including emoji expansion and URL/mention cleanup.
   - Embed each normalized row with MuRIL.
   - Store text, label, source, and metadata in a Chroma collection.

3. **Agent toolkit**
   - `normalization` node cleans text and optionally transliterates Romanized words.
   - `syntax` node computes simple code-mixing and word-salad signals. It is designed so a SyMCoM implementation can be swapped in.
   - `retrieval` node fetches the top similar historical comments from Chroma.
   - `reasoning` node asks the open-source LLM to classify using the retrieved examples.
   - `routing` node marks low-confidence or suspicious samples for human review.

4. **RAG long-term memory**
   - The vector store becomes the agent's historical memory.
   - Newly reviewed HITL examples can be appended with the ingestion CLI.
   - This improves future retrieval without changing model weights.

5. **Human-in-the-loop evaluation**
   - Low-confidence classifications are appended to a review CSV.
   - Human-reviewed rows can be imported back into the same Chroma collection.
   - This supports continuous improvement without fine-tuning.

## Install

From this folder:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Start Ollama separately and make sure your model is available:

```powershell
ollama pull gemma4:26b
ollama serve
```

## Configure

Copy `.env.example` to `.env` if you want to override defaults.

Important defaults:

- `OLLAMA_MODEL=gemma4:26b`
- `OLLAMA_BASE_URL=http://127.0.0.1:11434`
- `MURIL_MODEL=google/muril-base-cased`
- `CHROMA_DIR=./storage/chroma`
- `CHROMA_COLLECTION=code_mixed_hate_memory`

## Serve Ollama Through FastAPI

This project includes a separate FastAPI service for exposing local Ollama models on another port, useful with ngrok:

```powershell
python ..\ollama_fastapi_service.py --host 0.0.0.0 --port 8088
ngrok http 8088
```

The service always uses `gemma4:26b` and has no bearer-token auth. See `OLLAMA_FASTAPI_SERVICE.md` for endpoints and curl examples using `https://accent-copied-scrabble.ngrok-free.dev`.

## Ingest Examples Into RAG Memory

Use your existing classified file:

```powershell
python -m hate_rag_agents.ingest `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label
```

## Run Only The MuRIL Embedding Job

This does not call Ollama and does not run the LangGraph classifier. It only normalizes rows, creates tags, embeds with MuRIL, and writes vector plus metadata files.

```powershell
python -m hate_rag_agents.embed `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --output-dir .\outputs\embeddings
```

The command writes:

- `*_muril_vectors.npz`: compressed NumPy file containing `embeddings`, `ids`, `row_indices`, `normalized_texts`, and `tags`.
- `*_muril_metadata.csv`: row-level metadata, normalized text, label, tags, and syntax/code-mixing signals.

You can ingest a folder:

```powershell
python -m hate_rag_agents.ingest `
  --input .. `
  --recursive `
  --text-column Comment `
  --label-column hate_label
```

## Classify One Comment

```powershell
python -m hate_rag_agents.classify `
  --text "mee party vallu ila matladatam tappu"
```

## Classify A File

```powershell
python -m hate_rag_agents.classify `
  --input ..\Politics.csv `
  --output .\outputs\politics_agentic_classified.csv `
  --text-column Comment
```

Output columns include:

- `agent_normalized_text`
- `agent_hate_label`
- `agent_label_name`
- `agent_confidence`
- `agent_needs_review`
- `agent_explanation`
- `agent_retrieved_examples`
- `agent_raw_response`

## Add Human Reviewed Data Back To Memory

After reviewers fill labels in the HITL CSV:

```powershell
python -m hate_rag_agents.ingest `
  --input .\outputs\hitl_review_queue.csv `
  --text-column text `
  --label-column reviewer_label
```

## Notes

- MuRIL is loaded from Hugging Face via `transformers`; the first run downloads the model.
- The transliteration module supports optional backends and otherwise leaves Romanized text in place while preserving the normalized original.
- The syntax checker provides practical signals now and has a clear adapter point for SyMCoM.
- The LLM prompt asks for concise reasoning and JSON output, not hidden chain-of-thought.
- See `PROJECT_PLAN.md` for the full project flow, CSV format, embedding/tag rules, and module-level implementation.
- See `IMPLEMENTATION_PLAN.md` for the shorter phase-by-phase build plan.
