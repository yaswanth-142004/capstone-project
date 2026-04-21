# Implementation Plan

## Goal

Build a no-fine-tuning, agentic RAG classifier for Telugu-English and broader Indian code-mixed hate/offensive speech detection.

The system must use open-source instruction LLMs through Ollama, LangGraph for orchestration, LangChain for model/vector interfaces, MuRIL for multilingual embeddings, and a human review loop for uncertain or suspicious cases.

## Phase 1: Orchestrator And Model Runtime

**Implementation**

- `src/hate_rag_agents/graph.py`
  - Defines the LangGraph state machine.
  - Uses `ChatOllama` from LangChain for local instruction-model reasoning.
  - Supports model overrides from CLI/env.

**Default model**

- `llama3.1:8b`

**Supported alternatives**

- Gemma 2 instruction models available in Ollama.
- Aya-style multilingual instruction models available in your local runtime.
- Any Ollama chat/instruction model that follows JSON output.

## Phase 2: Agent Toolkit

**Normalization tool**

- `src/hate_rag_agents/normalization.py`
  - Unicode cleanup.
  - URL and mention replacement.
  - Emoji expansion.
  - Optional Romanized Telugu transliteration through `indic-transliteration`.
  - Designed so IndicTrans2 or another transliteration backend can replace the lightweight default.

**Syntactic checker**

- `src/hate_rag_agents/syntax.py`
  - Reports token count, script mix, repetition, short-token ratio, and suspicious low-structure text.
  - Provides a practical adapter point for a SyMCoM implementation.

**Embedding engine**

- `src/hate_rag_agents/embeddings.py`
  - LangChain `Embeddings` implementation for `google/muril-base-cased`.
  - Uses mean pooling and L2 normalization.

## Phase 3: RAG Long-Term Memory

**Vector database**

- `src/hate_rag_agents/rag_store.py`
  - Uses ChromaDB through `langchain-chroma`.
  - Stores normalized comment text, label, source file, row index, and original text.

**Ingestion**

- `src/hate_rag_agents/ingest.py`
  - Loads CSV/XLSX files or folders.
  - Auto-detects common text and label columns.
  - Converts examples into MuRIL vectors.
  - Adds examples to the Chroma collection.

## Phase 4: Agentic Execution Pipeline

Runtime order in `graph.py`:

1. `normalization`
   - Produces cleaned and optionally transliterated text.

2. `syntax`
   - Produces code-mixing and evasion signals.

3. `retrieval`
   - Uses MuRIL + Chroma to fetch top-k similar labelled examples.

4. `reasoning`
   - Builds an in-context prompt with target text, syntax report, and retrieved examples.
   - Asks the LLM for strict JSON with label, confidence, languages, rationale, and signals.

5. `hitl_routing`
   - Routes low-confidence, no-context, or suspicious word-salad samples to review.

## Phase 5: Human-In-The-Loop

**Review queue**

- `src/hate_rag_agents/hitl.py`
  - Appends uncertain cases to `outputs/hitl_review_queue.csv`.
  - Includes reviewer columns: `reviewer_label` and `reviewer_notes`.

**Memory update**

- Reviewed rows can be ingested back with:

```powershell
python -m hate_rag_agents.ingest `
  --input .\outputs\hitl_review_queue.csv `
  --text-column text `
  --label-column reviewer_label
```

## Execution Checklist

1. Install dependencies and the local package.
2. Pull/start the Ollama model.
3. Ingest labelled examples into Chroma.
4. Run single-text smoke classification.
5. Run file classification.
6. Review low-confidence rows.
7. Re-ingest reviewed rows into RAG memory.

## Future Extensions

- Replace the lightweight transliteration fallback with IndicTrans2 for stronger Romanized Telugu normalization.
- Add a real SyMCoM scoring backend under the same `syntax.py` interface.
- Add dataset-specific loaders for DravidianCodeMix and OffMix-3L schemas.
- Add evaluation scripts for precision, recall, F1, confusion matrix, and reviewer agreement.
- Add a Streamlit or Gradio review UI on top of the HITL CSV.
