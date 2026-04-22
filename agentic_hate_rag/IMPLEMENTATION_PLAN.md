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
  - Smart retrieval with distance-based filtering (`max_distance` threshold).
  - Label-diverse retrieval (`retrieve_diverse_examples()`) to prevent same-label bias.
  - Single-row auto-ingestion (`auto_ingest_row()`) for verified high-confidence predictions.

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
   - Filters out distant/irrelevant results via `max_retrieval_distance`.

4. `reasoning`
   - Builds an in-context prompt with target text, syntax report, retrieved examples, and any learned lessons.
   - Asks the LLM for strict JSON with label, confidence, languages, topic, rationale, and signals.

5. **conditional routing**: if confidence < threshold, route to `reflection`; otherwise go to `hitl_routing`.

6. `reflection` (new — triggered only for low-confidence predictions)
   - Re-retrieves with label-diverse strategy.
   - Sends a reflection prompt asking the LLM to reconsider its initial answer with new examples.
   - Updates the state with the revised classification.

7. `hitl_routing`
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

## Phase 6: Auto-Eval And Self-Improvement Loop

**Evaluation engine**

- `src/hate_rag_agents/eval.py`
  - `EvalTracker` class: accumulates per-row ground-truth vs prediction pairs.
  - Computes running accuracy, precision, recall, F1, confusion matrix.
  - `generate_eval_summary()`: writes a JSON evaluation report.
  - `print_eval_summary()`: prints a human-readable summary to console.

**Reflection and lessons**

- `src/hate_rag_agents/reflection.py`
  - `LessonStore` class: loads/saves/deduplicates learned lessons from `outputs/lessons.json`.
  - `analyze_errors()`: sends misclassified rows to the LLM to identify error patterns and extract lessons.
  - `build_reflection_prompt()`: builds a prompt for the self-correction second pass.
  - Lessons are injected into the reasoning prompt, improving future classifications within the same run.

**Auto-ingest**

- When the agent classifies a row with high confidence (≥0.85) and it matches ground truth:
  - The row is automatically ingested into ChromaDB.
  - This grows the RAG memory with verified examples.
  - Creates a virtuous cycle: better memory → better retrieval → better accuracy.

**Batch error analysis**

- Every N rows (default 50), if ground truth is available:
  - Collects misclassified rows from the batch.
  - Sends them to the LLM in a single "reflection" call.
  - Extracts lessons (e.g., "political criticism in Telugu is not hate speech").
  - Lessons are prepended to all future prompts for the rest of the run.

## Execution Checklist

1. Install dependencies and the local package.
2. Pull/start the Ollama model.
3. Ingest labelled examples into Chroma.
4. Run single-text smoke classification.
5. Run file classification with `--label-column hate_label` for auto-eval.
6. Review eval summary and lessons learned.
7. Review low-confidence rows in HITL queue.
8. Re-ingest reviewed rows into RAG memory.

## CLI Reference

### New classify flags

```text
--no-reflection       Disable the reflection loop for low-confidence rows.
--no-auto-ingest      Disable auto-ingestion of high-confidence verified rows.
--clear-lessons       Clear learned lessons from previous runs before starting.
--eval-batch-size N   Override batch size for error analysis (default: 50).
--label-column COL    Gold label column for auto-eval comparison.
```

### Example: full run with auto-eval

```powershell
python -m hate_rag_agents.classify `
  --input ..\..\datasets\Election_classified.csv `
  --output .\outputs\Election_agentic_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --limit 50
```

## Configuration Reference

### New environment variables

```text
REFLECTION_ENABLED=true          Enable/disable reflection loop
MAX_REFLECTION_RETRIES=1         How many reflection passes to attempt
MAX_RETRIEVAL_DISTANCE=1.2       Filter out retrieved examples beyond this distance
AUTO_INGEST_THRESHOLD=0.85       Confidence threshold for auto-ingestion
EVAL_BATCH_SIZE=50               Rows between error analysis batches
LESSONS_PATH=./outputs/lessons.json  Where to store learned lessons
```

## Future Extensions

- Replace the lightweight transliteration fallback with IndicTrans2 for stronger Romanized Telugu normalization.
- Add a real SyMCoM scoring backend under the same `syntax.py` interface.
- Add dataset-specific loaders for DravidianCodeMix and OffMix-3L schemas.
- Add a Streamlit or Gradio review UI on top of the HITL CSV.
