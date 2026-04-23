# Agentic RAG System for Telugu-English Code-Mixed Hate Speech Detection

## 1. Project Overview

This project implements a hate/offensive speech detection system for Telugu-English and other Indian code-mixed social media comments. Instead of training a new classifier from scratch or fine-tuning a transformer model, the system uses an agentic Retrieval-Augmented Generation (RAG) architecture.

The system combines:

- MuRIL embeddings for multilingual Indian-language semantic retrieval.
- ChromaDB as a local vector database and long-term RAG memory.
- LangGraph for a multi-step classification workflow.
- Ollama or vLLM for local open-source LLM inference.
- Human-in-the-loop (HITL) review for uncertain predictions.
- Progress logging and runtime diagnostics through `outputs/app.log`.

The main goal is to classify comments as:

- `0`: Non-hate / non-offensive
- `1`: Hate / offensive

The system also predicts a topic category, such as:

- `politics`
- `government`
- `religion`
- `sports`
- `entertainment`
- `caste_or_community`
- `gender_or_sexuality`
- `regional_or_nationality`
- `personal_abuse`
- `general`
- `unclear`

This makes the system more useful than a simple binary classifier because it gives both a safety label and a topic-level explanation of the content.

## 2. Motivation

Hate speech detection for Telugu-English comments is difficult because social media language is informal, code-mixed, transliterated, noisy, and context-dependent. A single comment may contain Telugu script, Romanized Telugu, English, emojis, political references, sarcasm, insults, and local cultural expressions.

Traditional supervised model training often requires:

- A large clean labelled dataset.
- Expensive annotation.
- Re-training when new examples or topics appear.
- GPU-heavy experimentation.
- Dataset-specific tuning.
- A fixed output label space.

This project avoids those constraints by using a no-fine-tuning agentic RAG approach. Labelled examples are stored in a retrievable memory, and the LLM uses those examples as context while classifying new comments.

## 3. Why This Approach Is Better Than Traditional Model Training

### 3.1 No Fine-Tuning Required

The system does not update model weights. This reduces training cost, avoids long GPU training cycles, and makes the project easier to run on local hardware.

### 3.2 Easy Knowledge Updates

In traditional training, adding new labelled examples usually requires retraining or fine-tuning. In this system, new examples can be added by ingesting them into ChromaDB:

```powershell
python -m hate_rag_agents.ingest `
  --input .\outputs\hitl_review_queue.csv `
  --text-column text `
  --label-column reviewer_label
```

The RAG memory improves without changing the LLM or MuRIL model weights.

### 3.3 Better Adaptation to Local Language and Context

The retrieval step provides similar historical examples from the project dataset. This helps the LLM understand local political, social, cultural, and linguistic context.

### 3.4 Transparent Reasoning Signals

The system stores:

- predicted label
- confidence
- topic
- explanation
- retrieved examples
- raw LLM response
- review reason
- errors, if any

This is more transparent than a standard black-box classifier that returns only a label.

### 3.5 Human Review Loop

Low-confidence or suspicious comments are automatically routed to a HITL CSV. Human reviewers can correct labels, and corrected data can be re-ingested into the RAG memory.

### 3.6 Flexible LLM Backend

The classifier can use:

- Ollama for local LLM inference.
- vLLM for faster OpenAI-compatible serving.

This makes the pipeline flexible for local development and stronger GPU-backed deployment.

## 4. System Architecture

The pipeline has three major modes:

1. Ingestion pipeline
2. Embedding-only pipeline
3. Classification pipeline

### 4.1 Ingestion Pipeline

The ingestion pipeline builds the RAG memory.

```text
Labelled CSV/XLSX
  -> read text and label columns
  -> normalize text
  -> optional transliteration
  -> MuRIL embeddings
  -> ChromaDB vector store
```

Command:

```powershell
python -m hate_rag_agents.ingest `
  --input .\datasets\final_aug.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --reset-store `
  --add-batch-size 5000
```

Important details:

- `--reset-store` deletes the existing Chroma memory and rebuilds it.
- `--add-batch-size` prevents Chroma batch-size errors during large ingestion.
- Ingestion embeds and stores labelled examples.
- This step must be done before classification if we want RAG retrieval to work.

### 4.2 Embedding-Only Pipeline

The embedding-only pipeline creates `.npz` vector files and metadata files. It does not populate ChromaDB and does not classify comments.

```text
CSV/XLSX
  -> normalize text
  -> create tags
  -> MuRIL embeddings
  -> .npz vector file
  -> metadata CSV
```

Command:

```powershell
python -m hate_rag_agents.embed `
  --input .\datasets\final_aug.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --output-dir .\outputs\embeddings
```

This is useful for analysis, vector inspection, and offline experiments.

### 4.3 Classification Pipeline

The classification pipeline predicts the hate label and topic for new comments.

```text
Input comment
  -> normalization
  -> syntax/code-mixing analysis
  -> MuRIL query embedding
  -> retrieve similar examples from ChromaDB
  -> build LLM prompt
  -> LLM returns JSON classification
  -> confidence/HITL routing
  -> output CSV
```

Command:

```powershell
python -m hate_rag_agents.classify `
  --input .\datasets\telugu_english_combined.v4.2_classified.csv `
  --output .\outputs\test_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --save-every 10
```

The classifier saves progress while running. If it crashes or is stopped, the output CSV contains the rows saved so far.

## 5. LangGraph Agent Workflow

The classifier is implemented as a LangGraph workflow with five nodes.

### 5.1 Normalization Node

This node cleans the comment text.

It handles:

- Unicode normalization.
- URL cleanup.
- mention cleanup.
- emoji expansion.
- optional transliteration.

### 5.2 Syntax Node

This node computes lightweight code-mixing and text quality signals.

It checks:

- Telugu token count.
- Latin token count.
- code-mix ratio.
- repetition ratio.
- short-token ratio.
- suspicious low-structure text.

### 5.3 Retrieval Node

This node embeds the normalized query with MuRIL and searches ChromaDB for similar labelled comments.

The default retrieval count is:

```text
RAG_TOP_K=5
```

The retrieved examples are passed to the LLM as contextual evidence.

### 5.4 Reasoning Node

This node builds a structured prompt and sends it to the selected LLM backend.

The LLM must return JSON with:

- `label`
- `label_name`
- `confidence`
- `languages`
- `primary_topic`
- `topic_tags`
- `rationale`
- `signals`

If the LLM returns malformed JSON, the system does not stop. It records the error, sets confidence to `0.0`, marks the row for review, and continues.

### 5.5 HITL Routing Node

This node decides whether a row needs human review.

Rows are routed to HITL when:

- confidence is below the threshold.
- the syntax checker finds suspicious text.
- the LLM returns invalid JSON.

The review queue is saved at:

```text
outputs/hitl_review_queue.csv
```

## 6. Models and Tools Used

### 6.1 MuRIL

MuRIL (`google/muril-base-cased`) is used for multilingual Indian-language embeddings.

MuRIL converts text into dense numerical vectors. Similar comments should have similar vector representations. These vectors are used for semantic search in ChromaDB.

The project supports GPU acceleration:

```text
MURIL_DEVICE=auto
```

When CUDA PyTorch is installed, MuRIL runs on GPU. On the available system, MuRIL resolves to:

```text
cuda (NVIDIA GeForce RTX 4080)
```

### 6.2 ChromaDB

ChromaDB is the local vector database used as RAG memory.

It stores:

- normalized text
- embedding vectors
- label metadata
- source file
- row index
- original text

### 6.3 LLM Backend

The system supports two LLM providers.

#### Ollama

Ollama is used for local model serving.

Example:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4:26b
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

#### vLLM

vLLM can be used for faster OpenAI-compatible inference.

Example:

```text
LLM_PROVIDER=vllm
VLLM_MODEL=google/gemma-2-9b-it
VLLM_BASE_URL=http://127.0.0.1:8090
```

The code includes a vLLM client and vLLM health check.

## 7. Configuration

Important `.env` settings:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4:26b
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_TIMEOUT_SECONDS=180

VLLM_MODEL=google/gemma-2-9b-it
VLLM_BASE_URL=http://127.0.0.1:8090
VLLM_TIMEOUT_SECONDS=180

MURIL_MODEL=google/muril-base-cased
MURIL_DEVICE=auto
MURIL_PROGRESS=true

CHROMA_DIR=./storage/chroma
CHROMA_COLLECTION=code_mixed_hate_memory
RAG_TOP_K=5
CONFIDENCE_THRESHOLD=0.68

HITL_QUEUE=./outputs/hitl_review_queue.csv
APP_LOG=./outputs/app.log
```

## 8. Output Files

### 8.1 Classification Output

The classified CSV contains the original dataset columns plus agent-generated columns:

- `agent_normalized_text`
- `agent_hate_label`
- `agent_label_name`
- `agent_confidence`
- `agent_primary_topic`
- `agent_topic_tags`
- `agent_needs_review`
- `agent_review_reason`
- `agent_explanation`
- `agent_retrieved_examples`
- `agent_raw_response`
- `agent_error`
- `gold_hate_label`
- `agent_label_matches_gold`

### 8.2 HITL Review Queue

The HITL CSV stores rows that need human review.

It includes:

- text
- normalized text
- predicted label
- confidence
- topic
- reason for review
- explanation
- retrieved examples
- reviewer label
- reviewer notes

### 8.3 App Log

Runtime logs are written to:

```text
outputs/app.log
```

The log records:

- startup configuration
- row start and row completion
- graph node timings
- MuRIL embedding timings
- Chroma retrieval timings
- LLM invocation timings
- save-progress timings
- errors and malformed LLM JSON

This makes it possible to diagnose why the pipeline is slow or where it stopped.

To watch logs live:

```powershell
Get-Content .\outputs\app.log -Wait -Tail 80
```

If the last log line is:

- `llm_invoke_start`: the delay is in the LLM backend.
- `node_retrieval_start`: the delay is in MuRIL query embedding or Chroma retrieval.
- `save_progress_start`: the delay is in writing the output CSV.
- `muril_model_load_start`: the delay is in loading the MuRIL model.

## 9. Error Handling and Robustness

The pipeline includes several robustness features:

- Output CSV is saved incrementally with `--save-every`.
- Malformed LLM JSON is captured instead of crashing the run.
- Per-row runtime errors are written into `agent_error`.
- Low-confidence rows are routed to HITL.
- App logs record the exact stage where delays or failures happen.
- Chroma ingestion uses batch insertion to avoid maximum batch-size errors.
- MuRIL device selection is printed and logged.

## 10. Recommended End-to-End Execution

### Step 1: Install

```powershell
cd C:\Users\nextt\Desktop\yaswanth_v\capstone-project
python -m pip install -r agentic_hate_rag\requirements.txt
python -m pip install -e .\agentic_hate_rag
```

### Step 2: Start LLM Backend

For Ollama:

```powershell
ollama serve
ollama pull gemma4:26b
```

For vLLM, start the vLLM service or the included FastAPI wrapper as configured in the project.

### Step 3: Build RAG Memory

```powershell
python -m hate_rag_agents.ingest `
  --input .\datasets\final_aug.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --reset-store `
  --add-batch-size 5000
```

### Step 4: Classify Dataset

```powershell
python -m hate_rag_agents.classify `
  --input .\datasets\telugu_english_combined.v4.2_classified.csv `
  --output .\outputs\test_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --save-every 10
```

### Step 5: Review HITL Rows

Open:

```text
outputs/hitl_review_queue.csv
```

Fill:

- `reviewer_label`
- `reviewer_notes`

### Step 6: Re-Ingest Human-Reviewed Data

```powershell
python -m hate_rag_agents.ingest `
  --input .\outputs\hitl_review_queue.csv `
  --text-column text `
  --label-column reviewer_label
```

## 11. Evaluation Strategy

Because the input file can include a gold label column, the classifier writes:

```text
gold_hate_label
agent_label_matches_gold
```

This allows later evaluation with:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- error analysis by topic
- error analysis by confidence
- HITL review rate

For fair evaluation, the same dataset should not be used both for RAG ingestion and testing. A better setup is:

- Use training/reference data for Chroma ingestion.
- Use separate unseen data for classification evaluation.

This prevents retrieval leakage.

## 12. Current Limitations

The current system is strong for a no-fine-tuning capstone prototype, but it has limitations:

- LLM inference can be slow for large datasets.
- Ollama may occasionally hang or return malformed JSON.
- Topic classification is prompt-based, not separately supervised.
- Syntax analysis is heuristic.
- Transliteration is lightweight and can be improved.
- The system quality depends on the quality and diversity of ingested examples.
- Full evaluation metrics still need a dedicated evaluation script.

## 13. Future Improvements

Recommended extensions:

- Add a formal evaluation script for precision, recall, F1, and confusion matrix.
- Add resume support to skip already-classified rows.
- Add timeout and retry logic per LLM call.
- Add stronger transliteration using IndicTrans2 or another dedicated model.
- Add SyMCoM-style code-mixing analysis.
- Add a reviewer UI using Streamlit or Gradio.
- Add topic-wise analytics for political, religious, caste/community, and gendered abuse.
- Compare Ollama and vLLM latency and output quality.
- Add prompt versioning for reproducibility.

## 14. Conclusion

This project demonstrates a practical agentic RAG alternative to traditional hate speech model training. It uses MuRIL and ChromaDB to retrieve similar labelled examples, then uses a local open-source LLM to classify comments with contextual support.

The system is especially suitable for Telugu-English and Indian code-mixed social media comments because it can adapt through retrieval memory rather than repeated model training. It supports human review, incremental improvement, GPU-accelerated embeddings, local/private inference, detailed logging, and interpretable output columns.

Compared with a conventional supervised classifier, this approach is more flexible, easier to update, more transparent, and better aligned with real-world moderation workflows where language and abuse patterns change over time.
