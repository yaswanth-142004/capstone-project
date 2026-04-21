# Agentic Code-Mixed Hate Speech Detection Project Plan

## 1. Project Objective

Build a no-fine-tuning hate/offensive speech detection system for Telugu-English and other Indian code-mixed social media comments.

The system uses:

- **LangGraph** as the orchestrator for the agent workflow.
- **LangChain** for LLM, embeddings, and vector database integration.
- **Ollama** for local open-source instruction LLM inference.
- **MuRIL** for multilingual Indian-language embeddings.
- **ChromaDB** for RAG long-term memory.
- **Human-in-the-loop review** for uncertain or suspicious comments.

The model weights are never fine-tuned. The system improves by adding reviewed examples into the RAG memory.

## 2. Implemented Folder Structure

```text
agentic_hate_rag/
  README.md
  PROJECT_PLAN.md
  IMPLEMENTATION_PLAN.md
  requirements.txt
  pyproject.toml
  .env.example
  src/hate_rag_agents/
    classify.py
    config.py
    embed.py
    embeddings.py
    graph.py
    hitl.py
    ingest.py
    io_utils.py
    labels.py
    normalization.py
    prompts.py
    rag_store.py
    syntax.py
```

## 3. Base CSV Format

The minimum CSV for embedding creation needs one text column.

```csv
Comment
"[Telugu comment text]"
"mee party vallu ila matladatam tappu"
```

For RAG ingestion and supervised retrieval examples, include a label column.

```csv
Comment,hate_label
"[Telugu non-offensive comment text]",0
"targeted abusive or hateful example",1
```

Recommended full schema:

```csv
id,Comment,normalized_text,hate_label,source,annotator_notes
1,"raw comment","cleaned or normalized comment",0,"Politics.csv","optional note"
2,"raw comment","cleaned or normalized comment",1,"Politics.csv","optional note"
```

Column meaning:

- `id`: optional stable row ID.
- `Comment`: raw text from the dataset.
- `normalized_text`: optional pre-cleaned text. If present, the tools can use it directly.
- `hate_label`: required for RAG memory ingestion; optional for embedding-only jobs.
- `source`: optional dataset/source name.
- `annotator_notes`: optional human notes.

Supported label values:

- `0`, `0.0`, `normal`, `neutral`, `non-hate`, `non-offensive`
- `1`, `1.0`, `hate`, `hateful`, `offensive`, `toxic`, `abusive`

## 4. Individual Implementation Details

### 4.1 Configuration

File: `src/hate_rag_agents/config.py`

Purpose:

- Loads `.env` values.
- Defines model, Chroma, MuRIL, top-k retrieval, confidence threshold, and HITL queue defaults.

Important settings:

```text
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://127.0.0.1:11434
MURIL_MODEL=google/muril-base-cased
CHROMA_DIR=./storage/chroma
CHROMA_COLLECTION=code_mixed_hate_memory
RAG_TOP_K=5
CONFIDENCE_THRESHOLD=0.68
```

### 4.2 Normalization Tool

File: `src/hate_rag_agents/normalization.py`

Purpose:

- Converts text to Unicode NFKC.
- Replaces URLs with `[URL]`.
- Replaces mentions with `[MENTION]`.
- Expands emojis into readable text such as `<emoji: face with tears of joy>`.
- Optionally transliterates Romanized Telugu into Telugu script through `indic-transliteration`.

Output fields:

- `original_text`
- `cleaned_text`
- `transliterated_text`
- `transliteration_backend`

### 4.3 Syntax And Code-Mixing Checker

File: `src/hate_rag_agents/syntax.py`

Purpose:

- Counts Latin tokens.
- Counts Telugu tokens.
- Computes code-mix ratio.
- Computes repetition ratio.
- Computes short-token ratio.
- Flags suspicious low-structure text that may be an evasion attempt.

Current implementation is a lightweight checker. It is designed so a SyMCoM backend can be added under the same interface later.

### 4.4 Label Utilities

File: `src/hate_rag_agents/labels.py`

Purpose:

- Normalizes different label formats into `0`, `1`, or `None`.
- Creates label tags:
  - `label:offensive`
  - `label:non_offensive`
  - `label:unknown`

### 4.5 MuRIL Embedding Engine

File: `src/hate_rag_agents/embeddings.py`

Purpose:

- Wraps `google/muril-base-cased` as a LangChain `Embeddings` class.
- Tokenizes input text with Hugging Face `AutoTokenizer`.
- Runs MuRIL with Hugging Face `AutoModel`.
- Mean-pools token embeddings using the attention mask.
- L2-normalizes each vector.
- Returns one vector per input row.

Embedding creation steps:

1. Read normalized text.
2. Tokenize with max length `256`.
3. Run MuRIL encoder.
4. Mean-pool hidden states across non-padding tokens.
5. Normalize the vector.
6. Save or send the vector to Chroma.

### 4.6 MuRIL-Only Embedding Job

File: `src/hate_rag_agents/embed.py`

Purpose:

- Runs only the embedding pipeline.
- Does not call Ollama.
- Does not call the LangGraph classifier.
- Does not require a hate label.

Command:

```powershell
python -m hate_rag_agents.embed `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --output-dir .\outputs\embeddings
```

Outputs:

- `Politics_sample_classified_muril_vectors.npz`
- `Politics_sample_classified_muril_metadata.csv`

The `.npz` file contains:

- `embeddings`: matrix of shape `(rows, vector_dimension)`
- `ids`: stable row IDs
- `row_indices`: source row numbers
- `normalized_texts`: text used for embedding
- `tags`: pipe-separated tags for each row

The metadata CSV contains:

- `id`
- `row_index`
- `source`
- `original_text`
- `cleaned_text`
- `normalized_text`
- `transliteration_backend`
- `label`
- `tags`
- `token_count`
- `latin_tokens`
- `telugu_tokens`
- `code_mix_ratio`
- `repetition_ratio`
- `short_token_ratio`
- `suspicious_word_salad`

### 4.7 Tag Creation

Implemented in: `src/hate_rag_agents/embed.py`

Tags are created from label, source file, and syntax signals.

Examples:

```text
source:Politics_sample_classified|label:non_offensive|script:telugu
source:Politics_sample_classified|label:offensive|script:code_mixed
source:Politics_sample_classified|label:unknown|script:latin|quality:suspicious_word_salad
```

Tag rules:

- `source:<file_stem>` always comes from the input filename.
- `label:offensive` when label is `1`.
- `label:non_offensive` when label is `0`.
- `label:unknown` when no label column exists.
- `script:code_mixed` when both Telugu and Latin tokens are present.
- `script:telugu` when Telugu tokens are present and Latin tokens are absent.
- `script:latin` when Latin tokens are present and Telugu tokens are absent.
- `script:other` when neither script is detected.
- `quality:suspicious_word_salad` when the syntax checker flags low-structure text.

### 4.8 Chroma RAG Memory

File: `src/hate_rag_agents/rag_store.py`

Purpose:

- Creates a Chroma collection.
- Stores embedded documents and metadata.
- Retrieves top-k similar examples for a new comment.

Stored metadata:

- label
- source
- row index
- original text

### 4.9 RAG Ingestion

File: `src/hate_rag_agents/ingest.py`

Purpose:

- Reads labelled CSV/XLSX data.
- Normalizes text.
- Embeds rows with MuRIL.
- Stores vectors and metadata in ChromaDB.

Command:

```powershell
python -m hate_rag_agents.ingest `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label
```

This is the command to build the searchable long-term memory used by the classifier.

### 4.10 Prompt Construction

File: `src/hate_rag_agents/prompts.py`

Purpose:

- Builds the in-context reasoning prompt.
- Includes:
  - target original text
  - normalized text
  - syntax/code-mixing report
  - top retrieved historical examples
  - strict JSON output schema

### 4.11 LangGraph Agent Workflow

File: `src/hate_rag_agents/graph.py`

Runtime nodes:

1. `normalization`
2. `syntax`
3. `retrieval`
4. `reasoning`
5. `hitl_routing`

State fields:

- input text
- normalized text
- syntax report
- retrieved examples
- LLM raw response
- final label
- confidence
- explanation
- review decision

### 4.12 Classification CLI

File: `src/hate_rag_agents/classify.py`

Purpose:

- Classifies one text or a CSV/XLSX file.
- Uses the full LangGraph + RAG + LLM pipeline.

Single comment:

```powershell
python -m hate_rag_agents.classify `
  --text "mee party vallu ila matladatam tappu"
```

CSV file:

```powershell
python -m hate_rag_agents.classify `
  --input ..\Politics.csv `
  --output .\outputs\politics_agentic_classified.csv `
  --text-column Comment
```

### 4.13 Human-In-The-Loop Review

File: `src/hate_rag_agents/hitl.py`

Purpose:

- Appends uncertain samples to `outputs/hitl_review_queue.csv`.

Rows are routed to review when:

- confidence is below threshold
- no RAG examples are retrieved
- syntax checker flags possible evasion or word-salad text

Human-reviewed rows can be ingested back into Chroma:

```powershell
python -m hate_rag_agents.ingest `
  --input .\outputs\hitl_review_queue.csv `
  --text-column text `
  --label-column reviewer_label
```

## 5. End-To-End Flow

### Offline Preparation Flow

```text
CSV dataset
  -> normalize text
  -> optional transliteration
  -> syntax/code-mixing signals
  -> MuRIL embeddings
  -> Chroma vector store
```

Command:

```powershell
python -m hate_rag_agents.ingest --input <labelled_csv> --text-column <text_col> --label-column <label_col>
```

### Embedding-Only Flow

```text
CSV dataset
  -> normalize text
  -> optional transliteration
  -> syntax/code-mixing signals
  -> tags
  -> MuRIL embeddings
  -> .npz vector file + metadata CSV
```

Command:

```powershell
python -m hate_rag_agents.embed --input <csv> --text-column <text_col> --output-dir .\outputs\embeddings
```

### Online Classification Flow

```text
new comment
  -> normalize
  -> syntax/code-mixing checker
  -> MuRIL query embedding
  -> retrieve top 5 Chroma examples
  -> LLM reasoning prompt
  -> JSON classification
  -> confidence routing
  -> optional HITL review queue
```

Command:

```powershell
python -m hate_rag_agents.classify --text "<comment>"
```

## 6. How CSV Embeddings And Tags Are Created

For each CSV row:

1. The selected text column is read.
2. Text is cleaned with Unicode normalization, URL cleanup, mention cleanup, and emoji expansion.
3. Romanized Telugu transliteration is attempted if the optional backend is available.
4. The syntax checker counts Telugu and Latin tokens.
5. Tags are generated from source, label, script type, and quality flags.
6. MuRIL converts the normalized text into a dense vector.
7. The vector is saved to `.npz` or inserted into ChromaDB.
8. Metadata is saved beside the vector or attached to the Chroma document.

## 7. Recommended Setup Commands

```powershell
cd .\agentic_hate_rag
python -m pip install -r requirements.txt
python -m pip install -e .
ollama pull llama3.1:8b
```

## 8. Recommended Execution Order

1. Run a small embedding-only smoke test:

```powershell
python -m hate_rag_agents.embed `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label `
  --limit 5
```

2. Build the Chroma RAG memory:

```powershell
python -m hate_rag_agents.ingest `
  --input ..\Politics_sample_classified.csv `
  --text-column normalized_text `
  --label-column hate_label
```

3. Classify one sample:

```powershell
python -m hate_rag_agents.classify --text "sample comment here"
```

4. Classify a full file:

```powershell
python -m hate_rag_agents.classify `
  --input ..\Politics.csv `
  --output .\outputs\Politics_agentic_classified.csv `
  --text-column Comment
```

5. Review HITL rows and ingest reviewed examples back into memory.

## 9. Current Limitations And Planned Extensions

Current:

- Transliteration uses a lightweight optional backend.
- Syntax checking is heuristic.
- Chroma ingestion requires labelled rows.
- First MuRIL run downloads model files from Hugging Face.

Planned:

- Add IndicTrans2 as a stronger transliteration backend.
- Add a true SyMCoM adapter.
- Add evaluation scripts for F1, precision, recall, confusion matrix, and reviewer agreement.
- Add dataset-specific import profiles for DravidianCodeMix and OffMix-3L.
- Add a Streamlit/Gradio review interface for HITL.
