# Research Paper Diagrams

This file contains paper-friendly architecture diagrams for the hate-speech detection project.

Recommended usage:

- Preview directly in any Markdown editor that supports Mermaid.
- Export each Mermaid block to `SVG` or `PDF` for the paper.
- Include the exported figure in LaTeX using `\includegraphics`.

---

## 1. Agentic RAG Model - Current Auto-Loop Architecture

**Suggested figure title:** Agentic RAG architecture with reflection, HITL, and self-improving memory

```mermaid
flowchart LR
    A[Input comment] --> B[Normalization]
    B --> C[Syntax and code-mix analysis]
    C --> D[MuRIL query embedding]
    D --> E[Chroma retrieval]
    E --> F[LLM reasoning]
    F --> G{Confidence below threshold?}
    G -- No --> H[HITL routing]
    G -- Yes --> I[Reflection retrieval]
    I --> J[Reflection prompt and second-pass LLM]
    J --> H
    H --> K{Needs review?}
    K -- Yes --> L[HITL review queue]
    K -- No --> M[Final prediction]
    M --> N[Output CSV or JSON]
```

---

## 2. Previous Architecture - Before Auto-Loop

**Suggested figure title:** Baseline agentic RAG pipeline before reflection and auto-evaluation

```mermaid
flowchart LR
    A[Input comment] --> B[Normalization]
    B --> C[Syntax analysis]
    C --> D[MuRIL query embedding]
    D --> E[Chroma retrieval]
    E --> F[LLM reasoning]
    F --> G[HITL routing]
    G --> H[Final output]
    G --> I[Review queue for low-confidence rows]
```

---

## 3. Auto-Loop Evaluation System

**Suggested figure title:** Auto-loop evaluation and self-improvement workflow

```mermaid
flowchart TD
    A[Labelled input file] --> B[Classification pipeline]
    B --> C[Prediction for each row]
    C --> D{Gold label available?}
    D -- No --> E[Write output only]
    D -- Yes --> F[Compare prediction with gold label]
    F --> G[Update running metrics]
    G --> H{High confidence and correct?}
    H -- Yes --> I[Auto-ingest into Chroma memory]
    H -- No --> J[Do not auto-ingest]
    G --> K{Batch boundary reached?}
    K -- No --> E
    K -- Yes --> L[Collect misclassified rows]
    L --> M[LLM-based error analysis]
    M --> N[Extract lessons]
    N --> O[Save lessons.json]
    O --> P[Inject lessons into later prompts]
```

---

## 4. Ingestion Pipeline

**Suggested figure title:** Labelled data ingestion into RAG memory

```mermaid
flowchart LR
    A[CSV or XLSX dataset] --> B[Detect text and label columns]
    B --> C[Normalize text]
    C --> D[Optional transliteration]
    D --> E[MuRIL document embeddings]
    E --> F[Create metadata]
    F --> G[ChromaDB vector store]
```

---

## 5. Embedding-Only Pipeline

**Suggested figure title:** Standalone embedding generation workflow

```mermaid
flowchart LR
    A[CSV or XLSX dataset] --> B[Normalize text]
    B --> C[Generate tags and syntax signals]
    C --> D[MuRIL embeddings]
    D --> E[NPZ vector file]
    C --> F[Metadata CSV]
```

---

## 6. Classification Pipeline - Paper Summary View

**Suggested figure title:** End-to-end classification workflow

```mermaid
flowchart LR
    A[New social media comment] --> B[Preprocessing]
    B --> C[Retrieve similar labelled examples]
    C --> D[Prompt construction]
    D --> E[Ollama or vLLM]
    E --> F[JSON prediction]
    F --> G[Confidence check]
    G --> H[Final label, topic, explanation]
```

---

## 7. Internal LangGraph Node Flow

**Suggested figure title:** LangGraph node execution order

```mermaid
flowchart TD
    A[normalization node] --> B[syntax node]
    B --> C[retrieval node]
    C --> D[reasoning node]
    D --> E{should reflect?}
    E -- Yes --> F[reflection node]
    E -- No --> G[hitl_routing node]
    F --> G
    G --> H[END]
```

---

## 8. Reflection Subsystem

**Suggested figure title:** Reflection loop for low-confidence predictions

```mermaid
flowchart LR
    A[First-pass prediction] --> B{Confidence below threshold?}
    B -- No --> C[Accept first-pass result]
    B -- Yes --> D[Retrieve more diverse examples]
    D --> E[Build reflection prompt]
    E --> F[Second-pass LLM]
    F --> G{Valid JSON?}
    G -- Yes --> H[Update prediction]
    G -- No --> I[Mark parse error]
    H --> J[Send to HITL routing]
    I --> J
```

---

## 9. Retrieval Memory Subsystem

**Suggested figure title:** Retrieval memory design with MuRIL and ChromaDB

```mermaid
flowchart LR
    A[Historical labelled examples] --> B[MuRIL embedding model]
    B --> C[ChromaDB collection]
    D[Incoming Telugu comment] --> E[MuRIL query embedding]
    E --> C
    C --> F[Top-k similar examples]
    F --> G[Used as LLM context]
```

---

## 10. HITL Feedback Loop

**Suggested figure title:** Human-in-the-loop improvement cycle

```mermaid
flowchart TD
    A[Low-confidence or suspicious row] --> B[HITL review queue]
    B --> C[Human reviewer]
    C --> D[Corrected label]
    D --> E[Re-ingest reviewed example]
    E --> F[Updated Chroma memory]
    F --> G[Better future retrieval]
```

---

## 11. Auto-Ingest Feedback Loop

**Suggested figure title:** Verified prediction feedback into long-term memory

```mermaid
flowchart LR
    A[Predicted row] --> B{Correct and high confidence?}
    B -- No --> C[Do not auto-ingest]
    B -- Yes --> D[Auto-ingest row]
    D --> E[Store in ChromaDB]
    E --> F[Improve future retrieval]
```

---

## 12. Deployment Options

**Suggested figure title:** Flexible deployment paths for the reasoning backend

```mermaid
flowchart LR
    A[Classification pipeline] --> B{LLM provider}
    B --> C[Ollama local model]
    B --> D[vLLM OpenAI-compatible endpoint]
    C --> E[JSON response]
    D --> E
```

---

## 13. Output Artifact Map

**Suggested figure title:** Output files generated by the system

```mermaid
flowchart TD
    A[Classification run] --> B[Classified CSV or XLSX]
    A --> C[HITL review queue CSV]
    A --> D[Evaluation summary JSON]
    A --> E[Lessons JSON]
    A --> F[Application log]
    G[Embedding-only run] --> H[NPZ vectors]
    G --> I[Embedding metadata CSV]
```

---

## 14. Baseline vs Auto-Loop Comparison

**Suggested figure title:** Architectural difference between baseline and auto-loop systems

```mermaid
flowchart LR
    subgraph P1[Previous baseline]
        A1[Normalize] --> B1[Retrieve]
        B1 --> C1[Reason]
        C1 --> D1[HITL]
    end

    subgraph P2[Current auto-loop]
        A2[Normalize] --> B2[Retrieve]
        B2 --> C2[Reason]
        C2 --> D2[Reflect if needed]
        D2 --> E2[HITL]
        E2 --> F2[Auto-eval]
        F2 --> G2[Lessons and auto-ingest]
    end
```

---

## Mermaid Source Export Commands

If you want publication-ready `SVG` or `PDF`, export the Mermaid diagrams first.

### Option 1: Mermaid CLI

Install once:

```powershell
npm install -g @mermaid-js/mermaid-cli
```

Export a diagram from a `.mmd` file:

```powershell
mmdc -i .\agentic_rag_model.mmd -o .\agentic_rag_model.svg -t neutral -b transparent
mmdc -i .\agentic_rag_model.mmd -o .\agentic_rag_model.pdf -t neutral -b white
```

### Option 2: Using `npx` without global install

```powershell
npx @mermaid-js/mermaid-cli -i .\agentic_rag_model.mmd -o .\agentic_rag_model.svg -t neutral -b transparent
```

### Recommended export settings for papers

```powershell
mmdc -i .\auto_loop_eval_system.mmd -o .\auto_loop_eval_system.pdf -t neutral -b white -w 2200
```

Notes:

- Use `-t neutral` for clean academic styling.
- Use `SVG` when the publisher accepts vector figures.
- Use `PDF` if your LaTeX workflow handles PDF figures more easily.

---

## LaTeX Figure Code for Research Papers

### Standard PDF include

```latex
\usepackage{graphicx}

\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{figures/agentic_rag_model.pdf}
    \caption{Agentic RAG architecture with reflection, HITL, and self-improving memory.}
    \label{fig:agentic-rag-model}
\end{figure}
```

### SVG include if your paper setup supports it

```latex
\usepackage{svg}

\begin{figure}[t]
    \centering
    \includesvg[width=\columnwidth]{figures/agentic_rag_model}
    \caption{Agentic RAG architecture with reflection, HITL, and self-improving memory.}
    \label{fig:agentic-rag-model}
\end{figure}
```

### Two small diagrams side by side

```latex
\usepackage{graphicx}
\usepackage{subcaption}

\begin{figure}[t]
    \centering
    \begin{subfigure}[t]{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/previous_architecture.pdf}
        \caption{Previous baseline architecture.}
    \end{subfigure}
    \hfill
    \begin{subfigure}[t]{0.48\columnwidth}
        \centering
        \includegraphics[width=\linewidth]{figures/auto_loop_eval_system.pdf}
        \caption{Auto-loop evaluation system.}
    \end{subfigure}
    \caption{Comparison between the baseline pipeline and the current self-improving architecture.}
    \label{fig:baseline-vs-autoloop}
\end{figure}
```

---

## Suggested Figure Names for the Paper

- `agentic_rag_model`
- `previous_architecture`
- `auto_loop_eval_system`
- `ingestion_pipeline`
- `embedding_only_pipeline`
- `classification_pipeline`
- `langgraph_node_flow`
- `reflection_subsystem`
- `retrieval_memory_subsystem`
- `hitl_feedback_loop`
- `auto_ingest_feedback_loop`
- `deployment_options`
- `output_artifact_map`
- `baseline_vs_autoloop`

---

## Practical Workflow

1. Copy one Mermaid diagram block into a `.mmd` file.
2. Export it using `mmdc`.
3. Save the exported figure in your paper's `figures/` folder.
4. Insert it in LaTeX using `\includegraphics`.
5. Use the suggested captions from this file and adjust wording if needed.
