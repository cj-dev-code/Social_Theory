# **Social Theory RAG Pipeline**

*A reproducible, latency-profiled retrieval-augmented generation system for feminist social theory.*

This repository contains a modular RAG stack designed for domain-specific reasoning over feminist theory, data feminism, and adjacent social-science texts. The project emphasizes **transparent system design**, **repeatable benchmarking**, and **clear separation of concerns** across retrieval, reranking, and generation components.

The implementation includes:

* a **retrieval layer** using HF embeddings + Qdrant with MMR diversification,
* **Cohere reranking** for higher-precision grounding,
* a **local LLM backend** (via Ollama) for controlled offline inference,
* a **Discord interface** for conversational access,
* and a **latency benchmarking harness** for evaluating end-to-end and per-stage performance.

The current codebase establishes a strong, inspectable baseline intended for subsequent optimization (e.g., migration from Ollama → vLLM → FlashInfer).

---

# **📐 Architecture Overview**

```
User Query
     ↓
Embedding-based Retrieval (Qdrant + MMR)
     ↓
Cohere Reranker (top-k consolidation)
     ↓
Prompt Assembly (context window management)
     ↓
Local LLM Inference (ChatOllama backend)
     ↓
Grounded Answer
```

### Key Modules

| Component         | File                        | Description                                                                                                                  |
| ----------------- | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Configuration** | `src/config.py`             | Centralizes model, embedding, index, reranking, and backend parameters. Favors version control over environment mutation.    |
| **Retrieval**     | `src/retrieval.py`          | Embedding generation → Qdrant MMR search → payload hydration. Implements retrieval-side abstraction boundaries.              |
| **RAG Graph**     | `src/generate.py`           | Orchestrates the end-to-end retrieval + generation flow with LangGraph. Encapsulates prompt construction and LLM invocation. |
| **Discord Bot**   | `src/discord_bot.py`        | Thin conversational interface bridging user messages to the RAG graph, with context-reveal utilities.                        |
| **Benchmarking**  | `benchmarks/latency_rag.py` | Runs controlled latency evaluations with per-stage timing (retrieval vs inference). Supports multi-query benchmarking.       |

A deliberate design choice is keeping algorithmic behavior (models, k-values, rerank depth, backend selection) **explicit in code** rather than obscured behind environment variables, enabling traceable diffs and predictable deployments.

---

# **⚡ Baseline Performance Profile**

Benchmark configuration:

* **Inference backend:** Ollama (local)
* **Model:** Qwen2.5-7B-Instruct (GGUF)
* **Embedding model:** `BAAI/bge-large-en-v1.5`
* **Retrieval settings:** MMR (k=15) + Cohere rerank (top-5)

Latency summary from `benchmarks/latency_rag.py` (5-run average):

| Stage              | Avg (ms) | Notes                            |
| ------------------ | -------- | -------------------------------- |
| **Total Pipeline** | ~4946 ms | End-to-end latency               |
| **Retrieval**      | ~1001 ms | Embeddings (CPU) + Qdrant access |
| **LLM Inference**  | ~3944 ms | Dominant source of latency       |

**~80% of total latency is attributable to inference**, matching expectations for Ollama-backed LLMs on consumer hardware. This provides a clear optimization path: migration to **vLLM**, **FlashInfer**, or a kernel-optimized backend will produce the largest gains.

---

# **🖥️ Hardware Context for Benchmarks**

To ensure reproducibility and correct interpretation of latency:

* **GPU:** NVIDIA GeForce RTX 2070 (8 GB)
* **Driver:** 570.xx
* **CUDA:** 12.8
* **CPU:** Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz
* **RAM:** 31 Gi
* **OS:** Linux Mint (or equivalent)
* **LLM Runtime:** Ollama
* **Model Format:** GGUF quantized model

**Why this matters:**
The RTX 2070 is not an inference-optimized architecture; Ollama does not use continuous batching or FlashAttention-style kernels. These constraints are reflected directly in the inference latency. On server-class GPUs (A100/L40) or vLLM, bottleneck proportions shift substantially.

---

# **🚀 Features**

### 🔍 Retrieval Layer

* BGE-large embeddings for high semantic recall
* MMR diversification for reduced redundancy
* Cohere reranking for precision-oriented selection
* Rich payload hydration for transparent context assembly

### 🤖 Generation Layer

* Local inference via Ollama (configurable model + parameters)
* Controlled prompt assembly with separation of system vs context vs query
* Deterministic, traceable generation flow

### 🧭 LangGraph Orchestration

* Pipeline expressed as an explicit graph
* Easy to extend with fallback nodes, observability, or evaluation branches

### 💬 Discord Integration

* Conversational UI (`!askchima`)
* Inline context reveal via reactions
* Lightweight response cache for interaction-scoped lookups

### 📊 Benchmarking & Instrumentation

* Stage-level timing (retrieval vs inference)
* Warmup control
* Uniform multi-query harness for reproducibility

---

# **📦 Getting Started**

### 1. Clone the repository

```bash
git clone <your_repo_url>
cd SOCIAL_THEORY
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example`:

```bash
cp .env.example .env
```

Provide:

* `COHERE_API_KEY`
* `DISCORD_BOT_TOKEN` (optional)
* `QDRANT_PATH` + `QDRANT_COLLECTION`
* `OLLAMA_MODEL_NAME`

### 4. Run a single RAG query

```bash
python - <<'EOF'
from src.generate import graph
print(graph.invoke({"question": "What does data feminism critique about objectivity?"}))
EOF
```

### 5. Run latency benchmarks

```bash
python benchmarks/latency_rag.py
```

### 6. Launch the Discord bot

```bash
python src/discord_bot.py
```

---

# **🧠 Design Principles**

### **1. Explicitness > Implicitness**

System behavior is encoded in code, not hidden in environment variables.
This supports code review, reproducibility, and controlled evolution of the pipeline.

### **2. Measurement-First Development**

Performance claims are tied to instrumentation, not intuition.
Latency harness includes stage-level breakdown to guide optimization work.

### **3. Separation of Concerns**

Retrieval, generation, configuration, orchestration, and UI live in distinct modules.
This keeps the surface area clean as the system evolves.

### **4. Real-World Deployability**

Discord integration ensures the pipeline operates in user-facing conversational settings, not only notebook-scale prototypes.

---

# **🛣️ Roadmap**

### 🔧 Inference Optimization (near-term)

* Migrate Ollama → **vLLM** for lower latency and higher throughput
* Evaluate **FlashInfer** for optimized attention kernels
* Enable **KV cache reuse** and **minibatching**

### 📚 Domain Adaptation (DaPT)

* Add LoRA/QLoRA training script (separate private module)
* Track perplexity before/after DaPT
* Integrate perplexity-based reranking or gating

### 📝 Evaluation Suite

* Retrieval quality scoring (recall@k, redundancy metrics)
* Hallucination & grounding confidence evaluation
* RAG-specific faithfulness scoring

### 🤖 Graph Extensions

* Introduce re-ask logic
* Add streaming support
* Add operational observability (token usage, timing traces)
