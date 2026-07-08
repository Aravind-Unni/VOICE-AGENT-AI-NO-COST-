# Production Grade Voice AI Agent for Customer Support

An enterprise-grade, ultra-low-latency Voice AI Agent designed to serve as an interactive Zendesk Customer Support Assistant. This system combines real-time WebRTC audio streaming with an advanced, multi-stage Hybrid Retrieval-Augmented Generation (RAG) architecture to deliver fluid, conversational, and highly accurate voice interactions.

---

## 🏗️ System Architecture Overview

The system runs a **dual-threaded runtime engine** managed under a single unified entry point (`main.py`):
1. **FastAPI Web Server:** Runs in a background thread to serve the static web frontend and securely generate transient JWT connection tokens via the LiveKit SDK.
2. **LiveKit Agent Worker:** Runs on the main thread, establishing a continuous stateful connection with LiveKit Cloud via WebRTC. It orchestrates the real-time AI pipeline by leveraging Deepgram for ultra-fast Speech-to-Text (STT) transcription, processing the user's intent through the hybrid RAG engine, and utilizing ElevenLabs for hyper-realistic Text-to-Speech (TTS) voice synthesis.

---

## 🧠 Advanced Hybrid RAG Pipeline Details

The underlying RAG engine (`rag_engine.py`) is decoupled from the streaming infrastructure, exposing a single executable tool to the voice pipeline. The execution lifecycle consists of the following phases:


### 1. Document Parsing & Indexing Framework
* **Parser:** Document ingestion utilizes **LlamaParse** to parse complex layout hierarchies, structural metadata, and visual features from PDF technical manuals into markdown.
* **Auto-Fallback DB Management:** The system automatically checks for database states at startup. If vector indices or serialized sparse models are missing, it triggers an automated processing script (`rag_preprocess.py`) to build dependencies directly from your source data.

### 2. Multi-Stage Query Optimization
* **Dual-Task Optimization:** Raw, volatile Speech-to-Text outputs are routed through a dedicated `ChatGroq` agent layout running Llama 3.3 70B.
* **Contextualization:** The optimizer evaluates conversation history to dynamically rewrite follow-ups and resolve unresolved pronouns into context-independent target search phrases.
* **STT Error Correction:** The agent cleans conversational artifacts, filter words ("um", "ah"), and domain-specific vocabulary variations (e.g., correcting "send desk" to "Zendesk", "tie kit" to "ticket").

### 3. Hybrid Retrieval Engine (Dense + Sparse)
* **Dense Stream:** Evaluates semantic meaning via `NVIDIAEmbeddings` leveraging the `nvidia/llama-nemotron-embed-1b-v2` representation model. Vector spaces are managed inside a local `Chroma` instance.
* **Sparse Stream:** Evaluates direct syntax patterns using a custom tokenized `BM25Retriever` serialized to disk via `pickle`.
* **Deduplication:** Aggregated content pools undergo comprehensive key deduplication before scoring to balance inference budgets.

### 4. Neural Cross-Encoder Reranking
* Candidates are passed through a local deep neural reranker running `cross-encoder/ms-marco-MiniLM-L-12-v2`. 
* Real-time text-pair relevance scoring filters low-confidence context chunks, selecting only the top $K$ document references ($K=5$) to limit the synthesis footprint.

### 5. Strict Guardrail Voice Generation
* Reranked contexts are passed to the target generation phase with deterministic instructions optimized for spoken interaction:
  * **Strict Context Adherence:** Instructs the LLM to provide instructions explicitly mapped to the manual text, explicitly prohibiting hallucinated paths.
  * **Voice Adaptability:** The model converts structure arrays and markdown tables into clean, natural verbal paths.
  * **Fallback Handling:** If data coverage is absent, it responds with a deterministic, safe fallback message: *"I don't have that information in my current knowledge base. Would you like me to connect you to a human agent?"*

---

## 📊 Observability, Tracing, & Evaluation

System observability and tracing are deeply integrated into the RAG lifecycle using **LangSmith** and **Langfuse**:
* **Granular Trace Mapping:** Every stage—from voice translation optimization to vector retrieval indices and cross-encoder sorting—is captured asynchronously.
* **Langfuse Framework:** `CallbackHandler` bindings route operational logs against specific `session_id` identifiers, linking separate calls into a singular observable session.
* **LangSmith Integration:** Provides structural evaluation, visual cost tracking, prompt latency measurement, and testing surfaces for fine-tuning system behavior.

---

## 🛠️ Project Configuration & Installation

### 1. Prerequisites
Ensure you are using Python 3.11 or later (Tested through Python 3.13).

### 2. Dependencies
Install required packages using the explicit environment configurations:
```bash
pip install uvicorn fastapi langchain-nvidia-ai-endpoints langchain-chroma langchain-community langchain-groq langfuse sentence-transformers livekit-agents livekit-plugins-deepgram livekit-plugins-elevenlabs livekit-plugins-openai livekit-api python-dotenv watchfiles
```

### 3. Environment Configuration

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=APIV...
LIVEKIT_API_SECRET=Mh63...
GROQ_API_KEY=gsk_...
DEEPGRAM_API_KEY=a627...
ELEVEN_API_KEY=sk_894...
NVIDIA_API_KEY=nvapi-...
LLAMA_CLOUD_API_KEY=llx-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
```
### 4. Running the Application

```bash
cd src
py main.py
```
