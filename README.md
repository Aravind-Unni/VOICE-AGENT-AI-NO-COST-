# 📞 Voice-First Zendesk Customer Support Agent with Advanced Hybrid RAG

An enterprise-grade, ultra-low-latency Voice AI Agent designed to serve as an interactive Zendesk Customer Support Assistant. This system combines real-time WebRTC audio streaming with an advanced, multi-stage Hybrid Retrieval-Augmented Generation (RAG) architecture to deliver fluid, conversational, and highly accurate voice interactions.

---

## 🏗️ System Architecture Overview

The system runs a **dual-threaded runtime engine** managed under a single unified entry point (`main.py`):
1. **FastAPI Web Server:** Runs in a background thread to serve the static web frontend and securely generate transient JWT connection tokens via the LiveKit SDK.
2. **LiveKit Agent Worker:** Runs on the main thread, establishing a continuous stateful connection with LiveKit Cloud via WebRTC to handle multi-modal transcription, processing, and synthesis loops.
