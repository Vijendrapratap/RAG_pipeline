"""rag_api — FastAPI backend for the custom transcript-RAG dashboard.

Replaces Open WebUI. Reuses the hybrid-retrieval logic that previously lived
inside the Open WebUI function tool, exposes it over HTTP, and (from Phase B
onward) adds bilingual answer synthesis via Ollama.
"""
