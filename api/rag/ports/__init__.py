"""Ports: abstract interfaces the RAG pipeline depends on.

Each port defines a contract (embedding, vector store, LLM) that the domain
and services rely on, without knowing which concrete adapter fulfils it.
"""
