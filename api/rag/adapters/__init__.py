"""Adapters: concrete implementations of the ports.

Each subpackage binds an external service (Mistral embeddings, Mistral chat,
Qdrant) to the abstract port it fulfils. The OpenAI adapters are kept alongside
them but are no longer wired in :mod:`rag.container`.
"""
