"""Retrieval-Augmented Generation (RAG) package for the Teps'out assistant.

Organised as a hexagonal architecture:

- ``domain``: framework-agnostic models, prompts, and exceptions.
- ``ports``: abstract interfaces the pipeline depends on.
- ``adapters``: concrete implementations of the ports (OpenAI, Qdrant).
- ``services``: application-level orchestration of the ports.
- ``config``/``container``: settings and dependency wiring.
"""
