"""LLM scaffolding for optional model-backed generation stages."""

from sitrep.llm.openai_client import (
    OpenAIConfigurationError,
    OpenAIResponsesClient,
    StructuredOutputError,
)

__all__ = [
    "OpenAIConfigurationError",
    "OpenAIResponsesClient",
    "StructuredOutputError",
]
