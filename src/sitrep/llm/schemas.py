from __future__ import annotations

from typing import Any

QUESTION_BATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string"},
                    "question_type": {"type": "string"},
                    "priority": {"type": "integer", "minimum": 1, "maximum": 100},
                    "rationale": {"type": "string"},
                },
                "required": ["question", "question_type", "priority", "rationale"],
            },
        }
    },
    "required": ["questions"],
}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "answer": {"type": "string"},
        "citation_paragraph_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
    "required": ["answer", "citation_paragraph_ids", "confidence", "rationale"],
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "summary_segments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "citation_paragraph_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 4,
                    },
                },
                "required": ["text", "citation_paragraph_ids"],
            },
        },
        "citation_paragraph_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 12,
        },
        "rationale": {"type": "string"},
    },
    "required": ["title", "summary", "summary_segments", "citation_paragraph_ids", "rationale"],
}

SDG_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "scores": {
            "type": "array",
            "minItems": 17,
            "maxItems": 17,
            "items": {"type": "integer", "enum": [0, 1]},
        },
        "rationale": {"type": "string"},
    },
    "required": ["scores", "rationale"],
}

SCHEMAS = {
    "question_batch": QUESTION_BATCH_SCHEMA,
    "answer": ANSWER_SCHEMA,
    "summary": SUMMARY_SCHEMA,
    "sdg_classification": SDG_CLASSIFICATION_SCHEMA,
}


def get_schema(schema_name: str) -> dict[str, Any]:
    if schema_name not in SCHEMAS:
        raise KeyError(f"Unknown schema: {schema_name}")
    return SCHEMAS[schema_name]
