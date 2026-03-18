from __future__ import annotations

from dataclasses import dataclass

from sitrep.llm.prompts import (
    ANSWER_GROUNDED_V1,
    QUESTION_CLUSTER_V1,
    QUESTION_CLUSTER_V2,
    QUESTION_CLUSTER_V3,
    SDG_CLASSIFY_V1,
    SUMMARY_CLUSTER_V1,
    SUMMARY_CLUSTER_V2,
    SUMMARY_EXECUTIVE_V1,
    SUMMARY_EXECUTIVE_V2,
    PromptTemplate,
)


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt: PromptTemplate
    schema_name: str
    default_model: str


PROMPT_REGISTRY: dict[str, PromptSpec] = {
    QUESTION_CLUSTER_V1.prompt_id: PromptSpec(
        prompt=QUESTION_CLUSTER_V1,
        schema_name="question_batch",
        default_model="gpt-5.4-mini",
    ),
    QUESTION_CLUSTER_V2.prompt_id: PromptSpec(
        prompt=QUESTION_CLUSTER_V2,
        schema_name="question_batch",
        default_model="gpt-5.4-mini",
    ),
    QUESTION_CLUSTER_V3.prompt_id: PromptSpec(
        prompt=QUESTION_CLUSTER_V3,
        schema_name="question_batch",
        default_model="gpt-5.4-mini",
    ),
    ANSWER_GROUNDED_V1.prompt_id: PromptSpec(
        prompt=ANSWER_GROUNDED_V1,
        schema_name="answer",
        default_model="gpt-5.4",
    ),
    SUMMARY_CLUSTER_V1.prompt_id: PromptSpec(
        prompt=SUMMARY_CLUSTER_V1,
        schema_name="summary",
        default_model="gpt-5.4",
    ),
    SUMMARY_CLUSTER_V2.prompt_id: PromptSpec(
        prompt=SUMMARY_CLUSTER_V2,
        schema_name="summary",
        default_model="gpt-5.4",
    ),
    SUMMARY_EXECUTIVE_V1.prompt_id: PromptSpec(
        prompt=SUMMARY_EXECUTIVE_V1,
        schema_name="summary",
        default_model="gpt-5.4",
    ),
    SUMMARY_EXECUTIVE_V2.prompt_id: PromptSpec(
        prompt=SUMMARY_EXECUTIVE_V2,
        schema_name="summary",
        default_model="gpt-5.4",
    ),
    SDG_CLASSIFY_V1.prompt_id: PromptSpec(
        prompt=SDG_CLASSIFY_V1,
        schema_name="sdg_classification",
        default_model="gpt-5.4-mini",
    ),
}


def get_prompt_spec(prompt_id: str) -> PromptSpec:
    if prompt_id not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt id: {prompt_id}")
    return PROMPT_REGISTRY[prompt_id]


def list_prompt_specs() -> list[dict[str, str]]:
    return [
        {
            "prompt_id": prompt_id,
            "description": spec.prompt.description,
            "schema_name": spec.schema_name,
            "default_model": spec.default_model,
        }
        for prompt_id, spec in sorted(PROMPT_REGISTRY.items())
    ]
