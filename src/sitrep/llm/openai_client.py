from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from sitrep.llm.prompts import RenderedPrompt
from sitrep.llm.registry import PromptSpec
from sitrep.llm.schemas import get_schema
from sitrep.settings import AppSettings, GenerationStageSettings

try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None  # type: ignore[misc,assignment]


class OpenAIConfigurationError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    pass


@dataclass(slots=True)
class OpenAIResponsesClient:
    api_key: str
    base_url: str | None = None

    @classmethod
    def from_settings(cls, app_settings: AppSettings) -> "OpenAIResponsesClient":
        if not app_settings.openai_api_key:
            raise OpenAIConfigurationError(
                "OPENAI_API_KEY is not configured. Set it in the environment or settings file to enable model-backed generation."
            )
        return cls(
            api_key=app_settings.openai_api_key,
            base_url=app_settings.openai_base_url,
        )

    def generate_structured(
        self,
        *,
        prompt_spec: PromptSpec,
        stage_settings: GenerationStageSettings,
        rendered_prompt: RenderedPrompt,
    ) -> dict[str, Any]:
        if OpenAI is None:
            raise OpenAIConfigurationError(
                "The openai package is not installed. Add it to the environment to enable model-backed generation."
            )
        model = stage_settings.model or prompt_spec.default_model
        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        request: dict[str, Any] = {
            "model": model,
            "instructions": rendered_prompt.system_prompt,
            "input": rendered_prompt.user_prompt,
            "max_output_tokens": stage_settings.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": prompt_spec.schema_name,
                    "schema": get_schema(prompt_spec.schema_name),
                    "strict": True,
                }
            },
        }
        if stage_settings.temperature is not None:
            request["temperature"] = stage_settings.temperature
        response = client.responses.create(**request)
        return _extract_json_payload(response)



def _extract_json_payload(response: Any) -> dict[str, Any]:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return _load_json(output_text)

    if hasattr(response, "model_dump"):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        payload = None

    if isinstance(payload, dict):
        text = _find_text_segment(payload)
        if text:
            return _load_json(text)

    raise StructuredOutputError("The OpenAI response did not contain structured JSON text.")



def _find_text_segment(payload: dict[str, Any]) -> str | None:
    output = payload.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict):
            continue
        contents = item.get("content")
        if not isinstance(contents, list):
            continue
        for content in contents:
            if not isinstance(content, dict):
                continue
            text_value = content.get("text")
            if isinstance(text_value, str) and text_value.strip():
                return text_value
    return None



def _load_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise StructuredOutputError(f"Failed to decode structured JSON: {error}") from error
    if not isinstance(payload, dict):
        raise StructuredOutputError("Structured response payload must be a JSON object.")
    return payload
