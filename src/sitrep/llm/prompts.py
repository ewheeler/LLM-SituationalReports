from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    prompt_id: str
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    prompt_id: str
    description: str
    system_prompt: str
    user_template: str

    def render(self, **context: Any) -> RenderedPrompt:
        rendered_context = {
            key: _render_value(value)
            for key, value in context.items()
        }
        return RenderedPrompt(
            prompt_id=self.prompt_id,
            system_prompt=self.system_prompt.strip(),
            user_prompt=self.user_template.format(**rendered_context).strip(),
        )


QUESTION_SYSTEM = (
    "You are an expert humanitarian analyst. Generate clear, specific, actionable questions "
    "based only on the supplied evidence. Maintain a neutral tone, avoid politics, define acronyms when possible, "
    "and return valid JSON only."
)


QUESTION_CLUSTER_V1 = PromptTemplate(
    prompt_id="questions.cluster.v1",
    description="Notebook-style prompt focused on grounded, actionable humanitarian questions.",
    system_prompt=QUESTION_SYSTEM,
    user_template="""
Generate 3 to 5 humanitarian analysis questions based exclusively on the supplied evidence.

Requirements:
- Use only the provided evidence and cluster metadata.
- Keep each question directly relevant to the event and cluster.
- Make questions precise, answerable, and useful for operational decision-making.
- Output only questions that can be answered from the provided evidence.

Event:
{event_name}

Cluster:
{cluster}

Evidence paragraphs:
{evidence}
""",
)


QUESTION_CLUSTER_V2 = PromptTemplate(
    prompt_id="questions.cluster.v2",
    description="Notebook-style prompt with explicit reasoning guidance for grounded question generation.",
    system_prompt=QUESTION_SYSTEM,
    user_template="""
Generate 3 to 5 grounded questions for a humanitarian situation report.

Process:
- Read the cluster headline and evidence carefully.
- Identify important developments, needs, affected groups, geography, risks, and response gaps.
- Ask focused questions that deepen understanding without needing external information.
- Prefer concise operational questions over generic ones.

Event:
{event_name}

Cluster:
{cluster}

Evidence paragraphs:
{evidence}
""",
)


QUESTION_CLUSTER_V3 = PromptTemplate(
    prompt_id="questions.cluster.v3",
    description="Notebook-style prompt with exemplars to encourage strategic and tactical questions.",
    system_prompt=QUESTION_SYSTEM,
    user_template="""
Generate 3 to 5 new grounded questions for this humanitarian cluster.

Question style examples:
- What patterns are emerging from the latest reported developments?
- Which groups or locations appear most affected according to the evidence?
- What operational gaps or response constraints are highlighted in the evidence?

Do not repeat the example wording unless it is directly supported by the evidence.
Use only the provided evidence.

Event:
{event_name}

Cluster:
{cluster}

Evidence paragraphs:
{evidence}
""",
)


ANSWER_GROUNDED_V1 = PromptTemplate(
    prompt_id="answers.grounded.v1",
    description="Synthesize a grounded answer from retrieved paragraphs.",
    system_prompt=(
        "You are a humanitarian analyst answering a question from supplied evidence only. "
        "Do not invent facts. Use only the provided paragraph identifiers for citations. "
        "Return valid JSON only."
    ),
    user_template="""
Answer the question using only the evidence paragraphs.
Prefer concise, source-grounded synthesis and cite only paragraph_ids present in the evidence.

Question:
{question}

Cluster:
{cluster}

Evidence paragraphs:
{evidence}
""",
)


SUMMARY_CLUSTER_V1 = PromptTemplate(
    prompt_id="summaries.cluster.v1",
    description="Write a grounded cluster summary.",
    system_prompt=(
        "You are a humanitarian analyst producing concise cluster summaries from structured inputs. "
        "Use only supplied evidence and return valid JSON only."
    ),
    user_template="""
Write a cluster summary with a short title. Keep it grounded in the supplied answers and citations.

Cluster:
{cluster}

Answers:
{answers}
""",
)


SUMMARY_CLUSTER_V2 = PromptTemplate(
    prompt_id="summaries.cluster.v2",
    description="Notebook-style cluster integration prompt with explicit citation preservation rules.",
    system_prompt=(
        "You integrate humanitarian answer snippets into a single cohesive narrative. "
        "Use only the provided answers and citation identifiers. Return valid JSON only."
    ),
    user_template="""
Your task is to integrate the following answer snippets into a single, cohesive, and flowing narrative.
The goal is to present as much of the original information as possible, not to summarize it too briefly.

Rules:
1. Integrate all key information from the provided answers into a coherent narrative.
2. Maintain original citation support. Every important statement in the summary should be supported.
3. Return `summary_segments` as 2-6 ordered sentence-like segments; each segment must include the exact `citation_paragraph_ids` supporting that segment.
4. When combining information from multiple answers, preserve all relevant support in both `summary_segments` and top-level `citation_paragraph_ids`.
5. Ensure logical flow and avoid adding outside knowledge.
6. Prefer a rich paragraph over a minimal summary when the evidence supports it.
7. Return `summary` as a single paragraph composed from the same segment order used in `summary_segments`.

Cluster:
{cluster}

Answers:
{answers}
""",
)


SUMMARY_EXECUTIVE_V1 = PromptTemplate(
    prompt_id="summaries.executive.v1",
    description="Write an executive summary from cluster summaries.",
    system_prompt=(
        "You are a humanitarian analyst drafting an executive summary for a situation report. "
        "Synthesize only from the supplied cluster summaries and return valid JSON only."
    ),
    user_template="""
Write an executive summary for this event. Keep it concise and decision-useful.

Event:
{event_name}

Cluster summaries:
{cluster_summaries}
""",
)


SUMMARY_EXECUTIVE_V2 = PromptTemplate(
    prompt_id="summaries.executive.v2",
    description="Notebook-style executive summary prompt with stronger citation-handling rules.",
    system_prompt=(
        "You are an AI assistant specialized in summarizing humanitarian situations strictly from provided source-derived summaries. "
        "Return valid JSON only."
    ),
    user_template="""
Carefully analyze the supplied cluster or SDG summaries and generate a concise, one-paragraph executive summary that provides an overview of the situation, focusing on the most important developments.

Instructions:
1. Base the summary solely on the provided summaries and their cited evidence.
2. Keep the summary concise, information-dense, and to one paragraph with a clear narrative flow.
3. Start with the most concrete, high-salience developments; do not open with hedging language if the supplied evidence is sufficient to support specific findings.
4. Return `summary_segments` as 3-6 ordered sentence-like segments; each segment must carry the exact `citation_paragraph_ids` supporting that statement.
5. Every important statement should be supported, and the top-level `citation_paragraph_ids` should be the union of all support used in `summary_segments`.
6. When the same statement is supported by multiple summaries, preserve all relevant support instead of collapsing to one citation.
7. Only use wording like "the available evidence offers only a limited overview" when the evidence is genuinely sparse or fragmentary. Treat the evidence as limited only if there are fewer than 2 grounded summaries or fewer than 4 unique cited paragraph ids overall. Otherwise, begin directly with concrete developments.
8. If the evidence is limited, explain what is known instead of making the whole paragraph about the limitation.
9. Do not cite paragraph identifiers that are not present in the supplied summaries.
10. Return `summary` as a single paragraph composed from the same segment order used in `summary_segments`.

Event:
{event_name}

Grounded summary count:
{grounded_summary_count}

Unique cited paragraph count:
{unique_citation_count}

Cluster summaries:
{cluster_summaries}
""",
)


SDG_CLASSIFY_V1 = PromptTemplate(
    prompt_id="sdg.classify.v1",
    description="Classify a question against the 17 Sustainable Development Goals.",
    system_prompt=(
        "You are an expert in Sustainable Development Goals (SDGs). "
        "Classify whether a humanitarian question directly relates to each SDG. "
        "Be precise, conservative, and return valid JSON only."
    ),
    user_template="""
Determine which Sustainable Development Goals are directly relevant to the question.

Classification criteria:
- Score 1 if the question directly addresses, mentions, or requires knowledge about this SDG's core themes.
- Score 0 if the question does not directly relate to this SDG, even if there are indirect associations.
- Evaluate each SDG independently.

Question:
{question}

SDG descriptions:
{sdg_descriptions}
""",
)


def _render_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str, sort_keys=True)
