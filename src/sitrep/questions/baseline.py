from __future__ import annotations

from sitrep.types import ClusterRecord, QuestionRecord


QUESTION_TEMPLATES = (
    "What are the latest developments related to {label}?",
    "Who is affected by {label}, and where are the impacts most severe?",
    "What humanitarian actions or gaps are most important for {label}?",
)


def generate_questions(cluster_records: list[ClusterRecord]) -> list[QuestionRecord]:
    questions: list[QuestionRecord] = []
    for cluster_index, cluster in enumerate(cluster_records):
        for template_index, template in enumerate(QUESTION_TEMPLATES):
            text = template.format(label=cluster.cluster_label.lower())
            questions.append(
                QuestionRecord(
                    question_id=f"{cluster.cluster_id}-{template_index + 1:02d}",
                    cluster_id=cluster.cluster_id,
                    event_label=cluster.event_label,
                    question=text,
                    question_type=_question_type(template_index),
                    priority=max(1, 100 - cluster_index),
                    metadata={
                        "cluster_label": cluster.cluster_label,
                        "top_terms": list(cluster.top_terms),
                    },
                )
            )
    return questions



def _question_type(index: int) -> str:
    return ("developments", "affected_groups", "actions_and_gaps")[index]
