from __future__ import annotations

from sitrep.retrieval.hybrid import retrieve_hybrid_top_k
from sitrep.types import AnswerRecord, ParagraphRecord, QuestionRecord



def answer_questions(
    question_records: list[QuestionRecord],
    paragraph_records: list[ParagraphRecord],
    *,
    top_k: int = 3,
    lexical_weight: float = 1.0,
    title_weight: float = 0.3,
    trust_weight: float = 0.2,
    recency_weight: float = 0.1,
) -> list[AnswerRecord]:
    answers: list[AnswerRecord] = []
    paragraphs_by_cluster = _paragraphs_by_cluster(paragraph_records)
    for question in question_records:
        candidate_paragraphs = paragraphs_by_cluster.get(question.cluster_id, paragraph_records)
        hits = retrieve_hybrid_top_k(
            question.question,
            candidate_paragraphs,
            top_k=top_k,
            lexical_weight=lexical_weight,
            title_weight=title_weight,
            trust_weight=trust_weight,
            recency_weight=recency_weight,
        )
        citation_ids = tuple(hit.paragraph_id for hit in hits)
        answer_text = _compose_answer(hits)
        answers.append(
            AnswerRecord(
                answer_id=f"{question.question_id}-answer",
                question_id=question.question_id,
                cluster_id=question.cluster_id,
                event_label=question.event_label,
                answer=answer_text,
                citation_paragraph_ids=citation_ids,
                confidence=round(hits[0].score, 4) if hits else 0.0,
                metadata={
                    **_question_metadata(question),
                    "retrieval_method": "hybrid_baseline",
                    "retrieval_hits": [
                        {
                            "paragraph_id": hit.paragraph_id,
                            "score": hit.score,
                            "title": hit.title,
                            "canonical_url": hit.canonical_url,
                            "source_connector": hit.source_connector,
                            "paragraph_index": hit.paragraph_index,
                            "components": hit.components or {},
                            "text_preview": hit.text[:240],
                        }
                        for hit in hits
                    ],
                    "citation_details": [
                        {
                            "paragraph_id": hit.paragraph_id,
                            "title": hit.title,
                            "url": hit.canonical_url,
                            "source_connector": hit.source_connector,
                            "context": hit.text,
                            "paragraph_index": hit.paragraph_index,
                        }
                        for hit in hits
                    ],
                },
            )
        )
    return answers



def _paragraphs_by_cluster(paragraph_records: list[ParagraphRecord]) -> dict[str, list[ParagraphRecord]]:
    grouped: dict[str, list[ParagraphRecord]] = {}
    for paragraph in paragraph_records:
        cluster_id = paragraph.metadata.get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            continue
        grouped.setdefault(cluster_id, []).append(paragraph)
    return grouped



def _question_metadata(question: QuestionRecord) -> dict[str, object]:
    metadata = dict(question.metadata)
    return {
        "question": question.question,
        "question_type": question.question_type,
        "priority": question.priority,
        "cluster_label": metadata.get("cluster_label"),
        "top_terms": metadata.get("top_terms", []),
        "sdg_numbers": metadata.get("sdg_numbers", []),
        "sdg_labels": metadata.get("sdg_labels", []),
        "sdg_scores": metadata.get("sdg_scores", []),
        "sdg_keys": [f"sdg-{number}" for number in metadata.get("sdg_numbers", []) if isinstance(number, int)],
    }



def _compose_answer(hits) -> str:
    if not hits:
        return "No grounded evidence was retrieved for this question in the current baseline pipeline."
    snippets = [hit.text for hit in hits[:2] if hit.text]
    if not snippets:
        return "Relevant source paragraphs were found, but no answer text could be synthesized."
    return " ".join(snippets)
