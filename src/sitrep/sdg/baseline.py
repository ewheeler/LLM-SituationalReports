from __future__ import annotations

import re
from difflib import SequenceMatcher

from sitrep.settings import SdgSettings
from sitrep.sdg.taxonomy import SDG_DEFINITIONS, SdgDefinition

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-/']+")


class BaselineSdgClassifier:
    def __init__(self, settings: SdgSettings) -> None:
        self.settings = settings

    def classify(self, question: str, context: str = "") -> dict[str, object]:
        normalized_question = question.lower().strip()
        normalized_context = context.lower().strip()
        classification_text = f"{normalized_question} {normalized_context}".strip()
        tokens = {match.group(0).lower() for match in TOKEN_RE.finditer(classification_text)}
        scored = [
            self._score_definition(definition, classification_text, tokens)
            for definition in SDG_DEFINITIONS
        ]
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        selected = [
            item for item in ranked
            if item["score"] >= self.settings.min_score
        ][: self.settings.max_labels_per_question]
        selected_numbers = {int(item["number"]) for item in selected}
        binary_scores = [1 if definition.number in selected_numbers else 0 for definition in SDG_DEFINITIONS]
        return {
            "sdg_scores": binary_scores,
            "sdg_labels": [str(item["name"]) for item in selected],
            "sdg_numbers": [int(item["number"]) for item in selected],
            "sdg_relevance": [round(float(item["score"]), 4) for item in scored],
            "sdg_matches": [
                {
                    "number": int(item["number"]),
                    "name": str(item["name"]),
                    "score": round(float(item["score"]), 4),
                    "matched_terms": list(item["matched_terms"]),
                }
                for item in selected
            ],
        }

    def _score_definition(
        self,
        definition: SdgDefinition,
        normalized_question: str,
        tokens: set[str],
    ) -> dict[str, object]:
        matched_terms: list[str] = []
        score = 0.0
        description_tokens = {match.group(0).lower() for match in TOKEN_RE.finditer(definition.description)}
        overlap = tokens & description_tokens
        if overlap:
            score += min(0.25, 0.05 * len(overlap))
            matched_terms.extend(sorted(overlap)[:5])
        for keyword in definition.keywords:
            keyword_normalized = keyword.lower()
            if " " in keyword_normalized:
                if keyword_normalized in normalized_question:
                    score += 0.5
                    matched_terms.append(keyword_normalized)
                    continue
            else:
                if keyword_normalized in tokens:
                    score += 0.35
                    matched_terms.append(keyword_normalized)
                    continue
            score = max(score, 0.2 * SequenceMatcher(None, normalized_question, keyword_normalized).ratio())
        if definition.name.lower() in normalized_question:
            score += 0.8
            matched_terms.append(definition.name.lower())
        if definition.number == 16 and any(term in normalized_question for term in ("conflict", "violence", "rights", "protection")):
            score += 0.2
        if definition.number == 3 and any(term in normalized_question for term in ("health", "cholera", "disease", "hospital")):
            score += 0.2
        if definition.number == 2 and any(term in normalized_question for term in ("food", "nutrition", "hunger", "famine")):
            score += 0.2
        if definition.number == 6 and any(term in normalized_question for term in ("water", "sanitation", "wash")):
            score += 0.2
        if definition.number == 13 and any(term in normalized_question for term in ("flood", "drought", "climate")):
            score += 0.2
        return {
            "number": definition.number,
            "name": definition.name,
            "score": min(score, 1.0),
            "matched_terms": tuple(dict.fromkeys(matched_terms)),
        }
