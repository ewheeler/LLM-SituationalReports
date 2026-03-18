from __future__ import annotations

import hashlib
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

from sitrep.settings import ClusteringSettings
from sitrep.types import ClusterAssignment, ClusterRecord, ParagraphRecord

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]{2,}")
STOPWORDS = {
    "about", "after", "amid", "among", "and", "are", "been", "being", "but",
    "for", "from", "have", "into", "more", "over", "such", "that", "than",
    "the", "their", "there", "they", "this", "those", "under", "with", "were",
    "will", "would", "response", "situation", "update", "humanitarian", "said",
    "reports", "report", "country", "people", "affected", "according", "also",
}


class NotebookClusteringUnavailableError(RuntimeError):
    pass


class NotebookClusteringError(RuntimeError):
    pass


def cluster_paragraphs_notebook_style(
    paragraph_records: list[ParagraphRecord],
    settings: ClusteringSettings,
) -> tuple[list[ClusterRecord], list[ClusterAssignment], dict[str, Any]]:
    if not paragraph_records:
        return [], [], {
            "provider": settings.provider,
            "status": "empty",
            "reason": "no_paragraphs",
        }

    np, SentenceTransformer, hdbscan, umap = _load_optional_dependencies()
    texts = [_prepare_text(paragraph.text) for paragraph in paragraph_records]
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    search_results: list[dict[str, Any]] = []
    best_run: dict[str, Any] | None = None
    for run_index, params in enumerate(_sample_parameter_sets(settings), start=1):
        labels, probabilities, relative_validity = _cluster_once(
            embeddings,
            params,
            settings,
            np=np,
            hdbscan=hdbscan,
            umap=umap,
        )
        cluster_count = len({label for label in labels if label >= 0})
        noise_ratio = float(np.mean(labels < 0)) if len(labels) else 0.0
        score = float(relative_validity) - noise_ratio * settings.noise_penalty
        if cluster_count < settings.min_clusters:
            score -= (settings.min_clusters - cluster_count) * 0.05
        run_result = {
            "run_index": run_index,
            **params,
            "cluster_count": cluster_count,
            "noise_ratio": round(noise_ratio, 4),
            "dbcv": round(float(relative_validity), 4),
            "score": round(score, 4),
            "labels": labels.tolist(),
            "probabilities": probabilities.tolist(),
        }
        search_results.append(run_result)
        if best_run is None or run_result["score"] > best_run["score"]:
            best_run = run_result

    if best_run is None:
        raise NotebookClusteringError("Failed to produce any HDBSCAN clustering runs.")

    labels = np.asarray(best_run["labels"], dtype=int)
    probabilities = np.asarray(best_run["probabilities"], dtype=float)
    if any(label < 0 for label in labels):
        labels = _reassign_noise_points(labels, embeddings, np=np)

    grouped: dict[int, list[tuple[ParagraphRecord, float]]] = defaultdict(list)
    for paragraph, label, probability in zip(paragraph_records, labels.tolist(), probabilities.tolist()):
        grouped[int(label)].append((paragraph, float(probability)))

    cluster_records: list[ClusterRecord] = []
    assignments: list[ClusterAssignment] = []
    for cluster_label, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        paragraphs = [paragraph for paragraph, _ in items]
        scores = {paragraph.paragraph_id: probability for paragraph, probability in items}
        cluster_id = _cluster_id(cluster_label, paragraphs)
        cluster_title, top_terms = _cluster_title(paragraphs)
        paragraph_ids = [paragraph.paragraph_id for paragraph in paragraphs]
        document_ids = sorted({paragraph.document_id for paragraph in paragraphs})
        source_mix = Counter(paragraph.source_connector for paragraph in paragraphs)
        representative_paragraph_id = max(
            paragraph_ids,
            key=lambda paragraph_id: scores.get(paragraph_id, 0.0),
        ) if paragraph_ids else None
        cluster_records.append(
            ClusterRecord(
                cluster_id=cluster_id,
                cluster_label=cluster_title,
                event_label=paragraphs[0].event_label if paragraphs else "",
                paragraph_count=len(paragraphs),
                document_count=len(document_ids),
                paragraph_ids=tuple(paragraph_ids),
                document_ids=tuple(document_ids),
                top_terms=tuple(top_terms),
                source_mix=dict(sorted(source_mix.items())),
                representative_paragraph_id=representative_paragraph_id,
                metadata={
                    "provider": settings.provider,
                    "cluster_label_index": cluster_label,
                    "best_run": {
                        key: value
                        for key, value in best_run.items()
                        if key not in {"labels", "probabilities"}
                    },
                    "search_runs": len(search_results),
                },
            )
        )
        for paragraph in paragraphs:
            assignments.append(
                ClusterAssignment(
                    cluster_id=cluster_id,
                    paragraph_id=paragraph.paragraph_id,
                    document_id=paragraph.document_id,
                    paragraph_index=paragraph.paragraph_index,
                    score=round(scores.get(paragraph.paragraph_id, 0.0), 4),
                )
            )

    cluster_records.sort(key=lambda cluster: (-cluster.paragraph_count, cluster.cluster_label, cluster.cluster_id))
    assignments.sort(key=lambda assignment: (assignment.cluster_id, assignment.document_id, assignment.paragraph_index))
    diagnostics = {
        "provider": settings.provider,
        "embedding_model": settings.embedding_model,
        "best_run": {
            key: value
            for key, value in best_run.items()
            if key not in {"labels", "probabilities"}
        },
        "search_results": [
            {
                key: value
                for key, value in result.items()
                if key not in {"labels", "probabilities"}
            }
            for result in sorted(search_results, key=lambda item: item["score"], reverse=True)
        ],
        "settings": asdict(settings),
    }
    return cluster_records, assignments, diagnostics



def _load_optional_dependencies():
    try:
        import hdbscan  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        import umap  # type: ignore
    except ModuleNotFoundError as error:
        raise NotebookClusteringUnavailableError(
            "Notebook-style clustering requires optional dependencies: sentence-transformers, numpy, umap-learn, and hdbscan."
        ) from error
    return np, SentenceTransformer, hdbscan, umap



def _cluster_once(embeddings, params, settings, *, np, hdbscan, umap):
    reducer = umap.UMAP(
        n_neighbors=int(params["n_neighbors"]),
        n_components=int(settings.umap_components),
        metric=settings.umap_metric,
        random_state=int(settings.random_state),
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(params["min_cluster_size"]),
        min_samples=int(params["min_samples"]),
        cluster_selection_epsilon=float(params["cluster_selection_epsilon"]),
        metric="euclidean",
        cluster_selection_method="eom",
        gen_min_span_tree=True,
    ).fit(reduced)
    labels = np.asarray(clusterer.labels_, dtype=int)
    probabilities = np.asarray(getattr(clusterer, "probabilities_", [1.0] * len(labels)), dtype=float)
    relative_validity = getattr(clusterer, "relative_validity_", 0.0) or 0.0
    return labels, probabilities, relative_validity



def _sample_parameter_sets(settings: ClusteringSettings) -> list[dict[str, Any]]:
    rng = random.Random(settings.random_state)
    seen: set[tuple[Any, ...]] = set()
    samples: list[dict[str, Any]] = []
    max_attempts = max(settings.random_search_evaluations * 5, settings.random_search_evaluations)
    while len(samples) < settings.random_search_evaluations and len(seen) < max_attempts:
        params = {
            "n_neighbors": rng.choice(tuple(settings.n_neighbors)),
            "min_cluster_size": rng.choice(tuple(settings.min_cluster_size)),
            "min_samples": rng.choice(tuple(settings.min_samples)),
            "cluster_selection_epsilon": rng.choice(tuple(settings.cluster_selection_epsilon)),
        }
        key = tuple(params.values())
        seen.add(key)
        if params not in samples:
            samples.append(params)
    if not samples:
        samples.append(
            {
                "n_neighbors": 9,
                "min_cluster_size": 6,
                "min_samples": 4,
                "cluster_selection_epsilon": 0.1,
            }
        )
    return samples



def _reassign_noise_points(labels, embeddings, *, np):
    cluster_ids = sorted({int(label) for label in labels.tolist() if label >= 0})
    if not cluster_ids:
        return np.arange(len(labels), dtype=int)
    centroids = {}
    for cluster_id in cluster_ids:
        points = embeddings[labels == cluster_id]
        centroids[cluster_id] = points.mean(axis=0)
    reassigned = labels.copy()
    for index, label in enumerate(labels.tolist()):
        if label >= 0:
            continue
        best_cluster = max(
            cluster_ids,
            key=lambda cluster_id: _cosine_similarity(embeddings[index], centroids[cluster_id], np=np),
        )
        reassigned[index] = best_cluster
    return reassigned



def _cosine_similarity(left, right, *, np) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))



def _cluster_id(cluster_label: int, paragraphs: list[ParagraphRecord]) -> str:
    seed = f"{cluster_label}:{paragraphs[0].event_label if paragraphs else ''}:{len(paragraphs)}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]



def _cluster_title(paragraphs: list[ParagraphRecord]) -> tuple[str, list[str]]:
    title_tokens: list[str] = []
    paragraph_tokens: list[str] = []
    for paragraph in paragraphs:
        title_tokens.extend(_tokenize(paragraph.title))
        paragraph_tokens.extend(token for token in _tokenize(paragraph.text) if token not in STOPWORDS)
        paragraph_tokens.extend(_normalize_term(tag) for tag in paragraph.tags)
    counts = Counter(title_tokens + paragraph_tokens)
    top_terms = [term for term, _ in counts.most_common(6)]
    if not top_terms:
        return "Miscellaneous Updates", []
    label_terms = top_terms[:2]
    label = " / ".join(term.replace("-", " ").title() for term in label_terms)
    return label, top_terms



def _prepare_text(text: str) -> str:
    cleaned = text.replace("\\xe2\\x80\\x9c", '"').replace("\\xe2\\x80\\x9d", '"')
    cleaned = cleaned.replace("\\xe2\\x80\\x98", "'").replace("\\xe2\\x80\\x99", "'")
    cleaned = cleaned.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    cleaned = cleaned.replace("\\xc2\\xa0", " ").replace("\xa0", " ")
    cleaned = cleaned.replace("\\xe2\\x80\\x94", "-").replace("\\xe2\\x80\\x93", "-")
    cleaned = cleaned.replace("\uf0b7", " ").replace("\t", " ").replace("\n", " ")
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
    cleaned = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\'\"-]', ' ', cleaned)
    cleaned = re.sub(r"\.{4,}", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()



def _tokenize(text: str) -> list[str]:
    return [_normalize_term(match.group(0)) for match in TOKEN_RE.finditer(text or "")]



def _normalize_term(term: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
