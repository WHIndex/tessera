from __future__ import annotations

from typing import Dict, Iterable, List

MODALITY_ORDER = ["text", "table", "graph"]


def _is_graph_chunk(chunk_id: str) -> bool:
    return chunk_id.startswith("m.") or chunk_id.startswith("g.")


def _is_table_chunk(chunk_id: str) -> bool:
    return chunk_id.startswith("ott_") or chunk_id.startswith("tat_")


def _is_text_chunk(chunk_id: str) -> bool:
    return chunk_id.startswith("nq_") or chunk_id.startswith("triviaqa_")


def infer_modalities_from_relevant_chunks(relevant_chunks: Dict[str, int] | Dict[str, float]) -> List[str]:
    has_text = False
    has_table = False
    has_graph = False

    for chunk_id, score in relevant_chunks.items():
        try:
            if float(score) <= 0:
                continue
        except Exception:
            continue
        if _is_table_chunk(chunk_id):
            has_table = True
        elif _is_text_chunk(chunk_id):
            has_text = True
        elif _is_graph_chunk(chunk_id):
            has_graph = True
        else:
            # 官方代码中对非 text/table 前缀常归入 KG 侧
            has_graph = True

    labels: List[str] = []
    if has_text:
        labels.append("text")
    if has_table:
        labels.append("table")
    if has_graph:
        labels.append("graph")
    return labels


def infer_modalities_from_dataset_score(dataset_score: Dict[str, int] | Dict[str, float]) -> List[str]:
    text_score = float(dataset_score.get("nq", 0)) + float(dataset_score.get("triviaqa", 0))
    table_score = float(dataset_score.get("ott", 0)) + float(dataset_score.get("tat", 0))
    graph_score = float(dataset_score.get("kg", 0))

    labels: List[str] = []
    if text_score > 0:
        labels.append("text")
    if table_score > 0:
        labels.append("table")
    if graph_score > 0:
        labels.append("graph")
    return labels


def modality_multihot(labels: Iterable[str]) -> List[int]:
    label_set = set(labels)
    return [1 if m in label_set else 0 for m in MODALITY_ORDER]
