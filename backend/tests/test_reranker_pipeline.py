from __future__ import annotations

import numpy as np

from backend.ml.scripts.common import compute_ranking_metrics
from backend.services.nlp import ml_reranker


class _FakeVectorizer:
    def transform(self, texts):
        return list(texts)


class _FakeModel:
    classes_ = np.array([0, 1, 2, 3], dtype=np.float32)

    def predict_proba(self, rows):
        output = []
        for row in rows:
            if "candidate=narcissist" in row:
                output.append([0.02, 0.08, 0.24, 0.66])
            elif "candidate=egotist" in row:
                output.append([0.08, 0.18, 0.42, 0.32])
            else:
                output.append([0.52, 0.28, 0.14, 0.06])
        return output


def test_reranker_respects_disable_flag(monkeypatch):
    monkeypatch.setenv("WORDCRAFT_DISABLE_RERANKER", "1")
    monkeypatch.setattr(ml_reranker, "_load_artifact", lambda: {"vectorizer": _FakeVectorizer(), "model": _FakeModel()})
    rows = [{"word": "egotist", "score": 0.6}, {"word": "narcissist", "score": 0.55}]
    reranked = ml_reranker.rerank_candidate_dicts(
        task="oneword",
        payload={"query": "self obsessed", "context": "formal"},
        candidates=rows,
    )
    assert reranked == rows
    monkeypatch.delenv("WORDCRAFT_DISABLE_RERANKER", raising=False)


def test_reranker_reorders_by_ml_score(monkeypatch):
    monkeypatch.setattr(ml_reranker, "_load_artifact", lambda: {"vectorizer": _FakeVectorizer(), "model": _FakeModel()})
    rows = [
        {"word": "egotist", "score": 0.8},
        {"word": "narcissist", "score": 0.55},
        {"word": "stone", "score": 0.7},
    ]
    reranked = ml_reranker.rerank_candidate_dicts(
        task="oneword",
        payload={"query": "self obsessed", "context": "formal"},
        candidates=rows,
        text_key="word",
        score_key="score",
        blend=0.85,
    )
    assert reranked[0]["word"] == "narcissist"
    assert "ml_score" in reranked[0]
    assert reranked[0]["score"] >= reranked[1]["score"]


def test_ranking_metrics_include_ndcg_and_hits():
    grouped = {
        "a": [
            {"label": 3, "pred_score": 0.9},
            {"label": 0, "pred_score": 0.1},
        ],
        "b": [
            {"label": 0, "pred_score": 0.8},
            {"label": 2, "pred_score": 0.7},
            {"label": 1, "pred_score": 0.2},
        ],
    }
    metrics = compute_ranking_metrics(grouped)
    assert metrics.samples == 2
    assert 0.0 <= metrics.ndcg_at_5 <= 1.0
    assert 0.0 <= metrics.hit_at_3 <= 1.0
