"""Feature-of-interest typing: maps the model's feature_kind phrase to a GWSW class by dense
retrieval over vocab/gwsw_foi.jsonl. Below the threshold the feature stays untyped."""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

_VOCAB = Path(__file__).resolve().parents[1] / "vocab" / "gwsw_foi.jsonl"
_MODEL = "BAAI/bge-m3"
_THRESHOLD = 0.42          # cosine similarity


def _doc(r: dict) -> str:
    en = r.get("label_en") or r["label_nl"]
    de = r.get("def_en") or ""
    return f"{en}. {de} ({r['label_nl']})".strip()


@lru_cache(maxsize=1)
def _index():
    from sentence_transformers import SentenceTransformer
    import torch
    recs = [json.loads(l) for l in _VOCAB.read_text(encoding="utf-8").splitlines() if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(_MODEL, device=device)
    emb = np.asarray(model.encode([_doc(r) for r in recs],
                                  normalize_embeddings=True, batch_size=64,
                                  show_progress_bar=False))
    return recs, emb, model


def resolve_feature_kind(kind: str, threshold: float = _THRESHOLD) -> Optional[dict]:
    """feature_kind -> {uri, label, label_nl, score}, or None."""
    if not kind or not kind.strip():
        return None
    recs, emb, model = _index()
    qv = model.encode([kind], normalize_embeddings=True)[0]
    sims = emb @ np.asarray(qv)
    i = int(sims.argmax())
    score = float(sims[i])
    if score < threshold:
        return None
    r = recs[i]
    return {"uri": r["uri"], "label": r.get("label_en") or r["label_nl"],
            "label_nl": r["label_nl"], "score": round(score, 3)}
