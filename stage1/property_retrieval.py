"""Observed-property candidates: hybrid dense + BM25 retrieval fused with RRF, top k from
QUDT quantity kinds and from GEMET separately."""
from __future__ import annotations
import re
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from stage2.property_index import _EF, _CHROMA

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_COLLECTIONS = ("qudt_quantitykind", "gemet")


@lru_cache(maxsize=1)
def _load():
    client = chromadb.PersistentClient(path=_CHROMA)
    packs = {}
    for name in _COLLECTIONS:
        col = client.get_collection(name, embedding_function=_EF)
        d = col.get(include=["documents", "metadatas"])
        labels = {i: m["label"] for i, m in zip(d["ids"], d["metadatas"])}
        bm25 = BM25Okapi([doc.lower().replace("_", " ").split() for doc in d["documents"]])
        packs[name] = (col, d["ids"], labels, bm25)
    return packs


def build_query(col_name: str, file_stem: str = "", metadata_text: str = "",
                query_term: str | None = None) -> str:
    parts = [(col_name if query_term is None else query_term).replace("_", " ")]
    if file_stem:
        parts.append(file_stem.replace("_", " ").replace("-", " "))
    if metadata_text:
        # short, URL-free metadata lines naming the column, plus the next two lines, capped
        tok = re.compile(rf"\b{re.escape(col_name)}\b", re.I)
        lines = metadata_text.splitlines()
        ctx, seen, chars = [], set(), 0
        for i, ln in enumerate(lines):
            if len(ln) > 160 or "http" in ln.lower():
                continue
            if tok.search(ln):
                for h in (x.strip() for x in lines[i:i + 3]):
                    if h and len(h) <= 160 and "http" not in h.lower() and h not in seen:
                        seen.add(h); ctx.append(h); chars += len(h)
            if chars > 200:
                break
        parts.extend(ctx)
    return _BGE_QUERY_PREFIX + " ".join(parts)


def _hybrid(pack, query: str, k: int) -> list[dict]:
    col, ids, labels, bm25 = pack
    n = len(ids)
    dense = col.query(query_texts=[query], n_results=n, include=["metadatas", "distances"])
    dist = {i: d for i, d in zip(dense["ids"][0], dense["distances"][0])}
    rrf: dict[str, float] = {}
    for rank, i in enumerate(dense["ids"][0]):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
    raw = query.replace(_BGE_QUERY_PREFIX, "")
    scores = bm25.get_scores(raw.lower().replace("_", " ").split())
    for rank, idx in enumerate(sorted(range(n), key=lambda x: scores[x], reverse=True)):
        rrf[ids[idx]] = rrf.get(ids[idx], 0.0) + 1.0 / (60 + rank + 1)
    top = sorted(rrf, key=rrf.get, reverse=True)[:k]
    src = "qudt" if pack is _load()["qudt_quantitykind"] else "gemet"
    return [{"uri": i, "label": labels[i], "source": src, "distance": round(dist.get(i, 1.0), 3)} for i in top]


def retrieve_candidates(col_name: str, file_stem: str = "", metadata_text: str = "",
                        k: int = 10, query_term: str | None = None) -> list[dict]:
    """Top k from each corpus, as {uri, label, source, distance}."""
    packs = _load()
    query = build_query(col_name, file_stem, metadata_text, query_term=query_term)
    out = []
    for name in _COLLECTIONS:
        out.extend(_hybrid(packs[name], query, k))
    return out
