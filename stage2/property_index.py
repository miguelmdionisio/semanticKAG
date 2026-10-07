"""Builds the retrieval index (chroma_db/): QUDT quantity kinds and GEMET concepts."""
from __future__ import annotations
from pathlib import Path

import rdflib
import chromadb
import torch
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rank_bm25 import BM25Okapi
from rdflib.namespace import RDF, RDFS, SKOS, DCTERMS, Namespace

QUDT = Namespace("http://qudt.org/schema/qudt/")
_QK_NS = "http://qudt.org/vocab/quantitykind/"

_ROOT = Path(__file__).resolve().parent.parent
_VOCAB = _ROOT / "vocab"
_CHROMA = str(_ROOT / "chroma_db")

_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_EF = SentenceTransformerEmbeddingFunction(model_name=_EMBEDDING_MODEL, device=_DEVICE)


def _first(g, s, *preds):
    for p in preds:
        for o in g.objects(s, p):
            return str(o)
    return ""


def _label_en(g, s):
    labels = list(g.objects(s, RDFS.label))
    for o in labels:
        if getattr(o, "language", None) in (None, "en"):
            return str(o)
    return str(labels[0]) if labels else ""


def _fresh_collection(name: str):
    client = chromadb.PersistentClient(path=_CHROMA)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(
        name=name, embedding_function=_EF, metadata={"hnsw:space": "cosine"}
    )


def _bm25(docs: list[str]) -> BM25Okapi:
    return BM25Okapi([d.lower().replace("_", " ").split() for d in docs])


def _add_all(col, ids, docs, metas, batch=5000):
    # Chroma caps the batch size
    for i in range(0, len(ids), batch):
        col.add(ids=ids[i:i + batch], documents=docs[i:i + batch], metadatas=metas[i:i + batch])


def build_qudt_qk():
    g = rdflib.Graph(); g.parse(_VOCAB / "qudt_quantitykind.ttl", format="turtle")
    ids, docs, metas = [], [], []
    for s in set(g.subjects(RDFS.label, None)):
        if not str(s).startswith(_QK_NS):
            continue
        label = _label_en(g, s)
        if not label:
            continue
        descr = _first(g, s, QUDT.plainTextDescription, DCTERMS.description, RDFS.comment)
        ids.append(str(s))
        docs.append(f"{label}. {descr}".strip())
        metas.append({"label": label, "source": "qudt"})
    col = _fresh_collection("qudt_quantitykind")
    _add_all(col, ids, docs, metas)
    print(f"qudt_quantitykind: indexed {len(ids)}")
    return col, _bm25(docs), ids


def build_gemet():
    g = rdflib.Graph(); g.parse(_VOCAB / "gemet_en.ttl", format="turtle")
    label_of = {s: _first(g, s, SKOS.prefLabel) for s in g.subjects(SKOS.prefLabel, None)}
    ids, docs, metas = [], [], []
    for s, label in label_of.items():
        if not label:
            continue
        defn = _first(g, s, SKOS.definition)
        parents = [label_of.get(p, "") for p in g.objects(s, SKOS.broader)]
        parents = " ".join(p for p in parents if p)
        doc = ". ".join(x for x in (label, defn, parents) if x)
        ids.append(str(s))
        docs.append(doc)
        metas.append({"label": label, "source": "gemet"})
    col = _fresh_collection("gemet")
    _add_all(col, ids, docs, metas)
    print(f"gemet: indexed {len(ids)}")
    return col, _bm25(docs), ids


if __name__ == "__main__":
    build_qudt_qk()
    build_gemet()
    print("done ->", _CHROMA)
