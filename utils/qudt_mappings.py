"""Unit string -> QUDT unit URI by exact match on ucumCode / symbol (no fuzzy matching)."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import Namespace

QUDT = Namespace("http://qudt.org/schema/qudt/")
_VOCAB = Path(__file__).resolve().parent.parent / "vocab" / "qudt_unit.ttl"

_ALIASES = {
    "l/s": "L/s", "lps": "L/s",
    "m3/s": "m3/s", "m^3/s": "m3/s",
    "m3/h": "m3/h", "m^3/h": "m3/h",
    "degc": "Cel", "deg c": "Cel", "celsius": "Cel",
    "us/cm": "uS/cm",
    "mg/l": "mg/L", "ug/l": "ug/L",
    "percent": "%", "pct": "%",
}


def _norm(s: str) -> str:
    """Fold unicode variants and whitespace; keep case (UCUM is case-sensitive)."""
    s = (s.strip()
           .replace("µ", "u").replace("μ", "u")
           .replace("²", "2").replace("³", "3")
           .replace("·", "."))
    return re.sub(r"\s+", "", s)


@lru_cache(maxsize=1)
def _index() -> dict[str, str]:
    g = Graph()
    g.parse(_VOCAB, format="turtle")

    key_to_uri: dict[str, str] = {}
    units = set(g.subjects(QUDT.ucumCode, None)) | set(g.subjects(QUDT.symbol, None))
    for unit in sorted(units, key=str):          # sorted: deterministic first match wins
        if not isinstance(unit, URIRef):
            continue
        uri = str(unit)
        keys = [str(o) for o in g.objects(unit, QUDT.ucumCode)]    # UCUM codes take priority
        keys += [str(o) for o in g.objects(unit, QUDT.symbol)]
        for k in keys:
            for variant in (k, _norm(k)):
                key_to_uri.setdefault(variant, uri)
    return key_to_uri


def resolve_unit(unit: str) -> str | None:
    if not unit:
        return None
    key_to_uri = _index()
    raw = unit.strip()
    norm = _norm(raw)
    canon = _ALIASES.get(norm.lower())
    for cand in (raw, norm, canon, _norm(canon) if canon else None):
        if cand and cand in key_to_uri:
            return key_to_uri[cand]
    return None


if __name__ == "__main__":
    for u in ["m", "L/s", "mg/L", "°C", "µS/cm", "NTU", "FNU", "%"]:
        print(f"{u:>7} -> {resolve_unit(u)}")
