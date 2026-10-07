"""Retrieval study on Graz: rank of the gold concept in its own corpus for raw vs enriched queries,
query content, and ranking mode (dense, BM25, hybrid). Makes four enrichment calls: three with
metadata (stability) and one without.

Run: python eval/retrieval_study.py  ->  eval/results/retrieval_study_graz[_t0].json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from score import DATASETS, load_gold, norm                        # noqa: E402
from stage1.models import SourceProfile, SSNRole, column_key        # noqa: E402
from stage1.evidence import gather_evidence                         # noqa: E402
from stage1.property_retrieval import _load, build_query, _BGE_QUERY_PREFIX  # noqa: E402
from stage1 import query_enrich                                     # noqa: E402

RESULTS = ROOT / "eval" / "results"
CACHE = ROOT / "eval" / "cache" / "graz_profile.json"
KS = (1, 5, 10, 20, 30, 50)

from config import load_config                                      # noqa: E402
TEMPERATURE = load_config(ROOT / "config.toml").temperature
SUFFIX = "" if TEMPERATURE is None else f"_t{TEMPERATURE:g}"


def _ranks(pack, query: str) -> dict[str, dict[str, int]]:
    """1-based rank of every corpus id under dense, BM25 and RRF-hybrid ranking."""
    col, ids, _labels, bm25 = pack
    n = len(ids)
    dense_ids = col.query(query_texts=[query], n_results=n, include=[])["ids"][0]
    scores = bm25.get_scores(query.replace(_BGE_QUERY_PREFIX, "").lower().replace("_", " ").split())
    bm25_ids = [ids[i] for i in sorted(range(n), key=lambda x: scores[x], reverse=True)]
    rrf: dict[str, float] = {}
    for lst in (dense_ids, bm25_ids):
        for r, i in enumerate(lst):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + r + 1)
    hybrid_ids = sorted(rrf, key=rrf.get, reverse=True)
    return {m: {i: r + 1 for r, i in enumerate(lst)}
            for m, lst in (("dense", dense_ids), ("bm25", bm25_ids), ("hybrid", hybrid_ids))}


def enrich(profile, include_metadata: bool) -> dict[str, str]:
    query_enrich._MEMO.clear()
    return query_enrich.enrich_queries(profile, model="claude-sonnet-4-6",
                                       include_metadata=include_metadata, temperature=TEMPERATURE,
                                       cache_prompt=True)


def main() -> None:
    gold, _ = load_gold(DATASETS["graz"])
    profile = SourceProfile.model_validate_json(CACHE.read_text(encoding="utf-8"))
    companion = "\n".join(c.text for c in gather_evidence(profile)
                          if not c.source.startswith("structure:"))

    RESULTS.mkdir(parents=True, exist_ok=True)
    phrases_path = RESULTS / f"enrichment_phrases_graz{SUFFIX}.json"
    if phrases_path.exists():
        runs = json.loads(phrases_path.read_text(encoding="utf-8"))
    else:
        runs = dict(meta_1=enrich(profile, True), meta_2=enrich(profile, True),
                    meta_3=enrich(profile, True), nometa=enrich(profile, False))
        phrases_path.write_text(json.dumps(runs, indent=1, ensure_ascii=False), encoding="utf-8")
    phr = runs["meta_1"]

    # enrichment stability + metadata dependence over ALL observable columns
    keys = sorted(set().union(*[set(v) for v in runs.values()]))
    same_3 = sum(1 for k in keys if len({runs[f"meta_{i}"].get(k) for i in (1, 2, 3)}) == 1)
    same_nometa = sum(1 for k in keys if runs["meta_1"].get(k) == runs["nometa"].get(k))

    packs = _load()
    sensors = {s.id: s for s in profile.sensors}
    features = {f.id: f for f in profile.features}
    rows = []
    for fp in profile.files:
        for c in fp.columns:
            key = column_key(profile.root_path, fp.path, c.name)
            g = gold.get(key)
            if not g or g["source"] != "reuse" or c.ssn_role not in (SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN):
                continue
            corpus = "qudt_quantitykind" if g["property"].startswith("qudt:") else "gemet"
            pack = packs[corpus]
            gid = next((i for i in pack[1] if norm(i) == g["property"]), None)
            s, f = sensors.get(c.sensor_id), features.get(c.feature_id)
            desc = " ".join(x for x in (s.label if s else "", s.attributes.get("type", "") if s else "",
                                        f.label if f else "") if x)
            stem, unit, term = Path(fp.path).stem, c.unit or "", phr.get(key)
            q = {
                "raw": build_query(c.name, stem, ""),
                "raw+meta": build_query(c.name, stem, companion),
                "enr": build_query(c.name, "", "", query_term=term),
                "enr+meta": build_query(c.name, "", companion, query_term=term),
            }
            q["raw+meta+unit"] = f"{q['raw+meta']} {unit}".strip()
            q["raw+meta+unit+desc"] = f"{q['raw+meta']} {unit} {desc}".strip()
            q["enr+meta+unit"] = f"{q['enr+meta']} {unit}".strip()
            r = {}
            for name, text in q.items():
                rk = _ranks(pack, text)
                modes = ("dense", "bm25", "hybrid") if name in ("raw+meta", "enr+meta") else ("hybrid",)
                for m in modes:
                    r[f"{name}/{m}"] = rk[m].get(gid) if gid else None
            rows.append(dict(key=key, gold=g["property"], phrase=term, unit=unit, ranks=r))

    n = len(rows)
    variants = list(rows[0]["ranks"])
    rec = lambda v, k: sum(1 for x in rows if x["ranks"][v] is not None and x["ranks"][v] <= k) / n
    table = {v: {k: rec(v, k) for k in KS} for v in variants}
    print(f"\nGraz, {n} reuse gold columns. Recall@k of the gold concept in its own corpus.\n")
    print(f"  {'variant':<26}" + "".join(f"{'@' + str(k):>7}" for k in KS))
    for v in variants:
        print(f"  {v:<26}" + "".join(f"{table[v][k]:>7.0%}" for k in KS))
    print(f"\n  enrichment: {len(keys)} columns; identical across 3 runs: {same_3}/{len(keys)}; "
          f"identical with vs without metadata: {same_nometa}/{len(keys)}")
    print("\n  per column: phrase | rank raw+meta -> enr+meta -> enr+meta+unit (hybrid)")
    for x in rows:
        R = x["ranks"]
        print(f"    {x['gold']:<22} {x['key'][-40:]:<40} {str(x['phrase'])[:28]:<28} "
              f"{R['raw+meta/hybrid']} -> {R['enr+meta/hybrid']} -> {R['enr+meta+unit/hybrid']}")
    diff = {k: {r: runs[r].get(k) for r in runs} for k in keys
            if len({runs[r].get(k) for r in runs}) > 1}
    (RESULTS / f"retrieval_study_graz{SUFFIX}.json").write_text(json.dumps(dict(
        temperature=TEMPERATURE, n=n, recall=table, enrichment=dict(columns=len(keys), identical_3_runs=same_3,
                                           identical_with_without_metadata=same_nometa,
                                           differing=diff),
        rows=rows), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote eval/results/retrieval_study_graz{SUFFIX}.json")


if __name__ == "__main__":
    main()
