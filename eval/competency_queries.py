"""Competency questions, answered with SPARQL over the merged graph of Graz and the two FlowBru
stations, built from cached main-configuration runs (no API).

Run: python eval/competency_queries.py  ->  eval/results/competency_queries.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rdflib                                                       # noqa: E402
from rdflib import Graph, Namespace, URIRef                         # noqa: E402
from stage1.models import SourceProfile                             # noqa: E402
from stage1.assemble import assemble_ir                             # noqa: E402
from stage3.materialize import materialize                          # noqa: E402

QUDT = Namespace("http://qudt.org/schema/qudt/")
RUNS = {
    "graz": "graz_profile__main_sampling_1.json",
    "flowbru_beco": "flowbru_beco_profile__main_sampling_1.json",
    "flowbru_belliard": "flowbru_belliard_profile__main_sampling_1.json",
}

PREFIXES = """
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX qudt: <http://qudt.org/schema/qudt/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
# the dataset of an observation is the path segment after /obs/
DS = 'BIND(REPLACE(STR(?obs), "^http://example.org/obs/([^/]+)/.*$", "$1") AS ?ds)'

QUERIES = {
    "Q1_shared_properties": (
        "Which observed properties are measured in more than one dataset?",
        PREFIXES + f"""
SELECT ?property (COUNT(DISTINCT ?ds) AS ?datasets) (COUNT(?obs) AS ?observations)
       (GROUP_CONCAT(DISTINCT ?ds; separator=", ") AS ?which)
WHERE {{ ?obs a sosa:Observation ; sosa:observedProperty ?property . {DS} }}
GROUP BY ?property HAVING (COUNT(DISTINCT ?ds) > 1)
ORDER BY DESC(?datasets) DESC(?observations)"""),
    "Q2_water_level_in_metres": (
        "All water-level observations of every dataset, in metres, whatever unit they were recorded in.",
        PREFIXES + f"""
SELECT ?ds ?unit (COUNT(?obs) AS ?n) (MIN(?m) AS ?min_m) (MAX(?m) AS ?max_m)
WHERE {{
  ?obs a sosa:Observation ;
       sosa:observedProperty <http://www.eionet.europa.eu/gemet/concept/9190> ;
       sosa:hasResult ?r .
  ?r qudt:numericValue ?v ; qudt:unit ?unit .
  ?unit qudt:conversionMultiplier ?f .
  BIND(?v * ?f AS ?m)
  {DS}
}}
GROUP BY ?ds ?unit ORDER BY ?ds ?unit"""),
    "Q3_precipitation_sensors": (
        "Which sensors measure precipitation, in any dataset?",
        PREFIXES + f"""
SELECT ?ds ?sensor ?label (COUNT(?obs) AS ?n)
WHERE {{
  ?obs a sosa:Observation ;
       sosa:observedProperty <http://www.eionet.europa.eu/gemet/concept/637> ;
       sosa:madeBySensor ?sensor .
  OPTIONAL {{ ?sensor rdfs:label ?label }}
  {DS}
}}
GROUP BY ?ds ?sensor ?label ORDER BY ?ds ?sensor"""),
}


def main() -> None:
    merged = Graph()
    for ds, fname in RUNS.items():
        prof = SourceProfile.model_validate_json((ROOT / "eval" / "cache" / fname).read_text(encoding="utf-8"))
        g = materialize(assemble_ir(prof, ROOT / "out" / "cq" / ds, max_rows=1000), ROOT / "out" / "cq" / ds)
        print(f"[cq] {ds}: {len(g):,} triples")
        merged += g
    # QUDT conversion multipliers of the units the merged graph uses
    units = set(merged.objects(None, QUDT.unit))
    qudt_units = Graph().parse(str(ROOT / "vocab" / "qudt_unit.ttl"))
    for u in units:
        for f in qudt_units.objects(u, QUDT.conversionMultiplier):
            merged.add((u, QUDT.conversionMultiplier, f))
    print(f"[cq] merged: {len(merged):,} triples, {len(units)} units")

    out = {"datasets": RUNS, "triples": len(merged), "queries": {}}
    for qid, (question, q) in QUERIES.items():
        rows = [{str(k): (str(v) if v is not None else None) for k, v in r.asdict().items()} for r in merged.query(q)]
        out["queries"][qid] = dict(question=question, sparql=q.strip(), rows=rows)
        print(f"\n{qid}: {question}")
        for r in rows:
            print("   ", {k: (v.rsplit('/', 1)[-1] if v and v.startswith("http") else v) for k, v in r.items()})
    path = ROOT / "eval" / "results" / "competency_queries.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
