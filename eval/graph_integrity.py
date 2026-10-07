"""Structural checks on a graph built from a cached profile (no API): one value per SOSA link per
observation, no dangling or unused declarations, a unit on every QuantityValue, typed timestamps,
and the Sample/Sampling shape when present.

Run: python eval/graph_integrity.py [eval/cache/<profile>.json] [tag]
     ->  eval/results/graph_integrity_graz{tag}.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Namespace, URIRef                              # noqa: E402
from rdflib.namespace import RDF, XSD                              # noqa: E402
from stage1.models import SourceProfile                            # noqa: E402
from stage1.assemble import assemble_ir                            # noqa: E402
from stage3.materialize import materialize                         # noqa: E402

SOSA = Namespace("http://www.w3.org/ns/sosa/")
QUDT = Namespace("http://qudt.org/schema/qudt/")


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "eval" / "cache" / "graz_profile.json"
    tag = sys.argv[2] if len(sys.argv) > 2 else ""
    profile = SourceProfile.model_validate_json(src.read_text(encoding="utf-8"))
    work = ROOT / "out" / f"integrity_graz{tag}"
    ir = assemble_ir(profile, work, max_rows=1000)
    g = materialize(ir, work)

    obs = set(g.subjects(RDF.type, SOSA.Observation))
    shape = Counter()
    for o in obs:
        for p in (SOSA.resultTime, SOSA.madeBySensor, SOSA.observedProperty, SOSA.hasFeatureOfInterest):
            if len(list(g.objects(o, p))) != 1:
                shape[f"not exactly one {p.split('/')[-1]}"] += 1
        n_res = len(list(g.objects(o, SOSA.hasResult))) + len(list(g.objects(o, SOSA.hasSimpleResult)))
        if n_res != 1:
            shape["not exactly one result"] += 1

    dangling = Counter()
    for p in (SOSA.madeBySensor, SOSA.observedProperty, SOSA.hasFeatureOfInterest, SOSA.hasResult,
              SOSA.isSampleOf, SOSA.isResultOf, SOSA.madeBySampler):
        for t in set(g.objects(None, p)):
            if isinstance(t, URIRef) and (t, RDF.type, None) not in g:
                dangling[p.split("/")[-1]] += 1

    unused = Counter()
    for cls, p in ((SOSA.Sensor, SOSA.madeBySensor), (SOSA.ObservableProperty, SOSA.observedProperty),
                   (SOSA.FeatureOfInterest, SOSA.hasFeatureOfInterest), (SOSA.Sampler, SOSA.madeBySampler)):
        for e in g.subjects(RDF.type, cls):
            if (None, p, e) not in g:
                unused[cls.split("/")[-1]] += 1

    qvs = set(g.subjects(RDF.type, QUDT.QuantityValue))
    qv_no_value = sum(1 for q in qvs if (q, QUDT.numericValue, None) not in g)
    qv_no_unit = sum(1 for q in qvs if (q, QUDT.unit, None) not in g)
    times = list(g.objects(None, SOSA.resultTime))
    untyped_time = sum(1 for t in times if getattr(t, "datatype", None) != XSD.dateTime)

    sampling = None
    sampled_files = [f for f in ir.files if f.sampling is not None]
    if sampled_files:
        samples = set(g.subjects(RDF.type, SOSA.Sample))
        samplings = set(g.subjects(RDF.type, SOSA.Sampling))
        sv = Counter()
        for s in samples:
            for p in (SOSA.isSampleOf, SOSA.isResultOf):
                if len(list(g.objects(s, p))) != 1:
                    sv[f"Sample: not exactly one {p.split('/')[-1]}"] += 1
        for s in samplings:
            for p in (SOSA.madeBySampler, SOSA.hasFeatureOfInterest, SOSA.resultTime, SOSA.hasResult):
                if len(list(g.objects(s, p))) != 1:
                    sv[f"Sampling: not exactly one {p.split('/')[-1]}"] += 1
        sampled_obs = [o for o in obs if any(f"/{f.file_id}/" in str(o) for f in sampled_files)]
        not_about_sample = sum(1 for o in sampled_obs
                               if not any(t in samples for t in g.objects(o, SOSA.hasFeatureOfInterest)))
        sampling = dict(sampled_files=[f.file_id for f in sampled_files], samples=len(samples),
                        samplings=len(samplings), samplers=len(ir.samplers),
                        sampled_observations=len(sampled_obs),
                        sampled_observations_not_about_a_sample=not_about_sample,
                        shape_violations=dict(sv))

    report = dict(source=str(src.relative_to(ROOT)), triples=len(g), observations=len(obs),
                  shape_violations=dict(shape), dangling_references=dict(dangling),
                  unreferenced_declarations=dict(unused), quantity_values=len(qvs),
                  quantity_values_without_numericValue=qv_no_value,
                  quantity_values_without_unit=qv_no_unit,
                  resultTime_not_xsd_dateTime=untyped_time,
                  sensors=len(ir.sensors), properties=len(ir.properties), features=len(ir.features))
    if sampling is not None:
        report["sampling"] = sampling
    out = ROOT / "eval" / "results" / f"graph_integrity_graz{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
