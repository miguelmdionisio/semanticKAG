"""Sample detection in the main configuration (no API, cached runs): the files the resolve call
marked as sample-based against the known ones (Graz: lab data/samples.csv; FlowBru: none), and
the graph shape of the samples.

Run: python eval/sampling_eval.py  ->  eval/results/sampling_eval.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdflib import Namespace                                       # noqa: E402
from rdflib.namespace import RDF                                    # noqa: E402
from stage1.models import SourceProfile                             # noqa: E402
from stage1.assemble import assemble_ir                             # noqa: E402
from stage3.materialize import materialize                          # noqa: E402

SOSA = Namespace("http://www.w3.org/ns/sosa/")
CACHE = ROOT / "eval" / "cache"
# dataset -> (cache pattern, runs, files expected to be sampled)
CASES = {
    "graz": ("graz_profile__main_sampling_{i}.json", 4, {"lab data/samples.csv"}),
    "flowbru_beco": ("flowbru_beco_profile__main_sampling_{i}.json", 2, set()),
    "flowbru_belliard": ("flowbru_belliard_profile__main_sampling_{i}.json", 2, set()),
}


def main() -> None:
    out = {}
    tp = fp = fn = 0
    for ds, (pat, n, expected) in CASES.items():
        runs = []
        for i in range(1, n + 1):
            f = CACHE / pat.format(i=i)
            if not f.exists():
                print(f"[sampling] missing {f.name}")
                continue
            prof = SourceProfile.model_validate_json(f.read_text(encoding="utf-8"))
            root = Path(prof.root_path).resolve()
            sampled = {Path(x.path).resolve().relative_to(root).as_posix(): x.sampling
                       for x in prof.files if x.sampling is not None}
            feats = {x.id: x for x in prof.features}
            samplers = {x.id: x for x in prof.samplers}
            got = set(sampled)
            tp += len(got & expected); fp += len(got - expected); fn += len(expected - got)
            detail = {}
            for path, s in sampled.items():
                smp = samplers.get(s.sampler_id) if s.sampler_id else None
                ft = feats.get(s.feature_id) if s.feature_id else None
                detail[path] = dict(
                    sampler=None if smp is None else dict(id=smp.id, label=smp.label,
                                                          identification=str(getattr(smp.identification, "value", smp.identification)),
                                                          evidence_source=smp.evidence_source),
                    feature=None if ft is None else dict(id=ft.id, label=ft.label, gwsw=ft.gwsw_class_uri))
            # graph shape of the samples (first 1,000 rows, as in the rest of the evaluation)
            shape = None
            if sampled:
                work = ROOT / "out" / "sampling_eval" / f"{ds}_{i}"
                g = materialize(assemble_ir(prof, work, max_rows=1000), work)
                samples = set(g.subjects(RDF.type, SOSA.Sample))
                samplings = set(g.subjects(RDF.type, SOSA.Sampling))
                obs_on_sample = sum(1 for o, ft in g.subject_objects(SOSA.hasFeatureOfInterest)
                                    if ft in samples)
                shape = dict(samples=len(samples), samplings=len(samplings),
                             samples_with_isSampleOf=sum(1 for s in samples if (s, SOSA.isSampleOf, None) in g),
                             samplings_with_sampler=sum(1 for s in samplings if (s, SOSA.madeBySampler, None) in g),
                             observations_about_a_sample=obs_on_sample, triples=len(g))
            runs.append(dict(run=i, sampled=sorted(got), expected=sorted(expected),
                             correct=got == expected, detail=detail, graph=shape))
            print(f"[{ds} run {i}] sampled {sorted(got) or '-'}  expected {sorted(expected) or '-'}  "
                  f"{'OK' if got == expected else 'WRONG'}  {json.dumps(detail, ensure_ascii=False)[:220]}"
                  + (f"  graph {shape}" if shape else ""))
        out[ds] = runs
    out["files"] = dict(true_positive=tp, false_positive=fp, false_negative=fn)
    print(f"[sampling] files: TP {tp}, FP {fp}, FN {fn}")
    path = ROOT / "eval" / "results" / "sampling_eval.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
