"""Rescores every cached run reported in Chapter 6 against the current gold (no API).

Run: python eval/ch6_summary.py  ->  eval/results/ch6_summary.json
"""
from __future__ import annotations
import json, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from score import DATASETS, load_gold, extract_pred, score          # noqa: E402
from stage1.models import SourceProfile                             # noqa: E402

CACHE = ROOT / "eval" / "cache"
SERIES = {  # name -> (dataset, cache-file pattern with {i}, runs)
    "graz/baseline":   ("graz", "graz_profile__t0_baseline_{i}.json", 4),
    "graz/enrich_k30": ("graz", "graz_profile__t0_enrich_k30_{i}.json", 4),
    "graz/sonnet46":   ("graz", "graz_profile__t0_main_{i}.json", 4),
    "graz/haiku45":    ("graz", "graz_profile__haiku45_{i}.json", 2),
    "graz/opus46":     ("graz", "graz_profile__opus46_{i}.json", 2),
}
for ds in ("flowbru_beco", "flowbru_belliard"):
    SERIES[f"{ds}/sonnet46"] = (ds, ds + "_profile_{i}.json", 2)
    SERIES[f"{ds}/haiku45"] = (ds, ds + "_profile__haiku45_{i}.json", 2)
    SERIES[f"{ds}/opus46"] = (ds, ds + "_profile__opus46_{i}.json", 2)
    SERIES[f"{ds}/shared_menu"] = (ds, ds + "_profile__shared_menu_{i}.json", 2)
SERIES["graz/shared_menu"] = ("graz", "graz_profile__shared_menu_{i}.json", 4)
SERIES["graz/main_sampling"] = ("graz", "graz_profile__main_sampling_{i}.json", 4)
for _ds in ("flowbru_beco", "flowbru_belliard"):
    SERIES[f"{_ds}/main_sampling"] = (_ds, _ds + "_profile__main_sampling_{i}.json", 2)


def metrics(r: dict) -> dict:
    L = r["layers"]
    l3 = L["L3_sensor"] or {}
    return dict(n_gold=r["n_gold"], n_matched=r["n_matched"], n_spurious=len(r["spurious"]),
                l1=L["L1_property"]["hits"], l1_len=L["L1_property"]["lenient_hits"],
                reuse_mint=L["reuse_mint"]["hits"], unit=L["L2_unit"]["hits"], feature=L["L4_feature"]["hits"],
                sensor_f1=l3.get("f1"),
                errors=[dict(key=x["key"], gold=x["gold"]["property"], pred=x["pred"]["property"],
                             lenient_ok=x["prop_ok_lenient"]) for x in r["rows"] if not x["prop_ok"]],
                unit_errors=[dict(key=x["key"], gold=x["gold"]["unit"], pred=x["pred"]["unit"])
                             for x in r["rows"] if not x["unit_ok"]],
                missed=r["missed"])


out = {}
for name, (ds, pat, n) in SERIES.items():
    cfg = DATASETS[ds]; gold, _ = load_gold(cfg)
    runs = []
    for i in range(1, n + 1):
        f = CACHE / pat.format(i=i)
        prof = SourceProfile.model_validate_json(f.read_text(encoding="utf-8"))
        runs.append(metrics(score(gold, extract_pred(prof), {}, cfg)))
    agg = {}
    for k in ("l1", "l1_len", "reuse_mint", "unit", "feature", "n_matched", "n_spurious"):
        vals = [r[k] for r in runs]
        agg[k] = dict(mean=statistics.mean(vals), runs=vals)
    f1 = [r["sensor_f1"] for r in runs if r["sensor_f1"] is not None]
    agg["sensor_f1"] = dict(mean=statistics.mean(f1) if f1 else None, runs=f1)
    out[name] = dict(dataset=ds, n_gold=runs[0]["n_gold"], runs=len(runs), agg=agg, run_detail=runs)
    a = agg
    print(f"{name:<26} gold {runs[0]['n_gold']:>2}  kept {a['n_matched']['runs']}  L1 {a['l1']['runs']}  "
          f"lenient {a['l1_len']['runs']}  unit {a['unit']['runs']}  sensorF1 {[round(x, 2) for x in a['sensor_f1']['runs']]}  "
          f"feat {a['feature']['runs']}  spurious {a['n_spurious']['runs']}")
(ROOT / "eval" / "results" / "ch6_summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote eval/results/ch6_summary.json")
