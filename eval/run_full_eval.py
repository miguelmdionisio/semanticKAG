"""Configuration ablation on Graz: each configuration's resolve call runs R times (cached, so the
script is resumable), is scored with score.py and aggregated. Also reports API usage, per-column
stability, column classification, and a retrieval-only baseline (top-1 candidate, no model choice).

Run: python eval/run_full_eval.py [--repeats 3] [--only main,baseline] [--run-name t0]
     ->  eval/results/<run-name>_eval_graz.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import anthropic                                                    # noqa: E402
from config import load_config, override                            # noqa: E402
from score import DATASETS, load_gold, get_profile, get_menus, extract_pred, score, norm  # noqa: E402
from stage1.models import SSNRole, column_key                       # noqa: E402

RESULTS = ROOT / "eval" / "results"
USAGE_PATH = RESULTS / "usage_graz.json"

CONFIGS = [
    ("baseline",     "eval/configs/baseline.toml"),       # k10, no enrichment, no classification
    ("noenrich_k30", "eval/configs/noenrich_k30.toml"),   # k30 only
    ("enrich_k10",   "eval/configs/enrich_k10.toml"),     # enrichment only
    ("enrich_k30",   "eval/configs/enrich_k30.toml"),     # enrichment + k30
    ("main",         "config.toml"),                      # enrichment + k30 + classification
]

_CTX = {"config": None, "run": None}
_USAGE: list[dict] = json.loads(USAGE_PATH.read_text(encoding="utf-8")) if USAGE_PATH.exists() else []
_orig_create = anthropic.resources.messages.Messages.create


def _create(self, *a, **kw):
    t = time.perf_counter()
    r = _orig_create(self, *a, **kw)
    _USAGE.append(dict(config=_CTX["config"], run=_CTX["run"], tool=kw["tool_choice"]["name"],
                       model=kw["model"], seconds=round(time.perf_counter() - t, 1),
                       input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                       cache_write=getattr(r.usage, "cache_creation_input_tokens", 0) or 0,
                       cache_read=getattr(r.usage, "cache_read_input_tokens", 0) or 0,
                       stop_reason=r.stop_reason))
    RESULTS.mkdir(parents=True, exist_ok=True)
    USAGE_PATH.write_text(json.dumps(_USAGE, indent=1), encoding="utf-8")
    return r


anthropic.resources.messages.Messages.create = _create


def run_metrics(r: dict, profile, gold: dict) -> dict:
    L = r["layers"]
    rows = r["rows"]
    strict = [x for x in rows if not x["gold"]["soft"]]
    kept = {column_key(profile.root_path, fp.path, c.name): c
            for fp in profile.files for c in fp.columns
            if c.ssn_role in (SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN)}
    demoted = [k for k, c in kept.items() if c.column_class and c.column_class != "measurement"]
    return dict(
        coverage=r["coverage"], n_pred=r["n_pred"], n_spurious=len(r["spurious"]),
        L0=L["L0_retrieval_recall"]["acc"],
        L1=L["L1_property"]["acc"],
        L1_strict=(sum(x["prop_ok"] for x in strict) / len(strict)) if strict else 0.0,
        reuse_mint=L["reuse_mint"]["acc"],
        L2=L["L2_unit"]["acc"],
        L3_f1=(L["L3_sensor"] or {}).get("f1"), L3_p=(L["L3_sensor"] or {}).get("precision"),
        L3_r=(L["L3_sensor"] or {}).get("recall"), L3_ari=(L["L3_sensor"] or {}).get("ari"),
        L4=L["L4_feature"]["acc"],
        harmonization=r["harmonization"]["rate"],
        n_sensors=len(profile.sensors),
        n_properties=len(profile.properties),
        property_sources=dict(Counter(p.source for p in profile.properties)),
        minted=[p.label for p in profile.properties if p.source == "minted"],
        demoted=len(demoted),
        wrong_demotions=[k for k in demoted if k in gold],
        spurious=[s["key"] for s in r["spurious"]],
        per_column={x["key"]: dict(prop_ok=x["prop_ok"], pred=x["pred"]["property"],
                                   gold=x["gold"]["property"], in_menu=x["in_menu"])
                    for x in rows},
    )


def retrieval_only(menus: dict, gold: dict) -> dict:
    """L1 accuracy if the property were simply the top-1 retrieved candidate."""
    a = b = n = 0
    for k, g in gold.items():
        cands = menus.get(k)
        if not cands:
            continue
        n += 1
        top_any = min(cands, key=lambda c: c["distance"])            # dense distance, both corpora
        a += norm(top_any["uri"]) == g["property"]
        corpus = "qudt" if g["property"].startswith("qudt:") else "gemet"
        own = [c for c in cands if c["source"] == corpus]              # RRF order within corpus
        b += bool(own) and norm(own[0]["uri"]) == g["property"]
    return dict(n=n, top1_any=a / n if n else 0.0, top1_gold_corpus=b / n if n else 0.0)


def agg(values: list) -> dict:
    vals = [v for v in values if v is not None]
    if not vals:
        return dict(mean=None, std=None, runs=values)
    return dict(mean=statistics.mean(vals),
                std=statistics.pstdev(vals) if len(vals) > 1 else 0.0, runs=values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", help="comma-separated config names")
    ap.add_argument("--run-name", default="full", help="names this series: cache tag + results file")
    args = ap.parse_args()
    todo = [c for c in CONFIGS if not args.only or c[0] in args.only.split(",")]

    out_path = RESULTS / f"{args.run_name}_eval_graz.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    for name, path in todo:
        cfg = dict(DATASETS["graz"])
        # repeats send the same prompt, so cache it
        cfg["pipe"] = override(load_config(ROOT / path), cache_prompt=args.repeats > 1 or None)
        cfg["tag"] = f"__{args.run_name}_{name}"
        gold, _ = load_gold(cfg)
        runs, menus = [], None
        for i in range(1, args.repeats + 1):
            _CTX.update(config=name, run=i)
            profile, cached = get_profile(cfg, refresh=False, run_idx=i)
            if menus is None:
                _CTX.update(run="menus")
                menus = get_menus(profile, cfg)
            r = score(gold, extract_pred(profile), menus, cfg)
            m = run_metrics(r, profile, gold)
            runs.append(m)
            print(f"[{name} run {i}{' cached' if cached else ''}] L0 {m['L0']:.0%}  L1 {m['L1']:.0%}  "
                  f"reuse/mint {m['reuse_mint']:.0%}  unit {m['L2']:.0%}  sensorF1 {m['L3_f1']:.0%}  "
                  f"feat {m['L4']:.0%}  cov {m['coverage']:.0%}  spurious {m['n_spurious']}  "
                  f"demoted {m['demoted']} (wrong {len(m['wrong_demotions'])})", flush=True)

        # per-column stability of the property decision across runs
        cols = sorted(runs[0]["per_column"])
        stability = {k: dict(correct_runs=sum(r["per_column"][k]["prop_ok"] for r in runs if k in r["per_column"]),
                             preds=sorted({r["per_column"][k]["pred"] for r in runs if k in r["per_column"]}),
                             gold=runs[0]["per_column"][k]["gold"],
                             in_menu=runs[0]["per_column"][k]["in_menu"])
                     for k in cols}
        scalar = ["coverage", "n_pred", "n_spurious", "L0", "L1", "L1_strict", "reuse_mint", "L2",
                  "L3_f1", "L3_p", "L3_r", "L3_ari", "L4", "harmonization", "n_sensors",
                  "n_properties", "demoted"]
        results[name] = dict(
            pipeline=asdict(cfg["pipe"]), repeats=len(runs),
            summary={k: agg([r[k] for r in runs]) for k in scalar},
            retrieval_only=retrieval_only(menus, gold),
            stability=stability,
            runs=[{k: v for k, v in r.items() if k != "per_column"} for r in runs],
        )
        RESULTS.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'config':<14}" + "".join(f"{h:>13}" for h in
          ("L0", "L1", "L1 strict", "reuse/mint", "unit", "sensor F1", "feature", "spurious")))
    for name, _ in CONFIGS:
        if name not in results:
            continue
        s = results[name]["summary"]
        cell = lambda k, pct=True: (f"{s[k]['mean']:.0%}±{s[k]['std']:.0%}" if pct
                                    else f"{s[k]['mean']:.1f}±{s[k]['std']:.1f}")
        print(f"{name:<14}" + "".join(f"{cell(k):>13}" for k in
              ("L0", "L1", "L1_strict", "reuse_mint", "L2", "L3_f1", "L4"))
              + f"{cell('n_spurious', False):>13}")
    print(f"\nwrote {out_path.relative_to(ROOT)}  (usage: {USAGE_PATH.relative_to(ROOT)})")


if __name__ == "__main__":
    main()
