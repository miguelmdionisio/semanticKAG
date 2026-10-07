"""Free-generation ablation on Graz: the resolve call without candidate menus. The model names
vocabulary URIs from its own knowledge; answers are scored without the membership guard, and URIs
missing from the vendored vocabularies count as hallucinated.

Run: python eval/free_generation.py [--repeats 2]  ->  eval/results/free_generation_graz[_t0].json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from config import load_config                                       # noqa: E402
from score import DATASETS, load_gold, norm                          # noqa: E402
from stage1.models import SourceProfile                              # noqa: E402
from stage1 import resolver                                          # noqa: E402
from stage1.property_retrieval import _load                          # noqa: E402

RESULTS = ROOT / "eval" / "results"
CACHE = ROOT / "eval" / "cache" / "graz_profile.json"

_SUBS = [
    ("and a CANDIDATE MENU of observed-property URIs retrieved from two vocabularies",
     "and you must name its observed property using one of two vocabularies"),
    ("pick the SINGLE best candidate URI that captures what the column measures.",
     "give the full URI of the SINGLE best concept that captures what the column measures, from "
     "your own knowledge of the two vocabularies: a QUDT quantity kind "
     "(http://qudt.org/vocab/quantitykind/<Name>) or a GEMET concept "
     "(http://www.eionet.europa.eu/gemet/concept/<number>)."),
    ("If NO candidate is a good", "If NO concept you know is a good"),
]


def build_free_request(profile, cfg):
    orig = resolver.retrieve_candidates
    resolver.retrieve_candidates = lambda *a, **kw: []          # no menus
    try:
        system, prompt, tool, _ = resolver.build_request(profile, config=cfg)
    finally:
        resolver.retrieve_candidates = orig
    for a, b in _SUBS:
        assert a in system, f"prompt drifted, substitution not found: {a[:40]}"
        system = system.replace(a, b)
    prompt = "\n".join(l for l in prompt.splitlines() if l.strip() != "candidates:")
    return system, prompt, tool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()
    gold, _ = load_gold(DATASETS["graz"])
    profile = SourceProfile.model_validate_json(CACHE.read_text(encoding="utf-8"))
    for fp in profile.files:
        for c in fp.columns:
            c.column_class = None
    cfg = replace(load_config(ROOT / "config.toml"), enrich_enabled=False)
    system, prompt, tool = build_free_request(profile, cfg)
    vocab = {norm(i) for pack in _load().values() for i in pack[1]}

    # a set temperature gets its own output file (e.g. _t0)
    suffix = "" if cfg.temperature is None else f"_t{cfg.temperature:g}"
    extra = {} if cfg.temperature is None else {"temperature": cfg.temperature}
    out_path = RESULTS / f"free_generation_graz{suffix}.json"
    runs = json.loads(out_path.read_text(encoding="utf-8"))["runs"] if out_path.exists() else []
    while len(runs) < args.repeats:
        resp = resolver._client().messages.create(
            model=cfg.model, max_tokens=16384, system=system, tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt,
                       "cache_control": {"type": "ephemeral"}}]}], **extra)
        ans = next(b for b in resp.content if b.type == "tool_use").input
        runs.append(dict(columns=ans.get("columns", {}), usage=dict(
            input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens)))
        RESULTS.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(dict(runs=runs), indent=1, ensure_ascii=False), encoding="utf-8")

    summary = []
    for i, run in enumerate(runs, 1):
        cols = run["columns"]
        n = ok = rm = hall = minted = 0
        detail = []
        for k, g in gold.items():
            r = cols.get(k)
            if not r:
                continue
            n += 1
            uri = r.get("property_uri", "MINT")
            is_mint = uri == "MINT"
            p = None if is_mint else norm(uri)
            exists = is_mint or p in vocab
            ok += (p == g["property"]) if not is_mint else 0
            rm += (g["source"] == "mint") == is_mint
            minted += is_mint
            hall += not exists
            detail.append(dict(key=k, gold=g["property"], pred=uri, correct=(p == g["property"]),
                               exists_in_vocab=exists))
        # all non-MINT URIs the model produced (not only gold columns)
        all_uris = [r.get("property_uri") for r in cols.values() if r.get("property_uri") != "MINT"]
        all_hall = [u for u in all_uris if norm(u) not in vocab]
        s = dict(run=i, n=n, L1=ok / n, reuse_mint=rm / n, minted=minted,
                 hallucinated_gold_cols=hall, uris_total=len(all_uris),
                 uris_hallucinated=len(all_hall), hallucinated_examples=sorted(set(all_hall))[:10],
                 usage=run["usage"], detail=detail)
        summary.append(s)
        print(f"[free run {i}] L1 {s['L1']:.0%}  reuse/mint {s['reuse_mint']:.0%}  minted {minted}/{n}  "
              f"non-existent URIs {len(all_hall)}/{len(all_uris)}  e.g. {s['hallucinated_examples'][:3]}")
        for d in detail:
            print(f"    {'OK ' if d['correct'] else '   '}{'   ' if d['exists_in_vocab'] else 'NX '}"
                  f"{d['key'][-42:]:<42} gold={d['gold']:<22} pred={d['pred']}")
    out_path.write_text(json.dumps(dict(runs=runs, summary=summary), indent=1, ensure_ascii=False),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
