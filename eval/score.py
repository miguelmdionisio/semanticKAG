"""Decision-level evaluation of the resolve call against a gold standard.

Layers, per observable column (joined on column_key):
  L0  retrieval recall   gold property in the candidate menu
  L1  observed property  exact match, plus reuse vs mint
  L2  unit               exact QUDT unit
  L3  sensor             grouping agreement (pairwise F1, ARI)
  L4  feature            GWSW class, including "untyped"
plus coverage, spurious columns and intra-dataset harmonization.

Resolved profiles are cached in eval/cache/, so re-scoring makes no API call unless --refresh.

Run: python eval/score.py <dataset> [--repeats N] [--refresh] [--config eval/configs/<name>.toml]
     python eval/score.py graz --sweep     (retrieval recall@k, no API)
"""
from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stage1.models import SourceProfile, SSNRole, column_key  # noqa: E402

EVAL = ROOT / "eval"
CACHE = EVAL / "cache"


# full URIs and gold CURIEs both reduce to one `prefix:localname` token
_FAMILIES = [                       # first match wins
    ("quantitykind/",         "qudt"),
    ("/vocab/unit/",          "unit"),
    ("/concept/",             "gemet"),
    ("gwsw.nl",               "gwsw"),
    ("example.org/property/", "mint"),   # minted observed-property (pipeline URI)
]


# QUDT units listed twice under different ids (same dimension and multiplier) score as one
_UNIT_EQUIV = {
    "unit:MicroS-PER-CentiM": "unit:MicroS-PER-CM",       # µS/cm, multiplier 1e-4, A0E2L-3I0M-1H0T3D0
}


def norm_unit(v: str | None) -> str | None:
    u = norm(v)
    return _UNIT_EQUIV.get(u, u)


def norm(v: str | None) -> str | None:
    """Canonical token for a URI or gold CURIE; unknown values pass through."""
    if not v:
        return None
    for needle, prefix in _FAMILIES:
        if needle in v:
            return f"{prefix}:{v.rstrip('/').rsplit('/', 1)[-1]}"
    if v.startswith("ex:property/"):                    # gold's minted CURIE form
        return f"mint:{v.rstrip('/').rsplit('/', 1)[-1]}"
    return v                                             # qudt:/gemet:/unit:/gwsw:/untyped


# deterministic query cleaning, used only by the recall sweep
_ABBREV = {
    "cod": "chemical oxygen demand", "tss": "total suspended solids",
    "bod": "biochemical oxygen demand", "do": "dissolved oxygen",
    "tn": "total nitrogen", "tp": "total phosphorus", "nh4": "ammonium",
    "temp": "temperature", "cond": "conductivity", "turb": "turbidity",
    "precip": "precipitation", "vel": "velocity", "lvl": "level",
}


def clean_term(name: str) -> str:
    """Split camelCase and separators, drop indices and numeric ids, expand abbreviations."""
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name.strip())     # camelCase -> spaces
    s = re.sub(r"[_\-/]+", " ", s)                             # separators -> spaces
    s = re.sub(r"(?<=[A-Za-z])\s*\d+\b", "", s)                # COD2 -> COD
    toks = [t for t in s.split() if not re.fullmatch(r"\d+", t)]   # drop bare ids
    toks = [_ABBREV.get(t.lower(), t) for t in toks]           # expand abbreviations
    return " ".join(toks).strip()


# datasets
DATASETS = {
    "graz": dict(
        dataset_id="graz",
        path="data/02 - Graz-West R05/data-set",   # root so column_key == gold "file"
        gold="gold_graz.json",
        key="file_column",
        sensor_layer=True,
        supported=True,
    ),
    # FlowBru: two stations without documentation, each resolved as its own dataset
    "flowbru_beco": dict(
        dataset_id="flowbru_beco",
        path="data/03 - FlowBru/Water_Quality/Canal_Beco",
        gold="gold_flowbru_beco.json",
        key="file_column",
        sensor_layer=True,
        supported=True,
    ),
    "flowbru_belliard": dict(
        dataset_id="flowbru_belliard",
        path="data/03 - FlowBru/Buffer_Basin_Streams/BT_Belliard",
        gold="gold_flowbru_belliard.json",
        key="file_column",
        sensor_layer=True,
        supported=True,
    ),
    "uwo": dict(
        dataset_id="uwo",
        path=None,
        gold="gold_uwo.json",
        key="variable",
        sensor_layer=True,
        supported=False,                            # relational; use the per-site adapter datasets
    ),
}


# UWO through eval/uwo_adapter.py: one dataset per site folder
_UWO_SITES = EVAL / "uwo_sites.json"
if _UWO_SITES.exists():
    for _ds, _m in json.loads(_UWO_SITES.read_text(encoding="utf-8"))["sites"].items():
        DATASETS[_ds] = dict(dataset_id=_ds, path=_m["path"], gold=_m["gold"], key="file_column",
                             sensor_layer=True, supported=True)


# gold and prediction, each as {column_key: record}
def load_gold(cfg: dict) -> tuple[dict, dict]:
    data = json.loads((EVAL / cfg["gold"]).read_text(encoding="utf-8"))
    gold: dict[str, dict] = {}
    for o in data["observations"]:
        key = f'{o["file"]}::{o["column"]}' if cfg["key"] == "file_column" else o["variable"]
        note = (o.get("note") or "").lower()
        gold[key] = dict(
            property=norm(o["observedProperty"]),
            source=o["decision"],                            # "reuse" | "mint"
            unit=norm_unit(o.get("unit")),
            feature=norm(o.get("feature_gwsw")) or "untyped",
            sensor=o.get("sensor"),                          # gold device id (grouping)
            # alternatives for the lenient score ("mint:*" = any mint)
            accept={a if a == "mint:*" else norm(a) for a in o.get("accept") or []},
            soft=("approximate" in note or "expert" in note),
            file=o.get("file"), column=o.get("column"),
            label=o.get("type") or o.get("variable"),
        )
    return gold, data


def extract_pred(profile: SourceProfile) -> dict:
    src_by_uri = {p.uri: p.source for p in profile.properties}
    gwsw_by_fid = {f.id: f.gwsw_class_uri for f in profile.features}
    pred: dict[str, dict] = {}
    for fp in profile.files:
        for col in fp.columns:
            if col.ssn_role not in (SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN):
                continue
            if not col.observed_property_uri:
                continue
            key = column_key(profile.root_path, fp.path, col.name)
            src = src_by_uri.get(col.observed_property_uri, "")
            pred[key] = dict(
                property=norm(col.observed_property_uri),
                source="mint" if src == "minted" else "reuse",
                unit=norm_unit(col.qudt_unit_uri),
                feature=norm(gwsw_by_fid.get(col.feature_id)) or "untyped",
                sensor=col.sensor_id,
                file=Path(fp.path).name, column=col.name,
            )
    return pred


def get_profile(cfg: dict, refresh: bool, run_idx: int = 0) -> tuple[SourceProfile, bool]:
    """Resolved profile, cached. Returns (profile, from_cache)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    suffix = "" if run_idx == 0 else f"_{run_idx}"
    cache_path = CACHE / f'{cfg["dataset_id"]}_profile{cfg.get("tag", "")}{suffix}.json'
    if cache_path.exists() and not refresh:
        return SourceProfile.model_validate_json(cache_path.read_text(encoding="utf-8")), True
    from stage1.stage1 import run1
    profile = run1(cfg["path"], dataset_id=cfg["dataset_id"], config=cfg.get("pipe"))
    cache_path.write_text(profile.model_dump_json(indent=1), encoding="utf-8")
    return profile, False


def get_menus(profile: SourceProfile, cfg: dict) -> dict:
    """Candidate menus for L0. Calls the model only when the config enables enrichment."""
    try:
        from stage1.resolver import build_request
        req = build_request(profile, config=cfg.get("pipe"))
        return req[3] if req else {}
    except Exception as e:
        print(f"[eval] L0 menus unavailable ({type(e).__name__}: {e})")
        return {}


# sensor grouping: pairwise P/R/F1 and adjusted Rand index
def _adjusted_rand(a: list, b: list) -> float:
    n = len(a)
    if n < 2:
        return 1.0
    cont = Counter(zip(a, b))
    idx = sum(comb(v, 2) for v in cont.values())
    sa = sum(comb(v, 2) for v in Counter(a).values())
    sb = sum(comb(v, 2) for v in Counter(b).values())
    exp = sa * sb / comb(n, 2)
    mx = 0.5 * (sa + sb)
    return 1.0 if mx == exp else (idx - exp) / (mx - exp)


def _partition_prf(a: list, b: list) -> dict:
    """a = gold labels, b = pred labels (aligned). Pair-counting P/R/F1."""
    tp = sum(comb(v, 2) for v in Counter(zip(a, b)).values())
    same_gold = sum(comb(v, 2) for v in Counter(a).values())     # TP + FN
    same_pred = sum(comb(v, 2) for v in Counter(b).values())     # TP + FP
    prec = tp / same_pred if same_pred else 1.0
    rec = tp / same_gold if same_gold else 1.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, ari=_adjusted_rand(a, b), n=len(a))


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score(gold: dict, pred: dict, menus: dict, cfg: dict) -> dict:
    gset, pset = set(gold), set(pred)
    matched = sorted(gset & pset)
    missed = sorted(gset - pset)
    spurious = sorted(pset - gset)

    rows, harm = [], defaultdict(list)
    l1 = l1_len = rm = l2 = l4 = 0
    l1_strict_total = 0
    l0_hits = l0_total = 0
    for k in matched:
        g, p = gold[k], pred[k]
        # a gold mint row accepts any minted property
        prop_ok = (p["source"] == "mint") if g["source"] == "mint" else g["property"] == p["property"]
        acc = g.get("accept") or set()
        prop_len = prop_ok or p["property"] in acc or ("mint:*" in acc and p["source"] == "mint")
        l1_len += prop_len
        rm_ok = g["source"] == p["source"]
        unit_ok = g["unit"] == p["unit"]
        feat_ok = g["feature"] == p["feature"]
        l1 += prop_ok
        rm += rm_ok
        l2 += unit_ok
        l4 += feat_ok
        if not g["soft"]:
            l1_strict_total += 1
        # L0 over gold reuse rows only
        in_menu = None
        if g["source"] == "reuse":
            cand = {norm(c["uri"]) for c in menus.get(k, [])}
            in_menu = g["property"] in cand
            l0_total += 1
            l0_hits += in_menu
        harm[g["property"]].append(k)
        rows.append(dict(key=k, gold={**g, "accept": sorted(acc)}, pred=p, prop_ok=prop_ok,
                         prop_ok_lenient=prop_len, rm_ok=rm_ok,
                         unit_ok=unit_ok, feat_ok=feat_ok, in_menu=in_menu))

    l3 = None
    if cfg["sensor_layer"] and matched:
        a = [str(gold[k]["sensor"]) for k in matched]
        b = [str(pred[k]["sensor"]) for k in matched]
        l3 = _partition_prf(a, b)

    # harmonization: do gold same-concept groups get one predicted URI?
    harm_groups, converged = [], 0
    for concept, ks in sorted(harm.items()):
        if len(ks) < 2:
            continue
        pred_uris = sorted({pred[k]["property"] for k in ks})
        ok = len(pred_uris) == 1
        converged += ok
        harm_groups.append(dict(concept=concept, n=len(ks), pred_uris=pred_uris, converged=ok))

    return dict(
        dataset=cfg["dataset_id"],
        n_gold=len(gold), n_pred=len(pred), n_matched=len(matched),
        coverage=_rate(len(matched), len(gold)),
        spurious_rate=_rate(len(spurious), len(pred)),
        layers=dict(
            L0_retrieval_recall=dict(hits=l0_hits, total=l0_total, acc=_rate(l0_hits, l0_total)),
            L1_property=dict(hits=l1, total=len(matched), acc=_rate(l1, len(matched)),
                             strict_total=l1_strict_total,
                             lenient_hits=l1_len, lenient_acc=_rate(l1_len, len(matched))),
            reuse_mint=dict(hits=rm, total=len(matched), acc=_rate(rm, len(matched))),
            L2_unit=dict(hits=l2, total=len(matched), acc=_rate(l2, len(matched))),
            L3_sensor=l3,
            L4_feature=dict(hits=l4, total=len(matched), acc=_rate(l4, len(matched))),
        ),
        harmonization=dict(groups=harm_groups,
                           rate=_rate(converged, len(harm_groups))),
        rows=rows, missed=missed,
        spurious=[dict(key=k, property=pred[k]["property"], file=pred[k]["file"]) for k in spurious],
    )


def summary_line(r: dict) -> str:
    L = r["layers"]
    s = (f'coverage {r["n_matched"]}/{r["n_gold"]} ({r["coverage"]:.0%})  '
         f'L1 prop {L["L1_property"]["acc"]:.0%} (lenient {L["L1_property"].get("lenient_acc", 0):.0%})  '
         f'reuse/mint {L["reuse_mint"]["acc"]:.0%}  '
         f'L2 unit {L["L2_unit"]["acc"]:.0%}  L4 feat {L["L4_feature"]["acc"]:.0%}')
    if L["L3_sensor"]:
        s += f'  L3 sensor F1 {L["L3_sensor"]["f1"]:.0%}'
    if L["L0_retrieval_recall"]["total"]:
        s += f'  L0 recall {L["L0_retrieval_recall"]["acc"]:.0%}'
    s += f'  spurious {len(r["spurious"])}'
    return s


_CSS = ("body{font:13px/1.5 system-ui,Arial;margin:24px;max-width:1300px;color:#1a1a1a}"
        "h1{font-size:20px}h2{border-bottom:2px solid #ccc;margin-top:26px;font-size:15px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px 7px;"
        "vertical-align:top;text-align:left}th{background:#f3f5f7}.mono{font-family:Consolas,monospace;"
        "font-size:11px}.src{color:#555;font-size:11px}.ok{color:#178a3a}.bad{color:#c0392b;font-weight:600}"
        ".card{display:inline-block;border:1px solid #ccc;border-radius:6px;padding:8px 12px;margin:4px 6px 4px 0;"
        "min-width:96px}.card b{font-size:20px;display:block}.prov{background:#fff7e6}"
        "tr.miss{background:#fff5f5}tr.soft{background:#fffdf2}")


def _card(label: str, value: str, sub: str = "", cls: str = "") -> str:
    return (f'<div class="card {cls}"><span class=src>{html.escape(label)}</span>'
            f'<b>{html.escape(value)}</b><span class=src>{html.escape(sub)}</span></div>')


def render_html(r: dict, cfg: dict, agg: dict | None) -> str:
    e = html.escape
    L = r["layers"]

    def pct(d):
        return f'{d["acc"]:.0%}' if d and d.get("total") else "—"

    cards = [
        _card("coverage", f'{r["coverage"]:.0%}', f'{r["n_matched"]}/{r["n_gold"]} gold cols'),
        _card("L1 property", pct(L["L1_property"]), f'{L["L1_property"]["hits"]}/{L["L1_property"]["total"]} exact'),
        _card("reuse/mint", pct(L["reuse_mint"]), f'{L["reuse_mint"]["hits"]}/{L["reuse_mint"]["total"]}'),
        _card("L2 unit", pct(L["L2_unit"]), f'{L["L2_unit"]["hits"]}/{L["L2_unit"]["total"]}'),
        _card("L4 feature", pct(L["L4_feature"]), "provisional", "prov"),
    ]
    if L["L3_sensor"]:
        cards.append(_card("L3 sensor F1", f'{L["L3_sensor"]["f1"]:.0%}',
                           f'ARI {L["L3_sensor"]["ari"]:.2f}'))
    if L["L0_retrieval_recall"]["total"]:
        d = L["L0_retrieval_recall"]
        cards.append(_card("L0 recall", f'{d["acc"]:.0%}', f'{d["hits"]}/{d["total"]} in menu'))
    cards.append(_card("spurious", str(len(r["spurious"])), "not in gold"))

    def mark(ok):
        return '<span class=ok>✓</span>' if ok else '<span class=bad>✗</span>'

    body = []
    for row in r["rows"]:
        g, p = row["gold"], row["pred"]
        cls = "soft" if g["soft"] else ""
        prop = (f'{e(g["property"])}' if row["prop_ok"]
                else f'<span class=bad>{e(g["property"])}</span> ≠ {e(p["property"])}')
        unit = e(str(g["unit"])) if row["unit_ok"] else f'<span class=bad>{e(str(g["unit"]))}</span> ≠ {e(str(p["unit"]))}'
        feat = e(g["feature"]) if row["feat_ok"] else f'<span class=bad>{e(g["feature"])}</span> ≠ {e(p["feature"])}'
        menu = "" if row["in_menu"] is None else (" · in menu" if row["in_menu"] else " · <span class=bad>not in menu</span>")
        body.append(
            f'<tr class="{cls}"><td class=mono>{e(row["key"])}{"  ⚠soft" if g["soft"] else ""}</td>'
            f'<td>{mark(row["prop_ok"])} {prop}<br><span class=src>{e(g["source"])}{mark(row["rm_ok"]) if not row["rm_ok"] else ""}{menu}</span></td>'
            f'<td>{mark(row["unit_ok"])} {unit}</td>'
            f'<td>{mark(row["feat_ok"])} {feat}</td>'
            f'<td class=src>{e(str(g["sensor"]))}<br>→ {e(str(p["sensor"]))}</td></tr>')

    miss_rows = "".join(f'<tr class=miss><td class=mono>{e(k)}</td></tr>' for k in r["missed"])
    spur_rows = "".join(f'<tr><td class=mono>{e(s["key"])}</td><td class=mono>{e(s["property"])}</td>'
                        f'<td class=mono>{e(s["file"])}</td></tr>' for s in r["spurious"])
    harm_rows = "".join(
        f'<tr><td class=mono>{e(g["concept"])}</td><td>{g["n"]}</td>'
        f'<td>{"<span class=ok>converged</span>" if g["converged"] else "<span class=bad>split</span>"}</td>'
        f'<td class=mono>{e(" | ".join(g["pred_uris"]))}</td></tr>' for g in r["harmonization"]["groups"])

    agg_html = ""
    if agg:
        agg_html = ("<h2>Variance across runs</h2><table><tr><th>layer</th><th>mean</th><th>std</th>"
                    "<th>runs</th></tr>"
                    + "".join(f'<tr><td>{e(k)}</td><td>{v["mean"]:.1%}</td><td>{v["std"]:.1%}</td>'
                              f'<td class=mono>{e(", ".join(f"{x:.0%}" for x in v["runs"]))}</td></tr>'
                              for k, v in agg.items()) + "</table>")

    l3 = L["L3_sensor"]
    l3_html = ("" if not l3 else
               f'<h2>L3 sensor — grouping agreement</h2><p class=src>Pairwise over {l3["n"]} matched '
               f'columns (the LLM mints its own ids, so grouping is scored, not the id): '
               f'precision {l3["precision"]:.0%} · recall {l3["recall"]:.0%} · '
               f'F1 {l3["f1"]:.0%} · adjusted Rand {l3["ari"]:.2f}.</p>')

    return (f"<!doctype html><meta charset=utf-8><title>{e(cfg['dataset_id'])} scorecard</title>"
            f"<style>{_CSS}</style>"
            f"<h1>{e(cfg['dataset_id'])} — decision scorecard</h1>"
            f'<p class=src>Pipeline decisions (one resolve call) vs gold, joined on '
            f'<span class=mono>file::column</span>. L4 feature is <b>provisional</b> '
            f'(tentative GWSW classes, expert-pending).</p>'
            f'<div>{"".join(cards)}</div>'
            f'<h2>Matched columns — {r["n_matched"]}</h2>'
            "<table><tr><th>column</th><th>observedProperty (gold ≠ pred) · source</th>"
            "<th>unit</th><th>feature GWSW</th><th>sensor gold→pred</th></tr>"
            + "".join(body) + "</table>"
            + (f'<h2>Missed — {len(r["missed"])} gold columns the pipeline did not produce</h2>'
               f'<table><tr><th>column (file::col)</th></tr>{miss_rows}</table>' if r["missed"] else "")
            + (f'<h2>Spurious — {len(r["spurious"])} predicted columns with no gold '
               f'(expected: evaluations/ derived-stat junk)</h2>'
               f'<table><tr><th>column</th><th>property</th><th>file</th></tr>{spur_rows}</table>'
               if r["spurious"] else "")
            + (f'<h2>Harmonization — same concept → one URI ({r["harmonization"]["rate"]:.0%} of '
               f'{len(r["harmonization"]["groups"])} multi-column concepts)</h2>'
               f'<table><tr><th>gold concept</th><th>#cols</th><th>result</th><th>pipeline URI(s)</th></tr>'
               f'{harm_rows}</table>' if r["harmonization"]["groups"] else "")
            + l3_html + agg_html)


def recall_sweep(cfg: dict, ks=(10, 15, 20, 30, 50)) -> None:
    """Recall@k over gold reuse rows, raw column name vs clean_term (no API)."""
    from stage1.evidence import gather_evidence
    from stage1.property_retrieval import retrieve_candidates
    profile, _ = get_profile(cfg, refresh=False)
    gold, _ = load_gold(cfg)
    chunks = gather_evidence(profile)
    companion = "\n".join(c.text for c in chunks if not c.source.startswith("structure:"))
    colmap = {}
    for fp in profile.files:
        for col in fp.columns:
            if col.ssn_role in (SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN):
                colmap[column_key(profile.root_path, fp.path, col.name)] = (col, fp)

    maxk = max(ks)

    def rank_of(gp: str, col_name: str, stem: str, query_term: str | None = None) -> int | None:
        cands = retrieve_candidates(col_name, stem, companion, k=maxk, query_term=query_term)
        src = "qudt" if gp.startswith("qudt:") else "gemet"
        seq = [norm(c["uri"]) for c in cands if c["source"] == src]   # gold's own corpus
        return seq.index(gp) + 1 if gp in seq else None

    rows = []   # (concept, raw_name, clean_query, raw_rank, clean_rank)
    for key, g in gold.items():
        if g["source"] != "reuse" or key not in colmap:   # mint rows are N/A for recall
            continue
        col, fp = colmap[key]
        stem, gp = Path(fp.path).stem, g["property"]
        cq = clean_term(col.name)
        # the metadata lookup still keys on the real column name
        rows.append((gp, col.name, cq,
                     rank_of(gp, col.name, stem),
                     rank_of(gp, col.name, clean_term(stem), query_term=cq)))

    n = len(rows)
    rec = lambda i, k: sum(1 for r in rows if r[i] is not None and r[i] <= k) / n
    print(f"\n[recall@k sweep] {cfg['dataset_id']} — {n} reuse gold columns  (raw vs deterministic-clean)\n")
    print(f"  {'k':<6}{'raw':<9}cleaned")
    for k in ks:
        print(f"  {k:<6}{rec(3, k):<9.0%}{rec(4, k):.0%}")
    fmt = lambda r: "MISS" if r is None else str(r)
    print(f"\n  problem columns — rank raw -> cleaned (in its vocab; MISS = not in top-{maxk}):")
    shown = [r for r in rows if (r[3] is None or r[3] > 10) or r[3] != r[4]]
    for gp, name, cq, raw, cln in sorted(shown, key=lambda r: r[3] if r[3] is not None else 10**9,
                                         reverse=True):
        tag = "  <- fixed" if (raw is None or raw > 10) and (cln is not None and cln <= 10) else ""
        print(f"    {fmt(raw):>5} -> {fmt(cln):<5}  {gp:<24} {name!r} -> {cq!r}{tag}")
    print()


def aggregate(results: list[dict]) -> dict:
    keys = [("L1_property", "L1 property"), ("reuse_mint", "reuse/mint"),
            ("L2_unit", "L2 unit"), ("L4_feature", "L4 feature")]
    out = {}
    for lk, label in keys:
        vals = [r["layers"][lk]["acc"] for r in results]
        out[label] = dict(mean=statistics.mean(vals),
                          std=statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                          runs=vals)
    extra = {"L1 lenient": [r["layers"]["L1_property"].get("lenient_acc") for r in results],
             "L3 sensor F1": [(r["layers"]["L3_sensor"] or {}).get("f1") for r in results]}
    for label, vals in extra.items():
        vals = [v for v in vals if v is not None]
        if vals:
            out[label] = dict(mean=statistics.mean(vals),
                              std=statistics.pstdev(vals) if len(vals) > 1 else 0.0, runs=vals)
    return out


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=sorted(DATASETS))
    ap.add_argument("--repeats", type=int, default=1, help="run resolve N times for variance")
    ap.add_argument("--refresh", action="store_true", help="ignore cache; re-run the resolve call")
    ap.add_argument("--resume", action="store_true",
                    help="with --repeats: reuse repeat profiles already cached, run only the missing ones")
    ap.add_argument("--sweep", action="store_true", help="recall@k retrieval sweep (no API) instead of scoring")
    ap.add_argument("--config", help="pipeline config .toml for this run; tags the cache and "
                                     "outputs with its name, so baseline files are never overwritten")
    args = ap.parse_args()
    cfg = dict(DATASETS[args.dataset])
    if args.config:
        from config import load_config
        cfg["pipe"] = load_config(args.config)
        cfg["tag"] = "__" + Path(args.config).stem
        print(f"[eval] pipeline config {args.config}: {cfg['pipe']}")
    tag = cfg.get("tag", "")
    if args.repeats > 1:                                # repeats send the same prompt
        from config import load_config, override
        cfg["pipe"] = override(cfg.get("pipe") or load_config(), cache_prompt=True)

    gold, gold_data = load_gold(cfg)
    if not cfg["supported"]:
        nfeat = len(gold_data.get("sites") or gold_data.get("features") or [])
        print(f"[eval] {args.dataset}: pipeline UNSUPPORTED (inert adapter). "
              f"Gold validated: {len(gold)} observations, {nfeat} features. "
              f"Nothing to score until the ingestion path lands.")
        return

    if args.sweep:
        recall_sweep(cfg)
        return

    # API usage, one record per model call
    import anthropic, time
    usage_path = EVAL / "results" / f"usage_{args.dataset}{tag}.json"
    usage_log = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else []
    _orig = anthropic.resources.messages.Messages.create

    def _logged(self, *a, **kw):
        t0 = time.perf_counter()
        r = _orig(self, *a, **kw)
        usage_log.append(dict(model=kw.get("model"), tool=(kw.get("tool_choice") or {}).get("name"),
                              seconds=round(time.perf_counter() - t0, 1),
                              input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                              cache_write=getattr(r.usage, "cache_creation_input_tokens", 0) or 0,
                              cache_read=getattr(r.usage, "cache_read_input_tokens", 0) or 0,
                              stop_reason=r.stop_reason))
        usage_path.parent.mkdir(parents=True, exist_ok=True)
        usage_path.write_text(json.dumps(usage_log, indent=1), encoding="utf-8")
        return r
    anthropic.resources.messages.Messages.create = _logged

    results = []
    menus: dict = {}
    for i in range(args.repeats):
        run_idx = 0 if args.repeats == 1 else i + 1
        profile, cached = get_profile(cfg, refresh=args.refresh or (args.repeats > 1 and not args.resume),
                                      run_idx=run_idx)
        if i == 0:
            menus = get_menus(profile, cfg)            # L0 on the first run only
        pred = extract_pred(profile)
        r = score(gold, pred, menus, cfg)
        results.append(r)
        print(f"[{args.dataset} run {i+1}/{args.repeats}{' cached' if cached else ''}] "
              + summary_line(r))

    agg = aggregate(results) if args.repeats > 1 else None
    primary = results[0]

    out = dict(primary=primary, variance=agg)
    from dataclasses import asdict
    from config import load_config
    out["pipeline"] = asdict(cfg.get("pipe") or load_config())

    # demoted columns; a demoted gold column is a wrong demotion
    demoted = [dict(key=column_key(profile.root_path, fp.path, c.name), column_class=c.column_class,
                    in_gold=column_key(profile.root_path, fp.path, c.name) in gold)
               for fp in profile.files for c in fp.columns
               if c.column_class and c.column_class != "measurement"]
    out["demoted"] = demoted
    if demoted:
        wrong = [d["key"] for d in demoted if d["in_gold"]]
        print(f"[eval] demoted {len(demoted)} column(s) "
              f"{dict(Counter(d['column_class'] for d in demoted))}; wrongly demoted gold columns: "
              f"{len(wrong)}" + (f" {wrong}" if wrong else ""))
    (EVAL / f"scores_{args.dataset}{tag}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (EVAL / f"{args.dataset}{tag}_scorecard.html").write_text(
        render_html(primary, cfg, agg), encoding="utf-8")
    print(f"[eval] wrote eval/scores_{args.dataset}{tag}.json + eval/{args.dataset}{tag}_scorecard.html")


if __name__ == "__main__":
    main()
