"""UWO adapter: turns the relational SQLite database (one long fact table) into one folder per site
with a wide CSV per device and a metadata.json written from the dimension tables, plus a column
gold per site derived from build_uwo_gold.py. A pre-processing step, not part of the pipeline.

Run: python eval/uwo_adapter.py [--start 2021-01-01 --end 2021-01-03]
     ->  data/04 - UWO/wide/<site>/, eval/gold_uwo_<site>.json, eval/uwo_sites.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from build_uwo_gold import OBS, EXCLUDED, SITES                      # noqa: E402

DB = ROOT / "data" / "data_uwo_2021.sqlite"
OUT = ROOT / "data" / "04 - UWO" / "wide"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2021-01-03")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro&immutable=1", uri=True)
    q = lambda sql, p=(): list(con.execute(sql, p))
    var = {i: dict(name=n, unit=u, description=d) for i, n, u, d in q("select variable_id,name,unit,description from variable")}
    stype = {i: dict(name=n, description=d, manufacturer=m)
             for i, n, d, m in q("select source_type_id,name,description,manufacturer from source_type")}
    src = {i: dict(name=n, description=d, serial=s, type=stype.get(t, {}))
           for i, t, n, d, s in q("select source_id,source_type_id,name,description,serial from source")}
    site = {i: dict(name=n, description=d) for i, n, d in q("select site_id,name,description from site")}
    special = defaultdict(set)                    # source_type_id -> declared special numeric values
    special_doc = defaultdict(list)
    for t, desc, cat, num in q("select source_type_id,description,categorical_value,numerical_value from special_value_definition"):
        if num is not None:
            special[t].add(float(num))
        special_doc[t].append(dict(value=num if num is not None else cat, meaning=desc))
    src_type_id = {i: t for i, t in q("select source_id,source_type_id from source")}

    obs_by_name = {v[0]: dict(variable_id=k, property=v[2], decision=v[4], unit=v[5], note=v[6]) for k, v in OBS.items()}
    excluded = {n for n, _, _ in EXCLUDED}

    per_site: dict[str, list] = defaultdict(list)
    unknown_vars = set()
    for sid in sorted(src):
        rows = q("select timestamp, variable_id, site_id, value from signal "
                 "where source_id=? and timestamp >= ? and timestamp < ?", (sid, args.start, args.end))
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp", "variable_id", "site_id", "value"])
        # naive timestamps in the DB, written as ISO 8601 without offset
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
        sp = special.get(src_type_id.get(sid), set())
        if sp:
            df.loc[df["value"].isin(sp), "value"] = None
        for site_id, part in df.groupby("site_id"):
            wide = part.pivot_table(index="timestamp", columns="variable_id", values="value", aggfunc="first")
            wide.columns = [var[c]["name"] for c in wide.columns]
            sname = site[site_id]["name"]
            path = OUT / sname / f"{src[sid]['name']}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            wide.reset_index().to_csv(path, index=False)
            per_site[sname].append(dict(source_id=sid, file=path.name, variables=list(wide.columns), rows=len(wide)))
            unknown_vars |= {c for c in wide.columns if c not in obs_by_name and c not in excluded}

    manifest = {}
    for sname, files in sorted(per_site.items()):
        site_id = next(k for k, v in site.items() if v["name"] == sname)
        used_vars = sorted({v for f in files for v in f["variables"]})
        meta = dict(
            site=dict(name=sname, description=site[site_id]["description"]),
            devices=[dict(name=src[f["source_id"]]["name"], file=f["file"], serial=src[f["source_id"]]["serial"],
                          description=src[f["source_id"]]["description"],
                          type=src[f["source_id"]]["type"].get("name"),
                          type_description=src[f["source_id"]]["type"].get("description"),
                          manufacturer=src[f["source_id"]]["type"].get("manufacturer"),
                          special_values=special_doc.get(src_type_id[f["source_id"]], []))
                     for f in files],
            variables=[dict(name=v, unit=next(x["unit"] for x in var.values() if x["name"] == v),
                            description=next(x["description"] for x in var.values() if x["name"] == v))
                       for v in used_vars],
        )
        (OUT / sname / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

        gwsw = (SITES.get(sname) or (None, ""))[0]
        observations = []
        for f in files:
            for v in f["variables"]:
                g = obs_by_name.get(v)
                if not g:
                    continue                       # excluded variable: no gold row
                observations.append(dict(
                    type=v, file=f["file"], column=v, sensor=src[f["source_id"]]["name"],
                    observedProperty=g["property"], decision=g["decision"], unit=g["unit"],
                    feature=sname, feature_gwsw=gwsw, accept=[],
                    note=("GWSW class tentative (UWO site sheet not expert-reviewed). " + (g["note"] or "")).strip()))
        ds = f"uwo_{sname}"
        (ROOT / "eval" / f"gold_{ds}.json").write_text(json.dumps(dict(
            dataset=ds, root=str((OUT / sname).relative_to(ROOT)), observations=observations), indent=1,
            ensure_ascii=False), encoding="utf-8")
        manifest[ds] = dict(site=sname, path=(OUT / sname).relative_to(ROOT).as_posix(), gold=f"gold_{ds}.json",
                            files=len(files), columns=sum(len(f["variables"]) for f in files),
                            gold_columns=len(observations), gold_variables=sorted({o["type"] for o in observations}))

    # greedy site cover of the gold measurement variables, fixed before any model output
    need = set().union(*(set(m["gold_variables"]) for m in manifest.values()))
    cover, left = [], set(need)
    while left:
        best = min(manifest, key=lambda d: (-len(left & set(manifest[d]["gold_variables"])), manifest[d]["columns"], d))
        if not left & set(manifest[best]["gold_variables"]):
            break
        cover.append(best)
        left -= set(manifest[best]["gold_variables"])
    out = dict(window=[args.start, args.end], sites=manifest, cover=cover,
               measurement_variables_in_window=sorted(need), unknown_variables=sorted(unknown_vars))
    (ROOT / "eval" / "uwo_sites.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    tot = sum(m["columns"] for m in manifest.values())
    print(f"[uwo] {len(manifest)} sites, {sum(m['files'] for m in manifest.values())} files, {tot} columns "
          f"({sum(m['gold_columns'] for m in manifest.values())} gold measurement columns)")
    print(f"[uwo] measurement variables in window: {len(need)}; unknown variables: {sorted(unknown_vars)}")
    print(f"[uwo] cover ({len(cover)} sites): " + ", ".join(f"{d}({manifest[d]['columns']} cols, "
          f"{manifest[d]['gold_columns']} gold)" for d in cover))


if __name__ == "__main__":
    main()
