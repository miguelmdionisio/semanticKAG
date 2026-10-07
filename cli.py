"""SemanticKAG command line.

profile never calls the model. prompt and retrieve call it only for query enrichment
(--no-enrich keeps them offline). resolve, build and build-multi make the resolve call.
Flags override config.toml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, override


def _cfg(args):
    return override(
        load_config(getattr(args, "config", None)),
        model=getattr(args, "model", None),
        max_rows=getattr(args, "max_rows", None),
        retrieval_k=getattr(args, "k", None),
        enrich_enabled=getattr(args, "enrich", None),
    )


def _load_profile(dataset, cfg, resolve):
    from stage1.stage1 import run1
    return run1(dataset, config=cfg, resolve=resolve)


def _dump_menu(menus):
    for key, cands in menus.items():
        print(f"  {key}")
        for c in cands:
            print(f"    - {c['uri']}  ({c['source']}: {c['label']}, d={c['distance']})")


def cmd_profile(args):
    prof = _load_profile(args.dataset, _cfg(args), resolve=False)
    print(f"dataset: {prof.id}   root: {prof.root_path}")
    print(f"files: {len(prof.files)}   companions: {len(prof.companion_files)}")
    for fp in prof.files:
        print(f"\n  {Path(fp.path).name}")
        for col in fp.columns:
            print(f"    - {col.name:<28} role={getattr(col.ssn_role, 'value', col.ssn_role)}"
                  f"  unit={col.unit or '?'}")


def cmd_prompt(args):
    from stage1.resolver import build_request
    cfg = _cfg(args)
    req = build_request(_load_profile(args.dataset, cfg, resolve=False), config=cfg)
    if not req:
        print("nothing to resolve")
        return
    system, user, tool, menus = req
    print("===== SYSTEM =====\n" + system)
    print("\n===== USER =====\n" + user)
    print(f"\n===== TOOL ===== {tool['name']}")
    print(f"\n===== MENUS ({len(menus)} columns; enrich="
          f"{'on' if cfg.enrich_enabled else 'off'}, k={cfg.retrieval_k}) =====")
    _dump_menu(menus)


def cmd_retrieve(args):
    from stage1.resolver import build_request
    cfg = _cfg(args)
    req = build_request(_load_profile(args.dataset, cfg, resolve=False), config=cfg)
    if not req:
        print("nothing to retrieve")
        return
    menus = {k: v for k, v in req[3].items() if not args.col or args.col.lower() in k.lower()}
    print(f"enrich={'on' if cfg.enrich_enabled else 'off'}, k={cfg.retrieval_k}, "
          f"{len(menus)} column(s)")
    _dump_menu(menus)


def cmd_resolve(args):
    prof = _load_profile(args.dataset, _cfg(args), resolve=True)
    print(f"\nsensors: {len(prof.sensors)}   properties: {len(prof.properties)}   "
          f"features: {len(prof.features)}")
    for fp in prof.files:
        for col in fp.columns:
            if col.observed_property_uri:
                print(f"  {col.name:<28} prop={col.observed_property_uri}  "
                      f"unit={col.qudt_unit_uri or col.unit or '?'}  "
                      f"sensor={col.sensor_id}  feature={col.feature_id}")


def cmd_build(args):
    from run import build_kg
    build_kg(args.dataset, workdir=args.workdir, out_path=args.out,
             dataset_id=args.id, config=_cfg(args))


def cmd_build_multi(args):
    from run import build_kg_multi
    build_kg_multi([(p, i) for p, i in args.pair], workdir=args.workdir,
                   out_path=args.out, config=_cfg(args))


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to a config.toml (default: ./config.toml)")
    common.add_argument("--model", help="override the resolve model")
    common.add_argument("--max-rows", type=int, dest="max_rows", help="override rows per file")
    common.add_argument("--k", type=int, help="override retrieval k (candidates per corpus)")
    common.add_argument("--enrich", dest="enrich", action="store_const", const=True, default=None,
                        help="turn LLM query enrichment ON for this run")
    common.add_argument("--no-enrich", dest="enrich", action="store_const", const=False,
                        help="turn LLM query enrichment OFF for this run")

    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in (("profile", cmd_profile), ("prompt", cmd_prompt),
                     ("resolve", cmd_resolve)):
        p = sub.add_parser(name, parents=[common], help=fn.__doc__)
        p.add_argument("--dataset", required=True)
        p.set_defaults(fn=fn)

    p = sub.add_parser("retrieve", parents=[common])
    p.add_argument("--dataset", required=True)
    p.add_argument("--col", help="only columns whose column_key contains this substring")
    p.set_defaults(fn=cmd_retrieve)

    p = sub.add_parser("build", parents=[common])
    p.add_argument("--dataset", required=True)
    p.add_argument("--id", help="dataset_id (default: folder name)")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", help="output .ttl path")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("build-multi", parents=[common])
    p.add_argument("--pair", nargs=2, action="append", metavar=("PATH", "ID"), required=True,
                   help="a dataset path and id; repeat for each dataset")
    p.add_argument("--workdir", required=True)
    p.add_argument("--out", help="combined output .ttl path")
    p.set_defaults(fn=cmd_build_multi)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
