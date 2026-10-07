"""Full-size materialization of Graz from a cached run (no API): every row, timed, with the
observation and result counts checked against the normalized files. Writes N-Triples to disk.

Run: python eval/full_scale.py [--processes N] [--keep]  ->  eval/results/full_scale_graz.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                                 # noqa: E402
import psutil                                                       # noqa: E402
from stage1.models import SourceProfile                             # noqa: E402
from stage1.assemble import assemble_ir                             # noqa: E402
from stage3.rml_builder import build_mapping, build_declarations    # noqa: E402

PROFILE = "graz_profile__main_sampling_1.json"
SOSA_OBS = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/ns/sosa/Observation>"
NUMERIC = "<http://qudt.org/schema/qudt/numericValue>"
SIMPLE = "<http://www.w3.org/ns/sosa/hasSimpleResult>"


def _run_tracked(cmd: list[str], cwd: Path) -> tuple[float, float]:
    """Run cmd, return (seconds, peak resident memory in GB over the whole process tree)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd)
    ps = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            rss = ps.memory_info().rss + sum(c.memory_info().rss for c in ps.children(recursive=True))
            peak = max(peak, rss)
        except psutil.Error:
            pass
        time.sleep(0.5)
    if proc.returncode:
        raise SystemExit(f"morph-kgc failed with exit code {proc.returncode}")
    return time.perf_counter() - t0, peak / 1e9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=1, help="morph-kgc worker processes (default 1)")
    ap.add_argument("--keep", action="store_true", help="keep the N-Triples file (it is large)")
    args = ap.parse_args()

    work = ROOT / "out" / "full_scale" / "graz"
    prof = SourceProfile.model_validate_json((ROOT / "eval" / "cache" / PROFILE).read_text(encoding="utf-8"))

    # 1. normalization, no row cap
    t0 = time.perf_counter()
    resolved = assemble_ir(prof, work, max_rows=None)
    t_norm = time.perf_counter() - t0
    print(f"[full] normalized {len(resolved.files)} files in {t_norm:.1f} s")

    # expected counts from the normalized files themselves
    files, exp_obs, exp_val = [], 0, 0
    for rf in resolved.files:
        cols = [o.value_col for o in rf.observations]
        df = pd.read_csv(rf.source_csv, usecols=cols, low_memory=False)
        n_val = int(df.notna().sum().sum())
        files.append(dict(file=rf.file_id, rows=len(df), columns=len(cols), values=n_val))
        exp_obs += len(df) * len(cols)
        exp_val += n_val
    print(f"[full] expecting {exp_obs:,} observations, {exp_val:,} with a value")

    # 2. materialization to N-Triples
    mapping = work / "mapping.ttl"
    mapping.write_text(build_mapping(resolved), encoding="utf-8")
    nt = work / "graz_full.nt"
    ini = work / "morph.ini"
    ini.write_text(
        "[CONFIGURATION]\n"
        "output_format=N-TRIPLES\n"
        f"output_file={nt.as_posix()}\n"
        f"number_of_processes={args.processes}\n\n"
        "[DataSource1]\n"
        f"mappings={mapping.as_posix()}\n", encoding="utf-8")
    t_mat, peak_gb = _run_tracked([sys.executable, "-m", "morph_kgc", str(ini)], work)
    print(f"[full] morph-kgc: {t_mat:.1f} s, peak memory {peak_gb:.1f} GB")

    # 3. declarations (built from the IR, as in the pipeline)
    t0 = time.perf_counter()
    decl = build_declarations(resolved)
    with nt.open("a", encoding="utf-8") as fh:
        fh.write(decl.serialize(format="nt"))
    t_decl = time.perf_counter() - t0

    # 4. count by streaming the file
    triples = obs = numeric = simple = 0
    with nt.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            triples += 1
            if SOSA_OBS in line:
                obs += 1
            elif NUMERIC in line:
                numeric += 1
            elif SIMPLE in line:
                simple += 1
    size_gb = nt.stat().st_size / 1e9
    print(f"[full] {triples:,} triples, {obs:,} observations, {numeric + simple:,} result values, {size_gb:.1f} GB")

    out = dict(
        profile=PROFILE, processes=args.processes,
        rows_total=sum(f["rows"] for f in files), files=files,
        seconds=dict(normalization=round(t_norm, 1), materialization=round(t_mat, 1),
                     declarations=round(t_decl, 1)),
        peak_memory_gb_morph=round(peak_gb, 2),
        triples=triples, declaration_triples=len(decl), observations=obs,
        result_values=numeric + simple, numeric_values=numeric, simple_results=simple,
        nt_size_gb=round(size_gb, 2),
        checks=dict(observations_expected=exp_obs, observations_match=obs == exp_obs,
                    values_expected=exp_val, values_match=numeric + simple == exp_val),
    )
    path = ROOT / "eval" / "results" / "full_scale_graz.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    print(json.dumps(out["checks"]))
    if not args.keep:
        nt.unlink()


if __name__ == "__main__":
    main()
