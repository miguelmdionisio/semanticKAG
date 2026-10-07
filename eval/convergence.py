"""Cross-dataset convergence (no API): for each gold concept found in several datasets, the
identifiers each independently resolved dataset chose.

Run: python eval/convergence.py  ->  eval/results/convergence.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))
from score import DATASETS, load_gold, extract_pred                 # noqa: E402
from stage1.models import SourceProfile                             # noqa: E402

CACHE = ROOT / "eval" / "cache"


def runs() -> list[tuple[str, str, str]]:
    out = [("graz", "graz", "graz_profile__main_sampling_1.json"),
           ("flowbru", "flowbru_beco", "flowbru_beco_profile__main_sampling_1.json"),
           ("flowbru", "flowbru_belliard", "flowbru_belliard_profile__main_sampling_1.json")]
    cover = json.loads((ROOT / "eval" / "uwo_sites.json").read_text(encoding="utf-8"))["cover"]
    out += [("uwo", ds, f"{ds}_profile__shared_menu_1.json") for ds in cover]
    return out


def main() -> None:
    by_concept: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for family, ds, fname in runs():
        gold, _ = load_gold(DATASETS[ds])
        pred = extract_pred(SourceProfile.model_validate_json((CACHE / fname).read_text(encoding="utf-8")))
        for key, g in gold.items():
            by_concept[g["property"]].append((family, pred[key]["property"] if key in pred else None))
    result = {}
    for concept, rows in sorted(by_concept.items()):
        families = sorted({f for f, _ in rows})
        if len(families) < 2:
            continue
        chosen = Counter(p for _, p in rows)
        top, n_top = chosen.most_common(1)[0]
        result[concept] = dict(datasets=families, columns=len(rows), most_common=top, on_most_common=n_top,
                               chosen=dict(chosen),
                               per_dataset={f: dict(Counter(p for ff, p in rows if ff == f)) for f in families})
        print(f"{concept:<26} {families}  {n_top}/{len(rows)} on {top}   {result[concept]['per_dataset']}")
    path = ROOT / "eval" / "results" / "convergence.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
