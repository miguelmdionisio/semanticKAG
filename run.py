"""End-to-end pipeline: profile -> resolve -> assemble IR -> materialize."""
from __future__ import annotations
from pathlib import Path

from config import PipelineConfig, load_config, override
from stage1.stage1 import run1
from stage1.assemble import assemble_ir
from stage3.materialize import materialize


def build_kg(dataset_path: str, workdir: str, out_path: str | None = None,
             dataset_id: str | None = None, model: str | None = None,
             max_rows: int | None = None, config: PipelineConfig | None = None):
    config = override(config or load_config(), model=model, max_rows=max_rows)
    profile = run1(dataset_path, dataset_id=dataset_id, config=config)
    resolved = assemble_ir(profile, workdir, max_rows=config.max_rows)
    graph = materialize(resolved, workdir, out_path=out_path)
    print(f"[run] {len(graph)} triples"
          + (f" -> {out_path}" if out_path else ""))
    return profile, resolved, graph


def build_kg_multi(datasets: list[tuple[str, str]], workdir: str,
                   out_path: str | None = None, model: str | None = None,
                   max_rows: int | None = None, config: PipelineConfig | None = None):
    """Build one graph from several (dataset_path, dataset_id) pairs.

    Sensor, feature and observation URIs are dataset-scoped; vocabulary URIs are shared,
    so the same concept converges on one node across datasets.
    """
    from rdflib import Graph
    config = override(config or load_config(), model=model, max_rows=max_rows)
    combined = Graph()
    per: dict[str, int] = {}
    for path, dsid in datasets:
        _, _, g = build_kg(path, workdir=str(Path(workdir) / dsid),
                           dataset_id=dsid, config=config)
        combined += g
        per[dsid] = len(g)
    if out_path:
        combined.serialize(destination=out_path, format="turtle")
    print(f"[run] combined KG: {len(combined)} triples from {len(datasets)} datasets"
          + (f" -> {out_path}" if out_path else ""))
    return combined, per
