"""IR -> RDF: write the RML mapping, run morph-kgc, add the declarations."""
from __future__ import annotations
from pathlib import Path

import morph_kgc

from stage1.models import ResolvedDataset
from stage3.rml_builder import build_mapping, build_declarations


def materialize(resolved: ResolvedDataset, workdir, out_path=None):
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    mapping_path = workdir / "mapping.ttl"
    mapping_path.write_text(build_mapping(resolved), encoding="utf-8")

    config = (
        "[CONFIGURATION]\n"
        "output_format=N-TRIPLES\n\n"
        "[DataSource1]\n"
        f"mappings={mapping_path.resolve().as_posix()}\n"
    )
    graph = morph_kgc.materialize(config)
    graph += build_declarations(resolved)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        graph.serialize(destination=str(out_path), format="turtle")
    return graph
