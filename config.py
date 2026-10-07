"""Pipeline configuration.

Precedence: PipelineConfig defaults < config.toml < CLI flags / explicit kwargs.
The defaults reproduce the original setup (no enrichment, k=10); config.toml holds the main one.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent / "config.toml"


@dataclass(frozen=True)
class PipelineConfig:
    model: str = "claude-sonnet-4-6"
    temperature: float | None = None      # None = not sent (API default)
    cache_prompt: bool = False            # prompt caching, only pays off for repeated calls
    use_documentation: bool = True        # False hides companion files (ablation)
    max_rows: int = 1000
    retrieval_k: int = 10                 # candidates per corpus
    menu_format: str = "lines"            # "lines" or "shared" (compact menus)
    enrich_enabled: bool = False
    enrich_include_metadata: bool = True
    enrich_model: str = "claude-sonnet-4-6"
    classify_columns: bool = False        # resolve call may demote non-measurement columns
    model_sampling: bool = False          # resolve call may mark sample-based files


def _merge(cfg: PipelineConfig, data: dict) -> PipelineConfig:
    changes: dict = {}
    if "model" in data:
        changes["model"] = data["model"]
    if "max_rows" in data:
        changes["max_rows"] = data["max_rows"]
    if "temperature" in data:
        changes["temperature"] = float(data["temperature"])
    if "cache_prompt" in data:
        changes["cache_prompt"] = bool(data["cache_prompt"])
    evidence = data.get("evidence", {})
    if "use_documentation" in evidence:
        changes["use_documentation"] = bool(evidence["use_documentation"])
    retrieval = data.get("retrieval", {})
    if "k" in retrieval:
        changes["retrieval_k"] = retrieval["k"]
    if "menu_format" in retrieval:
        if retrieval["menu_format"] not in ("lines", "shared"):
            raise ValueError(f"[retrieval] menu_format must be 'lines' or 'shared', got {retrieval['menu_format']!r}")
        changes["menu_format"] = retrieval["menu_format"]
    enrichment = data.get("enrichment", {})
    if "enabled" in enrichment:
        changes["enrich_enabled"] = enrichment["enabled"]
    if "include_metadata" in enrichment:
        changes["enrich_include_metadata"] = enrichment["include_metadata"]
    if "model" in enrichment:
        changes["enrich_model"] = enrichment["model"]
    resolution = data.get("resolution", {})
    if "classify_columns" in resolution:
        changes["classify_columns"] = resolution["classify_columns"]
    if "model_sampling" in resolution:
        changes["model_sampling"] = resolution["model_sampling"]
    return replace(cfg, **changes)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    p = Path(path) if path is not None else _DEFAULT_PATH
    if not p.exists():
        return cfg
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            raise RuntimeError(
                f"{p.name} exists but no TOML parser is available "
                "(need Python 3.11+ for tomllib, or `pip install tomli`)."
            )
    return _merge(cfg, tomllib.loads(p.read_text(encoding="utf-8")))


def override(cfg: PipelineConfig, **kwargs) -> PipelineConfig:
    """Replace the given fields, ignoring None values."""
    changes = {k: v for k, v in kwargs.items() if v is not None}
    return replace(cfg, **changes) if changes else cfg
