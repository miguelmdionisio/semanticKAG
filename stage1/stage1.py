from __future__ import annotations
from pathlib import Path

from config import PipelineConfig, load_config, override
from .discoverer import discover
from .profiler import profile_csv
from .models import SourceProfile


def run1(dataset_path: str, dataset_id: str | None = None,
         resolve: bool = True, model: str | None = None,
         config: PipelineConfig | None = None) -> SourceProfile:
    """Profile a dataset and, if `resolve`, run the resolve call on it."""
    config = override(config or load_config(), model=model)
    root = Path(dataset_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    dataset_id = dataset_id or root.name
    csv_files, readable_companions, skipped_companions = discover(str(root))
    if not csv_files:
        raise ValueError(f"No CSV files found in {root}")

    print(f"[stage1] Found {len(csv_files)} CSV files, "
          f"{len(readable_companions)} companion files, "
          f"{len(skipped_companions)} skipped files")

    file_profiles = [profile_csv(p) for p in sorted(csv_files)]
    profile = SourceProfile(
        id=dataset_id,
        root_path=str(root),
        companion_files=[str(p) for p in readable_companions + skipped_companions],
        files=file_profiles,
    )

    if resolve:
        from .resolver import resolve_dataset
        print("[stage1] Resolving (sensors + observedProperty + units) ...")
        resolve_dataset(profile, config=config)
        print(f"[stage1] {len(profile.sensors)} sensors, {len(profile.properties)} properties")

    return profile
