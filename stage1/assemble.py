"""Turn a resolved SourceProfile into the IR (ResolvedDataset) and write normalized CSVs."""
from __future__ import annotations
from pathlib import Path

from .models import (SourceProfile, SSNRole, ResolvedDataset, ResolvedFile,
                     ObservationMap, SensorDecl, FeatureDecl, SamplerDecl, SamplingMap)
from .normalize import normalize_csv
from utils.uri_policy import sensor_uri, feature_uri, sampler_uri, _slug

_OBSERVABLE = {SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN}
_NUMERIC = {"float64", "float32", "int64", "int32"}


def _file_id(root: str, path: str) -> str:
    rel = Path(path).resolve().relative_to(Path(root).resolve()).with_suffix("")
    return _slug(rel.as_posix())


def assemble_ir(profile: SourceProfile, workdir, max_rows: int = 1000) -> ResolvedDataset:
    norm_dir = Path(workdir) / "normalized"
    sensors = [
        SensorDecl(
            uri=sensor_uri(profile.id, e.id),
            label=e.label,
            identification=getattr(e.identification, "value", str(e.identification)),
        )
        for e in profile.sensors
    ]
    features = [
        FeatureDecl(uri=feature_uri(profile.id, f.id), label=f.label,
                    gwsw_class_uri=f.gwsw_class_uri)
        for f in profile.features
    ]
    samplers = [
        SamplerDecl(
            uri=sampler_uri(profile.id, s.id),
            label=s.label,
            identification=getattr(s.identification, "value", str(s.identification)),
        )
        for s in profile.samplers
    ]

    files: list[ResolvedFile] = []
    for fp in profile.files:
        obs_cols = [
            c for c in fp.columns
            if c.ssn_role in _OBSERVABLE and c.observed_property_uri and c.sensor_id
        ]
        if not obs_cols:
            continue
        fid = _file_id(profile.root_path, fp.path)
        out_csv = norm_dir / f"{fid}.csv"
        ts_iso = normalize_csv(fp, out_csv, max_rows=max_rows)
        # in a sample-based file the observation's feature of interest is the row's sample
        sampling = None
        if fp.sampling is not None:
            sampling = SamplingMap(
                sampler_uri=sampler_uri(profile.id, fp.sampling.sampler_id) if fp.sampling.sampler_id else None,
                feature_uri=feature_uri(profile.id, fp.sampling.feature_id) if fp.sampling.feature_id else None,
            )
        observations = [
            ObservationMap(
                value_col=c.name,
                value_dtype="xsd:double" if c.dtype in _NUMERIC else "xsd:string",
                sensor_uri=sensor_uri(profile.id, c.sensor_id),
                property_uri=c.observed_property_uri,
                unit_uri=c.qudt_unit_uri,
                foi_uri=(feature_uri(profile.id, c.feature_id)
                         if c.feature_id and sampling is None else None),
            )
            for c in obs_cols
        ]
        files.append(ResolvedFile(
            source_csv=str(out_csv.resolve()),
            file_id=fid,
            timestamp_col=fp.datetime_column,
            timestamp_is_iso=ts_iso,
            observations=observations,
            sampling=sampling,
        ))

    return ResolvedDataset(
        id=profile.id,
        sensors=sensors,
        properties=list(profile.properties),
        features=features,
        samplers=samplers,
        files=files,
    )
