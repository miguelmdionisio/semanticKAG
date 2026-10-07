from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum
from pathlib import Path


class EntityKind(str, Enum):
    SENSOR = "sensor"
    SAMPLER = "sampler"

class IdentificationMethod(str, Enum):
    DECLARED="declared"
    INFERRED="inferred"

class ResolvedEntity(BaseModel):
    kind: EntityKind = EntityKind.SENSOR
    id: str
    label: str
    attributes: dict[str, str] = Field(default_factory=dict)   # device attributes only
    identification: IdentificationMethod
    evidence_source: Optional[str] = None


# Resolved IR: every field is a decided URI, column or datatype, so the RML builder is a pure template.

class SensorDecl(BaseModel):
    uri: str
    label: str
    identification: str = "inferred"

class PropertyDecl(BaseModel):
    uri: str
    label: str
    source: str = "minted"                  # "qudt" | "gemet" | "minted"

class FeatureResolution(BaseModel):
    id: str
    label: str
    kind: Optional[str] = None
    gwsw_class_uri: Optional[str] = None    # None = untyped

class FeatureDecl(BaseModel):
    uri: str
    label: str
    gwsw_class_uri: Optional[str] = None

class SamplerDecl(BaseModel):
    uri: str
    label: str
    identification: str = "inferred"

class SamplingMap(BaseModel):
    """A sample-based file: each row is one sosa:Sample produced by one sosa:Sampling."""
    sampler_uri: Optional[str] = None
    feature_uri: Optional[str] = None

class ObservationMap(BaseModel):
    value_col: str
    value_dtype: str = "xsd:double"
    sensor_uri: str
    property_uri: str
    unit_uri: Optional[str] = None          # None -> sosa:hasSimpleResult
    foi_uri: Optional[str] = None

class ResolvedFile(BaseModel):
    source_csv: str
    file_id: str
    timestamp_col: Optional[str] = None
    timestamp_is_iso: bool = True
    observations: list[ObservationMap] = Field(default_factory=list)
    sampling: Optional[SamplingMap] = None

class ResolvedDataset(BaseModel):
    id: str
    sensors: list[SensorDecl] = Field(default_factory=list)
    properties: list[PropertyDecl] = Field(default_factory=list)
    features: list[FeatureDecl] = Field(default_factory=list)
    samplers: list[SamplerDecl] = Field(default_factory=list)
    files: list[ResolvedFile] = Field(default_factory=list)


class SSNRole(str, Enum):
    TIMESTAMP         = "timestamp"
    OBSERVABLE_PROP   = "observable_property"
    QUALITY_FLAG      = "quality_flag"
    ROW_INDEX         = "row_index"
    AUXILIARY         = "auxiliary"           # secondary timestamps and durations
    UNKNOWN           = "unknown"


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_pct: float
    min: Optional[float] = None
    max: Optional[float] = None
    sample_values: list[str] = Field(default_factory=list)
    ssn_role: SSNRole = SSNRole.UNKNOWN
    qudt_unit_uri: Optional[str] = None
    observation_type: Optional[str] = None
    unit: Optional[str] = None
    sensor_id: Optional[str] = None
    observed_property_uri: Optional[str] = None
    feature_id: Optional[str] = None
    column_class: Optional[str] = None      # measurement | derived | diagnostic | quality


class FileSampling(BaseModel):
    sampler_id: Optional[str] = None
    feature_id: Optional[str] = None


class FileProfile(BaseModel):
    path: str
    row_count: int
    row_count_sampled: bool = False
    delimiter: str
    decimal: str = "."
    encoding: str
    datetime_column: Optional[str] = None
    datetime_format: Optional[str] = None
    timezone: Optional[str] = None
    frequency: Optional[str] = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    sampling: Optional[FileSampling] = None


class SourceProfile(BaseModel):
    id: str
    root_path: str
    companion_files: list[str] = Field(default_factory=list)
    files: list[FileProfile] = Field(default_factory=list)
    sensors: list[ResolvedEntity] = Field(default_factory=list)
    properties: list[PropertyDecl] = Field(default_factory=list)
    features: list[FeatureResolution] = Field(default_factory=list)
    samplers: list[ResolvedEntity] = Field(default_factory=list)

def column_key(root_path: str, file_path: str, column_name: str) -> str:
    """Join key between resolver, IR and gold: '<root-relative posix path>::<column>'."""
    rel = Path(file_path).resolve().relative_to(Path(root_path).resolve()).as_posix()
    return f"{rel}::{column_name}"
