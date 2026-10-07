"""URI minting, shared by the RML mapping and the declarations so references always resolve.

Sensor, feature, sampler and observation URIs are dataset-scoped; minted property URIs are global.
"""
from __future__ import annotations
import re

BASE = "http://example.org"

ROW = "__row__"     # unique row key added to every normalized CSV


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def sensor_uri(dataset_id: str, sensor_id: str) -> str:
    return f"{BASE}/sensor/{_slug(dataset_id)}/{_slug(sensor_id)}"


def minted_property_uri(label: str) -> str:
    return f"{BASE}/property/{_slug(label)}"


def feature_uri(dataset_id: str, feature_id: str) -> str:
    return f"{BASE}/feature/{_slug(dataset_id)}/{_slug(feature_id)}"


def obs_template(dataset_id: str, file_id: str, value_col: str) -> str:
    return f"{BASE}/obs/{_slug(dataset_id)}/{_slug(file_id)}/{_slug(value_col)}/{{{ROW}}}"


def sampler_uri(dataset_id: str, sampler_id: str) -> str:
    return f"{BASE}/sampler/{_slug(dataset_id)}/{_slug(sampler_id)}"


def sample_template(dataset_id: str, file_id: str) -> str:
    """Keyed per file and row: all columns of a row share one sample."""
    return f"{BASE}/sample/{_slug(dataset_id)}/{_slug(file_id)}/{{{ROW}}}"


def sampling_template(dataset_id: str, file_id: str) -> str:
    return f"{BASE}/sampling/{_slug(dataset_id)}/{_slug(file_id)}/{{{ROW}}}"


def result_template(dataset_id: str, file_id: str, value_col: str) -> str:
    return f"{BASE}/result/{_slug(dataset_id)}/{_slug(file_id)}/{_slug(value_col)}/{{{ROW}}}"
