"""The resolve call: one model call per dataset decides sensors, features of interest,
and per observable column its observed property, unit, sensor and feature."""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from config import PipelineConfig, load_config, override
from .models import (SourceProfile, ResolvedEntity, IdentificationMethod,
                     EntityKind, PropertyDecl, FeatureResolution, FileSampling, column_key)
from .evidence import gather_evidence, _OBSERVABLE
from .property_retrieval import retrieve_candidates
from utils.uri_policy import minted_property_uri
from utils.qudt_mappings import resolve_unit

load_dotenv()
_CLIENT = None


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _CLIENT

_SYSTEM = """You resolve a dataset's observable columns. For EACH column you decide three things, and you also identify the physical sensor devices.

You are given, per column: its column_key, sample values, and a CANDIDATE MENU of observed-property URIs retrieved from two vocabularies — QUDT quantity kinds (physics: Temperature, VolumeFlowRate, ...) and GEMET (domain: chemical oxygen demand, water level, ...). You also get evidence chunks: `structure:*` (folder + samples, always present) and companion chunks (metadata files, GROUND TRUTH when present).

For each column, output {sensor_id, property_uri, property_label, unit, feature_id}:
- property_uri: pick the SINGLE best candidate URI that captures what the column measures. Prefer a QUDT quantity kind when it fits the physics (temperature, flow rate, velocity, conductivity, turbidity); prefer a GEMET concept when the property is domain-specific and QUDT only offers a generic dimension (COD, TSS, water level, precipitation). The property MUST be dimensionally consistent with the unit — a column in m/s is a velocity, never a mass or volume flow rate; a column in mg/L is a concentration. If NO candidate is a good, dimensionally-consistent fit, output the literal "MINT" rather than forcing a wrong one.
- property_label: the candidate's label if you reused one; if MINT, a concise lowercase name for the property (e.g. "total suspended solids").
- unit: the measurement unit as a short symbol string (e.g. "m", "L/s", "mg/L", "°C", "mm", "m/s"). Take it VERBATIM from a companion/metadata chunk when present; otherwise INFER it from the column meaning and sample magnitudes. Use "" only if genuinely undeterminable.

Sensors (device identity):
- One physical device often produces several columns (a flow meter reporting flow_rate AND velocity) -> ONE entity, bind multiple columns. Deduplicate a device shared across files into one entity.
- attributes are DEVICE-ONLY (manufacturer/model/type/position/serial). NEVER put unit or measured quantity there.
- identification = "declared" if grounded in a companion/metadata chunk (set evidence_source to its name); "inferred" if only structural evidence (evidence_source null).
- id = short snake_case token (e.g. inflow_flodar).

Features of interest (the real-world thing each observation is ABOUT — the monitored structure or water body, NOT the sensor):
- Identify the distinct features from the evidence: site descriptions, folder/station names, file names. Examples: a combined sewer overflow / CSO structure, a stormwater storage or buffer basin, a sewer manhole/chamber, a sewer collector or pipe, a pumping station, a river or canal, a sewer catchment area.
- Deduplicate the SAME physical feature across files into ONE feature (e.g. a "chamber" named in two files = one feature).
- For each feature output {id, label, feature_kind}:
  - id = short snake_case token (e.g. cso_chamber, buffer_basin, deversoir).
  - label = a human name taken from the evidence (the site/station name verbatim when present).
  - feature_kind = a SHORT, GENERIC English noun phrase for the KIND of asset (e.g. "combined sewer overflow", "stormwater storage basin", "sewer manhole", "sewer collector pipe", "pumping station", "river", "sewer catchment area"). NO proper nouns, NO measurand words (never "water level").
- feature_id (per column) = the id of the feature that column observes.

Bind EVERY listed column_key in the `columns` map, using the column_key strings verbatim."""

_TOOL = {
    "name": "resolve_dataset",
    "description": "Resolve sensors, features of interest, and per observable column its sensor binding + observedProperty + unit + feature.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                        "identification": {"type": "string", "enum": ["declared", "inferred"]},
                        "evidence_source": {"type": ["string", "null"]},
                    },
                    "required": ["id", "label", "identification"],
                },
            },
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "label": {"type": "string"},
                        "feature_kind": {"type": "string",
                                         "description": "short GENERIC English kind, no proper nouns, no measurand"},
                    },
                    "required": ["id", "label", "feature_kind"],
                },
            },
            "columns": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "sensor_id": {"type": "string", "description": "one of entities[].id"},
                        "property_uri": {"type": "string", "description": "a candidate URI, or 'MINT'"},
                        "property_label": {"type": "string"},
                        "unit": {"type": "string"},
                        "feature_id": {"type": "string", "description": "one of features[].id"},
                    },
                    "required": ["sensor_id", "property_uri", "property_label", "unit", "feature_id"],
                },
                "description": "map of column_key -> resolution, using exact column_key strings",
            },
        },
        "required": ["entities", "features", "columns"],
    },
}

# Optional rules: appended to the prompt and schema only when their config flag is on.
_CLASSES = ["measurement", "derived", "diagnostic", "quality"]

_CLASSIFY_RULE = """

Column class (per column): decide whether the column is a MEASUREMENT, a value that a sensor or a laboratory analysis actually observed in the world. If it is not, classify it as:
- derived: computed from other data after the fact (event statistics, sums, maxima, return periods, gap or availability analyses, model or simulation output);
- diagnostic: the state of the device or of the data transmission rather than of the environment (battery voltage, signal strength, counters, status codes);
- quality: a flag or score describing the quality of another value.
Use the evidence: folder and file names, column names, and the companion metadata. Only measurement columns become observations. When in doubt, choose measurement. Fill the other fields for every column regardless."""


def _classify_tool() -> dict:
    import copy
    tool = copy.deepcopy(_TOOL)
    col = tool["input_schema"]["properties"]["columns"]["additionalProperties"]
    col["properties"]["column_class"] = {
        "type": "string", "enum": _CLASSES,
        "description": "measurement, or why the column is not one (derived / diagnostic / quality)",
    }
    col["required"] = col["required"] + ["column_class"]
    return tool


_SAMPLING_RULE = """

Sampling (per file): some files do not hold readings of an in-situ sensor but the results of analysing physical SAMPLES taken from a feature (grab or composite samples of water, sediment or sludge, analysed in a laboratory or with a separate instrument). In such a file each ROW is one sample, and every column of the row is a measurement made on that same sample.
- For every such file add an entry to `sampled_files`, keyed by the file path exactly as it appears before '::' in its column_keys, with {sampler_id, feature_id}:
  - sampler_id = the device that TOOK the samples (e.g. an automatic or peristaltic sampler), declared in `samplers` with the same fields as a sensor; null if the evidence does not say how the samples were taken.
  - feature_id = the feature the samples were taken from (one of features[].id).
- The sensor_id of those columns remains the instrument that analysed the sample.
- Do NOT list files of sensors that measure in place (probes, sondes, in-situ spectrometers, gauges). A regular, high-frequency time step (every few minutes) indicates an in-situ sensor.
- If no file is sample-based, output an empty `sampled_files` object and an empty `samplers` list."""


def _sampling_tool(base: dict) -> dict:
    import copy
    tool = copy.deepcopy(base)
    props = tool["input_schema"]["properties"]
    props["samplers"] = copy.deepcopy(props["entities"])
    props["samplers"]["description"] = "sosa:Sampler devices that took samples (may be empty)"
    props["sampled_files"] = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "sampler_id": {"type": ["string", "null"], "description": "one of samplers[].id, or null"},
                "feature_id": {"type": "string", "description": "one of features[].id"},
            },
            "required": ["sampler_id", "feature_id"],
        },
        "description": "map of file path (the part of column_key before '::') -> sampling, "
                       "ONLY for files whose rows are physical samples; empty if none",
    }
    tool["input_schema"]["required"] = tool["input_schema"]["required"] + ["samplers", "sampled_files"]
    return tool


_SHARED_MENU_RULE = """

Compact candidate menus: every candidate concept appears ONCE under "Candidate concepts" as `identifier  label`; each distinct ranked candidate list appears ONCE under "Candidate lists" (best first; columns with the same candidates share a list); each column names its own list. For property_uri, write an identifier from the column's OWN list exactly as listed (e.g. gemet:9190, qudt:Temperature), or "MINT". If you reuse one, property_label is its label from "Candidate concepts"."""


def _curie(c: dict) -> str:
    """qudt:<Name> or gemet:<number>."""
    return f"{c['source']}:{c['uri'].rsplit('/', 1)[-1]}"


def _shared_menus(menus: dict[str, list[dict]]) -> tuple[str, dict[str, str]]:
    concepts: dict[str, str] = {}
    lists: dict[tuple, str] = {}
    ref: dict[str, str] = {}
    for key, cands in menus.items():
        curies = tuple(_curie(c) for c in cands)
        for c in cands:
            concepts.setdefault(_curie(c), c["label"])
        ref[key] = lists.setdefault(curies, f"L{len(lists) + 1}")
    text = ("## Candidate concepts (every candidate listed once: identifier, label)\n"
            + "\n".join(f"{cu}  {label}" for cu, label in concepts.items())
            + "\n\n## Candidate lists (ranked, best first; several columns may share a list)\n"
            + "\n".join(f"{name}: {', '.join(curies)}" for curies, name in lists.items()))
    return text, ref


def _expand_curies(result: dict, menus: dict[str, list[dict]]) -> None:
    """Map identifiers back to full URIs; one outside the column's list is later minted."""
    for key, r in (result.get("columns") or {}).items():
        if r and r.get("property_uri"):
            by_curie = {_curie(c): c["uri"] for c in menus.get(key, [])}
            r["property_uri"] = by_curie.get(r["property_uri"], r["property_uri"])


def _observable_cols(profile: SourceProfile):
    for fp in profile.files:
        for col in fp.columns:
            if col.ssn_role in _OBSERVABLE:
                yield column_key(profile.root_path, fp.path, col.name), col, fp


def build_request(profile: SourceProfile, config: PipelineConfig | None = None):
    """(system, prompt, tool, menus) for the resolve call, or None if nothing to resolve.
    Calls the model only for query enrichment, when enabled."""
    config = config or load_config()
    cols = list(_observable_cols(profile))
    chunks = gather_evidence(profile)
    if not config.use_documentation:
        chunks = [c for c in chunks if c.source.startswith("structure:")]
    if not cols or not chunks:
        return None

    companion_text = "\n".join(c.text for c in chunks if not c.source.startswith("structure:"))

    enriched: dict[str, str] = {}
    if config.enrich_enabled:
        from .query_enrich import enrich_queries
        enriched = enrich_queries(profile, model=config.enrich_model,
                                  include_metadata=config.enrich_include_metadata and config.use_documentation,
                                  temperature=config.temperature,
                                  cache_prompt=config.cache_prompt)

    menus: dict[str, list[dict]] = {}
    blocks = []
    for key, col, fp in cols:
        # an enriched phrase replaces both the column name and the file stem
        term = enriched.get(key)
        cands = retrieve_candidates(col.name, "" if term else Path(fp.path).stem,
                                    companion_text, k=config.retrieval_k, query_term=term)
        menus[key] = cands
        samples = ", ".join(col.sample_values[:3]) if col.sample_values else "?"
        lines = [f"### {key}", f"  samples: [{samples}]", "  candidates:"]
        lines += [f"    - {c['uri']}  ({c['source']}: {c['label']}, d={c['distance']})" for c in cands]
        blocks.append("\n".join(lines))

    shared = config.menu_format == "shared"
    head = ""
    if shared:
        head, ref = _shared_menus(menus)
        head += "\n\n"
        blocks = []
        for key, col, fp in cols:
            samples = ", ".join(col.sample_values[:3]) if col.sample_values else "?"
            blocks.append(f"### {key}\n  samples: [{samples}]\n  candidates: {ref[key]}")

    evidence_text = "\n\n".join(f"===== {c.source} =====\n{c.text}" for c in chunks)
    prompt = (
        f"Dataset: {profile.id}\n\n"
        + head
        + "## Observable columns to resolve (sensor + observedProperty + unit)\n"
        + "\n\n".join(blocks)
        + f"\n\n## Evidence\n{evidence_text}"
    )
    system, tool = _SYSTEM, _TOOL
    if shared:
        system = system + _SHARED_MENU_RULE
    if config.classify_columns:
        system, tool = system + _CLASSIFY_RULE, _classify_tool()
    if config.model_sampling:
        system, tool = system + _SAMPLING_RULE, _sampling_tool(tool)
    return system, prompt, tool, menus


def _user_content(prompt: str, cache: bool):
    if not cache:
        return prompt
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]


def resolve_dataset(profile: SourceProfile, model: str | None = None,
                    config: PipelineConfig | None = None) -> None:
    config = override(config or load_config(), model=model)
    req = build_request(profile, config=config)
    if req is None:
        return
    system, prompt, tool, menus = req

    extra = {} if config.temperature is None else {"temperature": config.temperature}
    # long prompts can outlast the SDK's default 10-minute timeout
    response = _client().with_options(timeout=1800.0).messages.create(
        model=config.model, max_tokens=16384, system=system,
        tools=[tool], tool_choice={"type": "tool", "name": "resolve_dataset"},
        messages=[{"role": "user", "content": _user_content(prompt, config.cache_prompt)}], **extra,
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    if config.menu_format == "shared":
        _expand_curies(tool_use.input, menus)
    _attach(profile, tool_use.input, menus)


def _attach(profile: SourceProfile, result: dict, menus: dict[str, list[dict]]) -> None:
    profile.sensors = [
        ResolvedEntity(
            kind=EntityKind.SENSOR, id=e["id"], label=e.get("label", e["id"]),
            attributes=e.get("attributes", {}) or {},
            identification=IdentificationMethod(e["identification"]),
            evidence_source=e.get("evidence_source"),
        )
        for e in result.get("entities", [])
    ]

    columns = result.get("columns", {})

    # demote columns classified as not a measurement; assemble_ir then drops them
    excluded: set[str] = set()
    for key, col, _ in _observable_cols(profile):
        r = columns.get(key) or {}
        cls = r.get("column_class")
        col.column_class = cls if cls in _CLASSES else None
        if cls in _CLASSES and cls != "measurement":
            excluded.add(key)
            col.sensor_id = col.observed_property_uri = col.feature_id = None
            col.unit = col.qudt_unit_uri = None
    if excluded:
        print(f"[resolve] {len(excluded)} column(s) demoted (not measurements)")

    # property and unit per column; a URI outside the column's menu is minted instead
    decided: dict[str, tuple[str, str, str]] = {}
    for key, col, _ in _observable_cols(profile):
        r = columns.get(key)
        if not r or key in excluded:
            continue
        col.sensor_id = r.get("sensor_id")
        col.unit = r.get("unit") or None
        col.qudt_unit_uri = resolve_unit(col.unit) if col.unit else None

        choice = r.get("property_uri", "MINT")
        cand = {c["uri"]: c for c in menus.get(key, [])}
        if choice in cand:
            decided[key] = (choice, cand[choice]["label"], cand[choice]["source"])
        else:
            label = r.get("property_label") or col.name
            decided[key] = (minted_property_uri(label), label, "minted")

    # a minted property whose label matches a reused external one takes the external URI
    ext_by_label: dict[str, tuple[str, str, str]] = {}
    for uri, label, source in decided.values():
        if source in ("qudt", "gemet"):
            ext_by_label.setdefault(label.strip().lower(), (uri, label, source))
    for key, (uri, label, source) in list(decided.items()):
        if source == "minted":
            hit = ext_by_label.get(label.strip().lower())
            if hit:
                decided[key] = hit

    props: dict[str, PropertyDecl] = {}
    for key, col, _ in _observable_cols(profile):
        if key not in decided:
            continue
        uri, label, source = decided[key]
        col.observed_property_uri = uri
        props.setdefault(uri, PropertyDecl(uri=uri, label=label, source=source))

    profile.properties = list(props.values())

    # features of interest; the GWSW class comes from retrieval on the model's feature_kind
    from .foi_retrieval import resolve_feature_kind
    profile.features = []
    feat_ids: set[str] = set()
    for fdef in result.get("features", []) or []:
        fid = fdef.get("id")
        if not fid:
            continue
        kind = (fdef.get("feature_kind") or fdef.get("label") or "").strip()
        hit = resolve_feature_kind(kind)
        profile.features.append(FeatureResolution(
            id=fid, label=fdef.get("label") or fid, kind=kind,
            gwsw_class_uri=hit["uri"] if hit else None,
        ))
        feat_ids.add(fid)
    for key, col, _ in _observable_cols(profile):
        r = columns.get(key)
        col.feature_id = (r.get("feature_id") if (r and key not in excluded
                                                  and r.get("feature_id") in feat_ids) else None)

    if "sampled_files" in result:
        _attach_sampling(profile, result, feat_ids)

    # declare only sensors and features that a kept column references
    if excluded:
        used_s = {c.sensor_id for _, c, _ in _observable_cols(profile) if c.observed_property_uri}
        used_f = {c.feature_id for _, c, _ in _observable_cols(profile) if c.observed_property_uri}
        dropped = [s.id for s in profile.sensors if s.id not in used_s]
        profile.sensors = [s for s in profile.sensors if s.id in used_s]
        profile.features = [f for f in profile.features if f.id in used_f]
        if dropped:
            print(f"[resolve] dropped sensor(s) with no measurement columns: {dropped}")


def _attach_sampling(profile: SourceProfile, result: dict, feat_ids: set[str]) -> None:
    """Mark sample-based files. An undeclared sampler id becomes None; an undeclared feature
    id falls back to the file's single bound feature, else None."""
    profile.samplers = [
        ResolvedEntity(
            kind=EntityKind.SAMPLER, id=e["id"], label=e.get("label", e["id"]),
            attributes=e.get("attributes", {}) or {},
            identification=IdentificationMethod(e.get("identification", "inferred")),
            evidence_source=e.get("evidence_source"),
        )
        for e in result.get("samplers", []) or [] if e.get("id")
    ]
    sampler_ids = {s.id for s in profile.samplers}
    sampled = {str(k).split("::")[0].strip(): v
               for k, v in (result.get("sampled_files") or {}).items() if isinstance(v, dict)}

    root = Path(profile.root_path).resolve()
    matched: set[str] = set()
    for fp in profile.files:
        fp.sampling = None
        rel = Path(fp.path).resolve().relative_to(root).as_posix()
        s = sampled.get(rel)
        if s is None:
            continue
        matched.add(rel)
        kept = [c for c in fp.columns if c.ssn_role in _OBSERVABLE and c.observed_property_uri]
        if not kept:
            continue
        fid = s.get("feature_id")
        if fid not in feat_ids:
            bound = {c.feature_id for c in kept if c.feature_id}
            fid = next(iter(bound)) if len(bound) == 1 else None
        sid = s.get("sampler_id")
        sid = sid if sid in sampler_ids else None
        fp.sampling = FileSampling(sampler_id=sid, feature_id=fid)
        for c in kept:
            c.feature_id = fid

    used = {fp.sampling.sampler_id for fp in profile.files if fp.sampling and fp.sampling.sampler_id}
    profile.samplers = [s for s in profile.samplers if s.id in used]
    unknown = sorted(set(sampled) - matched)
    if unknown:
        print(f"[resolve] sampled_files keys matching no file (ignored): {unknown}")
    n = sum(1 for fp in profile.files if fp.sampling)
    if n:
        print(f"[resolve] {n} sampled file(s), {len(profile.samplers)} sampler(s)")
