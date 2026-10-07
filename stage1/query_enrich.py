"""Query enrichment: one model call per dataset turns each observable column into a short
concept phrase, used as its retrieval query."""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
import anthropic

from .models import SourceProfile, column_key
from .evidence import gather_evidence, _OBSERVABLE

load_dotenv()
_CLIENT = None
# an identical request in the same process reuses the first answer
_MEMO: dict[tuple, dict[str, str]] = {}


def _client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _CLIENT


_SYSTEM = """Domain: you name the physical quantities and substances measured in environmental and urban-water monitoring datasets, that is, sensor time series and laboratory analyses from sewer systems, wastewater, drainage catchments, rivers, precipitation, and water quality.

Task: for each observable column of one dataset you are given its column_key (which encodes the folder/file path and the column name), a few sample values, and the dataset's companion metadata (ground truth when present). Output ONE short phrase naming the physical quantity or substance that column measures. The phrase is embedded by a retriever and matched against QUDT quantity kinds (physical dimensions) and GEMET concepts (environmental domain); it is the SEARCH QUERY, not the final answer, so phrase it the way a formal vocabulary would.

Rules for each phrase:
- Expand any abbreviation or acronym to its full standard scientific term, from your own domain knowledge.
- If the column name is an opaque code or identifier with no meaning of its own, infer the quantity from the folder name, file name, sample magnitudes, and metadata.
- Use standard scientific terminology for the quantity, not instrument or vendor slang, so it matches a formal ontology label.
- Keep it to 1-5 words, the concept ONLY: no units, no sensor/device words, no site or station names, no punctuation, no explanation.
- When the companion metadata states what a column measures, use that.
- If you genuinely cannot tell, fall back to the column name with separators turned into spaces.

Return the `queries` map keyed by the EXACT column_key strings you were given, one phrase per column."""

_TOOL = {
    "name": "enrich_queries",
    "description": "For each observable column, the short canonical concept phrase to use as its retrieval query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "map of column_key -> short concept phrase (1-5 words, concept only)",
            },
        },
        "required": ["queries"],
    },
}


def _observable_cols(profile: SourceProfile):
    for fp in profile.files:
        for col in fp.columns:
            if col.ssn_role in _OBSERVABLE:
                yield column_key(profile.root_path, fp.path, col.name), col, fp


def build_enrich_request(profile: SourceProfile, include_metadata: bool = True):
    """(system, prompt, tool, keys) for the enrichment call, or None if nothing to enrich."""
    cols = list(_observable_cols(profile))
    if not cols:
        return None
    chunks = gather_evidence(profile)
    companion_text = "\n".join(c.text for c in chunks if not c.source.startswith("structure:"))

    blocks = []
    for key, col, _ in cols:
        samples = ", ".join(col.sample_values[:3]) if col.sample_values else "?"
        blocks.append(f"- {key}\n    samples: [{samples}]")

    prompt = (
        f"Dataset: {profile.id}\n\n"
        "## Observable columns (write one query phrase for each)\n"
        + "\n".join(blocks)
        + (f"\n\n## Companion metadata (ground truth when present)\n{companion_text}"
           if include_metadata and companion_text.strip() else "")
    )
    return _SYSTEM, prompt, _TOOL, [k for k, _, _ in cols]


def enrich_queries(profile: SourceProfile, model: str = "claude-sonnet-4-6",
                   include_metadata: bool = True, temperature: float | None = None,
                   cache_prompt: bool = False) -> dict[str, str]:
    """column_key -> concept phrase; {} if there is nothing to enrich."""
    req = build_enrich_request(profile, include_metadata=include_metadata)
    if req is None:
        return {}
    system, prompt, tool, keys = req
    memo_key = (model, system, prompt, temperature)
    if memo_key in _MEMO:
        return dict(_MEMO[memo_key])

    extra = {} if temperature is None else {"temperature": temperature}
    response = _client().messages.create(
        model=model, max_tokens=4096, system=system,
        tools=[tool], tool_choice={"type": "tool", "name": "enrich_queries"},
        messages=[{"role": "user", "content": prompt if not cache_prompt else
                   [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]}],
        **extra,
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    raw = tool_use.input.get("queries", {}) or {}
    valid = set(keys)
    out = {k: v.strip() for k, v in raw.items() if k in valid and isinstance(v, str) and v.strip()}
    _MEMO[memo_key] = out
    return dict(out)
