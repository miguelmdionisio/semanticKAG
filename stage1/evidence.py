from __future__ import annotations
from pathlib import Path
from typing import Optional
from pydantic import BaseModel


from .models import SourceProfile, FileProfile, SSNRole, column_key

_OBSERVABLE = {SSNRole.OBSERVABLE_PROP, SSNRole.UNKNOWN}

_TEXT_EXT = {".json", ".csv", ".tsv", ".txt", ".md", ".xml", ".yaml", ".yml"}

class EvidenceChunk(BaseModel):
    source: str   # "structure:<relpath>" or the companion file name
    text: str


def _rel(profile: SourceProfile, file_path: str) -> str:
    return Path(file_path).resolve().relative_to(Path(profile.root_path).resolve()).as_posix()

def _structural_chunk(profile: SourceProfile, fp: FileProfile) -> Optional[EvidenceChunk]:
    rel = _rel(profile, fp.path)
    obs_cols = [c for c in fp.columns if c.ssn_role in _OBSERVABLE]
    if not obs_cols:
        return None
    lines = [f"file: {rel}", f"folder: {Path(rel).parent.as_posix()}"]
    for col in obs_cols:
        key = column_key(profile.root_path, fp.path, col.name)
        samples = ", ".join(col.sample_values[:3]) if col.sample_values else "?"
        lines.append(f"  - column_key: {key}")
        lines.append(
            f"    name={col.name}  observation={col.observation_type or '?'}  "
            f"unit={col.unit or '?'}  samples=[{samples}]"
        )
    return EvidenceChunk(source=f"structure:{rel}", text="\n".join(lines))


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""
    # failed font extraction yields /uniXXXX glyph noise
    if text.count("/uni") > 20:
        return ""
    return text


def _companion_chunk(path_str: str) -> Optional[EvidenceChunk]:
    path = Path(path_str)
    ext = path.suffix.lower()
    if ext == ".pdf":
        text = _extract_pdf(path)
    elif ext in _TEXT_EXT:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    else:
        return None
    text = text.strip()
    return EvidenceChunk(source=path.name, text=text) if text else None


def gather_evidence(profile: SourceProfile, max_chars: int = 20_000) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for fp in profile.files:
        chunk = _structural_chunk(profile, fp)
        if chunk:
            chunks.append(chunk)
    for comp in profile.companion_files:
        chunk = _companion_chunk(comp)
        if chunk:
            chunks.append(chunk)
    for c in chunks:
        if len(c.text) > max_chars:
            c.text = c.text[:max_chars] + "\n…[truncated]"
    return chunks