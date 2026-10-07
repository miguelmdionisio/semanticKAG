import os
from pathlib import Path

CSV_EXT = {".csv", ".tsv"}
READABLE_EXT = {".json", ".txt", ".md", ".xml", ".yaml", ".yml"}

def discover(root:str) -> tuple[list[Path], list[Path], list[Path]]:
    root_path = Path(root)
    csvs: list[Path] = []
    readables: list[Path] = []
    skipped: list[Path] = []

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            fpath = Path(dirpath) / filename
            ext = fpath.suffix.lower()
            if ext in CSV_EXT:
                csvs.append(fpath)
            elif ext in READABLE_EXT:
                readables.append(fpath)
            else: skipped.append(fpath)

    return csvs, readables, skipped