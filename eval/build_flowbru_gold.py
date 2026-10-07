"""FlowBru gold standard: one entry per column for two stations that ship no documentation
(Canal_Beco, BT_Belliard). Built from headers, units and values before looking at pipeline output.
Every IRI is checked against the vendored vocabularies.

Run: python eval/build_flowbru_gold.py
"""
from __future__ import annotations
import html
import json
from pathlib import Path

ROOT = Path("data/03 - FlowBru")

STATIONS = {
    "flowbru_beco": dict(
        path="Water_Quality/Canal_Beco",
        sensors={
            "exo2_sonde": "YSI EXO2 multiparameter sonde (in situ). Only the chlorophyll and phycocyanin "
                          "headers name EXO2; the other quality parameters are assumed to be the same sonde",
            "beco_level_sensor": "water level sensor (relative level and level above MSL)",
            "rain_gauge_ecluse": "rain gauge 'Pluvio Ecluse' (at the lock), stored in the level file",
        },
        features={
            "canal_beco": dict(label="Canal, Bassin Béco", gwsw="gwsw:Kanaal"),
            "rain_site_ecluse": dict(label="Rain gauge site, Ecluse", gwsw=None),
        },
        obs=[
            ("Chlorophyll.tsv", "EXO2 Chlorophyll [µg/l][µg/L]", "chlorophyll", "gemet:1391", "reuse",
             "unit:MicroGM-PER-L", "exo2_sonde", "canal_beco", [], ""),
            ("Chlorophyll.tsv", "EXO2 Chlorophyll [RFU][RFU]", "chlorophyll (fluorescence)", "gemet:1391", "reuse",
             None, "exo2_sonde", "canal_beco", ["mint:*"],
             "approximate: RFU = relative fluorescence units, not a QUDT unit; a minted 'chlorophyll fluorescence' is accepted (lenient)"),
            ("Conductivity.tsv", "Conductivity[µS/cm]", "electrical conductivity", "qudt:ElectrolyticConductivity", "reuse",
             "unit:MicroS-PER-CM", "exo2_sonde", "canal_beco",
             ["qudt:Conductivity", "qudt:ElectricConductivity", "gemet:1686"],
             "approximate: QUDT has three near-synonyms; ElectrolyticConductivity = conductivity of an electrolyte solution"),
            ("dissolved_oxygen.tsv", "Dissolved Oxygen [mg/l][mg/L]", "dissolved oxygen", "gemet:11788", "reuse",
             "unit:MilliGM-PER-L", "exo2_sonde", "canal_beco", [], ""),
            ("pH.tsv", "pH[]", "pH", "gemet:14636", "reuse",
             "unit:PH", "exo2_sonde", "canal_beco", ["qudt:Acidity"],
             "GEMET 'pH-value'; QUDT models pH as quantitykind:Acidity (unit:PH applies to it)"),
            ("phycocyanin.tsv", "EXO2 Phycocyanin [µg/l][µg/L]", "phycocyanin", "ex:property/phycocyanin", "mint",
             "unit:MicroGM-PER-L", "exo2_sonde", "canal_beco", [],
             "no phycocyanin concept in QUDT/GEMET (nearest: GEMET 939 'blue-green alga', an organism, not the pigment)"),
            ("phycocyanin.tsv", "EXO2 Phycocyanin [RFU][RFU]", "phycocyanin (fluorescence)", "ex:property/phycocyanin", "mint",
             None, "exo2_sonde", "canal_beco", [], "RFU = relative fluorescence units, not a QUDT unit"),
            ("Temperature.tsv", "Temperature[°C]", "water temperature", "qudt:Temperature", "reuse",
             "unit:DEG_C", "exo2_sonde", "canal_beco", [], ""),
            ("Turbidity.tsv", "Turbidity[NTU]", "turbidity", "qudt:Turbidity", "reuse",
             "unit:NTU", "exo2_sonde", "canal_beco", ["gemet:8711"], ""),
            ("Water_Level.tsv", "Level[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "beco_level_sensor", "canal_beco", [], ""),
            ("Water_Level.tsv", "Level (MSL)[m]", "water level (above mean sea level)", "gemet:9190", "reuse",
             "unit:M", "beco_level_sensor", "canal_beco", [], "datum: mean sea level (unit metre, datum not expressible in QUDT)"),
            ("Water_Level.tsv", "Pluvio Ecluse[mm]", "precipitation", "gemet:637", "reuse",
             "unit:MilliM", "rain_gauge_ecluse", "rain_site_ecluse", ["gemet:6947"],
             "a rain gauge stored in the water-level file; GEMET 6947 'rain' accepted (lenient)"),
        ],
    ),
    "flowbru_belliard": dict(
        path="Buffer_Basin_Streams/BT_Belliard",
        sensors={
            "rain_gauge_flagey": "rain gauge 'Pluvio Flagey' (another site than the basin)",
            "basin1_level": "level measurement of basin 1 ('moyenne' = averaged); the volume is assumed to derive from it",
            "basin2_level": "level sensor of basin 2",
            "entrance_left": "level sensor, sewer entrance, left culvert ('Pertuis Gauche')",
            "entrance_right": "level sensor, sewer entrance, right culvert ('Pertuis Droit')",
            "vegason_ai1": "VEGASON ultrasonic level sensor, analogue input 1 (column empty in this export)",
            "vegapuls_ai2": "VEGAPULS radar level sensor, analogue input 2 (column empty in this export)",
            "fmr20_dn40_right": "Endress+Hauser Micropilot FMR20 radar level sensor, DN40, right",
            "fmr20_dn80_left": "Endress+Hauser Micropilot FMR20 radar level sensor, DN80, left",
        },
        features={
            "basin_1": dict(label="Buffer basin Belliard, basin 1", gwsw="gwsw:Bergingsbassin"),
            "basin_2": dict(label="Buffer basin Belliard, basin 2", gwsw="gwsw:Bergingsbassin"),
            "sewer_entrance": dict(label="Sewer entrance of the buffer basin", gwsw="gwsw:Rioolleiding"),
            "sewer_outlet": dict(label="Sewer outlet of the buffer basin", gwsw="gwsw:Rioolleiding"),
            "rain_site_flagey": dict(label="Rain gauge site, Flagey", gwsw=None),
        },
        obs=[
            ("Rainfall.tsv", "Pluvio Flagey[mm]", "precipitation", "gemet:637", "reuse",
             "unit:MilliM", "rain_gauge_flagey", "rain_site_flagey", ["gemet:6947"],
             "GEMET 6947 'rain' accepted (lenient)"),
            ("Volume.tsv", "Volume Bassin 1[m3]", "stored volume", "qudt:Volume", "reuse",
             "unit:M3", "basin1_level", "basin_1", ["qudt:LiquidVolume"],
             "approximate: probably computed from the basin-1 level (stage-volume curve); kept as an observation (user decision)"),
            ("Water_Level_Buffer_Basin.tsv", "Bassin 1 moyenne[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "basin1_level", "basin_1", [], ""),
            ("Water_Level_Buffer_Basin.tsv", "Bassin 2[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "basin2_level", "basin_2", [], ""),
            ("Water_Level_Sewer_Entrance.tsv", "Pertuis Gauche[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "entrance_left", "sewer_entrance", [], ""),
            ("Water_Level_Sewer_Entrance.tsv", "Pertuis Droit[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "entrance_right", "sewer_entrance", [], ""),
            ("Water_Level_Sewer_Outlet.tsv", "Vegason AI1 [mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "vegason_ai1", "sewer_outlet", [], "column empty in this export (decided from the header)"),
            ("Water_Level_Sewer_Outlet.tsv", "Vegapuls AI2 [mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "vegapuls_ai2", "sewer_outlet", [], "column empty in this export (decided from the header)"),
            ("Water_Level_Sewer_Outlet.tsv", "Fmr20 DN40 droit[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "fmr20_dn40_right", "sewer_outlet", [], ""),
            ("Water_Level_Sewer_Outlet.tsv", "Fmr20 DN80 gauch[mm]", "water level", "gemet:9190", "reuse",
             "unit:MilliM", "fmr20_dn80_left", "sewer_outlet", [], ""),
        ],
    ),
}


def verify() -> None:
    import rdflib
    from rdflib.namespace import SKOS, RDFS
    g = rdflib.Graph(); g.parse("vocab/gemet_en.ttl", format="turtle")
    gemet = {str(s).rsplit("/", 1)[-1] for s in g.subjects(SKOS.prefLabel, None)}
    gq = rdflib.Graph(); gq.parse("vocab/qudt_quantitykind.ttl", format="turtle")
    qk = {str(s).rsplit("/", 1)[-1] for s in gq.subjects(RDFS.label, None) if "quantitykind" in str(s)}
    gu = rdflib.Graph(); gu.parse("vocab/qudt_unit.ttl", format="turtle")
    un = {str(s).rsplit("/", 1)[-1] for s in gu.subjects(None, None) if "vocab/unit/" in str(s)}
    gw = {json.loads(l)["uri"].rsplit("/", 1)[-1] for l in open("vocab/gwsw_foi.jsonl", encoding="utf-8") if l.strip()}

    def ok(curie: str) -> bool:
        pre, loc = curie.split(":", 1)
        return {"gemet": loc in gemet, "qudt": loc in qk, "unit": loc in un, "gwsw": loc in gw,
                "mint": True, "ex": True}[pre]

    bad = []
    for st in STATIONS.values():
        for fname, col, *_ in st["obs"]:
            if not (ROOT / st["path"] / fname).exists():
                bad.append(f"missing file {st['path']}/{fname}")
        for (_, _, _, prop, _, unit, sensor, feat, accept, _) in st["obs"]:
            for c in [prop, *(a for a in accept if a != "mint:*")] + ([unit] if unit else []):
                if not ok(c):
                    bad.append(c)
            if sensor not in st["sensors"]:
                bad.append(f"sensor {sensor}")
            if feat not in st["features"]:
                bad.append(f"feature {feat}")
        for f in st["features"].values():
            if f["gwsw"] and not ok(f["gwsw"]):
                bad.append(f["gwsw"])
    if bad:
        raise SystemExit(f"UNVERIFIED: {sorted(set(bad))}")
    print("all IRIs, files, sensors and features verified")


def first_value(path: Path, col: str) -> tuple[str, str]:
    """First non-empty (resultTime, value) of a column, for the review sheet."""
    import pandas as pd
    d = pd.read_csv(path, sep="\t", encoding="latin1", nrows=5000, dtype=str)
    d.columns = [c.encode("latin1").decode("latin1") for c in d.columns]
    if col not in d.columns:
        return "", "(column not found)"
    s = d[col].dropna()
    return (d.iloc[s.index[0], 0], s.iloc[0]) if len(s) else ("", "(empty)")


def emit() -> None:
    esc = lambda x: html.escape("" if x is None else str(x))
    sections = []
    for ds, st in STATIONS.items():
        obs = []
        rows = []
        for (fname, col, typ, prop, dec, unit, sensor, feat, accept, note) in st["obs"]:
            ts, val = first_value(ROOT / st["path"] / fname, col)
            f = st["features"][feat]
            obs.append(dict(type=typ, file=fname, column=col, value=val, resultTime=ts,
                            sensor=sensor, sensor_type=st["sensors"][sensor],
                            observedProperty=prop, property_source=prop.split(":", 1)[0] if dec == "reuse" else "minted",
                            decision=dec, unit=unit, feature=f["label"], feature_gwsw=f["gwsw"],
                            accept=accept, note=note))
            rows.append(f"<tr><td class=mono>{esc(fname)}<br><b>{esc(col)}</b><br>value {esc(val)} @ {esc(ts)}</td>"
                        f"<td><b>{esc(prop)}</b> ({esc(dec)})<br><span class=src>accept: {esc(', '.join(accept) or '-')}</span></td>"
                        f"<td>{esc(unit or 'none')}</td><td>{esc(sensor)}</td>"
                        f"<td>{esc(f['label'])}<br><span class=src>{esc(f['gwsw'] or 'untyped')}</span></td>"
                        f"<td class=src>{esc(note)}</td><td class=exp>______</td></tr>")
        Path(f"eval/gold_{ds}.json").write_text(json.dumps(dict(
            dataset=ds, root=str(ROOT / st["path"]).replace("\\", "/"), observations=obs,
            features=[dict(id=k, **v) for k, v in st["features"].items()],
            sensors=st["sensors"]), ensure_ascii=False, indent=1), encoding="utf-8")
        sections.append(f"<h2>{esc(ds)} &mdash; {esc(st['path'])} ({len(obs)} columns)</h2>"
                        "<table><tr><th>column</th><th>property</th><th>unit</th><th>sensor</th>"
                        "<th>feature / GWSW</th><th>note</th><th>expert</th></tr>" + "".join(rows) + "</table>")
        print(f"wrote eval/gold_{ds}.json ({len(obs)} columns)")
    style = ("body{font:13px/1.5 system-ui,Arial;margin:24px;max-width:1400px}table{border-collapse:collapse;width:100%}"
             "td,th{border:1px solid #ccc;padding:4px 7px;vertical-align:top;text-align:left}th{background:#f3f5f7}"
             ".mono{font-family:Consolas,monospace;font-size:11px}.src{color:#555;font-size:11px}.exp{color:#b8860b}")
    Path("eval/flowbru_review.html").write_text(
        "<!doctype html><meta charset=utf-8><title>FlowBru gold</title>"
        f"<style>{style}</style><h1>FlowBru gold (no companion metadata)</h1>"
        "<p>Author judgement from headers, units and values; fixed before looking at pipeline output. "
        "<b>accept</b> = alternatives counted only in the lenient property score. Sensor grouping and GWSW "
        "classes are provisional.</p>" + "".join(sections), encoding="utf-8")
    print("wrote eval/flowbru_review.html")


if __name__ == "__main__":
    verify()
    emit()
