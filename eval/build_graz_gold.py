"""Graz-West gold standard, curated from the dataset's meta_data.json: one entry per column
(observed property, unit, sensor, feature). Every IRI is checked against the vendored vocabularies.

Run: python eval/build_graz_gold.py
"""
from __future__ import annotations
import json, html
from pathlib import Path

DATASET = "graz-west"

# sensors (all declared in meta_data.json)
SENSORS = {
    "chamber_ultrasound": dict(type="ultrasonic water-level sensor", manufacturer="Sommer"),
    "inflow_flodar":      dict(type="radar flow sensor (FLO-DAR)", manufacturer="Hach / Marsh McBirney"),
    "overflow_ocmpro":    dict(type="ultrasonic cross-correlation flow sensor (OCM Pro)", manufacturer="NIVUS"),
    "scan_spectrometer":  dict(type="UV/VIS spectrometer (s::can)", manufacturer="s::can"),
    "hach_lab_photometer":dict(type="cuvette photometer (DR3900 / DR6000)", manufacturer="Hach"),
    # lab TSS is gravimetric (filtering, drying, weighing), not measured on the photometer
    "lab_tss_gravimetric":dict(type="gravimetric TSS analysis (filtration, drying, weighing); instrument not stated",
                               manufacturer="(unspecified)"),
    "rain_gauge_stremayrgasse": dict(type="tipping-bucket rain gauge (0.1 mm)", manufacturer="(unspecified)"),
    "rain_gauge_0033901401":    dict(type="tipping-bucket rain gauge (0.1 mm)", manufacturer="(unspecified)"),
    "rain_gauge_0033901601":    dict(type="tipping-bucket rain gauge (0.1 mm)", manufacturer="(unspecified)"),
}

# features of interest (GWSW classes tentative)
FEATURES = {
    "cso_chamber": dict(label="CSO chamber",
        evidence='meta: "chamber of the CSO structure, above dry-weather-flow path"',
        gwsw="gwsw:Overstortput", options="Overstortput | Overstortconstructie | Rioolput"),
    "inflow_channel": dict(label="Inflow channel of CSO",
        evidence='meta: "inflow chanel of CSO structure"',
        gwsw="gwsw:Rioolleiding", options="Rioolleiding | Leiding | Transportleiding"),
    "overflow_channel": dict(label="Overflow channel downstream CSO",
        evidence='meta: "overflow chanel downstream the CSO structure"',
        gwsw="gwsw:Overstortleiding", options="Overstortleiding | Uitlaatconstructie | Leiding"),
    "rain_gauge_stremayrgasse": dict(label="Rain gauge — Graz-Stremayrgasse",
        evidence='meta internal-label "Graz-Stremayrgasse"; 47.0647,15.4517', gwsw=None,
        options="untyped (GWSW is sewer-asset scoped; rain gauge likely out of scope)"),
    "rain_gauge_0033901401": dict(label="Rain gauge — 0033901401",
        evidence='precipitation data/0033901401.csv', gwsw=None, options="untyped"),
    "rain_gauge_0033901601": dict(label="Rain gauge — 0033901601",
        evidence='precipitation data/0033901601.csv', gwsw=None, options="untyped"),
}

# observation types: property x unit x sensor x feature
OBS = [
    # water level: GEMET, since QUDT only has generic Length/Height
    dict(type="water level", prop="gemet:9190", src="gemet", decision="reuse", unit="unit:M",
         sensor="chamber_ultrasound", feature="cso_chamber",
         sources=[("hydraulic data/chamber.csv", "water_level", "0.23")]),
    dict(type="water level", prop="gemet:9190", src="gemet", decision="reuse", unit="unit:M",
         sensor="inflow_flodar", feature="inflow_channel",
         sources=[("hydraulic data/inflow-analogue.csv","water_level","0.178"),
                  ("hydraulic data/inflow-digital.csv","water_level","0.1068")]),
    dict(type="water level", prop="gemet:9190", src="gemet", decision="reuse", unit="unit:M",
         sensor="overflow_ocmpro", feature="overflow_channel",
         sources=[("hydraulic data/overflow.csv","water_level","0.011")]),
    dict(type="flow rate", prop="qudt:VolumeFlowRate", src="qudt", decision="reuse", unit="unit:L-PER-SEC",
         sensor="inflow_flodar", feature="inflow_channel",
         sources=[("hydraulic data/inflow-analogue.csv","flow_rate","39.2"),
                  ("hydraulic data/inflow-digital.csv","flow_rate","183.0")]),
    dict(type="flow rate", prop="qudt:VolumeFlowRate", src="qudt", decision="reuse", unit="unit:L-PER-SEC",
         sensor="overflow_ocmpro", feature="overflow_channel",
         sources=[("hydraulic data/overflow.csv","flow_rate","0.5")]),
    # velocity: qudt:Velocity (expert decision). The digital unit is m/s as documented, although
    # its values (max 1706) look like mm/s.
    dict(type="flow velocity", prop="qudt:Velocity", src="qudt", decision="reuse", unit="unit:M-PER-SEC",
         sensor="inflow_flodar", feature="inflow_channel",
         sources=[("hydraulic data/inflow-analogue.csv","flow_velocity","0.519"),
                  ("hydraulic data/inflow-digital.csv","flow_velocity","329.0")],
         note="Expert: Velocity (not LinearVelocity). inflow-digital values (mean 200.6, max 1706) look like mm/s despite the documented m/s."),
    dict(type="chemical oxygen demand (COD)", prop="gemet:1536", src="gemet", decision="reuse",
         unit="unit:MilliGM-PER-L", sensor="scan_spectrometer", feature="cso_chamber",
         sources=[("pollutant data/spectrometer.csv","COD","616.9"),
                  ("pollutant data/calibrated.csv","COD","972.39")]),
    dict(type="chemical oxygen demand (COD)", prop="gemet:1536", src="gemet", decision="reuse",
         unit="unit:MilliGM-PER-L", sensor="hach_lab_photometer", feature="cso_chamber",
         sources=[("lab data/samples.csv","COD1 / COD2 / COD3","874 / 902 / 956")]),
    # TSS: GEMET 'suspended matter' is approximate
    dict(type="total suspended solids (TSS)", prop="gemet:12220", src="gemet", decision="reuse",
         unit="unit:MilliGM-PER-L", sensor="scan_spectrometer", feature="cso_chamber",
         sources=[("pollutant data/spectrometer.csv","TSS","303.8")],
         accept=["mint:*"],   # lenient score only
         note="GEMET has no 'total suspended solids'; 12220 = 'suspended matter' (approximate). Expert: reuse this or mint ex:property/total-suspended-solids?"),
    dict(type="total suspended solids (TSS)", prop="gemet:12220", src="gemet", decision="reuse",
         unit="unit:MilliGM-PER-L", sensor="lab_tss_gravimetric", feature="cso_chamber",
         sources=[("lab data/samples.csv","TSS1 / TSS2 / TSS3","296 / 304 / 308")],
         accept=["mint:*"],
         note="Same 'suspended matter' approximation as above. Sensor: gravimetric analysis, not the photometer (fixed 2026-10-02)."),
    dict(type="water temperature", prop="qudt:Temperature", src="qudt", decision="reuse", unit="unit:DEG_C",
         sensor="scan_spectrometer", feature="cso_chamber",
         sources=[("pollutant data/spectrometer.csv","temperature","(sample blank)")]),
    dict(type="precipitation", prop="gemet:637", src="gemet", decision="reuse", unit="unit:MilliM",
         sensor="rain_gauge_stremayrgasse", feature="rain_gauge_stremayrgasse",
         sources=[("precipitation data/0033900601.csv","0033900601","0.1")],
         accept=["gemet:6947"],   # 'rain', lenient score only
         note="GEMET 637 = 'atmospheric precipitation' (chosen over 6947 'rain' — measures precipitation depth)."),
    dict(type="precipitation", prop="gemet:637", src="gemet", decision="reuse", unit="unit:MilliM",
         sensor="rain_gauge_0033901401", feature="rain_gauge_0033901401",
         sources=[("precipitation data/0033901401.csv","0033901401","0.1")], accept=["gemet:6947"]),
    dict(type="precipitation", prop="gemet:637", src="gemet", decision="reuse", unit="unit:MilliM",
         sensor="rain_gauge_0033901601", feature="rain_gauge_0033901601",
         sources=[("precipitation data/0033901601.csv","0033901601","0.1")], accept=["gemet:6947"]),
]

# refuse to write if any IRI is unknown
def verify():
    import rdflib
    from rdflib.namespace import SKOS, RDFS
    g=rdflib.Graph(); g.parse('vocab/gemet_en.ttl',format='turtle')
    gemet={str(s).rsplit('/',1)[-1] for s in g.subjects(SKOS.prefLabel,None)}
    gq=rdflib.Graph(); gq.parse('vocab/qudt_quantitykind.ttl',format='turtle')
    qk={str(s).rsplit('/',1)[-1] for s in gq.subjects(RDFS.label,None) if 'quantitykind' in str(s)}
    gu=rdflib.Graph(); gu.parse('vocab/qudt_unit.ttl',format='turtle')
    un={str(s).rsplit('/',1)[-1] for s in gu.subjects(None,None) if 'vocab/unit/' in str(s)}
    gw={json.loads(l)['uri'].rsplit('/',1)[-1] for l in open('vocab/gwsw_foi.jsonl',encoding='utf-8') if l.strip()}
    bad=[]
    for o in OBS:
        p=o["prop"]; loc=p.split(":",1)[1]
        if p.startswith("gemet:") and loc not in gemet: bad.append(p)
        if p.startswith("qudt:") and loc not in qk: bad.append(p)
        if o["unit"].split(":",1)[1] not in un: bad.append(o["unit"])
    for f in FEATURES.values():
        if f["gwsw"] and f["gwsw"].split(":",1)[1] not in gw: bad.append(f["gwsw"])
    if bad: raise SystemExit(f"UNVERIFIED IRIs: {sorted(set(bad))}")
    print("all IRIs verified against QUDT / GEMET / GWSW")

def emit():
    Path("eval").mkdir(exist_ok=True)
    def esc(x): return html.escape(str(x) if x is not None else "")
    def iri(c):
        if not c: return "UNTYPED"
        base={"gemet":"https://www.eionet.europa.eu/gemet/en/concept/","qudt":"https://qudt.org/vocab/quantitykind/",
              "unit":"https://qudt.org/vocab/unit/","gwsw":"http://data.gwsw.nl/1.6/totaal/","ex":"http://example.org/"}[c.split(":",1)[0]]
        return f'<a href="{base}{esc(c.split(":",1)[1])}" target=_blank>{esc(c)}</a>'

    def real_row(fk, col):
        """Read the first data row of a Graz CSV -> (resultTime, value) for the given column."""
        p=Path("data/02 - Graz-West R05/data-set")/fk
        real_col=col.split(" / ")[0].strip()             # 'COD1 / COD2 / COD3' -> 'COD1'
        with open(p,encoding="utf-8-sig",errors="replace") as fh:
            header=fh.readline().rstrip("\n").split(","); row=fh.readline().rstrip("\n").split(",")
        ts=row[0].replace(" ","T") if row else ""
        val=row[header.index(real_col)] if real_col in header and len(row)>header.index(real_col) else ""
        return ts, val

    # review HTML: one row per observation type
    rows=[]
    for i,o in enumerate(OBS,1):
        s=SENSORS[o["sensor"]]; f=FEATURES[o["feature"]]
        fk,col,_=o["sources"][0]
        ts,val=real_row(fk,col)
        foi=iri(f["gwsw"])
        result=(f'qudt:numericValue {esc(val) or "—"} ; qudt:unit {iri(o["unit"])}')
        ttl=("<span class=mono>a sosa:Observation ;<br>"
            f'&nbsp;sosa:resultTime "{esc(ts)}"^^xsd:dateTime ;<br>'
            f'&nbsp;sosa:madeBySensor <i>{esc(o["sensor"])}</i> <span class=src>({esc(s["type"])}, {esc(s["manufacturer"])})</span> ;<br>'
            f'&nbsp;sosa:observedProperty <b>{iri(o["prop"])}</b> <span class=src>({esc(o["src"])} · {esc(o["decision"])})</span> ;<br>'
            f'&nbsp;sosa:hasFeatureOfInterest <i>{esc(f["label"])}</i> a {foi} ;<br>'
            f'&nbsp;sosa:hasResult [ {result} ] .</span>')
        note=o.get("note","")
        srccell=(f'<b>{esc(o["type"])}</b><br><span class=mono>{esc(fk)} · {esc(col)}</span><br>'
                 f'<span class=mono>value={esc(val) or "(blank)"} @ {esc(ts)}</span><br>'
                 f'<span class=src>sensor {esc(o["sensor"])} · feature {esc(f["label"])}</span>')
        rows.append(f"<tr><td class=data>{srccell}</td><td>{ttl}"
                    f"{('<br><span class=src>note: '+esc(note)+'</span>') if note else ''}</td></tr>")

    # gold JSON: one entry per physical column ("COD1 / COD2 / COD3" is split)
    obs_json=[]
    for o in OBS:
        s=SENSORS[o["sensor"]]; f=FEATURES[o["feature"]]
        for fk,colspec,_ in o["sources"]:
            for col in (c.strip() for c in colspec.split(" / ")):
                ts,val=real_row(fk,col)
                obs_json.append(dict(type=o["type"],file=fk,column=col,value=val,resultTime=ts,
                    sensor=o["sensor"],sensor_type=s["type"],observedProperty=o["prop"],
                    property_source=o["src"],decision=o["decision"],unit=o["unit"],
                    feature=f["label"],feature_gwsw=f["gwsw"],accept=o.get("accept",[]),
                    note=o.get("note","")))

    feat_rows=[]
    for fid,f in FEATURES.items():
        badge=f"<span class=b style='background:#D85A30'>{iri(f['gwsw'])}</span>" if f["gwsw"] else "<span class=src>untyped</span>"
        feat_rows.append(f"<tr><td>{esc(f['label'])}</td><td class=src>{esc(f['evidence'])}</td>"
                         f"<td>{badge}</td><td class=src>{esc(f['options'])}</td><td class=exp>__________</td></tr>")

    Path("eval/gold_graz.json").write_text(json.dumps(
        dict(dataset=DATASET, observations=obs_json,
             features=[dict(id=k,**v) for k,v in FEATURES.items()],
             sensors=SENSORS), ensure_ascii=False, indent=1), encoding="utf-8")

    style=("body{font:13px/1.5 system-ui,Arial;margin:24px;max-width:1300px;color:#1a1a1a}"
        "h1{font-size:20px}h2{border-bottom:2px solid #ccc;margin-top:26px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:4px 7px;vertical-align:top;text-align:left}th{background:#f3f5f7}"
        ".mono{font-family:Consolas,monospace;font-size:11px}.data{background:#f7fbff}"
        ".src{color:#555;font-size:11px}.exp{color:#b8860b;font-size:11px}"
        ".b{display:inline-block;padding:1px 6px;border-radius:4px;color:#fff;font-size:11px}")
    doc=("<!doctype html><meta charset=utf-8><title>Graz-West gold</title>"
        f"<style>{style}</style><h1>Graz-West R05 — observation gold</h1>"
        "<p>Each row is a <b>real observation</b> (a source CSV row): its data on the left, the gold "
        "<span class=mono>sosa:Observation</span> it should yield on the right, every vocabulary IRI clickable — "
        f"put the pipeline's output beside it to evaluate. {len(OBS)} observations (one per type), ground-truthed from "
        "<span class=mono>meta_data.json</span>. All IRIs verified. "
        "Non-observations (<span class=mono>evaluations/</span> gap tables &amp; event statistics, the SWMM model) are excluded.</p>"
        f"<h2>Observations — {len(OBS)}</h2>"
        "<table><tr><th style='width:34%'>source (real CSV row)</th><th>gold observation (clickable IRIs)</th></tr>"
        +"".join(rows)+"</table>"
        f"<h2>Features → GWSW feature-of-interest — {len(FEATURES)} (expert sheet)</h2>"
        "<table><tr><th>feature</th><th>evidence</th><th>tentative GWSW</th><th>options</th><th>expert_decision</th></tr>"
        +"".join(feat_rows)+"</table>")
    Path("eval/graz_review.html").write_text(doc,encoding="utf-8")
    print(f"wrote eval/gold_graz.json ({len(obs_json)} column-observations) + "
          f"eval/graz_review.html ({len(OBS)} types, {len(FEATURES)} features)")

if __name__=="__main__":
    verify(); emit()
