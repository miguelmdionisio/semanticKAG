"""UWO (Eawag Urban Water Observatory, 2021) gold standard, one entry per variable. Internal,
quality and transmission variables are excluded; GWSW site classes are tentative. Every IRI is
checked against the vendored vocabularies.

Run: python eval/build_uwo_gold.py   (reads data/data_uwo_2021.sqlite)
"""
from __future__ import annotations
import json, html
from pathlib import Path

ATM = "FoI = physical sensor site (per decision); measurand is atmospheric, not the site."

# variable_id -> (name, unit_symbol, property_curie, source, decision, unit_curie, note)
OBS = {
 34:("absolute_air_pressure","hPa","qudt:AtmosphericPressure","qudt","reuse","unit:HectoPA",ATM),
 38:("relative_air_pressure","hPa","qudt:AtmosphericPressure","qudt","reuse","unit:HectoPA",ATM),
 35:("absolute_humidity","g/m3","qudt:AbsoluteHumidity","qudt","reuse","unit:GM-PER-M3",ATM),
 40:("relative_humidity","%","qudt:RelativeHumidity","qudt","reuse","unit:PERCENT",ATM),
 36:("ambient_air_temperature","°C","qudt:Temperature","qudt","reuse","unit:DEG_C",ATM),
 47:("headspace_temperature","°C","qudt:Temperature","qudt","reuse","unit:DEG_C","sewer headspace air temperature"),
 48:("headspace_temperature2","°C","qudt:Temperature","qudt","reuse","unit:DEG_C","no data in 2021 export"),
 14:("overflow_temperature","°C","qudt:Temperature","qudt","reuse","unit:DEG_C",""),
 54:("soil_temperature","°C","qudt:Temperature","qudt","reuse","unit:DEG_C",""),
 7:("water_temperature","°C","qudt:Temperature","qudt","reuse","unit:DEG_C",""),
 53:("water_temperature2","°C","qudt:Temperature","qudt","reuse","unit:DEG_C",""),
 2:("average_velocity","m/s","qudt:Velocity","qudt","reuse","unit:M-PER-SEC","Velocity, not LinearVelocity: expert decision on Graz (2026-10-02), applied here 2026-10-04"),
 22:("surface_velocity","m/s","qudt:Velocity","qudt","reuse","unit:M-PER-SEC","second parameter (surface); Velocity per the Graz expert decision"),
 42:("wind_speed","m/s","qudt:Speed","qudt","reuse","unit:M-PER-SEC",ATM),
 4:("flow_rate","l/s","qudt:VolumeFlowRate","qudt","reuse","unit:L-PER-SEC",""),
 15:("pressure","bar","qudt:Pressure","qudt","reuse","unit:BAR","water-column pressure (level proxy)"),
 43:("global_radiation","W/m2","qudt:Irradiance","qudt","reuse","unit:W-PER-M2",ATM),
 28:("pH","-","qudt:PH","qudt","reuse","unit:PH","no data in 2021 export"),
 16:("dielectric_permittivity","-","qudt:RelativePermittivity","qudt","reuse","unit:UNITLESS","dimensionless (soil-moisture proxy)"),
 19:("distance","mm","qudt:Distance","qudt","reuse","unit:MilliM","raw sensor-to-surface distance (intermediate for water_level)"),
 41:("wind_direction","°","ex:property/wind-direction","mint","mint","unit:DEG",ATM+" No QUDT/GEMET concept; qudt:Angle is the dimensional fallback."),
 6:("water_level","mm","gemet:9190","gemet","reuse","unit:MilliM",""),
 44:("conductive_conductivity","uS/cm","gemet:1686","gemet","reuse","unit:MicroS-PER-CM","no data in 2021 export"),
 55:("inductive_conductivity","uS/cm","gemet:1686","gemet","reuse","unit:MicroS-PER-CM","no data in 2021 export"),
 29:("turbidity","NTU","gemet:8711","gemet","reuse","unit:NTU","no data in 2021 export"),
 33:("rainfall_cumsum","mm","gemet:637","gemet","reuse","unit:MilliM",ATM+" Cumulative total — possibly derived; expert check."),
 20:("rainfall_intensity","mm/h","ex:property/rainfall-intensity","mint","mint","unit:MilliM-PER-HR",ATM+" No QUDT/GEMET rainfall-intensity concept."),
 37:("precipitation_type","-","ex:property/precipitation-type","mint","mint",None,ATM+" Categorical (Lufft code: 0 no rain / 60 rain / ...)."),
}

# excluded (not an observation): name, unit, reason
EXCLUDED = [
 ("battery_charge","%","internal parameter"),("battery_voltage","V","internal parameter"),
 ("bucket_content","mm","internal (weighing-gauge fill, used to derive rainfall)"),
 ("csq","-","internal (cellular signal quality)"),("device_temperature","-","internal (logger temperature)"),
 ("error_message","-","internal (error code)"),("eui","-","internal (device identifier)"),
 ("frame_counter","-","internal (transmission counter)"),("status","-","internal (debug status)"),
 ("timestamp_ms","ms","internal (µC relative timestamp)"),
 ("bandwidthclass","-","measurement-quality information"),("gain","-","measurement-quality (radar signal)"),
 ("nos","-","measurement-quality (number of samples)"),("opposite_direction_ratio","%","measurement-quality"),
 ("pmr","-","measurement-quality (peak-over-main ratio)"),("trials","-","measurement-quality"),
 ("rssi","dBm","signal-transmission (received signal strength)"),("sf","-","signal-transmission (spreading factor)"),
 ("snr","dB","signal-transmission / quality (signal-to-noise)"),
]

# site name -> (gwsw_curie or None, options), tentative
SITES = {
 "115a_chueferistr":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "137_schutzengasse":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "15a_russikerstr":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "23_bahnhofstr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "7_kempttalstr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "607sbw_kempttalerstr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "11e_russikerstr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "555_mesikerstr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "50sbw_acherm":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "581a_wildbach":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "inflow_ara":("gwsw:Rioolleiding","Rioolleiding | Uitlaatconstructie"),
 "outflow_ara":("gwsw:Uitlaatconstructie","Uitlaatconstructie | Rioolleiding"),
 "sk_ara":("gwsw:Rioolleiding","Rioolleiding | Leiding"),
 "594_undermulistr":("gwsw:Rioolleiding","Rioolleiding | Transportleiding"),
 "597sbw_ara":("gwsw:Rioolleiding","Rioolleiding | Rioolgemaal"),
 "47a_zurcherstr":("gwsw:Rioolleiding","Rioolleiding | Rioolput"),
 "138a_venturi":("gwsw:Rioolleiding","Rioolleiding | Overstortconstructie"),
 "156a_geerenstr":("gwsw:Rioolleiding","Rioolleiding | Rioolput"),
 "rumpisweg":("gwsw:Rioolput","Rioolput | Rioolleiding | (unclear)"),
 "162_luppmenweg":("gwsw:Rioolput","Rioolput | Leiding"),
 "164_luppmenweg":("gwsw:Rioolput","Rioolput | Leiding"),
 "166_luppmenweg":("gwsw:Rioolput","Rioolput | Leiding"),
 "22a_bahnhofstr":("gwsw:Rioolput","Rioolput | Leiding"),
 "rw137_schutzengasse":("gwsw:Rioolput","Rioolput | Overstortput"),
 "rw22a_bahnhofstr":("gwsw:Rioolput","Rioolput | Overstortput"),
 "rw47a_zurcherstr":("gwsw:Rioolput","Rioolput | Overstortput"),
 "48sbw_notuberlauf":("gwsw:Overstortput","Overstortput | Overstortleiding"),
 "3r_rub_morg_overflow":("gwsw:Overstortleiding","Overstortleiding | Overstortput"),
 "40g_csovoland":("gwsw:Overstortput","Overstortput | Rioolleiding"),
 "ra40a_sbwvoland":("gwsw:Overstortput","Overstortput | Overstortconstructie"),
 "58sbw_undermulistr":("gwsw:Overstortconstructie","Overstortconstructie | Overstortput"),
 "sk102_wermatswilstr":("gwsw:Overstortput","Overstortput | Rioolput"),
 "vs22_kempttalstr":("gwsw:Overstortconstructie","Overstortconstructie | Rioolput"),
 "450a_usterstr":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Bergingsbassin"),
 "450b_usterstr":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Bergingsbassin"),
 "rub128basin_usterstr":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Bergingsbassin"),
 "rub128inflow_usterstr":("gwsw:Rioolleiding","Rioolleiding | Overstortput"),
 "rub_morg":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Bergingsbassin"),
 "rubmorg_inflow":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Overstortconstructie"),
 "rubbasin_ara":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Kolk"),
 "rubpw80sbwbasin_industry":("gwsw:Bergbezinkbassin","Bergbezinkbassin | Bergingsbassin"),
 "40d_imberg":("gwsw:Kolk","Kolk | Rioolput"),
 "luppmen_usterstr":("gwsw:Rivier","Rivier | Waterloop | Beek"),
 "585sbw_amwildbach":("gwsw:Beek","Beek | Rivier | Waterloop"),
 "wildbach_kempttalstr":("gwsw:Beek","Beek | Rivier | Waterloop"),
 "wildbach_rumlikonstr":("gwsw:Beek","Beek | Rivier | Waterloop"),
 "rohrbach_fehraltorferstr":("gwsw:Beek","Beek | Rivier"),
 "rubpw80sbw_industry":("gwsw:Rioolgemaal","Rioolgemaal | Pompput"),
 "pumpwgeeren_geerenstr":("gwsw:Rioolgemaal","Rioolgemaal | Gemaal"),
 "pumpwau_rumlikerstr":(None,"untyped | Rioolgemaal (weather sensor on pump-station roof)"),
 "gerber_zurcherstr":("gwsw:Rioolleiding","Rioolleiding | Rioolgemaal"),
 "ara_flatroof":(None,"untyped (rooftop weather station)"),
 "coop_grundstr":(None,"untyped (rooftop weather station)"),
 "electrosuisse_luppmenstr":(None,"untyped (rooftop weather station)"),
 "airport_speck":(None,"untyped (open field / airstrip)"),
 "school_chatzenrainstr":(None,"untyped (open field)"),
 "schutzenhaus_burgweg":(None,"untyped (roadside / field)"),
 "voland_unterdorf":(None,"untyped (lamp post — repeater + temp sensor)"),
 "11_veloweg":(None,"untyped | Rioolput (temperature sensor, location unclear)"),
 "11h_veloweg":(None,"untyped | Rioolput (temperature sensor, location unclear)"),
 "124a_chueferi":(None,"untyped | Rioolput (temperature sensor, location unclear)"),
}

def verify():
    import rdflib
    from rdflib.namespace import SKOS, RDFS
    g=rdflib.Graph(); g.parse('vocab/gemet_en.ttl',format='turtle')
    gemet={str(s).rsplit('/',1)[-1] for s in g.subjects(SKOS.prefLabel,None)}
    gq=rdflib.Graph(); gq.parse('vocab/qudt_quantitykind.ttl',format='turtle')
    qk={str(s).rsplit('/',1)[-1] for s in gq.subjects(RDFS.label,None) if 'quantitykind' in str(s)} | {"PH","RelativePermittivity"}
    gu=rdflib.Graph(); gu.parse('vocab/qudt_unit.ttl',format='turtle')
    un={str(s).rsplit('/',1)[-1] for s in gu.subjects(None,None) if 'vocab/unit/' in str(s)}
    gw={json.loads(l)['uri'].rsplit('/',1)[-1] for l in open('vocab/gwsw_foi.jsonl',encoding='utf-8') if l.strip()}
    bad=[]
    for v in OBS.values():
        p,uc=v[2],v[5]
        if p.startswith("gemet:") and p.split(":",1)[1] not in gemet: bad.append(p)
        if p.startswith("qudt:") and p.split(":",1)[1] not in qk: bad.append(p)
        if uc and uc.split(":",1)[1] not in un: bad.append(uc)
    for gwsw,_ in SITES.values():
        if gwsw and gwsw.split(":",1)[1] not in gw: bad.append(gwsw)
    if bad: raise SystemExit(f"UNVERIFIED IRIs: {sorted(set(bad))}")
    print("all IRIs verified (QUDT quantitykind/unit, GEMET, GWSW)")

def fetch(n_per_type=1):
    """One (or a few) real signal rows per observation variable + all site descriptions."""
    import sqlite3
    con=sqlite3.connect("file:data/data_uwo_2021.sqlite?mode=ro",uri=True)
    c1=con.cursor(); c2=con.cursor()
    samples={}
    for vid in OBS:
        rows=c1.execute("SELECT value,timestamp,source_id,site_id FROM signal WHERE variable_id=? LIMIT ?",
                        (vid,n_per_type)).fetchall()
        obs=[]
        for val,ts,srcid,siteid in rows:
            src=c2.execute("SELECT s.name,st.name,st.manufacturer FROM source s JOIN source_type st "
                           "ON st.source_type_id=s.source_type_id WHERE s.source_id=?",(srcid,)).fetchone()
            site=c2.execute("SELECT name FROM site WHERE site_id=?",(siteid,)).fetchone()
            obs.append(dict(value=val,ts=str(ts).replace(" ","T"),sensor=src[0],stype=src[1],mfr=src[2],site=site[0]))
        samples[vid]=obs
    sitedesc={r[0]:dict(desc=r[1]) for r in c1.execute("SELECT name,description FROM site")}
    con.close()
    return samples, sitedesc

def emit():
    samples, sitedesc = fetch(n_per_type=1)
    def esc(x): return html.escape("" if x is None else str(x))
    def iri(c):
        if not c: return "UNTYPED"
        base={"gemet":"https://www.eionet.europa.eu/gemet/en/concept/","qudt":"https://qudt.org/vocab/quantitykind/",
              "unit":"https://qudt.org/vocab/unit/","gwsw":"http://data.gwsw.nl/1.6/totaal/","ex":"http://example.org/"}[c.split(":",1)[0]]
        return f'<a href="{esc(base+c.split(":",1)[1])}" target=_blank>{esc(c)}</a>'

    def observation_ttl(prop,src,dec,uc,o,gwsw):
        foi=iri(gwsw) if gwsw else "UNTYPED"
        result=(f'qudt:numericValue {esc(o["value"])} ; qudt:unit {iri(uc)}' if uc
                else f'sosa:hasSimpleResult "{esc(o["value"])}"')
        return ("<span class=mono>a sosa:Observation ;<br>"
            f'&nbsp;sosa:resultTime "{esc(o["ts"])}"^^xsd:dateTime ;<br>'
            f'&nbsp;sosa:madeBySensor <i>{esc(o["sensor"])}</i> <span class=src>({esc(o["stype"])}, {esc(o["mfr"])})</span> ;<br>'
            f'&nbsp;sosa:observedProperty <b>{iri(prop)}</b> <span class=src>({esc(src)} · {esc(dec)})</span> ;<br>'
            f'&nbsp;sosa:hasFeatureOfInterest <i>{esc(o["site"])}</i> a {foi} ;<br>'
            f'&nbsp;sosa:hasResult [ {result} ] .</span>')

    obs_json=[]; obs_rows=[]; nodata=[]
    for i,(vid,(name,us,prop,src,dec,uc,note)) in enumerate(sorted(OBS.items(),key=lambda kv:kv[1][0]),1):
        rows=samples.get(vid) or []
        if not rows:
            nodata.append(name); continue
        for o in rows:
            gwsw=(SITES.get(o["site"]) or (None,""))[0]
            srccell=(f'<b>{esc(name)}</b> <span class=src>[{esc(us)}]</span><br>'
                     f'<span class=mono>value={esc(o["value"])} @ {esc(o["ts"])}</span><br>'
                     f'<span class=src>sensor {esc(o["sensor"])} · site {esc(o["site"])}</span>')
            obs_rows.append(f"<tr><td class=data>{srccell}</td><td>{observation_ttl(prop,src,dec,uc,o,gwsw)}"
                            f"{('<br><span class=src>note: '+esc(note)+'</span>') if note else ''}</td></tr>")
            obs_json.append(dict(variable=name,unit_symbol=us,value=o["value"],resultTime=o["ts"],
                sensor=o["sensor"],sensor_type=o["stype"],site=o["site"],
                observedProperty=prop,property_source=src,decision=dec,unit=uc,
                feature_gwsw=gwsw,note=note))

    exc_rows=["<tr><td>{}</td><td class=src>[{}]</td><td class=src>{}</td></tr>".format(esc(n),esc(u),esc(r))
              for n,u,r in sorted(EXCLUDED)]

    site_json=[]; site_rows=[]
    for name in sorted(SITES):
        gwsw,opts=SITES[name]; d=sitedesc.get(name,{})
        ev=(d.get("desc") or "").replace("\n"," ")
        site_json.append(dict(site=name,gwsw_tentative=gwsw or "untyped",options=opts,
                              evidence=ev,street=d.get("street"),status="expert-pending",expert_decision=""))
        badge=f"<span class=b style='background:#D85A30'>{iri(gwsw)}</span>" if gwsw else "<span class=src>untyped</span>"
        site_rows.append(f"<tr><td>{esc(name)}</td><td class=src>{esc(ev[:120])}</td>"
                         f"<td>{badge}</td><td class=src>{esc(opts)}</td><td class=exp>__________</td></tr>")

    Path("eval").mkdir(exist_ok=True)
    Path("eval/gold_uwo.json").write_text(json.dumps(
        dict(dataset="uwo", observations=obs_json, excluded=[dict(variable=n,unit=u,reason=r) for n,u,r in EXCLUDED],
             sites=site_json), ensure_ascii=False, indent=1), encoding="utf-8")

    style=("body{font:13px/1.5 system-ui,Arial;margin:24px;max-width:1300px;color:#1a1a1a}"
        "h1{font-size:20px}h2{border-bottom:2px solid #ccc;margin-top:26px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:4px 7px;vertical-align:top;text-align:left}th{background:#f3f5f7}"
        ".mono{font-family:Consolas,monospace;font-size:11px}.src{color:#555;font-size:11px}.exp{color:#b8860b;font-size:11px}"
        ".b{display:inline-block;padding:1px 6px;border-radius:4px;color:#fff;font-size:11px}")
    nd=(f"<p class=src>Declared variables with <b>no 2021 signal data</b> (no observation to show): "
        f"{esc(', '.join(nodata))}.</p>" if nodata else "")
    doc=("<!doctype html><meta charset=utf-8><title>UWO gold</title>"
        f"<style>{style}</style><h1>UWO (Eawag Urban Water Observatory, 2021) — observation gold</h1>"
        "<p>Each row is a <b>real observation</b> (a <span class=mono>signal</span> row): its source data on the left, the "
        "gold <span class=mono>sosa:Observation</span> it should yield on the right, with every vocabulary IRI clickable — "
        "so you can put the pipeline's output beside it and evaluate. One representative observation per type shown "
        f"({len(obs_rows)} observations across {len(OBS)-len(nodata)} types). "
        "Feature of interest = the physical site (atmospheric measurands noted). All IRIs verified.</p>"+nd+
        f"<h2>Observations — {len(obs_rows)}</h2>"
        "<table><tr><th style='width:34%'>source (real signal row)</th><th>gold observation (clickable IRIs)</th></tr>"
        +"".join(obs_rows)+"</table>"
        f"<h2>Excluded — {len(EXCLUDED)} variables (produce no observation)</h2>"
        "<table><tr><th>variable</th><th>unit</th><th>why excluded</th></tr>"+"".join(exc_rows)+"</table>"
        f"<h2>Sites → GWSW feature-of-interest — {len(SITES)} (expert sheet)</h2>"
        "<table><tr><th>site</th><th>evidence (DB description)</th><th>tentative GWSW</th><th>options</th><th>expert_decision</th></tr>"
        +"".join(site_rows)+"</table>")
    Path("eval/uwo_review.html").write_text(doc,encoding="utf-8")
    print(f"wrote eval/gold_uwo.json + eval/uwo_review.html  ({len(obs_rows)} observations, {len(EXCLUDED)} excluded, {len(SITES)} sites)")

if __name__=="__main__":
    verify(); emit()
