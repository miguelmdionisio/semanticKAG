"""RML mapping and declarations graph, as a pure template over the IR.

Per observable column and row:
    <obs/...> a sosa:Observation ; sosa:resultTime ; sosa:madeBySensor ; sosa:observedProperty ;
        sosa:hasFeatureOfInterest ; sosa:hasResult <result/...>   (sosa:hasSimpleResult without a unit)
    <result/...> a qudt:QuantityValue ; qudt:numericValue ; qudt:unit .
In a sample-based file every row is also one sosa:Sample (isSampleOf the feature) produced by
one sosa:Sampling, and the observations' feature of interest is that sample.
"""
from __future__ import annotations
from pathlib import Path

from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS

from stage1.models import ResolvedDataset, ResolvedFile, ObservationMap
from utils.uri_policy import (obs_template, result_template, sample_template,
                              sampling_template, _slug)

SOSA = Namespace("http://www.w3.org/ns/sosa/")

_PREFIXES = """\
@prefix rr:   <http://www.w3.org/ns/r2rml#> .
@prefix rml:  <http://semweb.mmlab.be/ns/rml#> .
@prefix ql:   <http://semweb.mmlab.be/ns/ql#> .
@prefix sosa: <http://www.w3.org/ns/sosa/> .
@prefix qudt: <http://qudt.org/schema/qudt/> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
"""


def _q(s: str) -> str:
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _source_block(csv_path: str) -> str:
    # forward slashes, so Windows paths need no escaping
    posix = Path(csv_path).as_posix()
    return f'rml:logicalSource [ rml:source {_q(posix)} ; rml:referenceFormulation ql:CSV ]'


def _time_pom(f: ResolvedFile) -> str:
    dt = ' ; rr:datatype xsd:dateTime' if f.timestamp_is_iso else ''
    return (f'  rr:predicateObjectMap [ rr:predicate sosa:resultTime ;\n'
            f'      rr:objectMap [ rml:reference {_q(f.timestamp_col)}{dt} ] ]')


def _sample_blocks(ds_id: str, f: ResolvedFile) -> list[str]:
    """One Sample and one Sampling per row; the row timestamp is the Sampling's resultTime."""
    s = f.sampling
    sample, sampling = sample_template(ds_id, f.file_id), sampling_template(ds_id, f.file_id)
    tm = _slug(f.file_id)

    sample_poms = [
        f'  rr:predicateObjectMap [ rr:predicate sosa:isResultOf ;\n'
        f'      rr:objectMap [ rr:template "{sampling}" ; rr:termType rr:IRI ] ]']
    if s.feature_uri:
        sample_poms.insert(0,
            f'  rr:predicateObjectMap [ rr:predicate sosa:isSampleOf ;\n'
            f'      rr:objectMap [ rr:constant <{s.feature_uri}> ] ]')

    sampling_poms = [
        f'  rr:predicateObjectMap [ rr:predicate sosa:hasResult ;\n'
        f'      rr:objectMap [ rr:template "{sample}" ; rr:termType rr:IRI ] ]']
    if s.sampler_uri:
        sampling_poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:madeBySampler ;\n'
            f'      rr:objectMap [ rr:constant <{s.sampler_uri}> ] ]')
    if s.feature_uri:
        sampling_poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:hasFeatureOfInterest ;\n'
            f'      rr:objectMap [ rr:constant <{s.feature_uri}> ] ]')
    if f.timestamp_col:
        sampling_poms.insert(0, _time_pom(f))

    def block(name: str, subj: str, cls: str, poms: list[str]) -> str:
        return (f'<#{name}> a rr:TriplesMap ;\n'
                f'  {_source_block(f.source_csv)} ;\n'
                f'  rr:subjectMap [ rr:template "{subj}" ; rr:class {cls} ] ;\n'
                + " ;\n".join(poms) + " .\n")

    return [block(f"sample__{tm}", sample, "sosa:Sample", sample_poms),
            block(f"sampling__{tm}", sampling, "sosa:Sampling", sampling_poms)]


def _observation_block(ds_id: str, f: ResolvedFile, o: ObservationMap) -> str:
    tm = f"obs__{_slug(f.file_id)}__{_slug(o.value_col)}"
    subj = obs_template(ds_id, f.file_id, o.value_col)
    poms = [
        f'  rr:predicateObjectMap [ rr:predicate sosa:madeBySensor ;\n'
        f'      rr:objectMap [ rr:constant <{o.sensor_uri}> ] ]',
        f'  rr:predicateObjectMap [ rr:predicate sosa:observedProperty ;\n'
        f'      rr:objectMap [ rr:constant <{o.property_uri}> ] ]',
    ]
    if f.sampling is not None:
        poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:hasFeatureOfInterest ;\n'
            f'      rr:objectMap [ rr:template "{sample_template(ds_id, f.file_id)}" ; rr:termType rr:IRI ] ]')
    elif o.foi_uri:
        poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:hasFeatureOfInterest ;\n'
            f'      rr:objectMap [ rr:constant <{o.foi_uri}> ] ]')
    if f.timestamp_col:
        poms.insert(0, _time_pom(f))

    if o.unit_uri:
        result_subj = result_template(ds_id, f.file_id, o.value_col)
        poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:hasResult ;\n'
            f'      rr:objectMap [ rr:template "{result_subj}" ; rr:termType rr:IRI ] ]')
    else:
        poms.append(
            f'  rr:predicateObjectMap [ rr:predicate sosa:hasSimpleResult ;\n'
            f'      rr:objectMap [ rml:reference {_q(o.value_col)} ; rr:datatype {o.value_dtype} ] ]')

    return (
        f'<#{tm}> a rr:TriplesMap ;\n'
        f'  {_source_block(f.source_csv)} ;\n'
        f'  rr:subjectMap [ rr:template "{subj}" ; rr:class sosa:Observation ] ;\n'
        + " ;\n".join(poms) + " .\n"
    )


def _quantityvalue_block(ds_id: str, f: ResolvedFile, o: ObservationMap) -> str:
    tm = f"qv__{_slug(f.file_id)}__{_slug(o.value_col)}"
    subj = result_template(ds_id, f.file_id, o.value_col)
    return (
        f'<#{tm}> a rr:TriplesMap ;\n'
        f'  {_source_block(f.source_csv)} ;\n'
        f'  rr:subjectMap [ rr:template "{subj}" ; rr:class qudt:QuantityValue ] ;\n'
        f'  rr:predicateObjectMap [ rr:predicate qudt:numericValue ;\n'
        f'      rr:objectMap [ rml:reference {_q(o.value_col)} ; rr:datatype {o.value_dtype} ] ] ;\n'
        f'  rr:predicateObjectMap [ rr:predicate qudt:unit ;\n'
        f'      rr:objectMap [ rr:constant <{o.unit_uri}> ] ] .\n'
    )


def build_mapping(resolved: ResolvedDataset) -> str:
    blocks: list[str] = [_PREFIXES]
    for f in resolved.files:
        if f.sampling is not None:
            blocks.extend(_sample_blocks(resolved.id, f))
        for o in f.observations:
            blocks.append(_observation_block(resolved.id, f, o))
            if o.unit_uri:
                blocks.append(_quantityvalue_block(resolved.id, f, o))
    return "\n".join(blocks)


def build_declarations(resolved: ResolvedDataset) -> Graph:
    """Typed, labelled sensor, property, feature and sampler nodes."""
    g = Graph()
    g.bind("sosa", SOSA)
    g.bind("rdfs", RDFS)
    for s in resolved.sensors:
        u = URIRef(s.uri)
        g.add((u, RDF.type, SOSA.Sensor))
        if s.label:
            g.add((u, RDFS.label, Literal(s.label)))
    for p in resolved.properties:
        u = URIRef(p.uri)
        g.add((u, RDF.type, SOSA.ObservableProperty))
        if p.label:
            g.add((u, RDFS.label, Literal(p.label, lang="en")))
    for feat in resolved.features:
        u = URIRef(feat.uri)
        g.add((u, RDF.type, SOSA.FeatureOfInterest))
        if feat.gwsw_class_uri:
            g.add((u, RDF.type, URIRef(feat.gwsw_class_uri)))
        if feat.label:
            g.add((u, RDFS.label, Literal(feat.label)))
    for s in resolved.samplers:
        u = URIRef(s.uri)
        g.add((u, RDF.type, SOSA.Sampler))
        if s.label:
            g.add((u, RDFS.label, Literal(s.label)))
    return g
