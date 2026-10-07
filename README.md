# SemanticKAG

Code for the master's thesis *Semantic Harmonization of Urban Water Monitoring Data into Knowledge Graphs with Retrieval-Grounded Language Models*.

The pipeline turns a folder of monitoring data (CSV or TSV files, plus any documentation that ships with them) into an RDF knowledge graph in the [SOSA](https://www.w3.org/TR/vocab-ssn/) observation model. Observed properties come from [QUDT](https://qudt.org) quantity kinds and [GEMET](https://www.eionet.europa.eu/gemet) concepts, units from QUDT, and features of interest are typed with [GWSW](https://data.gwsw.nl) classes. Because every dataset reuses the same vocabulary identifiers, the same quantity in two datasets ends up as the same node, and graphs from different sources can be merged and queried together.


## How it works

```
profile -> enrich -> retrieve -> resolve -> assemble -> materialize
```

1. **Profile** (`stage1/profiler.py`): detect delimiter, encoding, decimal separator, header rows and the timestamp column of every file, and give each column a role (timestamp, measurement, quality flag, ...).
2. **Enrich** (`stage1/query_enrich.py`): one model call rewrites each measurement column into a short concept phrase, for example `COD2` into "chemical oxygen demand".
3. **Retrieve** (`stage1/property_retrieval.py`): for each column, hybrid dense and BM25 retrieval returns candidate properties from QUDT and from GEMET.
4. **Resolve** (`stage1/resolver.py`): one model call fills a structured form. For each column it picks a candidate property (or mints a new one), the unit and the sensor, and for the dataset it names the sensors and features of interest. Units are then matched to QUDT, and features are typed with a GWSW class by retrieval (`stage1/foi_retrieval.py`).
5. **Assemble** (`stage1/assemble.py`): write a normalized CSV per file and an intermediate representation in which every decision is a URI.
6. **Materialize** (`stage3/`): generate an RML mapping from that representation and run [morph-kgc](https://github.com/morph-kgc/morph-kgc) to produce the graph.

## Setup

Requires Python 3.11 or later (tested with 3.13) and an [Anthropic API key](https://console.anthropic.com).

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

The retrieval index of QUDT and GEMET is included in `chroma_db/`. To rebuild it from the vocabularies in `vocab/`:

```bash
python -m stage2.property_index
```

The embedding models (`BAAI/bge-large-en-v1.5` and `BAAI/bge-m3`) are downloaded from Hugging Face on first use. A GPU is used when available.

## Data

The datasets are not included. A dataset is a folder of CSV or TSV files, in any layout. Readable files next to them (JSON, TXT, MD, XML, YAML, and PDF) are treated as documentation and given to the model as evidence.

The thesis uses three datasets. The evaluation scripts expect them at these paths:

| Dataset | Path |
|---|---|
| Graz-West R05 (sewer hydraulics and pollutants, documented) | `data/02 - Graz-West R05/data-set` |
| FlowBru, Canal Beco station (no documentation) | `data/03 - FlowBru/Water_Quality/Canal_Beco` |
| FlowBru, BT Belliard station (no documentation) | `data/03 - FlowBru/Buffer_Basin_Streams/BT_Belliard` |
| Eawag Urban Water Observatory 2021 (relational) | `data/data_uwo_2021.sqlite` |

The UWO database is relational, so `eval/uwo_adapter.py` first exports it to one folder of wide CSV files per site.

## Usage

```bash
# profile the files (no model call)
python cli.py profile --dataset "data/02 - Graz-West R05/data-set"

# show the resolve prompt and the candidate menus (--no-enrich: no model call)
python cli.py prompt --dataset "data/02 - Graz-West R05/data-set" --no-enrich
python cli.py retrieve --dataset "data/02 - Graz-West R05/data-set" --col water_level --no-enrich

# resolve and print the decisions
python cli.py resolve --dataset "data/02 - Graz-West R05/data-set"

# build the graph of one dataset
python cli.py build --dataset "data/02 - Graz-West R05/data-set" --id graz --workdir out/graz --out graz.ttl

# build one merged graph from several datasets
python cli.py build-multi --pair "data/02 - Graz-West R05/data-set" graz --pair "data/03 - FlowBru/Water_Quality/Canal_Beco" beco --workdir out --out combined.ttl
```

`build` writes `graz.ttl` (Turtle), plus the RML mapping and the normalized CSV files in the work directory. The same functions are available from Python as `build_kg` and `build_kg_multi` in `run.py`.

With the main configuration, a resolve call on the Graz dataset sends about 100,000 input tokens.

## Configuration

`config.toml` holds the main configuration. Every command also takes `--config`, `--model`, `--k`, `--max-rows` and `--enrich` / `--no-enrich`, which override it.

| Setting | Main value | Meaning |
|---|---|---|
| `model` | `claude-sonnet-4-6` | model for the resolve call |
| `temperature` | `0.0` | for both model calls |
| `max_rows` | `1000` | rows per file written to the graph |
| `retrieval.k` | `30` | candidates per vocabulary in each column's menu |
| `retrieval.menu_format` | `lines` | `shared` writes each candidate once, for a shorter prompt |
| `enrichment.enabled` | `true` | the enrichment call before retrieval |
| `resolution.classify_columns` | `true` | lets the model drop derived, diagnostic and quality columns |
| `resolution.model_sampling` | `true` | models files of physical samples as `sosa:Sample` and `sosa:Sampling` |
| `evidence.use_documentation` | `true` | `false` hides the documentation files from every step |

The configurations used in the evaluation are in `eval/configs/`.

## Evaluation

`eval/` scores the decisions of the resolve call (property, unit, sensor, feature) against gold standards built by hand:

```bash
python eval/build_graz_gold.py        # writes eval/gold_graz.json (also build_flowbru_gold.py, build_uwo_gold.py)
python eval/score.py graz             # one resolve call, scored
python eval/score.py graz --config eval/configs/baseline.toml
python eval/run_full_eval.py --run-name t0 --repeats 4
```

Resolved datasets are cached in `eval/cache/`, which is not included in the repository, so running the scripts makes new model calls. The results reported in the thesis are in `eval/results/`. Each script states what it measures and where it writes at the top of the file.

## Repository layout

```
cli.py, run.py        command line and Python entry points
config.py, .toml      configuration
stage1/               profiling, enrichment, retrieval, resolve call, assembly
stage2/               builds the retrieval index
stage3/               RML mapping and materialization
utils/                URI minting and unit matching
vocab/                QUDT, GEMET and GWSW vocabulary files
chroma_db/            retrieval index
eval/                 gold standards, evaluation scripts, configurations and results
```

## Vocabularies

`vocab/` contains copies of the QUDT quantity kind and unit vocabularies, the English GEMET thesaurus, and a slice of 430 GWSW feature classes with English translations of their labels and definitions. They remain subject to the terms of their publishers.

## License

The code is released under the MIT License (see `LICENSE`). The vocabularies in `vocab/` keep their own terms.
