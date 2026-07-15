# Automated Triage of GitHub Issues

NLP system that classifies incoming GitHub issues by **type** (bug / feature / question / docs), predicts **priority/severity**, and suggests an **owner/component** — taken end-to-end from data pipeline to a monitored, governed service.

## Getting the data

The dataset (~139 MB) is **not** committed to this repo. It is the public
[sharjeelyunus/github-issues-dataset](https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset)
on Hugging Face (114k labelled issues). Fetch it into the project root:

```bash
# Option A: clone the dataset repo (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/sharjeelyunus/github-issues-dataset

# Option B: via the datasets library
python -c "from datasets import load_dataset; load_dataset('sharjeelyunus/github-issues-dataset')"
```

The pipeline expects the parquet at `github-issues-dataset/github_issues_dataset.parquet`.
DVC verifies you have the exact right file: `dvc status` compares its hash against
the committed pointer file (`github-issues-dataset/github_issues_dataset.parquet.dvc`).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
```

## Reproducing the data pipeline

Data and preprocessing are versioned with **DVC**: git tracks the *code and
hashes* (`dvc.yaml`, `dvc.lock`, `*.dvc`, `params.yaml`), DVC tracks the *data
content*. One command rebuilds everything from the raw parquet:

```bash
dvc repro
```

Stage graph (`dvc.yaml`):

| Stage | Command | What it does | Output |
| --- | --- | --- | --- |
| `prepare` | `python -m src.make_dataset` | label taxonomy → text cleaning (code/traceback segmentation) → PII redaction **with zero-leftover proof** → bot & duplicate filtering | `data/processed/issues_clean.parquet` |
| `split` | `python -m src.make_splits` | frozen 80/10/10 train/val/test split, stratified by `type` (seed & ratios in `params.yaml`) | `data/splits/{train,val,test}.parquet` |

DVC only re-runs a stage when one of its declared dependencies (raw data, the
`src/` modules, or `params.yaml`) actually changed — and `dvc.lock` records the
hash of every input and output, so any historical model can be traced back to
the exact bytes it was trained on (`git checkout <rev> && dvc checkout`).

Artifact storage is a local DVC remote (`dvc remote list`); on a fresh clone
of this repo on the same machine, `dvc pull` restores the data without re-running
anything.

### Layout

```
src/preprocess.py    clean_body() + redact() — imported by BOTH the pipeline and
                     (later) the FastAPI service, so live inputs get the exact
                     same treatment as training data (no train/serve skew)
src/taxonomy.py      raw repo labels -> type, with documented precedence rules
src/make_dataset.py  DVC stage 1 (prepare)
src/make_splits.py   DVC stage 2 (split)
params.yaml          split seed + ratios (DVC-tracked parameters)
```

### Split design

- **Dedup happens before the split** (in `prepare`), so near-identical issues
  can't sit in both train and test.
- Splits are **stratified on `type`**, with unmapped issues (`type = None`,
  ~38%) kept as their own stratum: type models drop them with one `dropna()`,
  priority/severity models use every row — one split serves all three tasks.
- `test` is touched once, at the end, for the final baseline-vs-transformer
  comparison. Tuning happens on `val`.

## Project status

- [x] EDA — label/priority/severity distributions, text stats (`Untitled.ipynb`)
- [x] Label taxonomy — normalise 40k raw repo labels into `bug/feature/question/docs`
- [x] Text cleaning — code-block/stack-trace segmentation, markdown normalisation
- [x] PII redaction, bot & duplicate filtering
- [x] Reproducible data pipeline — DVC (`dvc repro`), versioned splits
- [ ] Baseline (TF-IDF + linear) vs DistilBERT, tracked in MLflow
- [ ] FastAPI service, Docker, cloud deployment
- [ ] Monitoring (Prometheus/Grafana, Evidently drift) & governance (model card, HITL queue)
