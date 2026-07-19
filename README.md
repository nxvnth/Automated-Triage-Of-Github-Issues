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
| `prepare` | `python -m src.data.make_dataset` | label taxonomy → text cleaning (code/traceback segmentation) → PII redaction **with zero-leftover proof** → bot & duplicate filtering | `data/processed/issues_clean.parquet` |
| `split` | `python -m src.data.make_splits` | frozen 80/10/10 train/val/test split, stratified by `type` (seed & ratios in `params.yaml`) | `data/splits/{train,val,test}.parquet` |

DVC only re-runs a stage when one of its declared dependencies (raw data, the
`src/` modules, or `params.yaml`) actually changed — and `dvc.lock` records the
hash of every input and output, so any historical model can be traced back to
the exact bytes it was trained on (`git checkout <rev> && dvc checkout`).

Artifact storage is a local DVC remote (`dvc remote list`); on a fresh clone
of this repo on the same machine, `dvc pull` restores the data without re-running
anything.

### Daily rhythm

After changing anything the pipeline depends on (code in `src/`, `params.yaml`,
or the raw data):

```bash
dvc repro      # rebuild only the stages whose inputs changed
dvc push       # send new data versions to the DVC remote
git add . && git commit && git push   # record the new hashes (dvc.lock) with the code
```

`dvc push` goes before `git push` so anyone checking out the commit can
immediately `dvc pull` the matching data. A commit's `dvc.lock` pins the exact
hash of every pipeline input and output, so any model is traceable to the
exact bytes it was trained on: `git checkout <rev> && dvc checkout`.

### Layout

```
src/
  data/                  everything that produces the datasets
    preprocess.py        clean_body() + redact() — imported by BOTH the pipeline
                         and (later) the FastAPI service, so live inputs get the
                         exact same treatment as training data (no train/serve skew)
    taxonomy.py          raw repo labels -> type, with documented precedence rules
    make_dataset.py      DVC stage 1 (prepare)
    make_splits.py       DVC stage 2 (split)
  models/                everything that trains/evaluates models
    train_baseline.py    TF-IDF + logistic regression baseline (MLflow-tracked)
params.yaml              split seed + ratios (DVC-tracked parameters)
models/                  trained model artifacts (gitignored)
mlruns/                  MLflow tracking store (gitignored)
```

Dependencies are **pinned** in `requirements.txt`: an unpinned install once
silently downgraded pandas/pyarrow and changed parquet hashes (content was
proven identical via the DVC cache — only writer metadata differed).

### Split design

- **Dedup happens before the split** (in `prepare`), so near-identical issues
  can't sit in both train and test.
- Splits are **stratified on `type`**, with unmapped issues (`type = None`,
  ~38%) kept as their own stratum: type models drop them with one `dropna()`,
  priority/severity models use every row — one split serves all three tasks.
- `test` is touched once, at the end, for the final baseline-vs-transformer
  comparison. Tuning happens on `val`.

## Modeling

### Baseline: TF-IDF + logistic regression (`type` task)

```bash
python -m src.models.train_baseline   # trains, evaluates on val, logs to MLflow
mlflow ui                             # browse runs at http://localhost:5000 (local, no account)
```

Word+bigram TF-IDF (min_df=2, sublinear TF, 300k features) into logistic
regression with `class_weight='balanced'` (macro-F1 weights all classes
equally, so the training loss should too). C ∈ {0.25, 1, 4} selected on the
**val** split; **test stays sealed** until the final baseline-vs-transformer
comparison. Every run logs params, metrics, the classification report, the
confusion matrix, and lineage (git SHA + `dvc.lock` hash of the splits).

Best run (C=4), validation split:

| metric | value |
| --- | --- |
| **macro-F1** | **0.651** |
| accuracy | 0.829 |
| bug F1 | 0.872 |
| feature F1 | 0.852 |
| docs F1 | 0.629 |
| question F1 | 0.249 |

The gap between accuracy (0.83) and macro-F1 (0.65) is exactly why the brief
bans accuracy as the headline metric. The baseline's weak spot is `question`
(rarest class, and lexically similar to bug reports — the confusion matrix
shows questions absorbed into `bug`); that's the gap the transformer needs to
justify its cost on.

## Project status

- [x] EDA — label/priority/severity distributions, text stats (`Untitled.ipynb`)
- [x] Label taxonomy — normalise 40k raw repo labels into `bug/feature/question/docs`
- [x] Text cleaning — code-block/stack-trace segmentation, markdown normalisation
- [x] PII redaction, bot & duplicate filtering
- [x] Reproducible data pipeline — DVC (`dvc repro`), versioned splits
- [x] Baseline (TF-IDF + linear, val macro-F1 0.651) — tracked in MLflow
- [ ] DistilBERT fine-tune on the same frozen splits, compared on test
- [ ] FastAPI service, Docker, cloud deployment
- [ ] Monitoring (Prometheus/Grafana, Evidently drift) & governance (model card, HITL queue)
