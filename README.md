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

The notebook expects the parquet at `github-issues-dataset/github_issues_dataset.parquet`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install pandas pyarrow matplotlib seaborn jupyter
```

## Project status

- [x] EDA — label/priority/severity distributions, text stats (`Untitled.ipynb`)
- [x] Label taxonomy — normalise 40k raw repo labels into `bug/feature/question/docs`
- [ ] Text cleaning — code-block/stack-trace segmentation, markdown normalisation
- [ ] PII redaction, bot & duplicate filtering
- [ ] Reproducible data pipeline
- [ ] Baseline (TF-IDF + linear) vs DistilBERT, tracked in MLflow
- [ ] FastAPI service, Docker, cloud deployment
- [ ] Monitoring (Prometheus/Grafana, Evidently drift) & governance (model card, HITL queue)
