# AUTOMATED TRIAGE

## 01 - YOUR BRIEF - The job, in one paragraph.

You are the ML team for a software company that maintains popular open-source projects. Maintainers are drowning in incoming GitHub issues. Build an NLP system that, for each new issue, (1) classifies its type (bug / feature / question / docs), (2) predicts its priority or severity, and (3) suggests which team or component should own it - using the free-text title and body plus structured signals like labels. Then take it all the way to a running, monitored, governed service.

### THE POINT OF THIS PROJECT

- You are not just fine-tuning a classifier. You are building and defending a complete, production-style NLP system. The modeling is roughly 20% of your grade; the other 80% is everything around it.

### Why this is harder than it looks

- The text is genuinely messy. Issues mix prose with code blocks, stack traces, logs and markdown - nothing like clean benchmark text.
- Labels are noisy and inconsistent. Every repo labels differently; you must normalise a taxonomy and confront label noise.
- It's multi-class and imbalanced. "Bug" swamps "docs"; macro-Fl matters far more than accuracy.
- New categories appear over time. A class you never trained on shows up in production - your monitoring has to notice.
- User-submitted text contains PII. Emails, tokens, names - handling them is part of governance, not an afterthought.

## 02- THE DATA- Where to get it & what's inside.

Start with a ready-labelled set to get moving fast, then optionally scale up to raw data you assemble yourself (which makes your corpus unique and harder to copy).

RECOMMENDED PRIMARY HUGGING FACE

sharjeelyunus/github-issues-dataset

114k issues. type + severity + priority labels . load_dataset(...)

ACADEMIC BENCHMARK NLBSE'23

Issue Report Classification (1.2M issues)

[github.com/nlbse2023/issue-report-classification](http://github.com/nlbse2023/issue-report-classification)

NLBSE ON HUGGING FACE USE THE SQLITE VERSION

NLBSE/SkillCompetition

[huggingface.co/datasets/NLBSE/SkillCompetition](http://huggingface.co/datasets/NLBSE/SkillCompetition)

SCALE-UP (ADVANCED) BUILD YOUR OWN CORPUS

GH Archive - every public GitHub event

[gharchive.org](http://gharchive.org/) queryable free via BigQuery

114k+

- labelled issues in the primary dataset

3-4

- core type classes: bug/ feature/question/docs

1.2M

- issues in the NLBSE benchmark
- training set

Mixed

- prose + code + stack traces in onefield

Imbalanced

- some classes far rarer than others

Free

- all sources open; GH Archive via BigQuery free tier

Known traps - you'll lose marks if you miss these

- Separate code from prose. Code fences and stack traces should be detected/handled, not fed in raw as if they were sentences.
- Normalise the labels. Repo labels are inconsistent ("bug" vs "type:bug" vs "defect"). Map them to one taxonomy and document it.
- Filter bots and duplicates. Bot-generated issues and near-duplicates will inflate your scores dishonestly.
- Redact Pll early. Emails, access tokens, usernames appear in issue text - strip them in the pipeline.
- Don't report accuracy on imbalanced classes. Use macro-Fl and per- class metrics.

## 03- WHAT YOU MUST DO Stage by stage.

Work through these in order. The checklist is the minimum, not the ceiling.

### STAGE 1 Data Acquisition & Engineering

1. Start with the labelled Hugging Face dataset; optionally pull raw issues from GH Archive via BigQuery to build a larger corpus.
2. Clean text: detect/segment code blocks and stack traces, normalise markdown, lowercase where appropriate.
3. Redact PII (emails, tokens, usernames) and filter bots + duplicates.
4. Version your data & preprocessing (DVC or a documented re- runnable script).

### STAGE 2 Labels & Taxonomy

1. Normalise inconsistent repo labels into one shared set of classes.
2. Decide single-label vs multi-label and document the choice.
3. Quantify label noise - and say how you handled it.

### STAGE 3 Modeling & Evaluation

1. Build a baseline first: TF-IDF + a linear classifier. This is the bar to beat.
2. Fine-tune a transformer (DistilBERT or similar) and compare on the e held-out split.
3. Evaluate with macro-F1, per-class precision/recall, and a confusion matrix- not accuracy.
4. Analyse where the transformer helps and where it doesn't justify its cost/latency.
5. Track every experiment in MLflow.

### STAGE 4 Optimise, Package & Deploy

1. Optimise serving: batching, quantization, or export to ONNX to cut latency; measure the latency/accuracy trade-off.
2. Wrap the model in a FastAPI service returning predicted type, priority, suggested owner, and confidence.
3. Containerize with Docker; run locally with Docker Compose.
4. Deploy to AWS or GCP (see Section 5); document a rollback plan.

### STAGE 5 Observability & Monitoring

1. Service metrics: latency, throughput, error rate via Prometheus + Grafana (and/or CloudWatch /Cloud Monitoring)
2. Model metrics: log predictions and track the confidence distribution; flag low-confidence outputs.
3. Drift: monitor embedding / text drift and prediction drift with Evidently; detect out-of-distribution / new-intent inputs.
4. Alerting + retrain trigger: define when s notified and when the model should be retrained.

### STAGE 6 Governance & Re-evaluation

1. PII handling: prove your redaction works and document what's stored vs discarded.
2. Bias audit: check routing isn't systematically wrong for certain projects/components.
3. Human-in-the-loop: route low-confidence predictions to a review queue instead of auto-acting.
4. Explainability + model card: show why an issue was classified as it s; document intended use, data, performance, limits.
5. Auditability & lineage: log every prediction with model version; track data/model versions in MLflow.

## 04 TOOLS YOU CAN USE - All free, all open source.

| STAGE | RECOMENDED TOOLS |
| --- | --- |
| Data engineering | pandas, Hugging Face datasets, BigQuery (free tier) for GH Archive, regex/PII redaction, DVC |
| Modeling | scikit-learn (TF-IDF baseline), Hugging Face Transformers (DistilBERT), MLflow |
| Serving | FastAPI, ONNX Runtime / quantization, Docker, Docker Compose |
| Deployment | Docker + registry (Amazon ECR / Google Artifact Registry), then AWS ECS Fargate (Express Mode) or Lambda, or GCP Cloud Run |
| Observability | Prometheus + Grafana, Evidently AI
(embedding/prediction drift); cloud-native: CloudWatch / Cloud Monitoring |
| Governance | PII redaction, Fairlearn, human-in-the-loop review queue, SHAP/attention attributions, model card |

Note: we are not using Hugging Face Spaces / Gradio here -you covered that in class. Deploy to AWS or GCP instead.

## 05 - HOW TO DEPLOY (AWS & GCP) From your laptop to a live cloud service.

Build the container once, then deploy to AWS or GCP (do at least one; both earns bonus credit and teaches portability). Transformers are heavier than tabular models - mind image size and cold starts.

1. Export your model (and tokenizer); write a FastAPI/predict endpoint returning type, priority, owner, and confidence.
2. Optimise first - quantize or export to ONNX- so the container is smaller and faster to start.
3. Write a Dockerfile; run it locally; then add Docker Compose with Prometheus + Grafana.
4. Push the image to Amazon ECR or Google Artifact Registry.
5. Deploy (pick a cloud -see below).
6. Add an Evidently drift report (embedding + prediction drift) and surface low-confidence/OOD inputs.
7. Document rollback: keep the previous image tag and state how you'd redeploy

### GCP PATH Google Cloud

- Cloud Run - deploy the container directly; scales to zero; allocate enough memory for the transformer. Best default.
- Artifact Registry for the image; Cloud Build to build it.
- Cloud Monitoring + Logging for metrics & logs.
- Vertex Al (optional, advanced) - managed endpoints + model monitoring.
- New accounts get $300 in credits for 90 days.

### PATH Amazon Web Services

- Amazon ECS on Fargate (Express Mode for the simplest setup) - run the container serverlessly. Current recommended path.
- AWS Lambda (container image) - viable for lighter/quantized models, pay-per-request.
- Amazon ECR for the registry.
- CloudWatch for metrics/logs/alarms; SageMaker Model Monitor (optional) for managed drift.
- 12-month free tier + always-free allowances on several services.

HEADS-UP - DON'T USE APP RUNNER

AWS App Runner is closed to new customers as of April 30, 2026. Use ECS on Fargate (Express Mode) or Lambda instead.

FREE-TIER NOTE

Every service has a free option or allowance, but limits and terms change - verify on each provider's pricing page, and set a billing alert. Never commit secrets, keys, or real credentials to your repo.

## 06 WHAT YOU HAND IN - Not a notebook. A system.

- A Git repository with a README that reproduces everything from raw data to running service.
- A reproducible data + training pipeline (with PIl redaction and label normalisation) and version tracking.
- Baseline + transformer models compared in MLflow with macro-Fl and per-class metrics, plus the latency/accuracy trade-off.
- A Dockerized FastAPI service + docker-compose, and a live (or one- command-deployable) endpoint on AWS or GCP,
- An observability setup - metrics, Grafana dashboard, Evidently drift + OOD detection - with a retraining trigger.
- Governance artefacts - PII handling evidence, bias audit, human-in-the- loop queue, explainability, audit log, model card.
- A 1-2 page reflection - key trade-offs, what you'd change in production, the limits of your model.

What "done" looks like: someone clones your repo, follows the README, rebuilds the corpus, retrains the model, launches the service in one or two commands, posts a messy real issue and gets back a sensible type + priority + owner with a confidence score, opens the dashboard, and reads your model card to understand the limits - without asking you anything.