"""Stage 3.1 — the baseline to beat: TF-IDF + logistic regression on `type`.

Trains on data/splits/train.parquet, selects the regularisation strength on
val, and logs every run to MLflow (params, macro-F1, per-class metrics,
confusion matrix, and data lineage: git SHA + dvc.lock hash of the splits).

The TEST split is not touched here — it's reserved for the one final
baseline-vs-transformer comparison.

Run: python -m src.models.train_baseline
"""
import subprocess
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)
from sklearn.pipeline import Pipeline

SPLIT_DIR = Path('data/splits')
MODEL_PATH = Path('models/baseline_type.joblib')
C_GRID = [0.25, 1.0, 4.0]


def lineage() -> dict:
    """Pin this run to exact code + data versions."""
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'],
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'],
                           capture_output=True, text=True).stdout.strip() != ''
    lock = yaml.safe_load(Path('dvc.lock').read_text())
    split_hash = next(o['md5'] for o in lock['stages']['split']['outs']
                      if o['path'] == 'data/splits')
    return {'git_sha': sha, 'git_dirty': dirty, 'splits_md5': split_hash}


def load_split(name: str):
    df = pd.read_parquet(SPLIT_DIR / f'{name}.parquet').dropna(subset=['type'])
    return df['text'], df['type']


def main():
    X_train, y_train = load_split('train')
    X_val, y_val = load_split('val')
    print(f'train: {len(X_train)}  val: {len(X_val)}')

    lin = lineage()
    mlflow.set_experiment('type-baseline')

    best = {'f1': -1.0}
    for C in C_GRID:
        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, max_features=300_000)),
            # class_weight='balanced': docs/question are rare and macro-F1
            # weights all classes equally, so the loss should too.
            ('clf', LogisticRegression(C=C, class_weight='balanced',
                                       max_iter=2000)),
        ])
        with mlflow.start_run(run_name=f'tfidf-logreg-C{C}'):
            mlflow.log_params({'model': 'tfidf+logreg', 'C': C,
                               'ngram_range': '1-2', 'min_df': 2,
                               'max_features': 300_000,
                               'class_weight': 'balanced', **lin})
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_val)

            macro_f1 = f1_score(y_val, pred, average='macro')
            report = classification_report(y_val, pred, digits=3)
            mlflow.log_metric('val_macro_f1', macro_f1)
            for cls, m in classification_report(
                    y_val, pred, output_dict=True).items():
                if isinstance(m, dict) and 'f1-score' in m:
                    mlflow.log_metric(f'val_f1_{cls}', m['f1-score'])
            mlflow.log_text(report, 'classification_report.txt')

            labels = sorted(y_val.unique())
            cm = pd.DataFrame(confusion_matrix(y_val, pred, labels=labels),
                              index=[f'true_{c}' for c in labels],
                              columns=[f'pred_{c}' for c in labels])
            mlflow.log_text(cm.to_string(), 'confusion_matrix.txt')

            print(f'\n=== C={C}  val macro-F1: {macro_f1:.4f} ===')
            print(report)
            if macro_f1 > best['f1']:
                best = {'f1': macro_f1, 'C': C, 'pipe': pipe, 'cm': cm}

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best['pipe'], MODEL_PATH)
    print(f"\nBest: C={best['C']}  val macro-F1={best['f1']:.4f} "
          f'-> {MODEL_PATH}')
    print('\nConfusion matrix (best run):')
    print(best['cm'].to_string())


if __name__ == '__main__':
    main()
