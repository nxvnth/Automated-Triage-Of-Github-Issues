"""DVC stage 1 (prepare): raw parquet -> cleaned, redacted, labelled dataset.

    raw issues
      -> taxonomy: labels -> type (src/taxonomy.py)
      -> clean:    code/traceback segmentation, markdown normalisation
      -> redact:   PII -> sentinel tokens, then PROVE it (re-scan must be 0)
      -> filter:   bot-signature issues, exact duplicates
      -> write data/processed/issues_clean.parquet

Run via `dvc repro` (or directly: `python -m src.make_dataset`).
"""
import sys
from pathlib import Path

import pandas as pd

from src.preprocess import PII_PATTERNS, clean_body, redact
from src.taxonomy import to_type

RAW_PATH = Path('github-issues-dataset/github_issues_dataset.parquet')
OUT_PATH = Path('data/processed/issues_clean.parquet')

# Bots are detected by text signature (this dataset has no author column).
# Only ~31 issues match here, but the filter stays: raw GH Archive data
# (the scale-up path) is full of bots.
import re
RE_BOT = re.compile(
    r'dependabot|renovate\s*bot|greenkeeper|snyk[- ]bot'
    r'|automated (?:pull request|issue)'
    r'|this (?:issue|pr) was (?:auto|automatically)',
    re.IGNORECASE)


def main():
    df = pd.read_parquet(RAW_PATH)
    n0 = len(df)
    print(f'raw issues          : {n0}')

    # --- taxonomy ---
    df['type'] = df['labels'].map(to_type)

    # --- clean + redact ---
    res = df['body'].map(clean_body)
    df['body_clean'] = res.map(lambda t: t[0]).map(redact)
    df['n_code_blocks'] = res.map(lambda t: t[1])
    df['n_tracebacks'] = res.map(lambda t: t[2])
    df['title_clean'] = df['title'].fillna('').map(redact)

    # --- prove redaction worked (Stage 6.1): re-scan with the same detectors ---
    scan = df['title_clean'] + '\n' + df['body_clean']
    leftovers = {tok: int(scan.str.contains(rx).sum()) for tok, rx in PII_PATTERNS}
    print(f'PII after redaction : {leftovers}')
    if any(leftovers.values()):
        sys.exit('FATAL: PII survived redaction — refusing to write output.')

    # --- bots ---
    bot_mask = scan.str.contains(RE_BOT)
    df = df[~bot_mask].copy()
    print(f'bots removed        : {int(bot_mask.sum())}')

    # --- exact duplicates (whitespace-normalised clean text) ---
    # Must happen HERE, before the split stage, or near-identical issues land
    # in both train and test and inflate scores.
    norm = ((df['title_clean'] + ' ' + df['body_clean'])
            .str.lower().str.replace(r'\s+', ' ', regex=True).str.strip())
    dup_mask = norm.duplicated(keep='first')
    df = df[~dup_mask].copy()
    print(f'duplicates removed  : {int(dup_mask.sum())}')

    # Final modeling text: title first — densest signal, transformers
    # truncate from the right. (Serving does the same via prepare_text.)
    df['text'] = (df['title_clean'] + '\n' + df['body_clean']).str.strip()

    keep = ['id', 'repo', 'text', 'type', 'priority', 'severity',
            'n_code_blocks', 'n_tracebacks']
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_parquet(OUT_PATH, index=False)
    print(f'{n0} -> {len(df)} issues written to {OUT_PATH}')
    print('type distribution   :', df['type'].value_counts(dropna=False).to_dict())


if __name__ == '__main__':
    main()
