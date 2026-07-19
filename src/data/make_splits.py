"""DVC stage 2 (split): frozen train / val / test splits.

Splitting lives in the DATA pipeline, not the training scripts, so the
TF-IDF baseline and the transformer read byte-identical files — that's what
makes "compare on the held-out split" meaningful.

Stratified on `type`, with unmapped (type=None) issues kept as their own
stratum: type training drops them with one dropna(), priority/severity
training uses every row, and all three tasks share one split.

Ratios and seed come from params.yaml (DVC tracks them; changing either
invalidates and re-runs this stage).
"""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

IN_PATH = Path('data/processed/issues_clean.parquet')
OUT_DIR = Path('data/splits')


def main():
    params = yaml.safe_load(Path('params.yaml').read_text())['split']
    seed, val_frac, test_frac = params['seed'], params['val_frac'], params['test_frac']

    df = pd.read_parquet(IN_PATH)
    strata = df['type'].fillna('unmapped')

    train_val, test = train_test_split(
        df, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(
        train_val, test_size=val_frac / (1 - test_frac),
        stratify=strata.loc[train_val.index], random_state=seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [('train', train), ('val', val), ('test', test)]:
        part.to_parquet(OUT_DIR / f'{name}.parquet', index=False)
        typed = part['type'].notna().sum()
        print(f'{name:5s}: {len(part):6d} rows ({typed} with type) '
              f'-> {OUT_DIR / f"{name}.parquet"}')

    # Sanity: per-class proportions must match across splits (stratification).
    print('\ntype share per split (should be ~equal):')
    shares = pd.DataFrame({
        name: part['type'].value_counts(normalize=True, dropna=False)
        for name, part in [('train', train), ('val', val), ('test', test)]})
    print(shares.round(4))


if __name__ == '__main__':
    main()
