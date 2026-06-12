"""Aggregate Belebele self-split fold XLT runs into unified per-target-lang results.

Walks PATH for `raw.csv` files (one per method/fold/seed cell), then:
  1. POOLS folds within (method, seed, target_lang) -> one accuracy over all
     900 items (weighted by per-fold n). Folds are item-disjoint partitions,
     so pooling = one 900-item evaluation, never an average-with-std.
  2. Takes mean +/- std ACROSS seeds within (method, target_lang). Seeds are
     the only independent replication, so std lives only here.

Optional --zero-shot-path pools an existing zero-shot run's folds into a
`zero_shot` column (no seed axis).

Outputs (beside PATH unless --out-prefix given):
  <prefix>.csv       wide: rows=target_lang, cols=method, cell=pooled-900 mean
  <prefix>_std.csv   wide: same shape, seed std (ddof=1)
  <prefix>_long.csv  tidy: target_lang, method, mean_acc, seed_std, n_seeds, n_items

Usage:
    python -m scripts.evals.unify_bebe_folds \\
        artefacts/evals/xlt_runs/bloom/joshi5/9_bebe_grid_bloomz \\
        --zero-shot-path artefacts/evals/xlt_runs/bloom/zero/0_zero_shot_eval
"""
import argparse
import re
from pathlib import Path

import pandas as pd

TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}_')
SEEDFOLD_RE = re.compile(r'_f\d+_s\d+$')


def parse_method(run_name: str) -> str:
    """Method tag from a run_name: strip leading timestamp + trailing _f<d>_s<d>."""
    name = TIMESTAMP_RE.sub('', run_name)
    return SEEDFOLD_RE.sub('', name)


def collect(path: Path) -> pd.DataFrame:
    """Concatenate every raw.csv under PATH, adding a parsed `method` column."""
    frames = []
    for raw in sorted(Path(path).rglob('raw.csv')):
        df = pd.read_csv(raw)
        df['method'] = df['run_name'].map(parse_method)
        frames.append(df)
    if not frames:
        raise SystemExit(f'No raw.csv found under {path}')
    return pd.concat(frames, ignore_index=True)


def _weighted(x: pd.DataFrame) -> float:
    return (x['accuracy'] * x['n']).sum() / x['n'].sum()


def pool_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Pool folds -> one acc per (method, seed, target_lang) over sum(n) items."""
    df = df.dropna(subset=['accuracy', 'n'])
    rows = []
    for (method, seed, lang), x in df.groupby(['method', 'seed', 'target_lang']):
        rows.append({
            'method': method, 'seed': seed, 'target_lang': lang,
            'acc': _weighted(x), 'n_items': int(x['n'].sum()),
            'n_folds': x['fold'].nunique(),
        })
    return pd.DataFrame(rows)


def seed_mean_std(pooled: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std (ddof=1) across seeds per (method, target_lang)."""
    rows = []
    for (method, lang), x in pooled.groupby(['method', 'target_lang']):
        rows.append({
            'method': method, 'target_lang': lang,
            'mean_acc': x['acc'].mean(),
            'seed_std': x['acc'].std(ddof=1),
            'n_seeds': len(x),
            'n_items': int(x['n_items'].max()),
        })
    return pd.DataFrame(rows)


def pool_zero_shot(df: pd.DataFrame) -> pd.DataFrame:
    """Pool zero-shot folds -> a `zero_shot` row per target_lang (no seed axis)."""
    df = df.dropna(subset=['accuracy', 'n'])
    rows = []
    for lang, x in df.groupby('target_lang'):
        rows.append({
            'method': 'zero_shot', 'target_lang': lang,
            'mean_acc': _weighted(x), 'seed_std': float('nan'),
            'n_seeds': 0, 'n_items': int(x['n'].sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', type=Path, help='root to walk for grid raw.csv files')
    ap.add_argument('--zero-shot-path', type=Path, default=None,
                    help='root to walk for existing zero-shot raw.csv files')
    ap.add_argument('--out-prefix', type=Path, default=None,
                    help='output prefix (default: PATH/unified_folds)')
    args = ap.parse_args()

    prefix = args.out_prefix or (args.path / 'unified_folds')
    long = seed_mean_std(pool_folds(collect(args.path)))

    if args.zero_shot_path is not None:
        zs = pool_zero_shot(collect(args.zero_shot_path))
        long = pd.concat([long, zs], ignore_index=True)

    long = long.sort_values(['target_lang', 'method'])
    wide = long.pivot(index='target_lang', columns='method', values='mean_acc')
    std = long.pivot(index='target_lang', columns='method', values='seed_std')

    long.to_csv(f'{prefix}_long.csv', index=False)
    wide.to_csv(f'{prefix}.csv')
    std.to_csv(f'{prefix}_std.csv')
    n = long.groupby('target_lang')['n_items'].max()
    bad = n[n != 900]
    if len(bad):
        print(f'warning: {len(bad)} target langs have n_items != 900 (incomplete folds):')
        print(bad.to_string())
    print(f'wrote {prefix}.csv / _std.csv / _long.csv  ({wide.shape[0]} langs x {wide.shape[1]} methods)')


if __name__ == '__main__':
    main()
