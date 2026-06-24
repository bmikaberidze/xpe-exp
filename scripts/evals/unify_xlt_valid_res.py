"""Unify per-run XLT *validation* results (valid_res.csv) into per-fold aggregates.

Each tune run writes a valid_res.csv with the swept hyperparameter already
materialised as a column, e.g.:

    run_name,method,run_group,llm,fold,seed,learning_rate,best_val_acc,...
    spt_lr5e1_s10,spt,9c_...,bloom,1,10,0.5,0.83,...

So this is meta-config agnostic: the LR (or whatever was swept) is read straight
off the CSV column -- no wandb, no run_name parsing, no hardcoded grid.

Aggregation = "means per fold, std across seeds":
  group by (method, <group-key>, fold) -> mean of best_val_acc over seeds,
  std (ddof=1) over seeds, n_seeds. Folds are kept SEPARATE (each fold is its
  own tuning signal); seeds are the replication axis, so std lives there.

Output (beside PATH unless --out given):
  valid_unified.csv  tidy: method, <group-key>, fold, n_seeds, mean_val_acc, seed_std

Usage:
    python -m scripts.evals.unify_xlt_valid_res \\
        artefacts/evals/xlt_runs/bloom/joshi5/9c_lr_search_bebe_bloomz
    python -m scripts.evals.unify_xlt_valid_res <path> --group-by weight_decay
"""
import argparse
from pathlib import Path

import pandas as pd


def collect(path: Path) -> pd.DataFrame:
    """Concatenate every valid_res.csv under PATH (one row per run)."""
    frames = []
    for vr in sorted(Path(path).rglob('valid_res.csv')):
        frames.append(pd.read_csv(vr))
    if not frames:
        raise SystemExit(f'No valid_res.csv found under {path}')
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, group_key: str = 'learning_rate') -> pd.DataFrame:
    """Mean +/- std (ddof=1) across seeds per (method, group_key, fold)."""
    missing = {'method', 'fold', 'seed', 'best_val_acc', group_key} - set(df.columns)
    if missing:
        raise ValueError(f'valid_res rows missing columns: {sorted(missing)} '
                         f'(have {list(df.columns)})')
    df = df.dropna(subset=['best_val_acc'])
    rows = []
    for (method, key, fold), x in df.groupby(['method', group_key, 'fold']):
        rows.append({
            'method': method, group_key: key, 'fold': fold,
            'n_seeds': len(x),
            'mean_val_acc': x['best_val_acc'].mean(),
            'seed_std': x['best_val_acc'].std(ddof=1),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(['method', 'fold', group_key]).reset_index(drop=True)


def _warn_incomplete(out: pd.DataFrame, group_key: str) -> None:
    full = out['n_seeds'].max()
    bad = out[out['n_seeds'] < full]
    if len(bad):
        print(f'warning: {len(bad)} cells have fewer than {full} seeds (incomplete):')
        print(bad[['method', group_key, 'fold', 'n_seeds']].to_string(index=False))


def _print_best(out: pd.DataFrame, group_key: str) -> None:
    print(f'\nbest {group_key} per (method, fold) by mean val acc:')
    for (method, fold), x in out.groupby(['method', 'fold']):
        top = x.loc[x['mean_val_acc'].idxmax()]
        std = top['seed_std']
        std_s = '   nan' if pd.isna(std) else f'{std:.4f}'
        print(f'  {method:<5} fold{int(fold)}  {group_key}={top[group_key]:<8g}  '
              f'mean={top["mean_val_acc"]:.4f} +/- {std_s}  (n={int(top["n_seeds"])})')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', type=Path, help='root to walk for valid_res.csv files')
    ap.add_argument('--group-by', default='learning_rate',
                    help='swept-hyperparameter column to group on (default: learning_rate)')
    ap.add_argument('--out', type=Path, default=None,
                    help='output CSV (default: PATH/valid_unified.csv)')
    args = ap.parse_args()

    out_path = args.out or (args.path / 'valid_unified.csv')
    out = aggregate(collect(args.path), group_key=args.group_by)
    out.to_csv(out_path, index=False)
    _warn_incomplete(out, args.group_by)
    _print_best(out, args.group_by)
    print(f'\nwrote {out_path}  ({len(out)} cells)')


if __name__ == '__main__':
    main()
