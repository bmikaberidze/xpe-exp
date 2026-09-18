"""Unify per-run XLT *validation* results into per-fold aggregates.

Each run's validation rows carry the entry's own keys as columns, so the swept
hyperparameter is materialised by the group config, e.g. `lr: 0.5` on every entry:

    group,name,index,config,time_id,seed,fold,method,source_group,lr,accuracy,step
    9c_lr_search...,spt_lr5e1_s10,3,spt,20260918_1130,10,1,spt,joshi5,0.5,0.83,400

So this is group-config agnostic: the LR (or whatever was swept) is read straight
off the result rows -- no wandb, no run_name parsing, no hardcoded grid.

Aggregation = "means per fold, std across seeds":
  group by (method, <group-key>, fold) -> mean of best_val_acc over seeds,
  std (ddof=1) over seeds, n_seeds. Folds are kept SEPARATE (each fold is its
  own tuning signal); seeds are the replication axis, so std lives there.

Output (beside PATH unless --out given):
  valid_unified.csv  tidy: method, <group-key>, fold, n_seeds, mean_val_acc, seed_std

Usage:
    python -m scripts.evals.unify_xlt_valid_res \\
        artefacts/runs/groups/9c_lr_search_bebe_bloomz.joshi5
    python -m scripts.evals.unify_xlt_valid_res <path> --group-by weight_decay
"""
import argparse
from pathlib import Path

import pandas as pd


# The tune phase's after-training evaluation of the best checkpoint, written when
# the unit config sets `eval.after_training: true` (0.4 writes no validation rows
# otherwise). `accuracy` here is the same number the old `valid_res.csv` recorded as
# `best_val_acc`: the trainer reloads the best checkpoint before this pass.
VALID_GLOB = 'eval_validation_after_train.csv'


def collect(path: Path) -> pd.DataFrame:
    """Concatenate every run's validation rows under a group directory.

    Reads the 0.4 layout only -- `runs/groups/{group}/{time_id}_{name}/`.
    """
    frames = []
    for vr in sorted(Path(path).rglob(VALID_GLOB)):
        frames.append(pd.read_csv(vr))
    if not frames:
        raise SystemExit(
            f'No {VALID_GLOB} found under {path} -- the tune config needs '
            '`eval.after_training: true`, or the runs predate the 0.4 migration')
    df = pd.concat(frames, ignore_index=True)
    if 'best_val_acc' not in df.columns and 'accuracy' in df.columns:
        df = df.rename(columns={'accuracy': 'best_val_acc'})
    return df


def _with_fold_axis(df: pd.DataFrame) -> pd.DataFrame:
    """Give fold-less runs a single implicit fold 0.

    Belebele runs are folded and pass through untouched. SIB-200 has no fold
    axis, and `run_xlt` writes `fold: None` for it, which reads back as NaN --
    pandas `groupby` then DROPS those rows, leaving an empty frame. Synthesising
    one fold keeps the grouping below identical for both benchmarks.
    """
    if 'fold' in df.columns and df['fold'].notna().any():
        return df
    return df.assign(fold=0)


def aggregate(df: pd.DataFrame, group_key: str = 'learning_rate') -> pd.DataFrame:
    """Mean +/- std (ddof=1) across seeds per (method, group_key, fold).

    `fold` is optional: fold-less benchmarks collapse to a single fold 0.
    """
    missing = {'method', 'seed', 'best_val_acc', group_key} - set(df.columns)
    if missing:
        raise ValueError(f'valid_res rows missing columns: {sorted(missing)} '
                         f'(have {list(df.columns)})')
    df = _with_fold_axis(df)
    df = df.dropna(subset=['best_val_acc'])
    # A crashed run and its completed re-run each write a valid_res.csv for the
    # SAME (method, group_key, fold, seed); collect() concatenates in path order
    # (ascending timestamp), so keep='last' retains the latest = the re-run and
    # counts every seed exactly once. (LR-search runs are unique per lr, so this
    # is a no-op there.)
    before = len(df)
    df = df.drop_duplicates(subset=['method', group_key, 'fold', 'seed'], keep='last')
    dropped = before - len(df)
    if dropped:
        # A crashed run and its re-run legitimately collide. But if MOST rows
        # collapse, the group key is almost certainly constant -- e.g. grouping
        # on `learning_rate` while the sweep actually varied `prompt_lr` -- and
        # the result would silently be an average over a near-arbitrary subset.
        msg = (f'note: dropped {dropped}/{before} duplicate rows on '
               f"['method', {group_key!r}, 'fold', 'seed'] (crashed runs + re-runs)")
        if dropped > before / 2:
            raise ValueError(
                f'{msg}\n'
                f'That is more than half the rows, so {group_key!r} is probably CONSTANT '
                f'across these runs and is the wrong --group-by column. Columns present: '
                f'{sorted(df.columns)}. If the sweep varied the prompt-embedding LR, use '
                f'--group-by prompt_lr.')
        print(msg)
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
    ap.add_argument('path', type=Path, help='group directory to walk for validation results')
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
