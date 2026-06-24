"""Unify per-run XLT *test* results (raw.csv) into one accuracy table.

Two modes, chosen automatically by what the runs carry:

  AGGREGATE mode -- raw.csv has `fold` AND `seed` columns:
    1. POOL folds within (method, seed, target_lang) -> one accuracy over all
       items (weighted by per-fold n). Folds are item-disjoint partitions, so
       pooling = one big evaluation, never an average-with-std.
    2. Mean +/- std ACROSS seeds within (method, target_lang). Seeds are the
       only independent replication, so std lives only here.
    Output (beside PATH unless --out-prefix given) -- a SINGLE file, one row per
    target_lang. A binary `seen` flag (1 iff the target lang is in the backbone's
    pretraining-seen set, per LANG_GROUPS bloom_seen/aya_seen keyed on the `llm`
    column) follows target_lang. Then ALL <method>_acc columns, ALL <method>_std
    columns, and the shared n_seeds and n_items (pooled item count, e.g. 3 folds
    x 300 = 900):
      <prefix>.csv  target_lang, seen, zero_shot_acc, xpe_acc, spt_acc, dual_acc,
                    xpe_std, spt_std, dual_std, n_seeds, n_items
    Optional --zero-shot-path pools an existing zero-shot run's folds into a
    `zero_shot_acc`/`zero_shot_std` pair (no seed axis -> std is NaN, n_seeds 0).

  FALLBACK mode -- raw.csv has no fold/seed columns (e.g. plain LR-search test
  runs): no aggregation is possible, so emit one column per run instead:
      <prefix>.csv       wide: rows=target_lang, one col per run folder
                         (leading timestamp stripped, e.g. 20260522_170730_xpe -> xpe)

Usage:
    python -m scripts.evals.unify_xlt_test_res \\
        artefacts/evals/xlt_runs/bloom/joshi5/9_bebe_grid_bloomz \\
        --zero-shot-path artefacts/evals/xlt_runs/bloom/zero/0_zero_shot_eval

    python -m scripts.evals.unify_xlt_test_res <path>   # fallback if no folds/seeds

    # BloomZ
    python -m scripts.evals.unify_xlt_test_res \
        artefacts/evals/xlt_runs/bloom/joshi5/11_bebe_grid_bloomz \
        --zero-shot-path /home/bmikaberidze/xpe-exp/artefacts/evals/xlt_runs/bloom/zero/0_zero_shot_eval/20260610_105627_zs_eval_seq

    # Aya
    python -m scripts.evals.unify_xlt_test_res \
        artefacts/evals/xlt_runs/aya/aya_seen/12_bebe_grid_aya \
        --zero-shot-path /home/bmikaberidze/xpe-exp/artefacts/evals/xlt_runs/aya/zero/0_zero_shot_eval/20260508_075457_zs_eval

"""
import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.run_xlt import LANG_GROUPS

TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}_')
SEEDFOLD_RE = re.compile(r'_f\d+_s\d+$')

# Which LANG_GROUPS entry is the pretraining-"seen" set for each backbone (the
# `llm` column in raw.csv). LANG_GROUPS (scripts/run_xlt.py) is the source of
# truth for the actual language lists.
LLM_SEEN_GROUP = {'bloom': 'bloom_seen', 'aya': 'aya_seen'}


def parse_method(run_name: str) -> str:
    """Method tag from a run_name: strip leading timestamp + trailing _f<d>_s<d>."""
    name = TIMESTAMP_RE.sub('', run_name)
    return SEEDFOLD_RE.sub('', name)


def column_name(run_dir: Path) -> str:
    """Run folder's last part with the leading timestamp stripped."""
    return TIMESTAMP_RE.sub('', run_dir.name)


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


def has_fold_seed(df: pd.DataFrame) -> bool:
    """True iff the runs carry usable fold AND seed columns (aggregate mode)."""
    if not {'fold', 'seed'}.issubset(df.columns):
        return False
    return bool(df['fold'].notna().any() and df['seed'].notna().any())


# --- aggregate mode ---------------------------------------------------------

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


# method column order: zero_shot (the baseline) first, then the studied PEFT
# methods, then anything else alphabetically.
METHOD_ORDER = ['zero_shot', 'xpe', 'spt', 'dual']


def _method_sort_key(m: str):
    return (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m)


def to_wide(long: pd.DataFrame) -> pd.DataFrame:
    """One row per target_lang: ALL <method>_acc columns first, then ALL
    <method>_std columns, then the shared n_seeds and n_items (max across
    methods; they coincide on a complete grid). A method gets a _std column only
    if it has a seed axis -- zero_shot has none, so it is acc-only. Returns a
    flat frame with target_lang a column."""
    methods = sorted(long['method'].unique(), key=_method_sort_key)
    acc = long.pivot(index='target_lang', columns='method', values='mean_acc')
    std = long.pivot(index='target_lang', columns='method', values='seed_std')
    out = pd.DataFrame(index=acc.index)
    for m in methods:
        out[f'{m}_acc'] = acc[m]
    for m in methods:
        if std[m].notna().any():  # no seed axis (e.g. zero_shot) -> no std col
            out[f'{m}_std'] = std[m]
    n_seeds = long.pivot(index='target_lang', columns='method', values='n_seeds')
    n_items = long.pivot(index='target_lang', columns='method', values='n_items')
    out['n_seeds'] = n_seeds.max(axis=1).astype(int)
    out['n_items'] = n_items.max(axis=1).astype(int)
    return out.reset_index()


def restrict_to_langs(df: pd.DataFrame, langs) -> pd.DataFrame:
    """Keep only rows whose target_lang is in `langs`."""
    return df[df['target_lang'].isin(set(langs))].reset_index(drop=True)


def backbone(df: pd.DataFrame) -> str:
    """The single backbone (`llm`) the runs were produced with."""
    llms = sorted(df['llm'].dropna().unique())
    if len(llms) != 1:
        raise ValueError(f'expected exactly one llm across runs, got {llms}')
    return llms[0]


def seen_langs_for(llm: str) -> set[str]:
    """Pretraining-seen Belebele langs for a backbone, from LANG_GROUPS."""
    group = LLM_SEEN_GROUP.get(llm)
    if group is None:
        raise ValueError(f'no seen-group mapping for llm {llm!r}; '
                         f'known: {sorted(LLM_SEEN_GROUP)}')
    return set(LANG_GROUPS[group])


def add_seen(wide: pd.DataFrame, seen_langs) -> pd.DataFrame:
    """Insert a binary `seen` column right after target_lang: 1 if the target
    language is in the backbone's pretraining-seen set, else 0."""
    flag = wide['target_lang'].isin(set(seen_langs)).astype(int)
    wide.insert(1, 'seen', flag)
    return wide


def write_aggregate(path: Path, prefix: Path, zero_shot_path: Path | None) -> None:
    grid = collect(path)
    long = seed_mean_std(pool_folds(grid))
    if zero_shot_path is not None:
        # zero-shot is evaluated on all langs; keep only the grid's target langs
        # so every output row has all methods (no zero-shot-only rows).
        zs = restrict_to_langs(pool_zero_shot(collect(zero_shot_path)),
                               long['target_lang'].unique())
        long = pd.concat([long, zs], ignore_index=True)

    long = long.sort_values(['target_lang', 'method'])
    wide = add_seen(to_wide(long), seen_langs_for(backbone(grid)))
    out = f'{prefix}.csv'
    wide.to_csv(out, index=False)

    n = long.groupby('target_lang')['n_items'].max()
    bad = n[n != n.max()]
    if len(bad):
        print(f'warning: {len(bad)} target langs have fewer items than the max '
              f'({int(n.max())}) -- incomplete folds:')
        print(bad.to_string())
    n_methods = long['method'].nunique()
    print(f'[aggregate] wrote {out}  ({wide.shape[0]} langs x {n_methods} methods)')


# --- fallback mode ----------------------------------------------------------

def unify_runs(path: Path, key: str = 'target_lang', value: str = 'accuracy') -> pd.DataFrame:
    """One column per run (no aggregation): rows aligned on `key`."""
    series = []
    for raw in sorted(Path(path).rglob('raw.csv')):
        name = column_name(raw.parent)
        df = pd.read_csv(raw)
        for col in (key, value):
            if col not in df.columns:
                raise ValueError(f"{raw}: missing '{col}' column (has {list(df.columns)})")
        col = df.set_index(key)[value].rename(name)
        if col.index.has_duplicates:
            raise ValueError(f"{raw}: duplicate '{key}' values, cannot align")
        series.append(col)
    if not series:
        raise SystemExit(f'No raw.csv found under {path}')

    seen: dict[str, int] = {}
    for s in series:
        seen[s.name] = seen.get(s.name, 0) + 1
        if seen[s.name] > 1:
            new = f'{s.name}#{seen[s.name]}'
            print(f"warning: duplicate column '{s.name}' -> '{new}'")
            s.rename(new, inplace=True)
    return pd.concat(series, axis=1)


def write_fallback(path: Path, prefix: Path, key: str, value: str) -> None:
    table = unify_runs(path, key=key, value=value)
    out = f'{prefix}.csv'
    table.to_csv(out)
    print(f'[fallback] no fold/seed columns -> one column per run; '
          f'wrote {out}  ({table.shape[0]} rows x {table.shape[1]} cols)')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', type=Path, help='root to walk for test raw.csv files')
    ap.add_argument('--zero-shot-path', type=Path, default=None,
                    help='root to walk for existing zero-shot raw.csv files (aggregate mode only)')
    ap.add_argument('--out-prefix', type=Path, default=None,
                    help='output prefix (default: PATH/test_unified)')
    ap.add_argument('--key', default='target_lang', help='row-alignment column (fallback mode)')
    ap.add_argument('--value', default='accuracy', help='value column (fallback mode)')
    args = ap.parse_args()

    prefix = args.out_prefix or (args.path / 'test_unified')

    if has_fold_seed(collect(args.path)):
        write_aggregate(args.path, prefix, args.zero_shot_path)
    else:
        if args.zero_shot_path is not None:
            print('warning: --zero-shot-path ignored in fallback mode (no fold/seed columns)')
        write_fallback(args.path, prefix, args.key, args.value)


if __name__ == '__main__':
    main()
