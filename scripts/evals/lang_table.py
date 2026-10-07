"""Unify per-run XLT *test* results into one accuracy table.

Two modes, chosen automatically by what the runs carry:

  AGGREGATE mode -- the runs carry `fold` AND `seed` columns:
    1. POOL folds within (method, seed, target_lang) -> one accuracy over all
       items (weighted by per-fold n). Folds are item-disjoint partitions, so
       pooling = one big evaluation, never an average-with-std.
    2. Mean +/- std ACROSS seeds within (method, target_lang). Seeds are the
       only independent replication, so std lives only here.
    Output (beside PATH unless --out-prefix given) -- a SINGLE file, one row per
    target_lang. A tri-state `seen` flag follows target_lang: 1 = in the
    backbone's pretraining-seen set (LANG_GROUPS bloom_seen/aya_seen, keyed on
    the `llm` column), 0 = unseen, -1 = unseen AND in the backbone's
    low-performing set (LOW_PERF_LANG_GROUPS). Low-perf refines unseen, so
    `seen <= 0` is the unseen slice and `seen == -1` the low-perf slice.
    Then ALL <method>_acc columns, ALL <method>_std
    columns, and the shared n_seeds and n_items (pooled item count, e.g. 3 folds
    x 300 = 900):
      <prefix>.csv  target_lang, seen, zero_shot_acc, xpe_acc, spt_acc, dual_acc,
                    xpe_std, spt_std, dual_std, n_seeds, n_items
    Optional --zero-shot-path pools an existing zero-shot run's folds into a
    `zero_shot_acc`/`zero_shot_std` pair (no seed axis -> std is NaN, n_seeds 0).
    It also adds a `perf` column after `seen` -- the performance-based target
    group, which needs no published pretraining list: every language of the
    zero-shot run is ranked by its accuracy, the top --high-frac (default 50%)
    are `high`, the bottom --low-frac (default 25%) are `low`, the rest `mid`.
    The ranking is over ALL the zero-shot run's languages, not just this grid's
    targets, so a language's group does not depend on the source group.

  FALLBACK mode -- the runs carry no seed column (e.g. plain LR-search test
  runs): no aggregation is possible, so emit one column per run instead:
      <prefix>.csv       wide: rows=target_lang, one col per run folder
                         (leading timestamp stripped, e.g. 20260522_170730_xpe -> xpe)

The path is one group directory -- one experiment, one source group, one table
(`docs/micm-nlp-0.4-migration.md`).

Usage:
    python -m scripts.evals.lang_table \\
        artefacts/runs/groups/14a_bebe_grid_full_aya.enarzho \\
        --zero-shot-path artefacts/runs/groups/zs_bebe_folds_aya

    # zero-shot from ONE run of a mixed group (rows need no method/seed column)
    python -m scripts.evals.lang_table \\
        artefacts/runs/groups/26a_bebe_grid_aya.aya_high \\
        --zero-shot-path artefacts/runs/groups/25_bebe_zeroshot/20260924_035112_probe_aya_zs

    python -m scripts.evals.lang_table <group dir>   # fallback if no folds/seeds

    # Runs dispatched before `llm:` was a group entry key
    python -m scripts.evals.lang_table <group dir> --llm aya

    # A pre-micm-nlp-0.4 grid (raw.csv per run), written into the 0.4 layout so
    # target_source_table can set it next to a 0.5-era grid
    python -m scripts.evals.lang_table \\
        artefacts/evals/xlt_runs/aya/enarzho/14a_bebe_grid_full_aya \\
        --zero-shot-path artefacts/runs/groups/25_bebe_zeroshot/20260924_035112_probe_aya_zs \\
        --out-prefix artefacts/runs/groups/14a_bebe_grid_full_aya.enarzho/lang_table
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.xlt_langs import LANG_GROUPS, LOW_PERF_LANG_GROUPS

TIMESTAMP_RE = re.compile(r'^\d{8}_\d{6}_')

# Which LANG_GROUPS entry is the pretraining-"seen" set for each backbone (the
# `llm` column stamped by the group entry). LANG_GROUPS (src/xlt_langs.py) is the source of
# truth for the actual language lists.
LLM_SEEN_GROUP = {
    'bloom': 'bloom_seen',
    'aya': 'aya_seen',
    # SIB-200 encoders; these runs carry no fold axis (see has_seed_axis).
    'mdeberta': 'mdeberta_seen',
    'mgte': 'mgte_seen',
    # XLM-R-large, the paper's own backbone (grid 21a). Same no-fold SIB-200
    # shape as the two encoders above.
    'xlmr': 'xlmr_seen',
}

# Every backbone that can appear in the `llm` column needs an entry above --
# seen_langs_for() RAISES on a miss (unlike low_perf_langs_for(), which degrades
# to an empty set). A missing key therefore surfaces only at lang_table time, i.e.
# after a whole grid has already been trained. test_llm_seen_group_covers_every_backbone
# pins this against LOW_PERF_LANG_GROUPS so a new backbone cannot be half-wired.


def column_name(run_dir: Path) -> str:
    """Run folder's last part with the leading timestamp stripped."""
    return TIMESTAMP_RE.sub('', run_dir.name)


# A tune-and-test run's test phase is the entry's `separate_test` config, so the
# trainer writes it under the `separate_` prefix; a zero-shot or replay entry has
# no second config and writes the plain name. Neither carries a stage suffix,
# having no training phase of its own. One row per metric group, and a metric
# group is a target language (scripts/xlt_runner.py).
RESULTS_GLOBS = ('separate_test*.csv', 'test.csv', 'raw.csv')
RESULTS_GLOB = RESULTS_GLOBS[0]   # what a tune-and-test grid writes; kept for messages

# `raw.csv` is the pre-micm-nlp-0.4 per-run test file (artefacts/evals/xlt_runs/
# {llm}/{src_tag}/{run_group}/{time_id}_{run_name}/raw.csv). It has no `method`
# column -- the method is the run name's first segment (`xpe_f0_s10`) -- and
# names the source group `src_tag`. Read so the era-bridged grids (14a) can be
# tabulated next to 0.5-era ones; the bridge itself is grid 26b.
LEGACY_COLUMNS = {'src_tag': 'source_group'}

# What the group entry stamps on every row (micm_nlp.group.scalar_columns) and
# what this script needs from it. `llm` and `source_group` are entry keys because
# the run directory no longer carries them as path segments.
REQUIRED = ('method', 'seed', 'target_lang', 'accuracy', 'n')
# A zero-shot run has no method and no seed axis: its rows are pooled per language.
ZERO_SHOT_REQUIRED = ('target_lang', 'accuracy', 'n')


def collect(path: Path, llm: str | None = None, required=REQUIRED) -> pd.DataFrame:
    """Every run's test rows under a group directory, one row per target language.

    Reads the 0.4 layout only -- `runs/groups/{group}/{time_id}_{name}/`. Results
    produced before the migration live under `evals/xlt_runs/` and are read by the
    `pre-micm-nlp-0.4` tree, which is what produced them.
    """
    frames = []
    for csv in sorted(f for g in RESULTS_GLOBS for f in Path(path).rglob(g)):
        df = pd.read_csv(csv)
        if csv.name == 'raw.csv':
            df = df.rename(columns=LEGACY_COLUMNS)
            # run_name = `{time_id}_{method}_f{fold}_s{seed}`
            df['method'] = df['run_name'].str.replace(TIMESTAMP_RE, '', regex=True).str.split('_').str[0]
        frames.append(df)
    if not frames:
        raise SystemExit(f'No {" / ".join(RESULTS_GLOBS)} found under {path}')
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={'metric_group': 'target_lang', 'name': 'run_name'})
    if 'llm' not in df.columns:
        if llm is None:
            raise SystemExit(
                f'{path}: rows carry no `llm` column -- add `llm:` to the group entries, '
                'or pass --llm for runs dispatched before it was added')
        df['llm'] = llm
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f'{path}: result rows lack {missing}; found {sorted(df.columns)}')
    return df


def has_seed_axis(df: pd.DataFrame) -> bool:
    """True iff the runs carry a usable `seed` column (aggregate mode).

    A `fold` column is OPTIONAL: Belebele runs are folded (pool folds, then
    average over seeds) while SIB-200 runs are not (one row per seed already),
    and both belong in aggregate mode. Only a missing seed axis -- e.g. a plain
    LR-search test sweep -- falls back to one-column-per-run.

    Note this also promotes any older seed-but-no-fold run dir from FALLBACK to
    AGGREGATE mode, which is the correct treatment but is a behaviour change.
    """
    if 'seed' not in df.columns:
        return False
    return bool(df['seed'].notna().any())


# --- aggregate mode ---------------------------------------------------------

def _weighted(x: pd.DataFrame) -> float:
    return (x['accuracy'] * x['n']).sum() / x['n'].sum()


def _n_folds(x: pd.DataFrame) -> int:
    """Distinct folds in a group; 1 for a fold-less benchmark (SIB-200)."""
    if 'fold' not in x.columns or not x['fold'].notna().any():
        return 1
    return int(x['fold'].nunique())


def check_unique(df: pd.DataFrame) -> None:
    """Raise if two runs report the same (method, seed, fold, target_lang): a
    duplicated or restarted run would otherwise be pooled as extra items."""
    key = [c for c in ('method', 'seed', 'fold', 'target_lang') if c in df.columns]
    dup = df[df.duplicated(key, keep=False)]
    if len(dup):
        runs = sorted(dup['run_name'].unique()) if 'run_name' in dup.columns else '?'
        raise ValueError(f'{len(dup)} rows share a {key} key -- duplicated runs: {runs}')


def pool_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Pool folds -> one acc per (method, seed, target_lang) over sum(n) items."""
    df = df.dropna(subset=['accuracy', 'n'])
    rows = []
    for (method, seed, lang), x in df.groupby(['method', 'seed', 'target_lang']):
        rows.append({
            'method': method, 'seed': seed, 'target_lang': lang,
            'acc': _weighted(x), 'n_items': int(x['n'].sum()),
            'n_folds': _n_folds(x),
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
METHOD_ORDER = ['zero_shot', 'spt', 'd30', 'd70', 'xpe', 'dual']


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


def low_perf_langs_for(llm: str) -> set[str]:
    """Low-performing Belebele target langs for a backbone, from
    LOW_PERF_LANG_GROUPS. Empty set when the backbone has no list yet."""
    return set(LOW_PERF_LANG_GROUPS.get(llm, ()))


def add_seen(wide: pd.DataFrame, seen_langs, low_perf_langs=()) -> pd.DataFrame:
    """Insert a tri-state `seen` column right after target_lang:

         1  target lang is in the backbone's pretraining-seen set
         0  unseen by pretraining
        -1  unseen AND in the backbone's low-performing set

    Low-perf is a strict subset of unseen, so -1 refines 0 (an `unseen` slice is
    `seen <= 0`, a `low-perf` slice is `seen == -1`). A lang appearing in both
    the seen and low-perf lists is a config error and raises."""
    seen_langs, low_perf_langs = set(seen_langs), set(low_perf_langs)
    clash = seen_langs & low_perf_langs
    if clash:
        raise ValueError(f'langs in both the seen and low-perf groups: {sorted(clash)}')
    flag = wide['target_lang'].isin(seen_langs).astype(int)
    flag = flag.mask(wide['target_lang'].isin(low_perf_langs), -1)
    wide.insert(1, 'seen', flag)
    return wide


HIGH_FRAC, LOW_FRAC = 0.5, 0.25


def perf_groups(zero_shot: pd.DataFrame, high_frac: float = HIGH_FRAC,
                low_frac: float = LOW_FRAC) -> dict[str, str]:
    """{lang: 'high' | 'mid' | 'low'} from a pooled zero-shot table.

    Languages are ranked by zero-shot accuracy (ties broken by name, so the cut
    is reproducible); the top `high_frac` are `high`, the bottom `low_frac` are
    `low`. Both counts round DOWN on the low side and to nearest on the high
    side: 122 Belebele langs -> 61 high, 30 low. A tie straddling a cut is
    reported, because the tie-break then decides a group by name alone."""
    if not 0 < low_frac <= 1 - high_frac < 1:
        raise ValueError(f'need 0 < low_frac <= 1 - high_frac < 1, got '
                         f'high_frac={high_frac}, low_frac={low_frac}')
    ranked = zero_shot.sort_values(['mean_acc', 'target_lang'],
                                   ascending=[False, True]).reset_index(drop=True)
    n = len(ranked)
    n_high, n_low = round(n * high_frac), int(n * low_frac)
    acc = ranked['mean_acc']
    for name, cut in (('high', n_high), ('low', n - n_low)):
        if 0 < cut < n and acc[cut - 1] == acc[cut]:
            print(f'warning: zero-shot tie at the {name} cut ({acc[cut]:.4f}): '
                  f'{ranked["target_lang"][cut - 1]} / {ranked["target_lang"][cut]} '
                  f'are split by name')
    group = ['high'] * n_high + ['mid'] * (n - n_high - n_low) + ['low'] * n_low
    return dict(zip(ranked['target_lang'], group))


def add_perf(wide: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """Insert the `perf` column (see perf_groups) right after `seen`. A target
    lang the zero-shot run did not evaluate has no rank, which raises."""
    missing = sorted(set(wide['target_lang']) - set(groups))
    if missing:
        raise ValueError(f'target langs absent from the zero-shot run: {missing}')
    wide.insert(wide.columns.get_loc('seen') + 1, 'perf', wide['target_lang'].map(groups))
    return wide


def restrict_to_seeds(df: pd.DataFrame, seeds) -> pd.DataFrame:
    """Keep only the given seeds. Used to reproduce an earlier, smaller-n table
    from a grid that has since been extended (e.g. the first 5 of 10 seeds) --
    the seed axis is the replication axis, so dropping seeds is a valid, if less
    powered, aggregate. Raises if a requested seed is absent, so a typo cannot
    silently shrink n."""
    seeds = {int(s) for s in seeds}
    missing = seeds - set(df['seed'].dropna().astype(int))
    if missing:
        raise ValueError(f'seeds not present in the runs: {sorted(missing)}')
    return df[df['seed'].astype(int).isin(seeds)]


def write_aggregate(path: Path, prefix: Path, zero_shot_path: Path | None,
                    seeds=None, llm: str | None = None,
                    high_frac: float = HIGH_FRAC, low_frac: float = LOW_FRAC) -> None:
    grid = collect(path, llm)
    if seeds:
        grid = restrict_to_seeds(grid, seeds)
    check_unique(grid)
    long = seed_mean_std(pool_folds(grid))
    groups = None
    if zero_shot_path is not None:
        zs = pool_zero_shot(collect(zero_shot_path, llm, ZERO_SHOT_REQUIRED))
        groups = perf_groups(zs, high_frac, low_frac)   # ranked over ALL langs
        # zero-shot is evaluated on all langs; keep only the grid's target langs
        # so every output row has all methods (no zero-shot-only rows).
        zs = restrict_to_langs(zs, long['target_lang'].unique())
        long = pd.concat([long, zs], ignore_index=True)

    long = long.sort_values(['target_lang', 'method'])
    llm = backbone(grid)
    wide = add_seen(to_wide(long), seen_langs_for(llm), low_perf_langs_for(llm))
    if groups is not None:
        wide = add_perf(wide, groups)
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
    for res in sorted(f for g in RESULTS_GLOBS for f in Path(path).rglob(g)):
        name = column_name(res.parent)
        df = pd.read_csv(res).rename(columns={'metric_group': 'target_lang'})
        for col in (key, value):
            if col not in df.columns:
                raise ValueError(f"{res}: missing '{col}' column (has {list(df.columns)})")
        col = df.set_index(key)[value].rename(name)
        if col.index.has_duplicates:
            raise ValueError(f"{res}: duplicate '{key}' values, cannot align")
        series.append(col)
    if not series:
        raise SystemExit(f'No {" / ".join(RESULTS_GLOBS)} found under {path}')

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
    ap.add_argument('path', type=Path, help='group directory to walk for test result files')
    ap.add_argument('--zero-shot-path', type=Path, default=None,
                    help='group directory of an existing zero-shot run (aggregate mode only)')
    ap.add_argument('--high-frac', type=float, default=HIGH_FRAC,
                    help=f'top fraction of the zero-shot ranking labelled `high` (default {HIGH_FRAC})')
    ap.add_argument('--low-frac', type=float, default=LOW_FRAC,
                    help=f'bottom fraction of the zero-shot ranking labelled `low` (default {LOW_FRAC})')
    ap.add_argument('--llm', default=None,
                    help='backbone tag, for runs dispatched before `llm:` was a group entry key')
    ap.add_argument('--out-prefix', type=Path, default=None,
                    help='output prefix (default: PATH/lang_table)')
    ap.add_argument('--seeds', default=None,
                    help='comma-separated seeds to keep, e.g. 10,11,12,13,14 '
                         '(aggregate mode only; default = every seed present)')
    ap.add_argument('--key', default='target_lang', help='row-alignment column (fallback mode)')
    ap.add_argument('--value', default='accuracy', help='value column (fallback mode)')
    args = ap.parse_args()

    prefix = args.out_prefix or (args.path / 'lang_table')
    seeds = [int(s) for s in args.seeds.split(',')] if args.seeds else None

    if has_seed_axis(collect(args.path, args.llm)):
        write_aggregate(args.path, prefix, args.zero_shot_path, seeds, args.llm,
                        args.high_frac, args.low_frac)
    else:
        if args.zero_shot_path is not None:
            print('warning: --zero-shot-path ignored in fallback mode (no fold/seed columns)')
        write_fallback(args.path, prefix, args.key, args.value)


if __name__ == '__main__':
    main()
