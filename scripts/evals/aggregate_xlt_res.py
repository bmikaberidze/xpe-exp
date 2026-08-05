"""Aggregate the per-language XLT tables into ONE result table per backbone.

Input  = the `test_unified.csv` files written by `unify_xlt_test_res.py`, one per
         SOURCE group, under artefacts/evals/xlt_runs/<llm>/<src_group>/<run_group>/
Output = artefacts/evals/xlt_runs/<llm>/aggr_res.csv  (one row per
         TARGET-group x SOURCE-group cell, all methods side by side)

TARGET groups are read off the tri-state `seen` column that unify writes
(1 = pretraining-seen, 0 = unseen, -1 = unseen AND low-performing):

    low-perf   seen == -1   (LOW_PERF_LANG_GROUPS[<llm>], per-backbone)
    unseen     seen <= 0    (low-perf is a strict SUBSET of unseen)
    seen       seen == 1
    all        every target lang

EXCLUSION RULE (the reason cells are comparable): the langs of `--exclude-group`
(default joshi5) are dropped from EVERY target group, for EVERY source group --
not just from the source group that trained on them. Without this, a source
group only drops its OWN source langs, so e.g. the enarzho table would keep
deu/fra/jpn/spa as targets while the joshi5 table would not, and the two rows
would be averaging different language sets. With it, `zero_shot_acc` is
identical across all source rows of a target group -- which is the invariant to
eyeball after any rebuild. Cells that end up empty (e.g. the `seen` row of the
max-source group, whose seen langs are all its own sources) are skipped.

Column order mirrors `test_unified.csv`: ALL <method>_acc columns first
(zero_shot, then the tuned methods), then ALL <method>_std. Accuracies are means
over the target langs of the cell, in percent. The `_std` columns are the mean of
the per-lang across-seed std -- a typical seed-to-seed spread, NOT the std of the
reported mean. zero_shot has no seed axis, so it gets no _std column at all.

Usage:
    python -m scripts.evals.aggregate_xlt_res artefacts/evals/xlt_runs/aya \
        --run-group 14a_bebe_grid_full_aya

    python -m scripts.evals.aggregate_xlt_res artefacts/evals/xlt_runs/bloom \
        --run-group 14b_bebe_grid_full_bloomz --out /tmp/bloom.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from scripts.run_xlt import LANG_GROUPS

SOURCE_ORDER = ['enarzho', 'joshi5']  # then the *_seen group, whatever it is

TABLE = 'test_unified.csv'  # per-lang table written by unify_xlt_test_res

# (label, predicate on the tri-state `seen` column). Order = row order in the
# output. `unseen` is `<= 0` because low-perf (-1) refines unseen, not replaces it.
TARGET_GROUPS = [
    ('low-perf', lambda s: s == -1),
    ('unseen', lambda s: s <= 0),
    ('seen', lambda s: s == 1),
    ('all', lambda s: s.notna()),
]


def source_sort_key(src: str) -> tuple[int, str]:
    """Row order = increasing number of source languages: enarzho (3), joshi5
    (7), then the backbone's pretraining-seen group (24/39), which is whatever
    is left. Keeps every table reading few-sources -> many-sources."""
    return (SOURCE_ORDER.index(src) if src in SOURCE_ORDER else len(SOURCE_ORDER), src)


def find_tables(root: Path, run_group: str, table: str = TABLE) -> dict[str, Path]:
    """{source_group: <table>} for every source group under a backbone root that
    has a table for this run_group, in source_sort_key order. 'zero' dirs hold
    zero-shot evals, not a source group, so they are skipped."""
    found = {}
    for csv in root.glob(f'*/{run_group}/{table}'):
        src = csv.parent.parent.name
        if src != 'zero':
            found[src] = csv
    if not found:
        raise SystemExit(f'no */{run_group}/{table} under {root}')
    return {src: found[src] for src in sorted(found, key=source_sort_key)}


def method_cols(df: pd.DataFrame) -> list[str]:
    """Method tags present, zero_shot first, then the rest in table order."""
    tags = [c[:-4] for c in df.columns if c.endswith('_acc')]
    return sorted(tags, key=lambda t: (t != 'zero_shot', tags.index(t)))


def has_std(df: pd.DataFrame, method: str) -> bool:
    """Whether a method carries a usable seed-std. zero_shot has no seed axis,
    so its std column is absent (or all-NaN) and gets no output column."""
    col = f'{method}_std'
    return col in df.columns and df[col].notna().any()


def cell(df: pd.DataFrame, methods: list[str], label: str, src: str) -> dict | None:
    """One output row: ALL <method>_acc columns first, then ALL <method>_std, so
    the accuracies read side by side. Means over `df`'s target langs, in percent."""
    if df.empty:
        return None
    row = {'target_group': label, 'source_group': src, 'n_langs': len(df)}
    for m in methods:
        row[f'{m}_acc'] = df[f'{m}_acc'].mean() * 100
    for m in methods:
        if has_std(df, m):
            row[f'{m}_std'] = df[f'{m}_std'].mean() * 100
    return row


def aggregate(root: Path, run_group: str, exclude_group: str | None,
              table: str = TABLE) -> pd.DataFrame:
    tables = find_tables(root, run_group, table)
    excluded = set(LANG_GROUPS[exclude_group]) if exclude_group else set()

    frames = {}
    methods = None
    for src, csv in tables.items():
        df = pd.read_csv(csv)
        methods = methods or method_cols(df)
        frames[src] = df[~df['target_lang'].isin(excluded)]

    rows = []
    for label, pred in TARGET_GROUPS:
        slices = {src: df[pred(df['seen'])] for src, df in frames.items()}
        # Reference set = every lang any source group can report for this target
        # group. A source group whose own source langs fall inside it cannot
        # cover the reference (they are absent from its table as targets), so its
        # cell would silently average a different, easier/harder set -- skip it.
        # This is what removes `seen`/`all` for the max-source group.
        ref = set().union(*(set(s['target_lang']) for s in slices.values()))
        for src, s in slices.items():
            if set(s['target_lang']) != ref:
                missing = len(ref) - len(s)
                print(f'  skip {label:9s} x {src:11s} '
                      f'({missing}/{len(ref)} langs are its own source langs)')
                continue
            rows.append(cell(s, methods, label, src))

    out = pd.DataFrame([r for r in rows if r is not None])
    if out.empty:
        raise SystemExit('every target-group x source-group cell was empty')
    return out


def check_zero_shot_consistency(agg: pd.DataFrame) -> list[str]:
    """The invariant: within a target group, zero-shot acc must not depend on the
    source group -- it is the same untrained model on the same langs. Any spread
    means the source rows are averaging different language sets."""
    if 'zero_shot_acc' not in agg.columns:
        return []
    bad = []
    for label, g in agg.groupby('target_group', sort=False):
        spread = g['zero_shot_acc'].max() - g['zero_shot_acc'].min()
        if spread > 1e-6:
            langs = g.set_index('source_group')['n_langs'].to_dict()
            bad.append(f'  {label}: zero-shot spread {spread:.2f}pp across sources, '
                       f'n_langs={langs}')
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', type=Path,
                    help='backbone root, e.g. artefacts/evals/xlt_runs/aya')
    ap.add_argument('--run-group', required=True,
                    help='metaconfig stem, e.g. 14a_bebe_grid_full_aya')
    ap.add_argument('--table-name', default=TABLE,
                    help=f'per-lang table to read in each run-group dir '
                         f'(default {TABLE}); use this to aggregate a seed-'
                         f'restricted unify, e.g. test_unified_s10-14.csv')
    ap.add_argument('--exclude-group', default='joshi5',
                    help="LANG_GROUPS name dropped from ALL target groups "
                         "(default joshi5); pass '' to disable")
    ap.add_argument('--out', type=Path, default=None,
                    help='output csv (default <root>/aggr_res.csv)')
    args = ap.parse_args()

    agg = aggregate(args.root, args.run_group, args.exclude_group or None,
                    args.table_name)
    out = args.out or args.root / 'aggr_res.csv'
    agg.to_csv(out, index=False)
    print(f'wrote {out}  ({len(agg)} cells)')

    if problems := check_zero_shot_consistency(agg):
        print('WARNING: zero-shot differs across source groups -- target sets '
              'are not aligned:')
        print('\n'.join(problems))


if __name__ == '__main__':
    main()
