"""Aggregate the per-language XLT tables into ONE result table per backbone.

Input  = the `lang_table.csv` files written by `lang_table.py`, one per
         SOURCE group, in artefacts/runs/groups/<grid>.<src_group>/
Output = artefacts/runs/groups/<grid>.target_source_table.csv  (one row per
         TARGET-group x SOURCE-group cell, all methods side by side)

TARGET groups come in two flavours (--target-groups, default `auto`):

  perf -- the `perf` column lang_table writes when given --zero-shot-path: the
  backbone's own zero-shot percentile ranking. Needs no published pretraining
  list, so it works for every backbone. `auto` picks it when every table has it.

    low-perf           perf == low    (bottom 25% of the ranking)
    all wo high-perf   perf != high   (low-perf is a strict SUBSET of it)
    high-perf wo j5    perf == high   (top 50%, minus the excluded group)
    all wo j5          every target lang (minus the excluded group)

  The `wo j5` suffix names --exclude-group (see below); the first two groups
  carry no suffix because joshi5 lies entirely inside high-perf.

  Source groups are labelled by role, not by backbone: a `{llm}_high` group is
  shown as `high-perf` so the Aya and BLOOM tables read the same.

  Several grids can feed ONE table (`--run-group A B`), e.g. the era-bridged
  14a enarzho/joshi5 rows next to the 0.5-era 26a high-perf row. Each cell then
  says which grid (`grid` column) and how many seeds (`n_seeds`) it came from.

  seen -- the tri-state `seen` column (1 = pretraining-seen, 0 = unseen,
  -1 = unseen AND low-performing), i.e. the published lists:

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

Column order mirrors `lang_table.csv`: ALL <method>_acc columns first
(zero_shot, then the tuned methods), then ALL <method>_std. Accuracies are means
over the target langs of the cell, in percent. The `_std` columns are the mean of
the per-lang across-seed std -- a typical seed-to-seed spread, NOT the std of the
reported mean. zero_shot has no seed axis, so it gets no _std column at all.

Usage:
    python -m scripts.evals.target_source_table artefacts/runs/groups \
        --run-group 14a_bebe_grid_full_aya

    python -m scripts.evals.target_source_table artefacts/runs/groups \
        --run-group 14b_bebe_grid_full_bloomz --out /tmp/bloom.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.xlt_langs import LANG_GROUPS

TABLE = 'lang_table.csv'  # per-lang table written by lang_table

# {flavour: (column, [(label, predicate on that column)])}. Order = row order in
# the output. `unseen` is `<= 0` because low-perf (-1) refines unseen, not
# replaces it; `bottom-50` contains low-perf the same way.
TARGET_GROUPS = {
    'seen': ('seen', [
        ('low-perf', lambda s: s == -1),
        ('unseen', lambda s: s <= 0),
        ('seen', lambda s: s == 1),
        ('all', lambda s: s.notna()),
    ]),
    'perf': ('perf', [
        ('low-perf', lambda s: s == 'low'),
        ('all wo high-perf', lambda s: s != 'high'),
        ('high-perf wo {excl}', lambda s: s == 'high'),
        ('all wo {excl}', lambda s: s.notna()),
    ]),
}

EXCLUDE_LABELS = {'joshi5': 'j5', None: 'excluded'}


def source_label(src: str) -> str:
    """Display name of a source group: `{llm}_high` -> `high-perf`, others as is."""
    return 'high-perf' if src.endswith('_high') else src


def source_sort_key(src: str) -> tuple[int, str]:
    """Row order = increasing number of source languages, read from
    LANG_GROUPS: enarzho (3), joshi5 (7), then the larger groups. Keeps every
    table reading few-sources -> many-sources. A source group LANG_GROUPS does
    not know sorts last."""
    return (len(LANG_GROUPS[src]) if src in LANG_GROUPS else 10**6, src)


def find_tables(root: Path, run_groups, table: str = TABLE) -> dict[str, Path]:
    """{source_group: <table>} for every source group of these grids, in
    source_sort_key order.

    A group directory is named ``{grid}.{source_group}`` -- one group config per
    source group, one output directory each -- so the source group is the stem's
    last dotted segment. A `.zero` directory holds a zero-shot eval, not a source
    group, so it is skipped. A source group found in two grids is ambiguous and
    raises: restrict `run_groups` to the grids that should feed the table.
    """
    if isinstance(run_groups, str):
        run_groups = [run_groups]
    found: dict[str, Path] = {}
    for run_group in run_groups:
        for csv in root.glob(f'{run_group}.*/{table}'):
            src = csv.parent.name.split('.')[-1]
            if src == 'zero':
                continue
            if src in found:
                raise SystemExit(f'source group {src!r} in both {found[src].parent.name} '
                                 f'and {csv.parent.name}')
            found[src] = csv
    if not found:
        raise SystemExit(f'no {{{", ".join(run_groups)}}}.*/{table} under {root}')
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


def cell(df: pd.DataFrame, methods: list[str], label: str, src: str,
         grid: str = '') -> dict | None:
    """One output row: ALL <method>_acc columns first, then ALL <method>_std, so
    the accuracies read side by side. Means over `df`'s target langs, in percent.
    `n_seeds` is the smallest seed count over the cell's langs (they coincide on
    a complete grid)."""
    if df.empty:
        return None
    row = {'target_group': label, 'source_group': source_label(src), 'grid': grid,
           'n_langs': len(df),
           'n_seeds': int(df['n_seeds'].min()) if 'n_seeds' in df.columns else 0}
    for m in methods:
        row[f'{m}_acc'] = df[f'{m}_acc'].mean() * 100
    for m in methods:
        if has_std(df, m):
            row[f'{m}_std'] = df[f'{m}_std'].mean() * 100
    return row


def target_groups_for(frames: dict[str, pd.DataFrame], flavour: str = 'auto'):
    """(column, [(label, predicate)]) for the requested flavour. `auto` = perf
    when EVERY source table carries the `perf` column, else seen."""
    if flavour == 'auto':
        flavour = 'perf' if all('perf' in df.columns for df in frames.values()) else 'seen'
    column, groups = TARGET_GROUPS[flavour]
    missing = [src for src, df in frames.items() if column not in df.columns]
    if missing:
        raise SystemExit(f'--target-groups {flavour}: no `{column}` column in the '
                         f'table of {missing} (rebuild lang_table with --zero-shot-path)')
    return column, groups


def aggregate(root: Path, run_groups, exclude_group: str | None,
              table: str = TABLE, target_groups: str = 'auto') -> pd.DataFrame:
    tables = find_tables(root, run_groups, table)
    excluded = set(LANG_GROUPS[exclude_group]) if exclude_group else set()
    excl = EXCLUDE_LABELS.get(exclude_group, exclude_group)

    frames, grids = {}, {}
    methods = None
    for src, csv in tables.items():
        df = pd.read_csv(csv)
        methods = methods or method_cols(df)
        frames[src] = df[~df['target_lang'].isin(excluded)]
        grids[src] = csv.parent.name.rsplit('.', 1)[0]

    column, groups = target_groups_for(frames, target_groups)
    rows = []
    for label, pred in groups:
        label = label.format(excl=excl)
        slices = {src: df[pred(df[column])] for src, df in frames.items()}
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
            rows.append(cell(s, methods, label, src, grids[src]))

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
                    help='the groups root, e.g. artefacts/runs/groups')
    ap.add_argument('--run-group', required=True, nargs='+',
                    help='grid name(s) = the group stem before the source group, e.g. '
                         '14a_bebe_grid_full_aya; several grids feed one table '
                         '(each source group must come from exactly one of them)')
    ap.add_argument('--table-name', default=TABLE,
                    help=f'per-lang table to read in each run-group dir '
                         f'(default {TABLE}); use this to aggregate a seed-'
                         f'restricted lang_table run, e.g. lang_table_s10-14.csv')
    ap.add_argument('--target-groups', default='auto', choices=['auto', *TARGET_GROUPS],
                    help='perf = zero-shot percentile groups, seen = published lists; '
                         'auto = perf when every table has the `perf` column')
    ap.add_argument('--exclude-group', default='joshi5',
                    help="LANG_GROUPS name dropped from ALL target groups "
                         "(default joshi5); pass '' to disable")
    ap.add_argument('--out', type=Path, default=None,
                    help='output csv (default <root>/{run-group}.target_source_table.csv; '
                         'required with several run groups)')
    args = ap.parse_args()
    if args.out is None and len(args.run_group) > 1:
        ap.error('--out is required when several --run-group grids feed one table')

    agg = aggregate(args.root, args.run_group, args.exclude_group or None,
                    args.table_name, args.target_groups)
    out = args.out or args.root / f'{args.run_group[0]}.target_source_table.csv'
    agg.to_csv(out, index=False)
    print(f'wrote {out}  ({len(agg)} cells)')

    if problems := check_zero_shot_consistency(agg):
        print('WARNING: zero-shot differs across source groups -- target sets '
              'are not aligned:')
        print('\n'.join(problems))


if __name__ == '__main__':
    main()
