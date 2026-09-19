"""Unit tests for the XLT validation-result unifier (per-fold, seed std)."""
import math

import pandas as pd
import pytest

from scripts.evals.unify_xlt_valid_res import VALID_GLOB, aggregate, collect


def _valid_df():
    # spt, lr=0.5, fold 0, two seeds; plus lr=1.0 fold 0 one seed
    return pd.DataFrame([
        {'method': 'spt', 'fold': 0, 'seed': 10, 'learning_rate': 0.5, 'best_val_acc': 0.80},
        {'method': 'spt', 'fold': 0, 'seed': 11, 'learning_rate': 0.5, 'best_val_acc': 0.90},
        {'method': 'spt', 'fold': 0, 'seed': 10, 'learning_rate': 1.0, 'best_val_acc': 0.85},
    ])


def test_aggregate_mean_and_seed_std():
    out = aggregate(_valid_df(), group_key='learning_rate')
    cell = out[(out['learning_rate'] == 0.5) & (out['fold'] == 0)].iloc[0]
    assert math.isclose(cell['mean_val_acc'], 0.85)
    assert math.isclose(cell['seed_std'], 0.0707106781, rel_tol=1e-6)  # ddof=1
    assert cell['n_seeds'] == 2


def test_aggregate_single_seed_std_is_nan():
    out = aggregate(_valid_df(), group_key='learning_rate')
    cell = out[(out['learning_rate'] == 1.0) & (out['fold'] == 0)].iloc[0]
    assert math.isclose(cell['mean_val_acc'], 0.85)
    assert math.isnan(cell['seed_std'])
    assert cell['n_seeds'] == 1


def test_aggregate_keeps_folds_separate():
    df = pd.DataFrame([
        {'method': 'spt', 'fold': 0, 'seed': 10, 'learning_rate': 0.5, 'best_val_acc': 0.70},
        {'method': 'spt', 'fold': 1, 'seed': 10, 'learning_rate': 0.5, 'best_val_acc': 0.80},
    ])
    out = aggregate(df, group_key='learning_rate')
    assert set(out['fold']) == {0, 1}
    assert len(out) == 2


def test_aggregate_drops_null_best_val_acc():
    df = _valid_df()
    df.loc[0, 'best_val_acc'] = None  # a failed run
    out = aggregate(df, group_key='learning_rate')
    cell = out[(out['learning_rate'] == 0.5) & (out['fold'] == 0)].iloc[0]
    assert cell['n_seeds'] == 1  # only the surviving seed
    assert math.isclose(cell['mean_val_acc'], 0.90)


def test_aggregate_custom_group_key():
    df = pd.DataFrame([
        {'method': 'spt', 'fold': 0, 'seed': 10, 'weight_decay': 0.0, 'best_val_acc': 0.70},
        {'method': 'spt', 'fold': 0, 'seed': 11, 'weight_decay': 0.0, 'best_val_acc': 0.80},
    ])
    out = aggregate(df, group_key='weight_decay')
    assert 'weight_decay' in out.columns
    cell = out.iloc[0]
    assert math.isclose(cell['mean_val_acc'], 0.75)
    assert cell['n_seeds'] == 2


def test_collect_reads_valid_res_files(tmp_path):
    for i, (seed, acc) in enumerate([(10, 0.80), (11, 0.90)]):
        d = tmp_path / f'20260101_00000{i}_spt_lr5e1_s{seed}'
        d.mkdir()
        pd.DataFrame([{
            'run_name': f'spt_lr5e1_s{seed}', 'method': 'spt', 'fold': 0,
            'seed': seed, 'learning_rate': 0.5, 'best_val_acc': acc,
        }]).to_csv(d / VALID_GLOB, index=False)
    df = collect(tmp_path)
    assert len(df) == 2
    assert set(df['seed']) == {10, 11}


# --- fold-less runs (SIB-200) -----------------------------------------------

def _valid_rows(fold=None):
    rows = []
    for lr in (1e-5, 5e-5):
        for seed, acc in ((11, 0.70), (12, 0.74)):
            r = {'method': 'xpe', 'learning_rate': lr, 'seed': seed,
                 'best_val_acc': acc + (0.05 if lr == 5e-5 else 0.0)}
            if fold is not None:
                r['fold'] = fold
            rows.append(r)
    return pd.DataFrame(rows)


def test_aggregate_without_fold_column():
    """SIB-200 runs have seeds but no folds; aggregation must still work."""
    out = aggregate(_valid_rows())
    assert len(out) == 2                      # one cell per learning_rate
    assert set(out['n_seeds']) == {2}
    best = out.loc[out['mean_val_acc'].idxmax()]
    assert best['learning_rate'] == 5e-5
    assert best['mean_val_acc'] == pytest.approx(0.77)


def test_aggregate_with_all_nan_fold_column():
    """run_xlt writes fold=None for SIB, which reads back as NaN."""
    df = _valid_rows()
    df['fold'] = float('nan')
    out = aggregate(df)
    assert len(out) == 2
    assert set(out['n_seeds']) == {2}


def test_aggregate_still_separates_real_folds():
    df = pd.concat([_valid_rows(fold=0), _valid_rows(fold=1)], ignore_index=True)
    out = aggregate(df)
    assert len(out) == 4                      # 2 lrs x 2 folds, kept separate
    assert sorted(out['fold'].unique()) == [0, 1]


# --- guard against grouping on a constant column ----------------------------

def _sweep_rows(group_col='prompt_lr'):
    """18 rows: 6 swept values x 3 seeds, with a CONSTANT learning_rate column."""
    rows = []
    for v in (1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2):
        for seed, acc in ((11, 0.70), (12, 0.72), (13, 0.74)):
            rows.append({'method': 'xpe', 'learning_rate': 5e-5, group_col: v,
                         'seed': seed, 'best_val_acc': acc + v})
    return pd.DataFrame(rows)


def test_aggregate_on_the_swept_column_keeps_every_cell():
    out = aggregate(_sweep_rows(), group_key='prompt_lr')
    assert len(out) == 6                 # one cell per swept value
    assert set(out['n_seeds']) == {3}    # no row silently dropped


def test_aggregate_raises_when_group_key_is_constant():
    # grouping on learning_rate (constant 5e-5) would collapse 18 rows to 3 and
    # average over a near-arbitrary subset -- that must not pass silently
    with pytest.raises(ValueError, match='probably CONSTANT'):
        aggregate(_sweep_rows(), group_key='learning_rate')


def test_aggregate_still_tolerates_a_genuine_rerun_duplicate():
    df = _sweep_rows()
    dup = df.iloc[[0]].copy()
    dup['best_val_acc'] = 0.99          # a crashed run's re-run
    out = aggregate(pd.concat([df, dup], ignore_index=True), group_key='prompt_lr')
    assert len(out) == 6
    assert set(out['n_seeds']) == {3}
