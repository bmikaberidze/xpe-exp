"""Unit tests for the XLT validation-result unifier (per-fold, seed std)."""
import math

import pandas as pd

from scripts.evals.unify_xlt_valid_res import collect, aggregate


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
        }]).to_csv(d / 'valid_res.csv', index=False)
    df = collect(tmp_path)
    assert len(df) == 2
    assert set(df['seed']) == {10, 11}
