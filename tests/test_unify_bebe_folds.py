"""Unit tests for the fold-aware aggregator pure functions."""
import math

import pandas as pd

from scripts.evals.unify_bebe_folds import (
    parse_method, pool_folds, seed_mean_std, pool_zero_shot,
)


def test_parse_method_new_naming():
    assert parse_method('20260610_105815_xpe_f0_s10') == 'xpe'


def test_parse_method_old_naming():
    assert parse_method('20260610_105815_xpe') == 'xpe'


def test_parse_method_zero_shot_runname():
    assert parse_method('20260610_145744_zs_eval_seq') == 'zs_eval_seq'


def _grid_df():
    # one method, one seed, three folds (n=300 each) for one target lang
    return pd.DataFrame([
        {'method': 'xpe', 'seed': 10, 'fold': 0, 'target_lang': 'deu_Latn', 'accuracy': 0.40, 'n': 300},
        {'method': 'xpe', 'seed': 10, 'fold': 1, 'target_lang': 'deu_Latn', 'accuracy': 0.50, 'n': 300},
        {'method': 'xpe', 'seed': 10, 'fold': 2, 'target_lang': 'deu_Latn', 'accuracy': 0.60, 'n': 300},
    ])


def test_pool_folds_weighted_mean_over_900():
    out = pool_folds(_grid_df())
    row = out.iloc[0]
    assert math.isclose(row['acc'], 0.50)
    assert row['n_items'] == 900
    assert row['n_folds'] == 3


def test_seed_mean_std_across_seeds():
    pooled = pd.DataFrame([
        {'method': 'xpe', 'seed': 10, 'target_lang': 'deu_Latn', 'acc': 0.50, 'n_items': 900},
        {'method': 'xpe', 'seed': 11, 'target_lang': 'deu_Latn', 'acc': 0.60, 'n_items': 900},
    ])
    out = seed_mean_std(pooled)
    row = out.iloc[0]
    assert math.isclose(row['mean_acc'], 0.55)
    assert math.isclose(row['seed_std'], 0.0707106781, rel_tol=1e-6)
    assert row['n_seeds'] == 2
    assert row['n_items'] == 900


def test_pool_folds_skips_null_accuracy_rows():
    df = _grid_df()
    df.loc[2, 'accuracy'] = None  # a failed lang/fold
    out = pool_folds(df)
    row = out.iloc[0]
    assert row['n_items'] == 600  # only the two good folds pooled
    assert row['n_folds'] == 2


def test_pool_zero_shot_labels_and_pools():
    df = pd.DataFrame([
        {'target_lang': 'deu_Latn', 'fold': 0, 'accuracy': 0.30, 'n': 300},
        {'target_lang': 'deu_Latn', 'fold': 1, 'accuracy': 0.30, 'n': 300},
        {'target_lang': 'deu_Latn', 'fold': 2, 'accuracy': 0.30, 'n': 300},
    ])
    out = pool_zero_shot(df)
    row = out.iloc[0]
    assert row['method'] == 'zero_shot'
    assert math.isclose(row['mean_acc'], 0.30)
    assert row['n_items'] == 900
    assert row['n_seeds'] == 0
