"""Unit tests for the XLT test-result unifier (aggregate + fallback modes)."""
import math

import pandas as pd

from scripts.evals.unify_xlt_test_res import (
    parse_method, has_fold_seed, pool_folds, seed_mean_std, pool_zero_shot,
    column_name, unify_runs, to_wide, restrict_to_langs,
    backbone, seen_langs_for, add_seen,
)
from scripts.run_xlt import LANG_GROUPS


# --- method parsing ---------------------------------------------------------

def test_parse_method_new_naming():
    assert parse_method('20260610_105815_xpe_f0_s10') == 'xpe'


def test_parse_method_old_naming():
    assert parse_method('20260610_105815_xpe') == 'xpe'


def test_parse_method_zero_shot_runname():
    assert parse_method('20260610_145744_zs_eval_seq') == 'zs_eval_seq'


# --- mode detection ---------------------------------------------------------

def test_has_fold_seed_true():
    df = pd.DataFrame([{'fold': 0, 'seed': 10, 'target_lang': 'x', 'accuracy': 0.5, 'n': 300}])
    assert has_fold_seed(df) is True


def test_has_fold_seed_missing_columns():
    df = pd.DataFrame([{'target_lang': 'x', 'accuracy': 0.5}])
    assert has_fold_seed(df) is False


def test_has_fold_seed_all_null():
    df = pd.DataFrame([{'fold': None, 'seed': None, 'target_lang': 'x', 'accuracy': 0.5}])
    assert has_fold_seed(df) is False


# --- aggregate mode ---------------------------------------------------------

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


# --- single wide table (one row per lang, <method>_acc/_std + counts) -------

def _long_df():
    # two methods, one lang; n_seeds/n_items shared
    return pd.DataFrame([
        {'method': 'xpe', 'target_lang': 'deu_Latn', 'mean_acc': 0.55, 'seed_std': 0.05, 'n_seeds': 3, 'n_items': 900},
        {'method': 'spt', 'target_lang': 'deu_Latn', 'mean_acc': 0.50, 'seed_std': 0.04, 'n_seeds': 3, 'n_items': 900},
        {'method': 'xpe', 'target_lang': 'fra_Latn', 'mean_acc': 0.60, 'seed_std': 0.03, 'n_seeds': 3, 'n_items': 900},
        {'method': 'spt', 'target_lang': 'fra_Latn', 'mean_acc': 0.58, 'seed_std': 0.02, 'n_seeds': 3, 'n_items': 900},
    ])


def test_to_wide_one_row_per_lang_with_method_cols():
    out = to_wide(_long_df())
    # one row per language
    assert list(out['target_lang']) == ['deu_Latn', 'fra_Latn']
    # all accs first (xpe before spt), then all stds, then shared counts
    assert list(out.columns) == [
        'target_lang', 'xpe_acc', 'spt_acc', 'xpe_std', 'spt_std', 'n_seeds', 'n_items'
    ]
    deu = out.set_index('target_lang').loc['deu_Latn']
    assert math.isclose(deu['xpe_acc'], 0.55) and math.isclose(deu['xpe_std'], 0.05)
    assert math.isclose(deu['spt_acc'], 0.50) and math.isclose(deu['spt_std'], 0.04)
    assert deu['n_seeds'] == 3 and deu['n_items'] == 900


def test_to_wide_zero_shot_first_and_acc_only():
    # zero_shot leads the table and carries NO std (no seed axis -> NaN std)
    rows = [
        {'method': m, 'target_lang': 'deu_Latn', 'mean_acc': 0.5, 'seed_std': 0.01, 'n_seeds': 3, 'n_items': 900}
        for m in ['dual', 'spt', 'xpe']
    ]
    rows.append({'method': 'zero_shot', 'target_lang': 'deu_Latn', 'mean_acc': 0.39,
                 'seed_std': float('nan'), 'n_seeds': 0, 'n_items': 900})
    out = to_wide(pd.DataFrame(rows))
    assert list(out.columns) == [
        'target_lang',
        'zero_shot_acc', 'xpe_acc', 'spt_acc', 'dual_acc',
        'xpe_std', 'spt_std', 'dual_std',
        'n_seeds', 'n_items',
    ]


def test_backbone_single_llm():
    df = pd.DataFrame([{'llm': 'bloom'}, {'llm': 'bloom'}])
    assert backbone(df) == 'bloom'


def test_backbone_rejects_mixed():
    import pytest
    df = pd.DataFrame([{'llm': 'bloom'}, {'llm': 'aya'}])
    with pytest.raises(ValueError):
        backbone(df)


def test_seen_langs_for_uses_lang_groups():
    # source of truth = LANG_GROUPS, not a duplicated list
    assert seen_langs_for('bloom') == set(LANG_GROUPS['bloom_seen'])
    assert seen_langs_for('aya') == set(LANG_GROUPS['aya_seen'])


def test_add_seen_flags_per_backbone():
    # deu_Latn: seen by aya, NOT by bloom -> backbone-dependent
    wide = pd.DataFrame([
        {'target_lang': 'deu_Latn', 'xpe_acc': 0.5},
        {'target_lang': 'acm_Arab', 'xpe_acc': 0.6},  # in neither seen set
    ])
    out = add_seen(wide.copy(), seen_langs_for('bloom'))
    assert list(out.columns)[:2] == ['target_lang', 'seen']  # right after target_lang
    assert out.set_index('target_lang')['seen'].to_dict() == {'deu_Latn': 0, 'acm_Arab': 0}
    out_aya = add_seen(wide.copy(), seen_langs_for('aya'))
    assert out_aya.set_index('target_lang')['seen']['deu_Latn'] == 1


def test_restrict_to_langs_keeps_only_given():
    zs = pd.DataFrame([
        {'method': 'zero_shot', 'target_lang': 'deu_Latn', 'mean_acc': 0.39},
        {'method': 'zero_shot', 'target_lang': 'eng_Latn', 'mean_acc': 0.80},  # seen src, drop
        {'method': 'zero_shot', 'target_lang': 'fra_Latn', 'mean_acc': 0.41},
    ])
    out = restrict_to_langs(zs, ['deu_Latn', 'fra_Latn'])
    assert sorted(out['target_lang']) == ['deu_Latn', 'fra_Latn']


# --- fallback mode (no fold/seed -> one column per run) ---------------------

def test_column_name_strips_timestamp():
    from pathlib import Path
    assert column_name(Path('a/b/20260522_170730_xpe_lr5e5')) == 'xpe_lr5e5'


def test_unify_runs_one_column_per_run(tmp_path):
    # two runs, no fold/seed columns -> wide table, rows=target_lang, cols=run
    for ts, acc in [('20260101_000000_xpe_lr5e5', 0.30), ('20260101_000001_spt_lr5e5', 0.40)]:
        d = tmp_path / ts
        d.mkdir()
        pd.DataFrame([
            {'target_lang': 'deu_Latn', 'accuracy': acc},
            {'target_lang': 'fra_Latn', 'accuracy': acc + 0.01},
        ]).to_csv(d / 'raw.csv', index=False)

    table = unify_runs(tmp_path)
    assert list(table.columns) == ['xpe_lr5e5', 'spt_lr5e5']
    assert math.isclose(table.loc['deu_Latn', 'xpe_lr5e5'], 0.30)
    assert math.isclose(table.loc['fra_Latn', 'spt_lr5e5'], 0.41)
