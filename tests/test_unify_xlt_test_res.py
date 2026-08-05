"""Unit tests for the XLT test-result unifier (aggregate + fallback modes)."""
import math

import pandas as pd
import pytest

from scripts.evals.unify_xlt_test_res import (
    parse_method, has_fold_seed, pool_folds, seed_mean_std, pool_zero_shot,
    column_name, unify_runs, to_wide, restrict_to_langs,
    backbone, seen_langs_for, add_seen, low_perf_langs_for, restrict_to_seeds,
)
from scripts.run_xlt import LANG_GROUPS, LOW_PERF_LANG_GROUPS


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
    # all accs first (METHOD_ORDER: spt before xpe), then all stds, then counts
    assert list(out.columns) == [
        'target_lang', 'spt_acc', 'xpe_acc', 'spt_std', 'xpe_std', 'n_seeds', 'n_items'
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
        'zero_shot_acc', 'spt_acc', 'xpe_acc', 'dual_acc',
        'spt_std', 'xpe_std', 'dual_std',
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


def test_low_perf_langs_for_uses_lang_groups():
    # source of truth = LOW_PERF_LANG_GROUPS, keyed on the llm column
    assert low_perf_langs_for('bloom') == set(LOW_PERF_LANG_GROUPS['bloom'])
    assert low_perf_langs_for('aya') == set(LOW_PERF_LANG_GROUPS['aya'])
    assert low_perf_langs_for('nosuchllm') == set()  # no list yet -> no -1 rows


def test_low_perf_groups_are_a_subset_of_unseen():
    # the -1 refinement only makes sense if low-perf never overlaps the seen set
    # (or joshi5, which is always a source group), for either backbone
    for llm, seen_group in [('aya', 'aya_seen'), ('bloom', 'bloom_seen')]:
        low = set(LOW_PERF_LANG_GROUPS[llm])
        assert not low & set(LANG_GROUPS[seen_group])
        assert not low & set(LANG_GROUPS['joshi5'])


def test_add_seen_marks_low_perf_minus_one():
    wide = pd.DataFrame([
        {'target_lang': 'deu_Latn', 'xpe_acc': 0.5},   # aya-seen
        {'target_lang': 'acm_Arab', 'xpe_acc': 0.6},   # unseen, not low-perf
        {'target_lang': 'amh_Ethi', 'xpe_acc': 0.3},   # unseen AND low-perf
    ])
    out = add_seen(wide.copy(), seen_langs_for('aya'), low_perf_langs_for('aya'))
    assert out.set_index('target_lang')['seen'].to_dict() == {
        'deu_Latn': 1, 'acm_Arab': 0, 'amh_Ethi': -1,
    }


def test_add_seen_low_perf_refines_unseen_not_replaces_it():
    # `seen <= 0` must still select every unseen lang, low-perf included
    wide = pd.DataFrame([{'target_lang': 'acm_Arab'}, {'target_lang': 'amh_Ethi'},
                         {'target_lang': 'deu_Latn'}])
    out = add_seen(wide.copy(), seen_langs_for('aya'), low_perf_langs_for('aya'))
    assert sorted(out[out['seen'] <= 0]['target_lang']) == ['acm_Arab', 'amh_Ethi']
    assert list(out[out['seen'] == -1]['target_lang']) == ['amh_Ethi']


def test_add_seen_rejects_lang_in_both_groups():
    wide = pd.DataFrame([{'target_lang': 'deu_Latn'}])
    with pytest.raises(ValueError, match='both'):
        add_seen(wide, {'deu_Latn'}, {'deu_Latn'})


def test_restrict_to_seeds_keeps_only_given():
    grid = pd.DataFrame([{'seed': s, 'accuracy': 0.5} for s in [10, 11, 12, 13, 14, 15]])
    out = restrict_to_seeds(grid, [10, 11, 12, 13, 14])
    assert sorted(out['seed']) == [10, 11, 12, 13, 14]


def test_restrict_to_seeds_rejects_absent_seed():
    # a typo must not silently shrink n instead of erroring
    grid = pd.DataFrame([{'seed': s} for s in [10, 11]])
    with pytest.raises(ValueError, match='not present'):
        restrict_to_seeds(grid, [10, 99])


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
