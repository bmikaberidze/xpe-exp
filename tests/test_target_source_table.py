"""Unit tests for the per-backbone XLT result aggregator."""
import pandas as pd
import pytest

from scripts.evals.target_source_table import (
    aggregate, find_tables, method_cols, check_zero_shot_consistency, source_sort_key,
)
from src.xlt_langs import LANG_GROUPS

RUN_GROUP = '14x_grid'

# deu/fra/spa are joshi5 (the default exclusion); amh is low-perf (-1);
# acm is plain unseen (0); ita is seen (1).
LANGS = [
    ('amh_Ethi', -1), ('kac_Latn', -1),
    ('acm_Arab', 0), ('ceb_Latn', 0),
    ('ita_Latn', 1), ('pol_Latn', 1),
    ('deu_Latn', 1), ('fra_Latn', 1),  # joshi5 -> excluded everywhere
]


def _write_table(root, src_group, langs, acc=0.5):
    """A minimal lang_table.csv for one source group, in its `{grid}.{source_group}` dir."""
    d = root / f'{RUN_GROUP}.{src_group}'
    d.mkdir(parents=True)
    pd.DataFrame([
        {'target_lang': lang, 'seen': flag,
         'zero_shot_acc': 0.30, 'spt_acc': acc, 'xpe_acc': acc + 0.02,
         'spt_std': 0.01, 'xpe_std': 0.01, 'n_seeds': 10, 'n_items': 900}
        for lang, flag in langs
    ]).to_csv(d / 'lang_table.csv', index=False)
    return d


def test_find_tables_skips_zero_shot_dir(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    _write_table(tmp_path, 'zero', LANGS)
    assert sorted(find_tables(tmp_path, RUN_GROUP)) == ['enarzho']


def test_source_groups_ordered_few_to_many_langs(tmp_path):
    # dirs are created/globbed in an order that is NOT the wanted one
    for src in ['aya_seen', 'joshi5', 'enarzho']:
        _write_table(tmp_path, src, LANGS)
    assert list(find_tables(tmp_path, RUN_GROUP)) == ['enarzho', 'joshi5', 'aya_seen']
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    low = agg[agg['target_group'] == 'low-perf']
    assert list(low['source_group']) == ['enarzho', 'joshi5', 'aya_seen']


def test_find_tables_errors_when_nothing_matches(tmp_path):
    with pytest.raises(SystemExit):
        find_tables(tmp_path, RUN_GROUP)


def test_method_cols_puts_zero_shot_first():
    df = pd.DataFrame(columns=['target_lang', 'spt_acc', 'zero_shot_acc', 'xpe_acc'])
    assert method_cols(df) == ['zero_shot', 'spt', 'xpe']


def test_target_groups_split_on_tri_state_flag(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    n = agg.set_index('target_group')['n_langs'].to_dict()
    assert n['low-perf'] == 2                 # seen == -1
    assert n['unseen'] == 4                   # seen <= 0, low-perf INCLUDED
    assert n['seen'] == 2                     # seen == 1, joshi5 dropped
    assert n['all'] == 6                      # 8 langs - 2 joshi5


def test_excluded_group_dropped_from_every_target_group(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    assert agg['n_langs'].max() == 6  # deu/fra never counted, in any row
    # ...and keeping them changes only the joshi5-containing groups
    kept = aggregate(tmp_path, RUN_GROUP, None)
    assert kept.set_index('target_group')['n_langs'].to_dict()['seen'] == 4


def test_column_order_is_all_accs_then_all_stds_no_zero_shot_std(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    assert list(agg.columns) == [
        'target_group', 'source_group', 'grid', 'n_langs', 'n_seeds',
        'zero_shot_acc', 'spt_acc', 'xpe_acc',   # accs together, zero_shot first
        'spt_std', 'xpe_std',                    # then stds; zero_shot has none
    ]
    assert set(agg['grid']) == {RUN_GROUP} and set(agg['n_seeds']) == {10}


def test_accuracies_are_percent_means_over_langs(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS, acc=0.40)
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    row = agg[agg['target_group'] == 'low-perf'].iloc[0]
    assert row['spt_acc'] == pytest.approx(40.0)
    assert row['xpe_acc'] == pytest.approx(42.0)
    assert row['zero_shot_acc'] == pytest.approx(30.0)


def test_source_group_skipped_when_it_cannot_cover_the_target_set(tmp_path):
    # aya_seen trained on ita/pol, so they are absent from its own table; its
    # `seen` row would average a different set -> the cell must be dropped.
    _write_table(tmp_path, 'enarzho', LANGS)
    _write_table(tmp_path, 'aya_seen',
                 [(l, f) for l, f in LANGS if l not in ('ita_Latn', 'pol_Latn')])
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    cells = set(zip(agg['target_group'], agg['source_group']))
    assert ('seen', 'aya_seen') not in cells
    assert ('all', 'aya_seen') not in cells
    assert ('unseen', 'aya_seen') in cells   # unseen is fully covered
    assert ('seen', 'enarzho') in cells


def test_zero_shot_is_identical_across_source_groups(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    _write_table(tmp_path, 'joshi5', LANGS, acc=0.6)
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    assert check_zero_shot_consistency(agg) == []


def test_zero_shot_check_flags_misaligned_target_sets():
    # same target group, different zero-shot per source -> different lang sets
    agg = pd.DataFrame([
        {'target_group': 'unseen', 'source_group': 'a', 'n_langs': 10, 'zero_shot_acc': 41.0},
        {'target_group': 'unseen', 'source_group': 'b', 'n_langs': 9, 'zero_shot_acc': 40.7},
    ])
    problems = check_zero_shot_consistency(agg)
    assert len(problems) == 1 and 'unseen' in problems[0]


def test_low_perf_groups_cover_langs_that_exist_as_targets():
    # guards against typos in LOW_PERF_LANG_GROUPS: every code must look like a
    # Belebele lang tag, and none may be a joshi5 source
    from src.xlt_langs import LOW_PERF_LANG_GROUPS
    for llm, langs in LOW_PERF_LANG_GROUPS.items():
        assert len(langs) == len(set(langs)), f'{llm} has duplicates'
        assert all(len(l.split('_')) == 2 for l in langs)
        assert not set(langs) & set(LANG_GROUPS['joshi5'])


# --- performance-based target groups ----------------------------------------

PERF = {'amh_Ethi': 'low', 'kac_Latn': 'low', 'acm_Arab': 'mid', 'ceb_Latn': 'mid',
        'ita_Latn': 'high', 'pol_Latn': 'high', 'deu_Latn': 'high', 'fra_Latn': 'high'}


def _add_perf(d):
    df = pd.read_csv(d / 'lang_table.csv')
    df.insert(2, 'perf', df['target_lang'].map(PERF))
    df.to_csv(d / 'lang_table.csv', index=False)


def test_perf_target_groups_used_when_every_table_has_the_column(tmp_path):
    _add_perf(_write_table(tmp_path, 'enarzho', LANGS))
    agg = aggregate(tmp_path, RUN_GROUP, 'joshi5')
    n = agg.set_index('target_group')['n_langs'].to_dict()
    assert n == {'low-perf': 2, 'all wo high-perf': 4, 'high-perf wo j5': 2, 'all wo j5': 6}


def test_high_source_group_is_labelled_by_role_and_grids_can_be_mixed(tmp_path):
    _add_perf(_write_table(tmp_path, 'enarzho', LANGS))
    d = _write_table(tmp_path, 'aya_high', LANGS)
    d.rename(tmp_path / '26x_other_grid.aya_high')
    _add_perf(tmp_path / '26x_other_grid.aya_high')
    agg = aggregate(tmp_path, [RUN_GROUP, '26x_other_grid'], 'joshi5')
    low = agg[agg['target_group'] == 'low-perf'].set_index('source_group')
    assert list(low.index) == ['enarzho', 'high-perf']
    assert low.loc['high-perf', 'grid'] == '26x_other_grid'
    assert low.loc['enarzho', 'grid'] == RUN_GROUP


def test_source_group_in_two_grids_is_ambiguous(tmp_path):
    _write_table(tmp_path, 'enarzho', LANGS)
    (tmp_path / f'{RUN_GROUP}.enarzho').rename(tmp_path / '26x_other_grid.enarzho')
    _write_table(tmp_path, 'enarzho', LANGS)
    with pytest.raises(SystemExit):
        find_tables(tmp_path, [RUN_GROUP, '26x_other_grid'])


def test_seen_target_groups_still_selectable_and_the_auto_fallback(tmp_path):
    _add_perf(_write_table(tmp_path, 'enarzho', LANGS))
    _write_table(tmp_path, 'joshi5', LANGS)                 # no perf column
    for flavour in ('auto', 'seen'):
        agg = aggregate(tmp_path, RUN_GROUP, 'joshi5', target_groups=flavour)
        assert list(agg['target_group'].unique()) == ['low-perf', 'unseen', 'seen', 'all']
    with pytest.raises(SystemExit):
        aggregate(tmp_path, RUN_GROUP, 'joshi5', target_groups='perf')


def test_source_order_comes_from_lang_group_sizes():
    order = sorted(['aya_high', 'not_a_group', 'aya_seen', 'joshi5', 'enarzho'],
                   key=source_sort_key)
    assert order == ['enarzho', 'joshi5', 'aya_seen', 'aya_high', 'not_a_group']
    assert [len(LANG_GROUPS[g]) for g in order[:4]] == sorted(len(LANG_GROUPS[g]) for g in order[:4])
