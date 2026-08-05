"""Unit tests for the per-backbone XLT result aggregator."""
import pandas as pd
import pytest

from scripts.evals.aggregate_xlt_res import (
    aggregate, find_tables, method_cols, check_zero_shot_consistency,
)
from scripts.run_xlt import LANG_GROUPS

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
    """A minimal test_unified.csv for one source group."""
    d = root / src_group / RUN_GROUP
    d.mkdir(parents=True)
    pd.DataFrame([
        {'target_lang': lang, 'seen': flag,
         'zero_shot_acc': 0.30, 'spt_acc': acc, 'xpe_acc': acc + 0.02,
         'spt_std': 0.01, 'xpe_std': 0.01, 'n_seeds': 10, 'n_items': 900}
        for lang, flag in langs
    ]).to_csv(d / 'test_unified.csv', index=False)
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
        'target_group', 'source_group', 'n_langs',
        'zero_shot_acc', 'spt_acc', 'xpe_acc',   # accs together, zero_shot first
        'spt_std', 'xpe_std',                    # then stds; zero_shot has none
    ]


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
    from scripts.run_xlt import LOW_PERF_LANG_GROUPS
    for llm, langs in LOW_PERF_LANG_GROUPS.items():
        assert len(langs) == len(set(langs)), f'{llm} has duplicates'
        assert all(len(l.split('_')) == 2 for l in langs)
        assert not set(langs) & set(LANG_GROUPS['joshi5'])
