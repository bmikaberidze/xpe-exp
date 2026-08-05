"""Unit tests for named source-language groups and run-dir naming in run_xlt."""
import pytest
from types import SimpleNamespace

from scripts.run_xlt import LANG_GROUPS, resolve_langs, src_tag_for
from scripts.run_xlt import build_run_paths

JOSHI5 = ['eng_Latn', 'spa_Latn', 'deu_Latn', 'fra_Latn', 'jpn_Jpan', 'zho_Hans', 'arb_Arab']


def test_joshi5_group_defined():
    assert LANG_GROUPS['joshi5'] == JOSHI5


def test_bloom_seen_group():
    bloom_seen = LANG_GROUPS['bloom_seen']
    # 39 = ROOTS langs intersected with Belebele's 122
    assert len(bloom_seen) == 39
    assert len(set(bloom_seen)) == 39  # no duplicates
    # native scripts only: no romanized *_Latn duplicates of native-script langs
    for romanized in ('hin_Latn', 'urd_Latn', 'ben_Latn', 'npi_Latn', 'arb_Latn'):
        assert romanized not in bloom_seen
    # MSA Arabic only, no dialectal variants
    for dialect in ('acm_Arab', 'apc_Arab', 'ars_Arab', 'ary_Arab', 'arz_Arab'):
        assert dialect not in bloom_seen
    assert 'arb_Arab' in bloom_seen
    # the 5 ROOTS langs absent from Belebele must not appear
    for absent in ('tum_Latn', 'kik_Latn', 'aka_Latn', 'fon_Latn', 'run_Latn'):
        assert absent not in bloom_seen
    assert resolve_langs('bloom_seen', '') == bloom_seen


def test_aya_seen_group():
    aya_seen = LANG_GROUPS['aya_seen']
    # 24 codes = Aya Expanse's 23 languages (Chinese as both scripts), all in Belebele
    assert len(aya_seen) == 24
    assert len(set(aya_seen)) == 24  # no duplicates
    assert {'zho_Hans', 'zho_Hant'} <= set(aya_seen)
    # native scripts / MSA Arabic / Western Persian
    assert 'arb_Arab' in aya_seen and 'pes_Arab' in aya_seen
    for romanized in ('hin_Latn', 'urd_Latn', 'arb_Latn'):
        assert romanized not in aya_seen
    assert resolve_langs('aya_seen', '') == aya_seen


def test_resolve_group_name():
    assert resolve_langs('joshi5', '') == JOSHI5


def test_resolve_csv_when_no_group():
    assert resolve_langs(None, 'en, ru ,zh') == ['en', 'ru', 'zh']


def test_resolve_unknown_group_raises():
    with pytest.raises(ValueError, match='joshi5'):  # message lists known groups
        resolve_langs('nope', '')


def test_resolve_both_set_raises():
    with pytest.raises(ValueError, match='both'):
        resolve_langs('joshi5', 'en,ru')


def test_resolve_neither_set_returns_empty():
    assert resolve_langs(None, '') == []


def test_src_tag_uses_group_name():
    assert src_tag_for('joshi5', sorted(JOSHI5)) == 'joshi5'


def test_src_tag_falls_back_to_joined_sorted_langs():
    assert src_tag_for(None, ['ar', 'en', 'zh']) == 'ar-en-zh'


def test_src_tag_zero_when_empty():
    assert src_tag_for(None, []) == 'zero'


# --- SIB-200 encoder groups (pinned to src/sib200_meta.py) -------------------

from scripts.run_xlt import LOW_PERF_LANG_GROUPS
from src.sib200_meta import (
    LABEL_NAMES, MGTE_TOKENS_M, SIB200_LANGS, low_perf_langs, seen_langs, sib200_codes,
)


def test_sib200_table_shape():
    assert len(SIB200_LANGS) == 205  # SIB-200 covers 205 language codes
    assert len(set(sib200_codes())) == 205
    # joshi5 must be derivable from the table's Joshi class column
    assert sorted(r['code'] for r in SIB200_LANGS if r['class'] == 5.0) == sorted(JOSHI5)


def test_label_names_are_canonical_order():
    # https://huggingface.co/datasets/Davlan/sib200/raw/main/data/eng_Latn/labels.txt
    # This ordering IS the label -> id map used by the published runs.
    assert LABEL_NAMES == [
        'science/technology', 'travel', 'politics',
        'sports', 'health', 'entertainment', 'geography',
    ]


@pytest.mark.parametrize('group,backbone,size', [
    ('mdeberta_seen', 'mdeberta', 92),
    ('mgte_seen', 'mgte', 76),
])
def test_encoder_seen_groups_match_table(group, backbone, size):
    langs = LANG_GROUPS[group]
    assert len(langs) == size
    assert len(set(langs)) == size  # no duplicates
    assert set(langs) == set(seen_langs(backbone))
    assert resolve_langs(group, '') == langs


def test_mdeberta_seen_reproduces_paper_seen_92():
    # mDeBERTa-v3 trains on CC100 like XLM-R, so this group is taken to be the
    # paper's Seen-92 (xpe.pdf sec. 4.2) -- the bridge to Table 1. See the
    # assumption caveat in src/sib200_meta.py's docstring.
    assert set(LANG_GROUPS['mdeberta_seen']) == set(seen_langs('xlmr'))
    assert len(seen_langs('xlmr')) == 92


def test_mgte_seen_relation_to_xlmr_seen():
    xlmr, mgte = set(seen_langs('xlmr')), set(LANG_GROUPS['mgte_seen'])
    # Seen by mGTE, not by XLM-R. NOTE tgl_Latn appears here only because the
    # legacy xlmr column marks it 0 while XLM-R's own tag list includes `tl`;
    # kept deliberately -- see src/sib200_meta.py.
    assert mgte - xlmr == {'ceb_Latn', 'hat_Latn', 'quy_Latn', 'tgl_Latn', 'yor_Latn'}
    # Seen by XLM-R, not by mGTE. 2 of these 21 are code-expansion bookkeeping
    # (nno_Latn vs the single `no` tag; azb_Arab vs the single `az` tag).
    assert len(xlmr - mgte) == 21
    assert set(MGTE_TOKENS_M) == mgte  # token counts cover exactly the seen set


def test_low_perf_group_partitions_the_benchmark():
    # 46 low-perf + 67 unseen-not-low-perf + 85 seen-wo-joshi5 = 198 = 205 - 7.
    lp = low_perf_langs()
    assert len(lp) == 46
    # the paper requires Low-Performing to be a strict subset of Unseen
    assert not (set(lp) & set(seen_langs('xlmr')))
    seen_wo_j5 = set(seen_langs('xlmr')) - set(JOSHI5)
    assert len(seen_wo_j5) == 85
    unseen = set(sib200_codes()) - set(seen_langs('xlmr'))
    assert len(unseen - set(lp)) == 67


def test_low_perf_lang_groups_match_the_table():
    assert LOW_PERF_LANG_GROUPS['mdeberta'] == low_perf_langs('mdeberta') == low_perf_langs()
    # yor_Latn is in mGTE's pretraining set (0.04M tokens), so it cannot also be
    # an unseen low-performer there -- add_seen() raises on the overlap.
    assert LOW_PERF_LANG_GROUPS['mgte'] == low_perf_langs('mgte')
    assert set(LOW_PERF_LANG_GROUPS['mgte']) == set(low_perf_langs()) - {'yor_Latn'}


@pytest.mark.parametrize('backbone', ['mdeberta', 'mgte'])
def test_low_perf_is_disjoint_from_seen(backbone):
    # unify_xlt_test_res.add_seen() raises if a lang is in both groups.
    seen_group = LANG_GROUPS[f'{backbone}_seen']
    assert not (set(LOW_PERF_LANG_GROUPS[backbone]) & set(seen_group))


def test_all_seen_groups_are_sib200_codes():
    codes = set(sib200_codes())
    for group in ('mdeberta_seen', 'mgte_seen', 'joshi5', 'enarzho'):
        assert set(LANG_GROUPS[group]) <= codes, group
    for backbone in ('mdeberta', 'mgte'):
        assert set(LOW_PERF_LANG_GROUPS[backbone]) <= codes, backbone


def _cfg(arch='bloom'):
    return SimpleNamespace(model=SimpleNamespace(architecture=arch))


def test_build_run_paths_uses_group_name_as_src_tag(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_xlt.evals_dir', lambda: tmp_path)
    run_dir, llm, src_tag, run_group, run_name = build_run_paths(
        tune_config=None, test_config=_cfg(), source_langs_sorted=sorted(JOSHI5),
        run_group='g', slurm_task_id=0, interactive=True, source_group='joshi5',
    )
    assert src_tag == 'joshi5'
    assert 'joshi5' in str(run_dir)


def test_build_run_paths_falls_back_to_joined_langs(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_xlt.evals_dir', lambda: tmp_path)
    _, _, src_tag, _, _ = build_run_paths(
        tune_config=None, test_config=_cfg(), source_langs_sorted=['ar', 'en'],
        run_group='g', slurm_task_id=0, interactive=True, source_group=None,
    )
    assert src_tag == 'ar-en'
