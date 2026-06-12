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
