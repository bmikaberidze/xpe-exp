"""Unit tests for named source-language groups and run-dir naming in run_xlt."""
import pytest
from types import SimpleNamespace

from scripts.run_xlt import LANG_GROUPS, resolve_langs, src_tag_for
from scripts.run_xlt import build_run_paths

ANCHORS7 = ['eng_Latn', 'spa_Latn', 'fra_Latn', 'zho_Hans', 'hin_Deva', 'arb_Arab', 'ind_Latn']


def test_anchors7_group_defined():
    assert LANG_GROUPS['anchors7'] == ANCHORS7


def test_resolve_group_name():
    assert resolve_langs('anchors7', '') == ANCHORS7


def test_resolve_csv_when_no_group():
    assert resolve_langs(None, 'en, ru ,zh') == ['en', 'ru', 'zh']


def test_resolve_unknown_group_raises():
    with pytest.raises(ValueError, match='anchors7'):  # message lists known groups
        resolve_langs('nope', '')


def test_resolve_both_set_raises():
    with pytest.raises(ValueError, match='both'):
        resolve_langs('anchors7', 'en,ru')


def test_resolve_neither_set_returns_empty():
    assert resolve_langs(None, '') == []


def test_src_tag_uses_group_name():
    assert src_tag_for('anchors7', sorted(ANCHORS7)) == 'anchors7'


def test_src_tag_falls_back_to_joined_sorted_langs():
    assert src_tag_for(None, ['ar', 'en', 'zh']) == 'ar-en-zh'


def test_src_tag_zero_when_empty():
    assert src_tag_for(None, []) == 'zero'


def _cfg(arch='bloom'):
    return SimpleNamespace(model=SimpleNamespace(architecture=arch))


def test_build_run_paths_uses_group_name_as_src_tag(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_xlt.evals_dir', lambda: tmp_path)
    run_dir, llm, src_tag, run_group, run_name = build_run_paths(
        tune_config=None, test_config=_cfg(), source_langs_sorted=sorted(ANCHORS7),
        run_group='g', slurm_task_id=0, interactive=True, source_group='anchors7',
    )
    assert src_tag == 'anchors7'
    assert 'anchors7' in str(run_dir)


def test_build_run_paths_falls_back_to_joined_langs(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.run_xlt.evals_dir', lambda: tmp_path)
    _, _, src_tag, _, _ = build_run_paths(
        tune_config=None, test_config=_cfg(), source_langs_sorted=['ar', 'en'],
        run_group='g', slurm_task_id=0, interactive=True, source_group=None,
    )
    assert src_tag == 'ar-en'
