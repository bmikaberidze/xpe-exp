"""Unit tests for named source-language groups and run-dir naming in run_xlt."""
import pytest

from scripts.run_xlt import LANG_GROUPS, resolve_langs, src_tag_for

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
