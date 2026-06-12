"""Unit tests for excluding source langs from the discovered target set."""
from scripts.run_xlt import target_langs_excluding


ALL = ['arb_Arab', 'deu_Latn', 'eng_Latn', 'spa_Latn', 'zho_Hans']


def test_excludes_source_langs():
    assert target_langs_excluding(ALL, ['eng_Latn', 'spa_Latn']) == [
        'arb_Arab', 'deu_Latn', 'zho_Hans'
    ]


def test_empty_exclude_returns_all():
    assert target_langs_excluding(ALL, []) == ALL


def test_preserves_input_order():
    assert target_langs_excluding(['zho_Hans', 'eng_Latn', 'arb_Arab'], ['eng_Latn']) == [
        'zho_Hans', 'arb_Arab'
    ]


def test_exclude_not_present_is_noop():
    assert target_langs_excluding(ALL, ['xxx_Yyyy']) == ALL
