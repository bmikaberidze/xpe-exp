"""Unit tests for the SIB-200 download/shaping logic (no network)."""
import pytest

from scripts.datasets.download_sib import COLUMN_RENAMES, HF_DATASET, language_configs, shape_split
from src.sib200_meta import LABEL_NAMES


def test_hf_dataset_and_renames():
    assert HF_DATASET == 'Davlan/sib200'
    # `index_id` is deliberately NOT renamed: run_xlt injects its own `task_ids`
    # (the language index) and `datasets` raises on a duplicate column.
    assert COLUMN_RENAMES == {'category': 'labels', 'text': 'inputs'}


def test_language_configs_drops_the_junk_zip_config():
    # the hub lists 206 configs; 'nqo_Nkoo.zip' is not a language
    raw = ['eng_Latn', 'nqo_Nkoo', 'nqo_Nkoo.zip', 'kat_Geor']
    assert language_configs(raw) == ['eng_Latn', 'kat_Geor', 'nqo_Nkoo']


def test_language_configs_rejects_unknown_codes():
    with pytest.raises(ValueError, match='not in the SIB-200 table'):
        language_configs(['eng_Latn', 'xxx_Zzzz'])


def test_shape_split_renames_and_maps_labels():
    records = [
        {'index_id': 431, 'category': 'geography', 'text': 'Turkey is encircled by seas.'},
        {'index_id': 7, 'category': 'science/technology', 'text': 'Protons are positive.'},
    ]
    assert shape_split(records) == [
        {'index_id': 431, 'labels': LABEL_NAMES.index('geography'),
         'inputs': 'Turkey is encircled by seas.'},
        {'index_id': 7, 'labels': 0, 'inputs': 'Protons are positive.'},
    ]


def test_shape_split_label_ids_follow_canonical_order():
    # https://huggingface.co/datasets/Davlan/sib200/raw/main/data/eng_Latn/labels.txt
    records = [{'index_id': i, 'category': name, 'text': name}
               for i, name in enumerate(LABEL_NAMES)]
    assert [r['labels'] for r in shape_split(records)] == list(range(7))


def test_shape_split_never_emits_task_ids():
    # run_xlt.load_concat_dataset does add_column('task_ids', ...) on every split;
    # a pre-existing column of that name makes `datasets` raise.
    shaped = shape_split([{'index_id': 1, 'category': 'travel', 'text': 'x'}])
    assert 'task_ids' not in shaped[0]
    assert set(shaped[0]) == {'index_id', 'labels', 'inputs'}


def test_shape_split_rejects_unknown_category():
    with pytest.raises(ValueError, match='unknown category'):
        shape_split([{'index_id': 1, 'category': 'weather', 'text': 'hot'}])
