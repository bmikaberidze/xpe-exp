"""Tests for the Belebele item-disjoint fold splitter.

The splitter assigns each parallel item (keyed on (link, question_number),
identical across all 122 languages) to train/val/test for each of 3 folds, and
buckets reframed records into a DatasetDict with the canonical HF split names.

Correctness invariants the experiment depends on:
  - per fold: exactly train/val/test counts (500/100/300 for 900 items)
  - per fold: train/val/test are disjoint and cover every item
  - across folds: test blocks tile all items (each item tested exactly once)
  - deterministic and independent of input ordering (parallel-safe)
"""

import random

from scripts.datasets.split_bebe_folds import assign_fold_splits, fold_datasetdict


def _keys(n=900):
    # Belebele-shaped keys: (link, question_number). 90 passages x 10 questions.
    return [(f"https://example.org/passage_{p}", q)
            for p in range(n // 10) for q in range(1, 11)]


# --- assign_fold_splits -----------------------------------------------------

def test_per_fold_counts_are_500_100_300():
    folds = assign_fold_splits(_keys(), n_splits=3, train=500, val=100, test=300)
    assert len(folds) == 3
    for fmap in folds:
        counts = {s: sum(1 for v in fmap.values() if v == s) for s in ("train", "val", "test")}
        assert counts == {"train": 500, "val": 100, "test": 300}


def test_each_fold_partitions_all_items_disjointly():
    keys = _keys()
    folds = assign_fold_splits(keys, n_splits=3, train=500, val=100, test=300)
    for fmap in folds:
        assert set(fmap) == set(keys)
        assert set(fmap.values()) <= {"train", "val", "test"}


def test_test_blocks_tile_all_items_across_folds():
    keys = _keys()
    folds = assign_fold_splits(keys, n_splits=3, train=500, val=100, test=300)
    test_of = [{k for k, s in fmap.items() if s == "test"} for fmap in folds]
    assert test_of[0] & test_of[1] == set()
    assert test_of[0] & test_of[2] == set()
    assert test_of[1] & test_of[2] == set()
    assert test_of[0] | test_of[1] | test_of[2] == set(keys)


def test_deterministic_and_order_independent():
    keys = _keys()
    shuffled = keys[:]
    random.Random(123).shuffle(shuffled)
    a = assign_fold_splits(keys, n_splits=3, train=500, val=100, test=300, seed=7)
    b = assign_fold_splits(shuffled, n_splits=3, train=500, val=100, test=300, seed=7)
    assert a == b


# --- fold_datasetdict -------------------------------------------------------

def _records():
    return [
        {"text": "t0", "answer_label": "A", "link": "L1", "question_number": 1},
        {"text": "t1", "answer_label": "B", "link": "L1", "question_number": 2},
        {"text": "t2", "answer_label": "C", "link": "L2", "question_number": 1},
        {"text": "t3", "answer_label": "D", "link": "L2", "question_number": 2},
    ]


def test_uses_canonical_hf_split_names():
    fold_map = {("L1", 1): "train", ("L1", 2): "val", ("L2", 1): "test", ("L2", 2): "train"}
    dd = fold_datasetdict(_records(), fold_map)
    assert set(dd.keys()) == {"train", "validation", "test"}


def test_records_routed_to_their_mapped_split():
    fold_map = {("L1", 1): "train", ("L1", 2): "val", ("L2", 1): "test", ("L2", 2): "train"}
    dd = fold_datasetdict(_records(), fold_map)
    assert {r["text"] for r in dd["train"]} == {"t0", "t3"}
    assert [r["text"] for r in dd["validation"]] == ["t1"]
    assert [r["text"] for r in dd["test"]] == ["t2"]


def test_fields_preserved():
    fold_map = {("L1", 1): "test", ("L1", 2): "test", ("L2", 1): "test", ("L2", 2): "test"}
    dd = fold_datasetdict(_records(), fold_map)
    row = dd["test"][0]
    assert set(row) >= {"text", "answer_label", "link", "question_number"}
